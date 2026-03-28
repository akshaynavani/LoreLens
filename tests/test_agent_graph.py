"""End-to-end graph tests with scripted LLMs and the real local tools over a fake index."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from lorelens.agent.graph import LoreLensAgent, build_graph
from lorelens.agent.nodes import AgentNodes, extract_citations, parse_tool_payload
from lorelens.agent.prompts import OUT_OF_SCOPE_ANSWER
from lorelens.agent.state import GradeResult, QueryAnalysis
from lorelens.models import SearchFilters
from lorelens.tools.local import build_local_tools
from tests.conftest import FakeAnswerLLM, FakeFastLLM, FakeSearchService, make_chunk


def _agent(settings, prompts, fast, answer=None, service=None) -> tuple[LoreLensAgent, FakeSearchService]:
    service = service or FakeSearchService()
    nodes = AgentNodes(
        fast_llm=fast,
        answer_llm=answer or FakeAnswerLLM(),
        tools=build_local_tools(service),  # type: ignore[arg-type]
        prompts=prompts,
        settings=settings,
        catalog_fn=service.catalog,
    )
    return LoreLensAgent(build_graph(nodes), settings), service


def _analysis(**kw) -> QueryAnalysis:
    base = dict(
        standalone_question="How long do Nimbus v2 tokens last?",
        in_scope=True,
        complexity="simple",
        product="nimbus",
        version="v2",
        search_queries=["nimbus v2 access token expiry"],
    )
    base.update(kw)
    return QueryAnalysis(**base)


async def test_simple_question_skips_research_and_cites(settings, prompts):
    fast = FakeFastLLM(_analysis())
    agent, service = _agent(settings, prompts, fast)

    answer, state = await agent.arun("how long do tokens last?")

    assert "3600" in answer.answer
    assert [c.index for c in answer.citations] == [1]
    assert answer.citations[0].url.endswith("/page1")
    assert answer.grounded
    assert "research" not in fast.calls  # fast path: confident single-hop question
    query, filters = service.queries[0]
    assert query == "nimbus v2 access token expiry"
    assert filters == SearchFilters(product="nimbus", version="v2")
    assert {"analyze", "prefetch", "grade", "generate"} <= set(state["node_timings"])


async def test_complex_question_uses_tool_calls(settings, prompts):
    tool_turn = AIMessage(
        "",
        tool_calls=[{"name": "read_page", "args": {"doc_id": "doc1"}, "id": "call_1"}],
    )
    fast = FakeFastLLM(_analysis(complexity="complex"), research_turns=[tool_turn])
    agent, _ = _agent(settings, prompts, fast)

    answer, state = await agent.arun("compare token handling")

    assert answer.tool_calls == 1
    assert fast.calls.count("research") == 2  # tool call, then DONE
    assert any(d.chunk_id == "doc1#page" for d in state["documents"])


async def test_insufficient_evidence_triggers_bounded_rewrites(settings, prompts):
    fast = FakeFastLLM(
        _analysis(complexity="complex"),
        grades=[GradeResult(relevant=[], sufficient=False, missing="refresh behaviour")],
    )
    agent, service = _agent(settings, prompts, fast)

    answer, state = await agent.arun("how do I refresh tokens?")

    assert answer.rewrites == settings.max_rewrites
    assert any(q.startswith("rewritten query") for q, _ in service.queries)
    assert not answer.citations  # grader rejected everything -> not found
    assert "couldn't find" in answer.answer


async def test_caller_filters_override_model_guesses(settings, prompts):
    fast = FakeFastLLM(_analysis(product="unknown-product", version=None))
    agent, service = _agent(settings, prompts, fast)

    await agent.arun("tokens?", filters=SearchFilters(version="v1"))

    _, filters = service.queries[0]
    assert filters.product is None  # not in the catalog -> dropped
    assert filters.version == "v1"


async def test_out_of_scope_short_circuits(settings, prompts):
    fast = FakeFastLLM(_analysis(in_scope=False))
    agent, service = _agent(settings, prompts, fast)

    answer, _ = await agent.arun("banana bread recipe?")

    assert answer.answer == OUT_OF_SCOPE_ANSWER
    assert service.queries == []


async def test_stream_yields_tokens_or_updates(settings, prompts):
    agent, _ = _agent(settings, prompts, FakeFastLLM(_analysis()))
    kinds = [kind async for kind, _ in agent.astream("tokens?")]
    assert "update" in kinds


def test_parse_tool_payload_and_citations():
    payload = json.dumps({"results": [make_chunk(1).model_dump(), {"bad": "row"}]})
    chunks = parse_tool_payload(payload)
    assert [c.chunk_id for c in chunks] == ["doc1#0"]
    assert parse_tool_payload("not json") == []

    cites = extract_citations("A [2]. B [1][2]. C [9].", [make_chunk(1), make_chunk(2)])
    assert [c.index for c in cites] == [2, 1]


async def test_collect_ignores_tool_results_from_before_a_rewrite(settings, prompts):
    nodes = AgentNodes(
        fast_llm=FakeFastLLM(_analysis()),
        answer_llm=FakeAnswerLLM(),
        tools=build_local_tools(FakeSearchService()),  # type: ignore[arg-type]
        prompts=prompts,
        settings=settings,
    )
    old = json.dumps({"results": [make_chunk(9).model_dump()]})
    new = json.dumps({"results": [make_chunk(3).model_dump()]})
    call = {"name": "search_docs", "args": {"query": "q"}, "id": "c1"}
    messages = [
        AIMessage("", tool_calls=[call]),
        ToolMessage(old, tool_call_id="c1"),
        AIMessage("DONE"),
        HumanMessage("Evidence is still missing: x. Continue gathering evidence, then reply DONE."),
        AIMessage("", tool_calls=[{**call, "id": "c2"}]),
        ToolMessage(new, tool_call_id="c2"),
        AIMessage("DONE"),
    ]

    update = await nodes.collect({"messages": messages, "documents": []}, {})

    assert [d.chunk_id for d in update["documents"]] == ["doc3#0"]  # doc9 was graded already
