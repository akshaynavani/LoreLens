"""LangGraph wiring and the `LoreLensAgent` facade used by the CLI, API and evals.

    START -> analyze --(out of scope)--> out_of_scope -> END
                  \\
                   -> prefetch --(simple & confident)--> grade
                          \\                                ^
                           -> research <-> tools           |
                                  \\-> collect -------------+
    grade --(sufficient | budget spent)--> generate -> END
          \\--(insufficient)--> rewrite -> prefetch
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from lorelens.agent.nodes import AgentNodes, timed
from lorelens.agent.prompts import PromptProvider
from lorelens.agent.state import AgentState, ChatTurn
from lorelens.config import Settings, get_settings
from lorelens.models import Answer, SearchFilters


def build_graph(nodes: AgentNodes):  # noqa: ANN201 - CompiledStateGraph
    g = StateGraph(AgentState)
    g.add_node("analyze", timed("analyze")(nodes.analyze))
    g.add_node("out_of_scope", nodes.out_of_scope)
    g.add_node("prefetch", timed("prefetch")(nodes.prefetch))
    g.add_node("research", timed("research")(nodes.research))
    g.add_node("tools", ToolNode(nodes.tools, handle_tool_errors=True))
    g.add_node("collect", nodes.collect)
    g.add_node("grade", timed("grade")(nodes.grade))
    g.add_node("rewrite", timed("rewrite")(nodes.rewrite))
    g.add_node("generate", timed("generate")(nodes.generate))

    g.add_edge(START, "analyze")
    g.add_conditional_edges(
        "analyze",
        lambda s: "prefetch" if s["analysis"].in_scope else "out_of_scope",
        {"prefetch": "prefetch", "out_of_scope": "out_of_scope"},
    )
    g.add_edge("out_of_scope", END)
    g.add_conditional_edges(
        "prefetch", nodes.route_after_prefetch, {"grade": "grade", "research": "research"}
    )
    g.add_conditional_edges(
        "research", nodes.route_after_research, {"tools": "tools", "collect": "collect"}
    )
    g.add_edge("tools", "research")
    g.add_edge("collect", "grade")
    g.add_conditional_edges(
        "grade", nodes.route_after_grade, {"generate": "generate", "rewrite": "rewrite"}
    )
    g.add_edge("rewrite", "prefetch")
    g.add_edge("generate", END)
    return g.compile()


class LoreLensAgent:
    def __init__(self, graph: Any, settings: Settings) -> None:
        self.graph = graph
        self.settings = settings

    # ------------------------------------------------------------ construction
    @classmethod
    async def create(
        cls,
        settings: Settings | None = None,
        *,
        tools: Sequence[BaseTool] | None = None,
        fast_llm: BaseChatModel | None = None,
        answer_llm: BaseChatModel | None = None,
        prompts: PromptProvider | None = None,
        catalog_fn: Any = None,
    ) -> LoreLensAgent:
        settings = settings or get_settings()
        if tools is None:
            if settings.tool_mode == "mcp":
                from lorelens.tools.mcp_client import load_mcp_tools

                tools = await load_mcp_tools(settings)
            else:
                from lorelens.retrieval.service import get_search_service
                from lorelens.tools.local import build_local_tools

                service = get_search_service()
                tools = build_local_tools(service)
                catalog_fn = catalog_fn or service.catalog
        if fast_llm is None or answer_llm is None:
            from lorelens.agent.llm import get_chat_model

            fast_llm = fast_llm or get_chat_model("fast")
            answer_llm = answer_llm or get_chat_model("answer")
        if prompts is None:
            from lorelens.observability import get_prompt_provider

            prompts = get_prompt_provider()
        if catalog_fn is None:
            catalog_fn = _catalog_from_manifest(settings)

        nodes = AgentNodes(
            fast_llm=fast_llm,
            answer_llm=answer_llm,
            tools=tools,
            prompts=prompts,
            settings=settings,
            catalog_fn=catalog_fn,
        )
        return cls(build_graph(nodes), settings)

    # ------------------------------------------------------------------- input
    @staticmethod
    def _input(
        question: str, history: list[ChatTurn] | None, filters: SearchFilters | None
    ) -> AgentState:
        return {
            "question": question,
            "history": history or [],
            "filters": filters or SearchFilters(),
            "messages": [],
        }

    def _config(self, config: RunnableConfig | None) -> RunnableConfig:
        cfg: RunnableConfig = dict(config or {})  # type: ignore[assignment]
        # Each tool round trip is research+tools (2 steps); leave headroom for rewrites.
        cfg.setdefault(
            "recursion_limit",
            12 + 2 * self.settings.max_tool_calls + 6 * self.settings.max_rewrites,
        )
        return cfg

    # --------------------------------------------------------------------- run
    async def arun(
        self,
        question: str,
        *,
        history: list[ChatTurn] | None = None,
        filters: SearchFilters | None = None,
        config: RunnableConfig | None = None,
    ) -> tuple[Answer, AgentState]:
        start = time.perf_counter()
        state: AgentState = await self.graph.ainvoke(
            self._input(question, history, filters), self._config(config)
        )
        answer = Answer(
            question=question,
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            grounded=state.get("grounded", False),
            rewrites=state.get("rewrites", 0),
            tool_calls=state.get("tool_calls", 0),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return answer, state

    async def astream(
        self,
        question: str,
        *,
        history: list[ChatTurn] | None = None,
        filters: SearchFilters | None = None,
        config: RunnableConfig | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Yield ("update", {node: update}) for node progress and ("token", str) for
        answer tokens streamed from the generate node."""
        async for mode, chunk in self.graph.astream(
            self._input(question, history, filters),
            self._config(config),
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                message, meta = chunk
                if meta.get("langgraph_node") == "generate" and getattr(message, "content", None):
                    content = message.content
                    yield "token", content if isinstance(content, str) else str(content)
            else:
                yield "update", chunk


def _catalog_from_manifest(settings: Settings):  # noqa: ANN202
    from lorelens.ingestion.manifest import Manifest

    def _fn() -> dict:
        return Manifest.load(
            settings.manifest_path, settings.pinecone_namespace, settings.embedding_model
        ).catalog()

    return _fn
