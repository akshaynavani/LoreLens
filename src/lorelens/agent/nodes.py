"""LangGraph node implementations."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from functools import wraps
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from lorelens.agent.prompts import OUT_OF_SCOPE_ANSWER, PromptProvider
from lorelens.agent.state import AgentState, GradeResult, QueryAnalysis, RewriteResult
from lorelens.config import Settings
from lorelens.log import get_logger
from lorelens.models import Citation, RetrievedChunk, SearchFilters

logger = get_logger(__name__)

NodeFn = Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]
NOT_FOUND_ANSWER = (
    "I couldn't find this in the documentation. Try rephrasing with the product name, "
    "the feature or API name, or the exact error message."
)
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


# ----------------------------------------------------------------------------- helpers
def timed(name: str) -> Callable[[NodeFn], NodeFn]:
    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        async def wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            start = time.perf_counter()
            update = await fn(state, config)
            update = dict(update or {})
            update["node_timings"] = {name: (time.perf_counter() - start) * 1000}
            return update

        return wrapper

    return decorator


def tool_output_text(result: Any) -> str:
    """Normalize tool results from LangChain tools and MCP adapters to a string."""
    if isinstance(result, ToolMessage):
        result = result.content
    if isinstance(result, tuple) and result:
        result = result[0]  # (content, artifact)
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "\n".join(parts)
    return str(result)


def parse_tool_payload(text: str) -> list[RetrievedChunk]:
    """Turn `search_docs` / `read_page` JSON payloads into RetrievedChunks."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    if "results" in data:
        chunks = []
        for r in data["results"]:
            try:
                chunks.append(RetrievedChunk(**r))
            except Exception:  # noqa: BLE001 - skip malformed rows
                continue
        return chunks
    if data.get("found") and data.get("text"):
        return [
            RetrievedChunk(
                chunk_id=f"{data['doc_id']}#page",
                doc_id=data["doc_id"],
                text=data["text"],
                score=1.0,
                title=data.get("title"),
                url=data.get("url"),
                source=data.get("source"),
                product=data.get("product"),
                version=data.get("version"),
            )
        ]
    return []


def merge_documents(*groups: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}
    for group in groups:
        for c in group:
            if c.chunk_id not in merged or c.best_score > merged[c.chunk_id].best_score:
                merged[c.chunk_id] = c
    return sorted(merged.values(), key=lambda c: c.best_score, reverse=True)


def format_history(history: Sequence[dict] | None, max_turns: int = 6) -> str:
    if not history:
        return "(none)"
    turns = list(history)[-max_turns:]
    return "\n".join(f"{t['role']}: {t['content'][:1000]}" for t in turns)


def _label(c: RetrievedChunk) -> str:
    where = " > ".join(p for p in (c.title, c.section) if p) or c.doc_id
    tags = " ".join(p for p in (c.product, c.version) if p)
    return f"{where} ({tags})" if tags else where


def extract_citations(answer: str, sources: Sequence[RetrievedChunk]) -> list[Citation]:
    seen: list[int] = []
    for m in _CITATION_RE.finditer(answer):
        idx = int(m.group(1))
        if 1 <= idx <= len(sources) and idx not in seen:
            seen.append(idx)
    return [
        Citation(
            index=i,
            title=sources[i - 1].title,
            url=sources[i - 1].url,
            source=sources[i - 1].source,
            section=sources[i - 1].section,
            chunk_id=sources[i - 1].chunk_id,
        )
        for i in seen
    ]


# ------------------------------------------------------------------------------- nodes
class AgentNodes:
    def __init__(
        self,
        *,
        fast_llm: BaseChatModel,
        answer_llm: BaseChatModel,
        tools: Sequence[BaseTool],
        prompts: PromptProvider,
        settings: Settings,
        catalog_fn: Callable[[], dict] | None = None,
    ) -> None:
        self.fast_llm = fast_llm
        self.answer_llm = answer_llm
        self.tools = list(tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        if "search_docs" not in self.tools_by_name:
            raise ValueError("A `search_docs` tool is required")
        self.prompts = prompts
        self.settings = settings
        self.catalog_fn = catalog_fn or (lambda: {})

    @staticmethod
    def _cfg(config: RunnableConfig, metadata: dict[str, Any], run_name: str) -> RunnableConfig:
        merged = dict(config or {})
        merged["metadata"] = {**(config or {}).get("metadata", {}), **metadata}
        merged["run_name"] = run_name
        return merged

    # ---------------------------------------------------------------- analyze
    async def analyze(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        try:
            catalog = await asyncio.to_thread(self.catalog_fn)
        except Exception:  # noqa: BLE001 - catalog is advisory only
            catalog = {}
        prompt = self.prompts.render(
            "lorelens-analyze",
            catalog=json.dumps(catalog) if catalog else "(unknown)",
            history=format_history(state.get("history")),
            question=state["question"],
        )
        structured = self.fast_llm.with_structured_output(QueryAnalysis)
        analysis: QueryAnalysis = await structured.ainvoke(
            [HumanMessage(prompt.text)], self._cfg(config, prompt.metadata, "analyze_query")
        )

        product = analysis.product.lower() if analysis.product else None
        if catalog and product and product not in catalog:
            product = None  # the model guessed a product that is not indexed
        caller = state.get("filters") or SearchFilters()
        filters = SearchFilters(
            product=caller.product or product,
            version=caller.version or (analysis.version.lower() if analysis.version else None),
            doc_type=caller.doc_type or analysis.doc_type,
            tags=caller.tags,
        )
        queries = [q for q in analysis.search_queries if q.strip()][:4] or [
            analysis.standalone_question
        ]
        return {
            "analysis": analysis,
            "filters": filters,
            "search_queries": queries,
            "tried_queries": [],
            "documents": [],
            "rewrites": 0,
            "tool_calls": 0,
        }

    async def out_of_scope(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        return {"answer": OUT_OF_SCOPE_ANSWER, "citations": [], "grounded": False}

    # --------------------------------------------------------------- prefetch
    async def prefetch(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Run every planned query in parallel through the search tool, skipping the LLM
        tool-calling round trip for the first retrieval pass (main latency saver)."""
        search = self.tools_by_name["search_docs"]
        filters = (state.get("filters") or SearchFilters()).model_dump(
            exclude_none=True, exclude={"tags"}
        )
        tried = set(state.get("tried_queries", []))
        queries = [q for q in state.get("search_queries", []) if q not in tried]
        calls = [
            search.ainvoke({"query": q, "top_k": self.settings.top_k, **filters}, config)
            for q in queries
        ]
        results = await asyncio.gather(*calls, return_exceptions=True)
        found: list[RetrievedChunk] = []
        for q, r in zip(queries, results, strict=True):
            if isinstance(r, Exception):
                logger.warning("search_docs failed for %r: %s", q, r)
                continue
            found.extend(parse_tool_payload(tool_output_text(r)))
        return {
            "documents": merge_documents(state.get("documents", []), found),
            "tried_queries": [*state.get("tried_queries", []), *queries],
        }

    def route_after_prefetch(self, state: AgentState) -> str:
        docs = state.get("documents", [])
        analysis = state.get("analysis")
        if (
            docs
            and analysis is not None
            and analysis.complexity == "simple"
            and state.get("rewrites", 0) == 0
            and docs[0].best_score >= 0.5
        ):
            return "grade"  # confident single-hop question: skip the research loop
        return "research"

    # --------------------------------------------------------------- research
    async def research(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        if state.get("tool_calls", 0) >= self.settings.max_tool_calls:
            return {}
        docs = state.get("documents", [])[:12]
        evidence = "\n".join(
            f"- {_label(c)} [doc_id={c.doc_id}] score={c.best_score:.2f}: {c.text[:160]!r}"
            for c in docs
        ) or "(nothing yet)"
        filters = (state.get("filters") or SearchFilters()).model_dump(exclude_none=True)
        prompt = self.prompts.render(
            "lorelens-research",
            question=state["analysis"].standalone_question,
            filters=json.dumps(filters) if filters else "(none)",
            evidence=evidence + "\n\nQueries already run:\n"
            + "\n".join(f"- {q}" for q in state.get("tried_queries", [])),
        )
        messages = [
            SystemMessage(prompt.text),
            HumanMessage("Gather any missing evidence, then reply DONE."),
            *state.get("messages", []),
        ]
        response = await self.fast_llm.bind_tools(self.tools).ainvoke(
            messages, self._cfg(config, prompt.metadata, "research")
        )
        n_calls = len(getattr(response, "tool_calls", []) or [])
        return {"messages": [response], "tool_calls": state.get("tool_calls", 0) + n_calls}

    @staticmethod
    def route_after_research(state: AgentState) -> str:
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return "tools"
        return "collect"

    async def collect(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        found: list[RetrievedChunk] = []
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage):
                found.extend(parse_tool_payload(tool_output_text(msg)))
        return {"documents": merge_documents(state.get("documents", []), found)}

    # ------------------------------------------------------------------ grade
    async def grade(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        docs = state.get("documents", [])[:12]
        if not docs:
            return {
                "documents": [],
                "grade": GradeResult(sufficient=False, missing="no documents were found"),
            }
        numbered = "\n\n".join(
            f"[{i}] {_label(c)}\n{c.text[:900]}" for i, c in enumerate(docs, start=1)
        )
        prompt = self.prompts.render(
            "lorelens-grade", question=state["analysis"].standalone_question, documents=numbered
        )
        structured = self.fast_llm.with_structured_output(GradeResult)
        result: GradeResult = await structured.ainvoke(
            [HumanMessage(prompt.text)], self._cfg(config, prompt.metadata, "grade_documents")
        )
        keep = [docs[i - 1] for i in dict.fromkeys(result.relevant) if 1 <= i <= len(docs)]
        return {"documents": keep, "grade": result}

    def route_after_grade(self, state: AgentState) -> str:
        grade = state.get("grade")
        if grade is not None and grade.sufficient:
            return "generate"
        if state.get("rewrites", 0) >= self.settings.max_rewrites:
            return "generate"
        return "rewrite"

    # ---------------------------------------------------------------- rewrite
    async def rewrite(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        grade = state.get("grade")
        missing = grade.missing if grade else ""
        tried = state.get("tried_queries", [])
        prompt = self.prompts.render(
            "lorelens-rewrite",
            question=state["analysis"].standalone_question,
            missing=missing or "(unspecified)",
            tried="\n".join(f"- {q}" for q in tried),
        )
        structured = self.fast_llm.with_structured_output(RewriteResult)
        result: RewriteResult = await structured.ainvoke(
            [HumanMessage(prompt.text)], self._cfg(config, prompt.metadata, "rewrite_query")
        )
        queries = [q for q in result.queries if q.strip() and q not in tried][:3]
        if not queries:
            queries = [f"{state['analysis'].standalone_question} {missing}".strip()]
        note = HumanMessage(
            f"Evidence is still missing: {missing or 'unspecified'}. "
            f"New searches were run: {queries}. Continue gathering evidence, then reply DONE."
        )
        return {
            "search_queries": queries,
            "rewrites": state.get("rewrites", 0) + 1,
            "messages": [note],
        }

    # --------------------------------------------------------------- generate
    async def generate(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        sources = state.get("documents", [])[: self.settings.top_k]
        if not sources:
            return {"answer": NOT_FOUND_ANSWER, "citations": [], "grounded": False}
        sources_text = "\n\n".join(
            f"[{i}] {_label(c)}\nURL: {c.url or c.source or 'n/a'}\n{c.text}"
            for i, c in enumerate(sources, start=1)
        )
        question = state["analysis"].standalone_question
        prompt = self.prompts.render(
            "lorelens-answer",
            sources=sources_text,
            history=format_history(state.get("history")),
            question=question,
        )
        response = await self.answer_llm.ainvoke(
            [SystemMessage(prompt.text), HumanMessage(question)],
            self._cfg(config, prompt.metadata, "generate_answer"),
        )
        answer = tool_output_text(response.content)
        citations = extract_citations(answer, sources)
        return {"answer": answer, "citations": citations, "grounded": bool(citations)}
