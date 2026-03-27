"""Shared fakes: no network, no API keys, no Pinecone."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from lorelens.agent.prompts import LocalPromptProvider
from lorelens.agent.state import GradeResult, QueryAnalysis, RewriteResult
from lorelens.config import Settings
from lorelens.models import RetrievedChunk, SearchFilters


def make_chunk(i: int, score: float = 0.9, **kw: Any) -> RetrievedChunk:
    defaults = dict(
        chunk_id=f"doc{i}#0",
        doc_id=f"doc{i}",
        text=f"Chunk {i}: access tokens expire after 3600 seconds.",
        score=score,
        title=f"Page {i}",
        source=f"nimbus/v2/guides/page{i}.md",
        url=f"https://docs.example.com/nimbus/v2/guides/page{i}",
        section="Auth > Tokens",
        product="nimbus",
        version="v2",
        doc_type="guide",
    )
    defaults.update(kw)
    return RetrievedChunk(**defaults)


class FakeSearchService:
    """Duck-typed DocSearchService used by the real local tools."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else [make_chunk(1), make_chunk(2, 0.7)]
        self.queries: list[tuple[str, SearchFilters | None]] = []

    def search(self, query, filters=None, top_k=None, namespace=None, **_):  # noqa: ANN001, ANN201
        self.queries.append((query, filters))
        return self.chunks[: top_k or len(self.chunks)]

    def get_page(self, doc_id: str) -> dict:
        return {"doc_id": doc_id, "found": True, "title": "Full page", "text": "Full page text."}

    def catalog(self) -> dict:
        return {"nimbus": {"v2": 4, "v1": 1}, "relay": {"v1": 2}}


class _Responder:
    def __init__(self, fn: Callable[[Any], Any]) -> None:
        self.fn = fn

    async def ainvoke(self, messages: Any, config: Any = None) -> Any:
        return self.fn(messages)


class FakeFastLLM:
    """Scripted stand-in for the fast chat model (structured output + tool calling)."""

    def __init__(
        self,
        analysis: QueryAnalysis,
        grades: list[GradeResult] | None = None,
        research_turns: list[AIMessage] | None = None,
    ) -> None:
        self.analysis = analysis
        self.grades = list(grades or [GradeResult(relevant=[1, 2], sufficient=True)])
        self.research_turns = list(research_turns or [])
        self.calls: list[str] = []

    def with_structured_output(self, schema: type) -> _Responder:
        def respond(_messages: Any) -> Any:
            self.calls.append(schema.__name__)
            if schema is QueryAnalysis:
                return self.analysis
            if schema is GradeResult:
                return self.grades.pop(0) if len(self.grades) > 1 else self.grades[0]
            if schema is RewriteResult:
                return RewriteResult(queries=[f"rewritten query {len(self.calls)}"])
            raise AssertionError(f"unexpected schema {schema}")

        return _Responder(respond)

    def bind_tools(self, tools: Any) -> _Responder:
        def respond(_messages: Any) -> AIMessage:
            self.calls.append("research")
            return self.research_turns.pop(0) if self.research_turns else AIMessage("DONE")

        return _Responder(respond)


class FakeAnswerLLM:
    def __init__(self, text: str = "Access tokens expire after 3600 seconds [1].") -> None:
        self.text = text
        self.prompts: list[Any] = []

    async def ainvoke(self, messages: Any, config: Any = None) -> AIMessage:
        self.prompts.append(messages)
        return AIMessage(self.text)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        max_rewrites=1,
        max_tool_calls=3,
        top_k=5,
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )


@pytest.fixture
def prompts() -> LocalPromptProvider:
    return LocalPromptProvider()


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    import lorelens.observability as obs

    monkeypatch.setattr(obs, "get_langfuse", lambda: None)
