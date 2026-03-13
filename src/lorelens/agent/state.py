"""Agent state and structured-output schemas."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from lorelens.models import Citation, DocType, RetrievedChunk, SearchFilters


class QueryAnalysis(BaseModel):
    """Structured output of the analyze node."""

    standalone_question: str = Field(description="The question rewritten to be self-contained.")
    in_scope: bool = Field(description="Whether the question is about the documented products.")
    complexity: Literal["simple", "complex"] = "simple"
    product: str | None = Field(default=None, description="Product filter, if stated or implied.")
    version: str | None = Field(default=None, description="Version filter like 'v2', if stated.")
    doc_type: DocType | None = Field(default=None, description="Doc type filter, if implied.")
    search_queries: list[str] = Field(
        default_factory=list, description="1-4 focused search queries covering the question."
    )


class GradeResult(BaseModel):
    relevant: list[int] = Field(
        default_factory=list, description="Numbers of the chunks that are relevant."
    )
    sufficient: bool = Field(description="True if the relevant chunks fully answer the question.")
    missing: str = Field(default="", description="What information is still missing, if any.")


class RewriteResult(BaseModel):
    queries: list[str] = Field(description="1-3 new search queries.")


class ChatTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict, total=False):
    # ---- input
    question: str
    history: list[ChatTurn]
    filters: SearchFilters  # caller-provided filters (override the model's guesses)

    # ---- working memory
    analysis: QueryAnalysis
    messages: Annotated[list[AnyMessage], add_messages]  # research tool-calling loop
    search_queries: list[str]
    tried_queries: list[str]
    documents: list[RetrievedChunk]
    grade: GradeResult
    rewrites: int
    tool_calls: int
    node_timings: dict[str, float]

    # ---- output
    answer: str
    citations: list[Citation]
    grounded: bool
