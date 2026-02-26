"""Shared domain models used across ingestion, retrieval, agent and API layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

DocType = Literal["api_reference", "guide", "tutorial", "release_notes", "faq", "concept", "other"]


class SourceDocument(BaseModel):
    """A single documentation page as loaded from disk or the web."""

    doc_id: str
    text: str
    source: str  # file path or URL
    url: str | None = None
    title: str | None = None
    format: Literal["markdown", "html", "rst", "text"] = "markdown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    last_modified: datetime | None = None


class DocMetadata(BaseModel):
    """Normalized, filterable metadata attached to every chunk in Pinecone."""

    doc_id: str
    source: str
    url: str | None = None
    title: str | None = None
    product: str | None = None
    version: str | None = None
    doc_type: DocType = "other"
    tags: list[str] = Field(default_factory=list)
    content_hash: str

    def to_pinecone(self) -> dict[str, Any]:
        """Pinecone metadata must be flat str/number/bool/list[str]; drop Nones."""
        data = self.model_dump()
        return {k: v for k, v in data.items() if v is not None and v != []}


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    section: str | None = None  # breadcrumb, e.g. "Auth > OAuth > Refresh tokens"
    position: int = 0
    token_count: int = 0
    metadata: DocMetadata

    def embedding_text(self) -> str:
        """Text that is embedded: title + section breadcrumb give the vector context."""
        parts = [p for p in (self.metadata.title, self.section) if p]
        header = " | ".join(parts)
        return f"{header}\n\n{self.text}" if header else self.text


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    title: str | None = None
    url: str | None = None
    source: str | None = None
    section: str | None = None
    product: str | None = None
    version: str | None = None
    doc_type: str | None = None
    rerank_score: float | None = None

    @property
    def best_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


class SearchFilters(BaseModel):
    """Metadata filters the agent (or API caller) can apply to retrieval."""

    product: str | None = None
    version: str | None = None
    doc_type: DocType | None = None
    tags: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.product or self.version or self.doc_type or self.tags)

    def to_pinecone(self) -> dict[str, Any] | None:
        clauses: list[dict[str, Any]] = []
        if self.product:
            clauses.append({"product": {"$eq": self.product}})
        if self.version:
            clauses.append({"version": {"$eq": self.version}})
        if self.doc_type:
            clauses.append({"doc_type": {"$eq": self.doc_type}})
        if self.tags:
            clauses.append({"tags": {"$in": self.tags}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class Citation(BaseModel):
    index: int
    title: str | None = None
    url: str | None = None
    source: str | None = None
    section: str | None = None
    chunk_id: str


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    rewrites: int = 0
    tool_calls: int = 0
    latency_ms: float = 0.0
    trace_id: str | None = None
