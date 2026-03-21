"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from lorelens.models import Citation, RetrievedChunk, SearchFilters


class ChatTurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    history: list[ChatTurnIn] = Field(default_factory=list, max_length=20)
    filters: SearchFilters | None = None
    session_id: str | None = Field(default=None, max_length=128)
    user_id: str | None = Field(default=None, max_length=128)
    include_sources: bool = False


class SourceOut(BaseModel):
    chunk_id: str
    title: str | None = None
    section: str | None = None
    url: str | None = None
    product: str | None = None
    version: str | None = None
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool
    standalone_question: str | None = None
    filters: SearchFilters | None = None
    queries: list[str] = Field(default_factory=list)
    rewrites: int = 0
    tool_calls: int = 0
    latency_ms: float
    node_timings_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None
    sources: list[SourceOut] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    filters: SearchFilters | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]
    latency_ms: float


class IngestRequest(BaseModel):
    path: str = Field(description="Directory relative to the server's ingest root.")
    base_url: str | None = None
    namespace: str | None = None
    force: bool = False
    prune: bool = False


class IngestJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stats: dict | None = None
    error: str | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    score: Literal[0, 1] = Field(description="1 = helpful, 0 = not helpful")
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    recorded: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    tool_mode: str
    langfuse: bool
    cache: dict | None = None
