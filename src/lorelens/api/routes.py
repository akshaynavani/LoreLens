"""HTTP routes: ask (JSON + SSE), raw search, ingest jobs, feedback, health."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from lorelens import __version__
from lorelens.agent.graph import LoreLensAgent
from lorelens.api.deps import agent_dep, require_api_key, search_service_dep, settings_dep
from lorelens.api.schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestJob,
    IngestRequest,
    SearchRequest,
    SearchResponse,
    SourceOut,
)
from lorelens.config import Settings
from lorelens.log import get_logger
from lorelens.models import Citation
from lorelens.observability import get_langfuse, score_trace, traced_run
from lorelens.retrieval.service import DocSearchService

logger = get_logger(__name__)

public = APIRouter()
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

_JOBS: dict[str, IngestJob] = {}


@public.get("/health", response_model=HealthResponse)
async def health(request: Request, settings: Settings = Depends(settings_dep)) -> HealthResponse:
    service: DocSearchService | None = getattr(request.app.state, "search_service", None)
    return HealthResponse(
        status="ok",
        version=__version__,
        tool_mode=settings.tool_mode,
        langfuse=get_langfuse() is not None,
        cache=service.caches.stats() if service else None,
    )


# --------------------------------------------------------------------------- ask
@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, agent: LoreLensAgent = Depends(agent_dep)) -> AskResponse:
    history = [t.model_dump() for t in req.history]
    async with traced_run(
        "lorelens-ask",
        input={"question": req.question, "filters": req.filters.model_dump() if req.filters else None},
        user_id=req.user_id,
        session_id=req.session_id,
        tags=["api", agent.settings.tool_mode],
    ) as trace:
        answer, state = await agent.arun(
            req.question, history=history, filters=req.filters, config=trace.config
        )
        trace.set_output(
            {"answer": answer.answer, "citations": [c.model_dump() for c in answer.citations]},
            grounded=answer.grounded,
            rewrites=answer.rewrites,
        )

    analysis = state.get("analysis")
    sources = None
    if req.include_sources:
        sources = [
            SourceOut(
                chunk_id=c.chunk_id, title=c.title, section=c.section, url=c.url,
                product=c.product, version=c.version, score=c.best_score, text=c.text,
            )
            for c in state.get("documents", [])
        ]
    return AskResponse(
        answer=answer.answer,
        citations=answer.citations,
        grounded=answer.grounded,
        standalone_question=analysis.standalone_question if analysis else None,
        filters=state.get("filters"),
        queries=state.get("tried_queries", []),
        rewrites=answer.rewrites,
        tool_calls=answer.tool_calls,
        latency_ms=answer.latency_ms,
        node_timings_ms=state.get("node_timings", {}),
        trace_id=trace.trace_id,
        sources=sources,
    )


def _status_event(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Compact, client-friendly progress payloads for each graph step."""
    info: dict[str, Any] = {"node": node}
    if node == "analyze" and update.get("analysis") is not None:
        info["standalone_question"] = update["analysis"].standalone_question
        info["queries"] = update.get("search_queries", [])
        f = update.get("filters")
        info["filters"] = f.model_dump(exclude_none=True) if f else {}
    elif node in ("prefetch", "collect", "grade"):
        info["documents"] = len(update.get("documents", []))
    elif node == "tools":
        info["tools"] = [getattr(m, "name", None) for m in update.get("messages", [])]
    elif node == "rewrite":
        info["queries"] = update.get("search_queries", [])
    return info


@router.post("/ask/stream")
async def ask_stream(
    req: AskRequest, request: Request, agent: LoreLensAgent = Depends(agent_dep)
) -> EventSourceResponse:
    """Server-Sent Events: `status` per graph step, `token` for answer tokens, then `done`."""
    history = [t.model_dump() for t in req.history]

    async def events():  # noqa: ANN202
        start = time.perf_counter()
        final: dict[str, Any] = {}
        async with traced_run(
            "lorelens-ask-stream",
            input={"question": req.question},
            user_id=req.user_id,
            session_id=req.session_id,
            tags=["api", "stream", agent.settings.tool_mode],
        ) as trace:
            try:
                async for kind, payload in agent.astream(
                    req.question, history=history, filters=req.filters, config=trace.config
                ):
                    if await request.is_disconnected():
                        logger.info("Client disconnected; stopping stream")
                        return
                    if kind == "token":
                        yield {"event": "token", "data": payload}
                        continue
                    for node, update in payload.items():
                        if not isinstance(update, dict):
                            continue
                        if "answer" in update:
                            final = update
                        yield {"event": "status", "data": json.dumps(_status_event(node, update))}
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming ask failed")
                yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
                return
            citations: list[Citation] = final.get("citations", [])
            trace.set_output(final.get("answer"))
            yield {
                "event": "done",
                "data": json.dumps(
                    {
                        "answer": final.get("answer", ""),
                        "citations": [c.model_dump() for c in citations],
                        "grounded": final.get("grounded", False),
                        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                        "trace_id": trace.trace_id,
                    }
                ),
            }

    return EventSourceResponse(events())


# ------------------------------------------------------------------------ search
@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest, service: DocSearchService = Depends(search_service_dep)
) -> SearchResponse:
    start = time.perf_counter()
    results = await service.asearch(req.query, req.filters, req.top_k)
    return SearchResponse(
        query=req.query, results=results, latency_ms=round((time.perf_counter() - start) * 1000, 1)
    )


# ------------------------------------------------------------------------ ingest
def _resolve_ingest_path(settings: Settings, rel: str) -> Path:
    root = settings.ingest_root.resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root) or not target.is_dir():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"{rel!r} is not a directory under the ingest root")
    return target


def _run_ingest(job_id: str, path: Path, req: IngestRequest, service: DocSearchService) -> None:
    from lorelens.ingestion.loaders import DirectoryLoader
    from lorelens.ingestion.pipeline import IngestionPipeline

    job = _JOBS[job_id]
    job.status = "running"
    try:
        pipeline = IngestionPipeline(
            service.store, service.embeddings, service.settings, namespace=req.namespace
        )
        stats = pipeline.run(
            DirectoryLoader(path, req.base_url or service.settings.docs_base_url),
            force=req.force,
            prune=req.prune,
        )
        service.caches.results.clear()  # results may be stale after re-indexing
        job.status, job.stats = "succeeded", stats.as_dict()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest job %s failed", job_id)
        job.status, job.error = "failed", str(exc)


@router.post("/ingest", response_model=IngestJob, status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    req: IngestRequest,
    background: BackgroundTasks,
    settings: Settings = Depends(settings_dep),
    service: DocSearchService = Depends(search_service_dep),
) -> IngestJob:
    path = _resolve_ingest_path(settings, req.path)
    job = IngestJob(job_id=uuid.uuid4().hex, status="queued")
    _JOBS[job.job_id] = job
    background.add_task(asyncio.to_thread, _run_ingest, job.job_id, path, req, service)
    return job


@router.get("/ingest/{job_id}", response_model=IngestJob)
async def ingest_status(job_id: str) -> IngestJob:
    if job_id not in _JOBS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown job")
    return _JOBS[job_id]


# ---------------------------------------------------------------------- feedback
@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record end-user thumbs up/down as a LangFuse score on the answer's trace."""
    recorded = await asyncio.to_thread(
        score_trace, req.trace_id, "user_feedback", req.score,
        comment=req.comment, data_type="BOOLEAN",
    )
    return FeedbackResponse(recorded=recorded)
