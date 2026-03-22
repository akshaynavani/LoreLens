"""FastAPI application factory.

Run:  uvicorn lorelens.api.app:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lorelens import __version__
from lorelens.config import get_settings
from lorelens.log import configure_logging, get_logger
from lorelens.observability import flush

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from lorelens.agent.graph import LoreLensAgent
    from lorelens.retrieval.service import get_search_service

    settings = get_settings()
    # Tests (or alternative deployments) may pre-populate app.state.
    if getattr(app.state, "search_service", None) is None:
        app.state.search_service = get_search_service()
    if getattr(app.state, "agent", None) is None:
        app.state.agent = await LoreLensAgent.create(settings)
    logger.info("LoreLens API ready (tool_mode=%s)", settings.tool_mode)
    yield
    flush()


def create_app() -> FastAPI:
    from lorelens.api.routes import public, router

    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="LoreLens",
        version=__version__,
        description="Agentic RAG search over technical documentation.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(public)
    app.include_router(router)
    return app


app = create_app()
