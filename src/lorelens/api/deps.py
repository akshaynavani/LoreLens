"""FastAPI dependencies: settings, the shared agent/search service, API-key auth."""

from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, Request, status

from lorelens.agent.graph import LoreLensAgent
from lorelens.config import Settings, get_settings
from lorelens.retrieval.service import DocSearchService


def settings_dep() -> Settings:
    return get_settings()


def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(settings_dep),
) -> None:
    expected = settings.api_key.get_secret_value() if settings.api_key else ""
    if not expected:
        return  # auth disabled
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing X-API-Key")


def agent_dep(request: Request) -> LoreLensAgent:
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Agent is not initialised")
    return agent


def search_service_dep(request: Request) -> DocSearchService:
    service = getattr(request.app.state, "search_service", None)
    if service is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Search is not initialised")
    return service
