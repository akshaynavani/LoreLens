"""API tests with a fake agent/search service injected into app.state."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from lorelens.api import routes
from lorelens.api.app import create_app
from lorelens.api.deps import settings_dep
from tests.conftest import FakeFastLLM, FakeSearchService
from tests.test_agent_graph import _agent, _analysis


class _Caches:
    def stats(self) -> dict:
        return {"results": {"hits": 0, "misses": 0, "size": 0}}


class _SearchService(FakeSearchService):
    caches = _Caches()

    async def asearch(self, query, filters=None, top_k=None, namespace=None):  # noqa: ANN001, ANN201
        return self.search(query, filters, top_k)


@pytest.fixture
def client_factory(settings, prompts, monkeypatch):
    monkeypatch.setattr(routes, "get_langfuse", lambda: None)

    def make(api_key: str | None = None) -> TestClient:
        cfg = settings.model_copy(update={"api_key": SecretStr(api_key) if api_key else None})
        app = create_app()
        app.dependency_overrides[settings_dep] = lambda: cfg
        agent, _ = _agent(cfg, prompts, FakeFastLLM(_analysis()))
        app.state.agent = agent
        app.state.search_service = _SearchService()
        return TestClient(app)

    return make


def test_health(client_factory):
    with client_factory() as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["langfuse"] is False


def test_ask_returns_cited_answer(client_factory):
    with client_factory() as client:
        resp = client.post(
            "/v1/ask", json={"question": "How long do tokens last?", "include_sources": True}
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "3600" in body["answer"]
    assert body["citations"][0]["index"] == 1
    assert body["filters"]["product"] == "nimbus"
    assert body["sources"] and body["sources"][0]["chunk_id"] == "doc1#0"
    assert body["trace_id"] is None  # LangFuse disabled in tests


def test_search_endpoint(client_factory):
    with client_factory() as client:
        resp = client.post("/v1/search", json={"query": "tokens", "top_k": 1})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


def test_api_key_is_enforced(client_factory):
    with client_factory(api_key="s3cret") as client:
        assert client.post("/v1/search", json={"query": "x"}).status_code == 401
        ok = client.post("/v1/search", json={"query": "x"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
        assert client.get("/health").status_code == 200  # health stays public


def test_ingest_rejects_paths_outside_root(client_factory):
    with client_factory() as client:
        resp = client.post("/v1/ingest", json={"path": "../../etc"})
    assert resp.status_code == 400


def test_feedback_without_langfuse_is_not_recorded(client_factory):
    with client_factory() as client:
        resp = client.post("/v1/feedback", json={"trace_id": "abc", "score": 1})
    assert resp.json() == {"recorded": False}
