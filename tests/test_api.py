import json

import pytest
from fastapi.testclient import TestClient

from app.generation.generator import RAGService
from conftest import FakeLLM


@pytest.fixture
def client(settings, retriever, monkeypatch):
    from app import main
    from app.core import config

    original = config.get_settings
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    app = main.app
    app.dependency_overrides[original] = lambda: settings
    with TestClient(app) as c:
        app.state.service = RAGService(settings, retriever, llm=FakeLLM())
        app.state.startup_error = None
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["chunks"] > 0


def test_chat_returns_citations(client):
    r = client.post("/chat", json={"question": "هل يشترط رضا الزوجين؟"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    assert body["citations"][0]["article_label"] == "3"
    assert body["citations"][0]["text"].startswith("الفصل 3")
    assert body["disclaimer"]


def test_chat_validation_errors(client):
    assert client.post("/chat", json={"question": " "}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 1001}).status_code == 422
    assert client.post("/chat", json={"question": "سؤال", "mode": "magic"}).status_code == 422


def test_search_modes(client):
    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        r = client.post("/search", json={"query": "المهر", "mode": mode, "top_k": 3})
        assert r.status_code == 200 and r.json()["hits"], mode


def test_documents_requires_manifest(client, settings):
    assert client.get("/documents").status_code == 503
    settings.index_dir.mkdir(parents=True)
    (settings.index_dir / "manifest.json").write_text(json.dumps({
        "documents": [{"doc_id": "csp", "title": "مجلة", "legal_status": "consolidated", "authority": "official",
                       "pages": 35, "page_range": [5, 39], "articles": 213, "chunks": 220}],
        "total_pages": 35, "total_chunks": 220}), encoding="utf-8")
    settings.sources_file.parent.mkdir(parents=True, exist_ok=True)
    settings.sources_file.write_text("documents: []\n", encoding="utf-8")
    r = client.get("/documents")
    assert r.status_code == 200 and r.json()["documents"][0]["articles"] == 213


def test_feedback_is_stored(client, settings):
    r = client.post("/feedback", json={"question": "q", "answer": "a", "rating": "down", "issue": "bad_citation"})
    assert r.status_code == 200
    rec = json.loads(settings.feedback_file.read_text(encoding="utf-8").splitlines()[0])
    assert rec["rating"] == "down" and rec["issue"] == "bad_citation"


def test_service_unavailable_when_index_missing(client):
    client.app.state.service = None
    client.app.state.startup_error = "FileNotFoundError: chunks.jsonl"
    assert client.post("/chat", json={"question": "سؤال"}).status_code == 503
    assert client.get("/health").json()["status"] == "degraded"
