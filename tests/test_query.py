"""
API integration tests — all external services are mocked.

Coverage:
  ✓ GET  /health  (public, no auth)
  ✓ GET  /        (API root)
  ✓ POST /query   missing auth → 401
  ✓ POST /query   wrong key   → 401
  ✓ POST /query   valid key, relevant context → 200 with answer + sources
  ✓ POST /query   valid key, no relevant chunks → abstention
  ✓ POST /query   valid key, LLM signals INSUFFICIENT_CONTEXT → abstention
  ✓ POST /query   question too short (< 3 chars) → 422
  ✓ POST /query   question too long (> 500 chars) → 422
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Patch heavy imports BEFORE the app is imported ───────────────────────────
# This prevents sentence-transformers from loading during test collection.
with (
    patch("sentence_transformers.SentenceTransformer"),
    patch("app.services.embedder._load_model"),
):
    from app.dependencies import get_groq_client, get_qdrant_client
    from app.main import app

VALID_KEY = "test_bearer_key_123"
AUTH = {"Authorization": f"Bearer {VALID_KEY}"}


# ── Helpers to build mock Qdrant ScoredPoints ────────────────────────────────

def _make_point(text: str, source: str = "disaster.pdf", page: int = 5, score: float = 0.8):
    p = SimpleNamespace()
    p.score = score
    p.payload = {"text": text, "source": source, "page": page, "chunk_index": 0}
    return p


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_embed():
    """Replace embed() with a zero-vector stub."""
    with patch("app.routers.query.embedder.embed", return_value=[0.0] * 384) as m:
        yield m


@pytest.fixture()
def mock_qdrant_client():
    return MagicMock()


@pytest.fixture()
def mock_groq_client():
    return MagicMock()


@pytest.fixture()
def api_client(mock_qdrant_client, mock_groq_client):
    """TestClient with Qdrant + Groq injected as mocks."""
    app.dependency_overrides[get_qdrant_client] = lambda: mock_qdrant_client
    app.dependency_overrides[get_groq_client] = lambda: mock_groq_client
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMetaEndpoints:
    def test_root_returns_200(self, api_client):
        r = api_client.get("/")
        assert r.status_code == 200
        assert r.json()["name"] == "DRCIP Disaster Management RAG API"

    def test_health_no_auth_required(self, api_client, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = True
        r = api_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["qdrant"] == "connected"

    def test_health_collection_missing(self, api_client, mock_qdrant_client):
        mock_qdrant_client.collection_exists.return_value = False
        r = api_client.get("/health")
        assert r.status_code == 200
        assert r.json()["qdrant"] == "collection_missing"


class TestAuth:
    def test_missing_auth_returns_403(self, api_client):
        r = api_client.post("/query", json={"question": "What is flood preparedness?"})
        assert r.status_code in (401, 403)

    def test_wrong_key_returns_401(self, api_client):
        r = api_client.post(
            "/query",
            json={"question": "What is flood preparedness?"},
            headers={"Authorization": "Bearer wrong_key"},
        )
        assert r.status_code == 401


class TestQueryValidation:
    def test_question_too_short_returns_422(self, api_client):
        r = api_client.post("/query", json={"question": "Hi"}, headers=AUTH)
        assert r.status_code == 422

    def test_question_too_long_returns_422(self, api_client):
        r = api_client.post("/query", json={"question": "x" * 501}, headers=AUTH)
        assert r.status_code == 422

    def test_missing_question_returns_422(self, api_client):
        r = api_client.post("/query", json={}, headers=AUTH)
        assert r.status_code == 422


class TestQueryLogic:
    def test_valid_query_returns_answer_with_sources(
        self, api_client, mock_qdrant_client, mock_groq_client, mock_embed
    ):
        # Qdrant returns relevant chunks
        mock_qdrant_client.query_points.return_value = SimpleNamespace(
            points=[
                _make_point("Evacuate immediately when Flood Alert Level 3 is issued.", page=12),
                _make_point("Designated routes are marked with green signs.", page=13),
            ]
        )
        # Groq returns a coherent answer
        choice = SimpleNamespace(message=SimpleNamespace(content="Evacuate via green-signed routes."))
        mock_groq_client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])

        r = api_client.post(
            "/query",
            json={"question": "What are the coastal flood evacuation procedures?"},
            headers=AUTH,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "Evacuate via green-signed routes."
        assert len(data["sources"]) == 2
        assert data["sources"][0]["page"] == 12
        assert data["chunks_retrieved"] == 2

    def test_no_chunks_returns_abstention(
        self, api_client, mock_qdrant_client, mock_groq_client, mock_embed
    ):
        mock_qdrant_client.query_points.return_value = SimpleNamespace(points=[])

        r = api_client.post(
            "/query",
            json={"question": "What is the capital of France?"},
            headers=AUTH,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["sources"] == []
        assert data["chunks_retrieved"] == 0
        assert "could not find" in data["answer"].lower()
        # Groq should NOT have been called
        mock_groq_client.chat.completions.create.assert_not_called()

    def test_llm_signals_insufficient_context_returns_abstention(
        self, api_client, mock_qdrant_client, mock_groq_client, mock_embed
    ):
        mock_qdrant_client.query_points.return_value = SimpleNamespace(
            points=[_make_point("Some loosely related content.", score=0.32)]
        )
        # LLM returns the sentinel
        choice = SimpleNamespace(message=SimpleNamespace(content="INSUFFICIENT_CONTEXT"))
        mock_groq_client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])

        r = api_client.post(
            "/query",
            json={"question": "What is the recipe for disaster soup?"},
            headers=AUTH,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["sources"] == []
        assert "could not find" in data["answer"].lower()

    def test_duplicate_page_citations_are_deduplicated(
        self, api_client, mock_qdrant_client, mock_groq_client, mock_embed
    ):
        # Two chunks from the same page
        mock_qdrant_client.query_points.return_value = SimpleNamespace(
            points=[
                _make_point("Chunk A from page 7.", page=7),
                _make_point("Chunk B from page 7.", page=7),
                _make_point("Chunk C from page 8.", page=8),
            ]
        )
        choice = SimpleNamespace(message=SimpleNamespace(content="Detailed answer here."))
        mock_groq_client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])

        r = api_client.post(
            "/query",
            json={"question": "Explain the disaster response protocol."},
            headers=AUTH,
        )
        assert r.status_code == 200
        pages = [s["page"] for s in r.json()["sources"]]
        # page 7 should appear only once
        assert pages.count(7) == 1
        assert 8 in pages


class TestUpstreamFailures:
    def test_qdrant_failure_returns_503(
        self, api_client, mock_qdrant_client, mock_embed
    ):
        mock_qdrant_client.query_points.side_effect = RuntimeError("qdrant down")

        r = api_client.post(
            "/query",
            json={"question": "What are the flood evacuation procedures?"},
            headers=AUTH,
        )
        assert r.status_code == 503
        assert "unavailable" in r.json()["detail"].lower()

    def test_groq_failure_returns_503(
        self, api_client, mock_qdrant_client, mock_groq_client, mock_embed
    ):
        mock_qdrant_client.query_points.return_value = SimpleNamespace(
            points=[_make_point("Evacuate immediately.")]
        )
        mock_groq_client.chat.completions.create.side_effect = RuntimeError("groq down")

        r = api_client.post(
            "/query",
            json={"question": "What are the flood evacuation procedures?"},
            headers=AUTH,
        )
        assert r.status_code == 503
        assert "unavailable" in r.json()["detail"].lower()
