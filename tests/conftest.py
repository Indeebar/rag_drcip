"""
Shared pytest fixtures and environment setup.

All external services (Qdrant, Groq) are mocked — tests run offline.
"""
from __future__ import annotations

import os

import pytest

# Set env vars before any app module is imported
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test_qdrant_key")
os.environ.setdefault("GROQ_API_KEY", "test_groq_key")
os.environ.setdefault("API_KEYS", "test_bearer_key_123")
os.environ.setdefault("QDRANT_COLLECTION", "test_collection")


@pytest.fixture(autouse=True)
def _never_load_embedding_model(monkeypatch):
    """
    Keep the suite offline and fast.

    ``app.main.lifespan`` calls ``preload()``, which would otherwise load
    all-MiniLM-L6-v2 (~80 MB download on a cold cache). Query-time ``embed()``
    is mocked in the tests themselves.
    """
    monkeypatch.setattr("app.services.embedder._load_model", lambda: None)
