"""
Application-wide singletons injected via FastAPI's dependency system.
All objects are created once and reused across requests.
"""
from __future__ import annotations

from functools import lru_cache

from groq import Groq
from qdrant_client import QdrantClient

from app.config import Settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (reads env vars once)."""
    return Settings()


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a cached Qdrant Cloud client."""
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Return a cached Groq client."""
    s = get_settings()
    return Groq(api_key=s.groq_api_key)
