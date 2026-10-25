"""Shared test fixtures.

Environment variables are set *before* ``app.main`` is imported so the module-level
application builds against dummy credentials. Tests never touch real services.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

TEST_ENV = {
    "ENV": "test",
    "LOG_LEVEL": "WARNING",
    "DOCS_ENABLED": "true",
    "QDRANT_URL": "https://test-cluster.example.qdrant.io:6333",
    "QDRANT_API_KEY": "test-qdrant-api-key-0000000000",
    "QDRANT_COLLECTION": "drcip_disaster_docs_v1",
    "GROQ_API_KEY": "test-groq-api-key-0000000000",
    "GROQ_MODEL": "qwen/qwen3.8-27b",
    "GROQ_MODEL_FALLBACK": "openai/gpt-oss-20b",
    "API_KEYS": "test-api-key-0000000000,test-api-key-1111111111",
}

for _key, _value in TEST_ENV.items():
    os.environ[_key] = _value


@pytest.fixture
def settings_kwargs() -> dict[str, object]:
    """Minimal valid settings, with the dotenv file disabled for isolation."""
    return {
        "env": "test",
        "log_level": "WARNING",
        "qdrant_url": "https://test-cluster.example.qdrant.io:6333",
        "qdrant_api_key": "test-qdrant-api-key-0000000000",
        "groq_api_key": "test-groq-api-key-0000000000",
        "api_keys": "test-api-key-0000000000,test-api-key-1111111111",
    }


@pytest.fixture
def settings(settings_kwargs: dict[str, object]):
    from app.config import Settings

    return Settings(_env_file=None, **settings_kwargs)  # type: ignore[arg-type]


@pytest.fixture
def client(settings) -> Iterator[TestClient]:
    from app.main import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client
