"""Configuration validation and secret-handling tests."""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from app.config import Settings

QDRANT_KEY = "qdrant-secret-value-abcdefghij"
GROQ_KEY = "groq-secret-value-abcdefghij"
API_KEY = "api-key-secret-value-abcdefghij"

# conftest sets these so that ``app.main`` can be imported; a defaults test must not
# see them, otherwise it would assert against the test harness rather than the code.
_SETTING_NAMES = {
    "SERVICE_NAME",
    "API_VERSION",
    "ENV",
    "LOG_LEVEL",
    "DOCS_ENABLED",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "EMBED_MODEL",
    "EMBED_DIM",
    "EMBED_BACKEND",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GROQ_MODEL_FALLBACK",
    "GROQ_TIMEOUT_S",
    "TOP_K",
    "MIN_SCORE",
    "MAX_QUESTION_CHARS",
    "MAX_OUTPUT_TOKENS",
    "REQUEST_TIMEOUT_S",
    "API_KEYS",
    "RATE_LIMIT_PER_MIN",
    "ALLOWED_ORIGINS",
}


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ambient setting so assertions target the real defaults."""
    for name in list(os.environ):
        if name.upper() in _SETTING_NAMES:
            monkeypatch.delenv(name, raising=False)


def build(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "qdrant_url": "https://example.qdrant.io:6333",
        "qdrant_api_key": QDRANT_KEY,
        "groq_api_key": GROQ_KEY,
        "api_keys": API_KEY,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_defaults_match_the_approved_plan() -> None:
    settings = build()

    assert settings.qdrant_collection == "drcip_disaster_docs_v1"
    assert settings.embed_model == "intfloat/multilingual-e5-small"
    assert settings.embed_dim == 384
    assert settings.embed_backend == "qdrant_cloud"
    assert settings.groq_model == "qwen/qwen3.8-27b"
    assert settings.groq_model_fallback == "openai/gpt-oss-20b"
    assert settings.top_k == 6
    assert 0.0 <= settings.min_score <= 1.0
    assert settings.env == "production"  # safe default when ENV is unset
    assert settings.docs_enabled is True


@pytest.mark.parametrize("missing", ["qdrant_url", "qdrant_api_key", "groq_api_key", "api_keys"])
def test_required_settings_fail_fast(missing: str) -> None:
    values: dict[str, object] = {
        "qdrant_url": "https://example.qdrant.io:6333",
        "qdrant_api_key": QDRANT_KEY,
        "groq_api_key": GROQ_KEY,
        "api_keys": API_KEY,
    }
    values.pop(missing)

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)  # type: ignore[arg-type]


def test_qdrant_url_requires_https() -> None:
    with pytest.raises(ValidationError, match="https"):
        build(qdrant_url="http://example.qdrant.io:6333")


def test_qdrant_url_allows_http_for_local_development() -> None:
    assert build(qdrant_url="http://localhost:6333").qdrant_url == "http://localhost:6333"


def test_qdrant_url_must_be_absolute() -> None:
    with pytest.raises(ValidationError):
        build(qdrant_url="example.qdrant.io")


def test_trailing_slash_is_removed() -> None:
    assert build(qdrant_url="https://example.qdrant.io:6333/").qdrant_url == (
        "https://example.qdrant.io:6333"
    )


def test_multiple_api_keys_are_parsed_and_deduplicated() -> None:
    settings = build(api_keys=f"{API_KEY}, second-api-key-abcdefghij ,{API_KEY}")

    assert settings.api_key_set == frozenset({API_KEY, "second-api-key-abcdefghij"})
    assert len(settings.api_key_set) == 2


def test_short_api_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 16"):
        build(api_keys="short-key")


def test_empty_api_key_list_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        build(api_keys="  ,  ")


def test_fallback_model_must_differ_from_primary() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        build(groq_model="qwen/qwen3.8-27b", groq_model_fallback="qwen/qwen3.8-27b")


def test_empty_fallback_is_allowed() -> None:
    assert build(groq_model_fallback="").groq_model_fallback == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 0),
        ("top_k", 11),
        ("min_score", -0.1),
        ("min_score", 1.5),
        ("embed_dim", 0),
        ("rate_limit_per_min", 0),
        ("max_question_chars", 4),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        build(**{field: value})


def test_allowed_origins_are_split_and_trimmed() -> None:
    settings = build(allowed_origins=" https://a.example , https://b.example ,")

    assert settings.allowed_origin_list == ["https://a.example", "https://b.example"]


def test_blank_allowed_origins_means_no_cors() -> None:
    assert build(allowed_origins="").allowed_origin_list == []


def test_secrets_are_never_exposed_in_repr_or_summary() -> None:
    settings = build()

    assert QDRANT_KEY not in repr(settings)
    assert GROQ_KEY not in repr(settings)
    assert API_KEY not in repr(settings)

    summary = json.dumps(settings.safe_summary())
    for secret in (QDRANT_KEY, GROQ_KEY, API_KEY):
        assert secret not in summary


def test_safe_summary_reports_key_count_not_values() -> None:
    settings = build(api_keys=f"{API_KEY},second-api-key-abcdefghij")

    summary = settings.safe_summary()
    assert summary["api_key_count"] == 2
    assert summary["qdrant_collection"] == "drcip_disaster_docs_v1"
    assert summary["qdrant_host"] == "example.qdrant.io"


def test_unknown_environment_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_UNRELATED_VAR", "1")

    assert build().service_name == "drcip-rag-api"
