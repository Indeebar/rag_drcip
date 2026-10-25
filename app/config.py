"""Application configuration.

All credentials come from the environment (or a local, git-ignored ``.env``).
Secrets are wrapped in :class:`pydantic.SecretStr` so they cannot leak through
``repr()``, logs or tracebacks, and are never serialised by ``safe_summary()``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
EmbedBackend = Literal["qdrant_cloud", "local"]

MIN_API_KEY_LENGTH = 16


class Settings(BaseSettings):
    """Validated runtime configuration (fail fast at startup)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Service ---------------------------------------------------------
    service_name: str = "drcip-rag-api"
    api_version: str = "1.0"
    env: Environment = "production"
    log_level: str = "INFO"
    docs_enabled: bool = True

    # --- Qdrant ----------------------------------------------------------
    qdrant_url: str
    qdrant_api_key: SecretStr
    qdrant_collection: str = "drcip_disaster_docs_v1"

    # --- Embeddings ------------------------------------------------------
    embed_model: str = "intfloat/multilingual-e5-small"
    embed_dim: int = Field(default=384, ge=1, le=4096)
    embed_backend: EmbedBackend = "qdrant_cloud"

    # --- Groq ------------------------------------------------------------
    groq_api_key: SecretStr
    groq_model: str = "qwen/qwen3.8-27b"
    groq_model_fallback: str = "openai/gpt-oss-20b"
    groq_timeout_s: float = Field(default=20.0, gt=0, le=60)

    # --- Retrieval / answer ----------------------------------------------
    top_k: int = Field(default=6, ge=1, le=10)
    min_score: float = Field(default=0.30, ge=0.0, le=1.0)
    max_question_chars: int = Field(default=500, ge=16, le=4000)
    max_output_tokens: int = Field(default=700, ge=64, le=4096)
    request_timeout_s: float = Field(default=25.0, gt=0, le=120)

    # --- Public API ------------------------------------------------------
    api_keys: SecretStr
    rate_limit_per_min: int = Field(default=10, ge=1, le=600)
    allowed_origins: str = ""

    # ---------------------------------------------------------------- validators
    @field_validator("qdrant_url")
    @classmethod
    def _validate_qdrant_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("QDRANT_URL must be a full URL, e.g. https://<cluster>.qdrant.io:6333")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
            raise ValueError("QDRANT_URL must use https (plain http is allowed only for localhost)")
        return value

    @field_validator("groq_model", "groq_model_fallback", "embed_model", "qdrant_collection")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("api_keys")
    @classmethod
    def _normalise_api_keys(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        keys = [key.strip() for key in raw.split(",") if key.strip()]
        if not keys:
            raise ValueError("API_KEYS must contain at least one key")
        too_short = [index for index, key in enumerate(keys) if len(key) < MIN_API_KEY_LENGTH]
        if too_short:
            raise ValueError(
                f"every API key must be at least {MIN_API_KEY_LENGTH} characters "
                f"(offending positions: {too_short})"
            )
        # Preserve order but drop duplicates, so rotation lists stay readable.
        unique = list(dict.fromkeys(keys))
        return SecretStr(",".join(unique))

    @model_validator(mode="after")
    def _check_fallback_differs(self) -> Settings:
        if self.groq_model_fallback and self.groq_model_fallback == self.groq_model:
            raise ValueError("GROQ_MODEL_FALLBACK must differ from GROQ_MODEL (or be empty)")
        return self

    # ---------------------------------------------------------------- derived
    @property
    def api_key_set(self) -> frozenset[str]:
        """Configured API keys, for constant-time membership checks."""
        return frozenset(
            key.strip() for key in self.api_keys.get_secret_value().split(",") if key.strip()
        )

    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def qdrant_host(self) -> str:
        return urlparse(self.qdrant_url).hostname or ""

    def safe_summary(self) -> dict[str, object]:
        """Non-secret view of the configuration, safe to log at startup."""
        return {
            "service": self.service_name,
            "api_version": self.api_version,
            "env": self.env,
            "docs_enabled": self.docs_enabled,
            "qdrant_host": self.qdrant_host,
            "qdrant_collection": self.qdrant_collection,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "embed_backend": self.embed_backend,
            "groq_model": self.groq_model,
            "groq_model_fallback": self.groq_model_fallback or None,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "rate_limit_per_min": self.rate_limit_per_min,
            "api_key_count": len(self.api_key_set),
        }


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (also used as a FastAPI dependency)."""
    return Settings()
