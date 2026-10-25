"""
Centralised settings loaded from environment variables.
All secrets come from the environment; this file never contains real values.
"""
from __future__ import annotations

from functools import cached_property
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Required ──────────────────────────────────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str
    groq_api_key: str
    api_keys: str  # comma-separated bearer tokens

    # ── Optional with sensible defaults ───────────────────────────────────────
    qdrant_collection: str = "disaster_docs"
    groq_model: str = "openai/gpt-oss-20b"
    top_k: int = 5
    score_threshold: float = 0.30  # cosine similarity floor for abstention

    # ── Future use (Neon PostgreSQL) ──────────────────────────────────────────
    database_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        # utf-8-sig transparently strips a UTF-8 BOM (common on Windows editors),
        # which would otherwise be read as part of the first variable name.
        env_file_encoding="utf-8-sig",
        case_sensitive=False,
        extra="ignore",
    )

    @cached_property
    def api_key_set(self) -> set[str]:
        """Parse comma-separated API_KEYS into a set for O(1) lookup."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}
