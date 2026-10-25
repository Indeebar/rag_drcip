"""Public response/request models.

These models are the API contract. Field names here are what the integrating
developer sees, so treat them as frozen once released: add fields, never rename.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness probe. Makes no network calls, so it is safe for uptime pingers."""

    status: Literal["ok"] = "ok"
    service: str = Field(examples=["drcip-rag-api"])
    version: str = Field(examples=["1.0"])
    environment: str = Field(examples=["production"])


class ErrorDetail(BaseModel):
    code: str = Field(examples=["unauthorized"])
    message: str = Field(examples=["Missing or invalid API key."])
    request_id: str = Field(examples=["req_01jb7z3k9v00"])


class ErrorResponse(BaseModel):
    error: ErrorDetail


# Attached to routers so the error contract appears in /openapi.json (FastAPI only
# publishes models that a path operation references).
ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    500: {"model": ErrorResponse, "description": "Unexpected server error."},
}
