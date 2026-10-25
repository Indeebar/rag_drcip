"""Health and service-metadata endpoints (unauthenticated)."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, RedirectResponse, Response

from app.deps import SettingsDep
from app.schemas import ERROR_RESPONSES, HealthResponse

router = APIRouter(tags=["service"], responses=ERROR_RESPONSES)


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    """Return 200 as soon as the process is up.

    Deliberately does **not** contact Qdrant or Groq: Render's health check and
    external uptime pingers must not fail because a downstream provider blipped.
    A dependency-aware readiness probe arrives in a later phase as ``/health/ready``.
    """
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.api_version,
        environment=settings.env,
    )


@router.get("/", include_in_schema=False)
async def root(settings: SettingsDep) -> Response:
    """Tiny landing document, so hitting the base URL is not a bare 404."""
    if settings.docs_enabled:
        return RedirectResponse(url="/docs")
    return JSONResponse(
        {
            "service": settings.service_name,
            "version": settings.api_version,
            "endpoints": {"health": "/health"},
        }
    )
