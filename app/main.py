"""FastAPI application factory.

Importing this module builds the default app, so a missing or invalid environment
fails at boot (loudly, once) instead of on the first user request. Tests build
isolated instances through :func:`create_app` with explicit settings.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health
from app.config import Settings, get_settings
from app.errors import install_error_handlers
from app.observability import RequestContextMiddleware, configure_logging

__version__ = "1.0.0"

logger = logging.getLogger("drcip.startup")

DESCRIPTION = """
Retrieval-augmented question answering over a disaster-management reference
corpus. Answers are grounded in retrieved passages and always carry citations;
when the corpus does not support an answer the API abstains instead of guessing.

**Authentication:** send `X-API-Key` on every `/v1/*` request.
`/health` is public so uptime monitors can reach it.
""".strip()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="DRCIP RAG API",
        version=settings.api_version,
        description=DESCRIPTION,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        contact={"name": "DRCIP RAG API"},
        license_info={"name": "Proprietary"},
    )

    # Route handlers resolve settings through this dependency, so tests (and any
    # future multi-tenant variant) can supply a different instance.
    app.state.settings = settings
    app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(RequestContextMiddleware)

    if settings.allowed_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origin_list,
            allow_credentials=False,
            allow_methods=["POST", "GET"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-Response-Time-Ms"],
        )

    install_error_handlers(app)
    app.include_router(health.router)

    logger.info("startup %s", settings.safe_summary())
    if settings.embed_backend == "local" and settings.env == "production":
        logger.warning(
            "EMBED_BACKEND=local in production: a local embedding model may exceed "
            "Render's 512 MB free-tier memory. Prefer 'qdrant_cloud'."
        )

    return app


app = create_app()
