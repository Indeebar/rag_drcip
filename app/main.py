"""
FastAPI application entry point.

Startup sequence (lifespan):
  1. Pre-load embedding model into RAM once (avoids slow first request).
  2. Initialise Qdrant and Groq clients (cached via lru_cache).

Routes:
  GET  /          — API info
  GET  /health    — liveness probe (no auth; used by Render & keep-alive cron)
  POST /query     — RAG question answering (bearer auth required)
  GET  /docs      — Interactive Swagger UI (auto-generated)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from app.dependencies import get_groq_client, get_qdrant_client, get_settings
from app.routers import query
from app.services.embedder import preload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info("── Startup ──────────────────────────────────────────")

    logger.info("Loading embedding model (all-MiniLM-L6-v2)…")
    preload()
    logger.info("Embedding model ready.")

    logger.info("Initialising Qdrant client…")
    get_qdrant_client()
    logger.info("Qdrant client ready.")

    logger.info("Initialising Groq client…")
    get_groq_client()
    logger.info("Groq client ready.")

    logger.info("All services up — API is accepting requests.")
    logger.info("─────────────────────────────────────────────────────")

    yield  # ── application runs ──

    logger.info("── Shutdown ─────────────────────────────────────────")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="DRCIP Disaster Management RAG API",
    version="1.0.0",
    description=(
        "RAG-powered Q&A API for disaster-management documents. "
        "Send a question, receive a grounded answer with page citations."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — open for public consumption; restrict origins in production if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(query.router, tags=["Query"])


# ── Meta endpoints ────────────────────────────────────────────────────────────

@app.get("/", tags=["Meta"], summary="API info")
def root():
    return {
        "name": "DRCIP Disaster Management RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["Meta"], summary="Liveness probe")
def health(qdrant: QdrantClient = Depends(get_qdrant_client)):
    """
    Public liveness endpoint — no authentication required.
    Hit this every ~13 minutes to prevent Render cold starts and
    every ~6 days to keep the Qdrant free cluster from sleeping.
    """
    settings = get_settings()
    qdrant_status = "unknown"
    try:
        exists = qdrant.collection_exists(settings.qdrant_collection)
        qdrant_status = "connected" if exists else "collection_missing"
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        qdrant_status = "error"

    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "collection": settings.qdrant_collection,
        "model": "all-MiniLM-L6-v2",
        "model_loaded": True,
    }
