"""
POST /query — the sole public endpoint.

Request  → { "question": "..." }
Response → { "answer", "sources", "model", "chunks_retrieved" }

Authentication: Bearer token checked against API_KEYS env var.
Abstention:     Triggered by score threshold (pre-LLM) or LLM sentinel.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from groq import Groq
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.config import Settings
from app.dependencies import get_groq_client, get_qdrant_client, get_settings
from app.services import embedder, generator, retriever

router = APIRouter()
_bearer = HTTPBearer()
logger = logging.getLogger(__name__)


def _unavailable(service: str, exc: Exception) -> HTTPException:
    """Map an upstream failure to a 503 without leaking internal details."""
    logger.exception("%s request failed: %s", service, exc)
    return HTTPException(
        status_code=503,
        detail=f"{service} is currently unavailable. Please try again later.",
    )

# ── Human-readable abstention message returned to the consumer ────────────────
_ABSTENTION = (
    "I could not find sufficient information in the available "
    "disaster-management documents to answer this question reliably. "
    "Please consult official sources or rephrase your query."
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language disaster-management question.",
        examples=["What are the evacuation procedures for coastal flooding?"],
    )


class SourceCitation(BaseModel):
    document: str = Field(description="Source filename")
    page: int = Field(description="1-based page number inside the document")
    excerpt: str = Field(description="First 250 characters of the matched chunk")


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    model: str
    chunks_retrieved: int


# ── Auth dependency ───────────────────────────────────────────────────────────

def _verify_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
    settings: Settings = Depends(get_settings),
) -> str:
    if credentials.credentials not in settings.api_key_set:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return credentials.credentials


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a disaster-management question",
    description=(
        "Embeds the question, retrieves relevant document chunks from Qdrant, "
        "and generates a grounded answer via Groq. Returns an abstention "
        "response when the documents do not support an answer."
    ),
)
def query_endpoint(
    body: QueryRequest,
    settings: Settings = Depends(get_settings),
    qdrant: QdrantClient = Depends(get_qdrant_client),
    groq_client: Groq = Depends(get_groq_client),
    _key: str = Depends(_verify_key),
) -> QueryResponse:
    # 1. Embed question
    try:
        vector = embedder.embed(body.question)
    except Exception as exc:  # pragma: no cover - defensive
        raise _unavailable("Embedding service", exc) from exc

    # 2. Retrieve chunks (returns [] when nothing meets score_threshold)
    try:
        chunks = retriever.search(
            client=qdrant,
            collection=settings.qdrant_collection,
            vector=vector,
            top_k=settings.top_k,
            score_threshold=settings.score_threshold,
        )
    except Exception as exc:
        raise _unavailable("Vector store", exc) from exc

    # 3. Layer-1 abstention: no chunks above threshold
    if not chunks:
        return QueryResponse(
            answer=_ABSTENTION,
            sources=[],
            model=settings.groq_model,
            chunks_retrieved=0,
        )

    # 4. Generate answer via Groq
    try:
        answer = generator.generate(
            client=groq_client,
            model=settings.groq_model,
            question=body.question,
            chunks=chunks,
        )
    except Exception as exc:
        raise _unavailable("Generation service", exc) from exc

    # 5. Layer-2 abstention: LLM signalled insufficient context
    if generator.is_insufficient(answer):
        return QueryResponse(
            answer=_ABSTENTION,
            sources=[],
            model=settings.groq_model,
            chunks_retrieved=len(chunks),
        )

    # 6. Build citation list from chunk payloads
    sources: list[SourceCitation] = []
    seen_pages: set[tuple[str, int]] = set()
    for chunk in chunks:
        p = chunk.payload or {}
        doc = p.get("source", "unknown")
        page = p.get("page", 0)
        key = (doc, page)
        if key not in seen_pages:           # deduplicate same-page citations
            seen_pages.add(key)
            text = p.get("text", "")
            sources.append(
                SourceCitation(
                    document=doc,
                    page=page,
                    excerpt=text[:250] + ("…" if len(text) > 250 else ""),
                )
            )

    return QueryResponse(
        answer=answer,
        sources=sources,
        model=settings.groq_model,
        chunks_retrieved=len(chunks),
    )
