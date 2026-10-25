"""
Embedding service.

Uses all-MiniLM-L6-v2 (384-dim, ~22 MB weights) for query encoding.
The model is loaded once at startup and kept in memory for the process lifetime.
On Render's 512 MB free tier this consumes ~150-200 MB RAM, leaving headroom
for FastAPI, Uvicorn, and library overhead.

CPU-only PyTorch is installed in the Dockerfile to avoid the ~2 GB GPU variant.
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model (called once at startup)."""
    return SentenceTransformer(MODEL_NAME)


def embed(text: str) -> list[float]:
    """
    Encode *text* into a normalised 384-dimensional vector.

    Normalisation (L2) ensures cosine similarity equals dot product,
    which is what Qdrant uses for COSINE distance collections.
    """
    model = _load_model()
    vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


def preload() -> None:
    """Trigger model loading at startup so the first request is not slow."""
    _load_model()
