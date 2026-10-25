"""
Retrieval service — vector search against Qdrant Cloud.

Returns up to `top_k` chunks whose cosine similarity to the query vector
is ≥ `score_threshold`.  An empty list triggers abstention upstream.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint


def search(
    client: QdrantClient,
    collection: str,
    vector: list[float],
    top_k: int,
    score_threshold: float,
) -> list[ScoredPoint]:
    """
    Perform cosine-similarity search and return matching points.

    Each returned ScoredPoint carries:
      .score   — cosine similarity (0-1, higher = more relevant)
      .payload — {"text", "source", "page", "chunk_index"}
    """
    result = client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
    )
    return result.points
