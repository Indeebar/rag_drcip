#!/usr/bin/env python3
"""
Ingest PDF documents into Qdrant Cloud.

Run this script LOCALLY before deploying the API.
The API on Render is read-only — it never ingests.

Usage
─────
    # First time (creates collection)
    python -m ingest.ingest --pdf disaster.pdf

    # Re-ingest with collection reset (useful after PDF updates)
    python -m ingest.ingest --pdf disaster.pdf --reset

    # Custom collection name
    python -m ingest.ingest --pdf disaster.pdf --collection my_collection

Environment variables required (loaded from .env automatically):
    QDRANT_URL
    QDRANT_API_KEY
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from ingest.chunker import CHUNK_OVERLAP, CHUNK_SIZE, extract_chunks

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 32   # upsert batch size; keep low to avoid Qdrant payload limits


def _get_env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"[ERROR] Environment variable '{key}' is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into Qdrant Cloud for the DRCIP RAG API."
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument(
        "--collection",
        default=os.environ.get("QDRANT_COLLECTION", "disaster_docs"),
        help="Qdrant collection name (default: disaster_docs).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before ingesting.",
    )
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.isfile(args.pdf):
        print(f"[ERROR] File not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    qdrant_url = _get_env("QDRANT_URL")
    qdrant_api_key = _get_env("QDRANT_API_KEY")

    # ── Connect to Qdrant ─────────────────────────────────────────────────────
    print(f"[1/5] Connecting to Qdrant at {qdrant_url} ...")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # ── Create / reset collection ─────────────────────────────────────────────
    print(f"[2/5] Setting up collection '{args.collection}' ...")
    exists = client.collection_exists(args.collection)

    if args.reset and exists:
        print(f"      Deleting existing collection '{args.collection}' ...")
        client.delete_collection(args.collection)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=qmodels.VectorParams(
                size=EMBEDDING_DIM,
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"      Collection '{args.collection}' created (dim={EMBEDDING_DIM}, COSINE).")
    else:
        print(f"      Collection '{args.collection}' already exists — upserting into it.")

    # ── Extract chunks ────────────────────────────────────────────────────────
    print(f"[3/5] Extracting chunks from '{args.pdf}' ...")
    print(f"      chunk_size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars")
    t0 = time.perf_counter()
    chunks = list(extract_chunks(args.pdf))
    elapsed = time.perf_counter() - t0
    print(f"      -> {len(chunks)} chunks extracted in {elapsed:.2f}s")

    if not chunks:
        print("[WARN] No chunks extracted - the PDF may be empty or image-only.")
        sys.exit(0)

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"[4/5] Loading embedding model '{EMBEDDING_MODEL}' ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Embed and upsert in batches ───────────────────────────────────────────
    print(f"[5/5] Embedding and upserting {len(chunks)} chunks (batch_size={BATCH_SIZE}) ...")
    total = len(chunks)
    upserted = 0
    t0 = time.perf_counter()

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE,
        )

        points = [
            qmodels.PointStruct(
                id=c["id"],          # deterministic UUID string
                vector=v.tolist(),
                payload={
                    "text": c["text"],
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                },
            )
            for c, v in zip(batch, vectors)
        ]

        client.upsert(collection_name=args.collection, points=points)
        upserted += len(points)

        pct = upserted / total * 100
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"\r      [{bar}] {upserted}/{total} ({pct:.0f}%)", end="", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n\nDone! Ingestion complete in {elapsed:.1f}s")
    print(f"  Collection : {args.collection}")
    print(f"  Chunks     : {upserted}")
    print(f"  Model      : {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"\nYour API is ready. Deploy to Render and set QDRANT_COLLECTION={args.collection}")


if __name__ == "__main__":
    main()
