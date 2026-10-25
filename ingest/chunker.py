"""
PDF chunker — extracts page-aware text chunks suitable for RAG ingestion.

Strategy
────────
• Extract text per page with pypdf.
• Apply a character-level sliding window *within* each page so chunks never
  span page boundaries — this keeps source citations (page numbers) accurate.
• Skip pages with fewer than MIN_CHARS useful characters (blank / image-only).
• Normalise whitespace (collapse newlines / extra spaces from PDF extraction).

Chunk IDs
─────────
Each chunk gets a deterministic UUID derived from (source, page, char_offset)
so re-running ingestion on the same PDF is idempotent (Qdrant upsert).
"""
from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from typing import TypedDict

from pypdf import PdfReader

# ── Tuneable constants ────────────────────────────────────────────────────────
CHUNK_SIZE = 900       # characters  (~180-220 tokens for English prose)
CHUNK_OVERLAP = 120    # characters  (~13 % overlap to preserve context at seams)
MIN_CHARS = 60         # skip chunks / pages shorter than this


class Chunk(TypedDict):
    id: str            # deterministic UUID string
    text: str
    source: str        # filename
    page: int          # 1-based
    chunk_index: int   # global sequence number across the whole PDF


def _make_id(source: str, page: int, char_offset: int) -> str:
    """Create a deterministic UUID5 from (source, page, offset)."""
    key = f"{source}::{page}::{char_offset}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def extract_chunks(
    pdf_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Generator[Chunk, None, None]:
    """
    Yield Chunk dicts from *pdf_path*.

    Parameters
    ----------
    pdf_path:     Path to the PDF file.
    chunk_size:   Maximum characters per chunk.
    chunk_overlap: Characters shared between adjacent chunks on the same page.
    """
    source_name = os.path.basename(pdf_path)
    reader = PdfReader(pdf_path)
    global_index = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        # Normalise: collapse runs of whitespace/newlines to a single space
        text = " ".join(raw_text.split())

        if len(text) < MIN_CHARS:
            continue  # blank or image-only page

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if len(chunk_text) >= MIN_CHARS:
                yield Chunk(
                    id=_make_id(source_name, page_num, start),
                    text=chunk_text,
                    source=source_name,
                    page=page_num,
                    chunk_index=global_index,
                )
                global_index += 1

            if end >= len(text):
                break
            start = end - chunk_overlap
