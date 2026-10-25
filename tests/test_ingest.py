"""
Unit tests for the PDF chunker.

These tests run fully offline — no Qdrant, no Groq, no model loading.
"""
from __future__ import annotations

from ingest.chunker import CHUNK_OVERLAP, CHUNK_SIZE, extract_chunks


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChunkerBasic:
    def test_single_short_page_yields_one_chunk(self, monkeypatch):
        text = "Evacuate immediately when a Flood Alert Level 3 is declared."

        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader([text]),
        )
        chunks = list(extract_chunks("fake.pdf"))
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["page"] == 1
        assert chunks[0]["source"] == "fake.pdf"

    def test_long_page_yields_multiple_chunks(self, monkeypatch):
        # Create text definitely longer than CHUNK_SIZE
        text = "word " * 300  # 1500 chars
        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader([text]),
        )
        chunks = list(extract_chunks("fake.pdf"))
        assert len(chunks) > 1

    def test_chunk_size_respected(self, monkeypatch):
        text = "A" * (CHUNK_SIZE * 3)
        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader([text]),
        )
        chunks = list(extract_chunks("fake.pdf"))
        for c in chunks:
            assert len(c["text"]) <= CHUNK_SIZE

    def test_empty_page_skipped(self, monkeypatch):
        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader(["", "   ", "Real content here for testing."]),
        )
        chunks = list(extract_chunks("fake.pdf"))
        # Only the third page should produce chunks
        assert all(c["page"] == 3 for c in chunks)

    def test_page_metadata_is_accurate(self, monkeypatch):
        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader([
                "Page one content about flood evacuation procedures and emergency response.",
                "Page two content about earthquake safety guidelines and shelter locations.",
            ]),
        )
        chunks = list(extract_chunks("fake.pdf"))
        pages = [c["page"] for c in chunks]
        assert 1 in pages
        assert 2 in pages

    def test_chunk_ids_are_deterministic(self, monkeypatch):
        text = "Some consistent text for testing determinism."
        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader([text]),
        )
        run1 = [c["id"] for c in extract_chunks("fake.pdf")]
        run2 = [c["id"] for c in extract_chunks("fake.pdf")]
        assert run1 == run2

    def test_chunk_index_is_global_sequential(self, monkeypatch):
        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader(
                ["First page with some content.", "Second page with more content."]
            ),
        )
        chunks = list(extract_chunks("fake.pdf"))
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_shares_content(self, monkeypatch):
        # Long enough page to get at least two chunks
        text = ("The quick brown fox jumped over the lazy dog. " * 30)
        monkeypatch.setattr(
            "ingest.chunker.PdfReader",
            lambda path: _FakePdfReader([text]),
        )
        chunks = list(extract_chunks("fake.pdf"))
        if len(chunks) >= 2:
            # End of chunk 0 and start of chunk 1 should overlap
            tail = chunks[0]["text"][-CHUNK_OVERLAP:]
            # They share some characters (overlap region)
            assert tail[:20] in chunks[1]["text"]


# ── Fake pypdf reader ─────────────────────────────────────────────────────────

class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    def __init__(self, page_texts: list[str]):
        self.pages = [_FakePage(t) for t in page_texts]
