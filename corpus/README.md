# Corpus

Source documents for ingestion live here. **PDFs in this folder are git-ignored on
purpose** (`corpus/*.pdf`): redistribution rights have not been established, so the
corpus is not published with the repository. Only this note and the generated
`manifest.json` are tracked.

## `disaster.pdf` — provenance inspection (Phase 1, 25 Oct 2026)

| Property | Finding |
|---|---|
| Title (PDF metadata) | *DRCIP Knowledge Base – Disaster Management in India (RAG Corpus)* |
| Author (PDF metadata) | `DRCIP project` |
| Producer / creator | ReportLab PDF Library; creator "(unspecified)" |
| Created / modified | 2026-10-24 20:14 UTC |
| Pages | 37 |
| Extracted text | ~138,700 characters / ~22,200 words (real text layer — not a scan) |
| Structure | `Part N` → `N.M Section title` → `[SEC N.M \| HAZARD: … \| PHASE: … \| AUTHORITY: … \| STATUS: … \| SOURCE: …]` markers, plus many tables |
| Official status | **Not an official government publication** (stated on every page header) |
| Copyright / licence / redistribution statement | **None found.** Searched for "copyright", "licence", "redistribute", "all rights", "©", "public domain" — no rights statement in the document. |
| `STATUS` labels used | `VERIFIED` and `CURATED`, section by section |

### Consequences for the product

1. **Do not redistribute the PDF** from this repository until the corpus owner
   confirms rights in writing. Keep it local, or move it to private storage later.
2. **Do not describe answers as official government guidance.** Every response
   carries a disclaimer, and citations reproduce each section's own `STATUS` and
   `SOURCE` labels verbatim. The API never upgrades `CURATED` to `VERIFIED`.
3. `VERIFIED` in this corpus means "verified by the corpus compiler", not "verified
   by a government authority" — the response wording must reflect that distinction.
4. Because a text layer exists, no OCR is required. Table-heavy pages still need
   layout-aware extraction (pdfplumber) to keep rows readable for retrieval.

### Getting the file

Place the PDF at `corpus/disaster.pdf` before running ingestion (Phase 2):

```powershell
# already in place in this working copy
python -m ingest --pdf corpus/disaster.pdf --recreate --verify
```
