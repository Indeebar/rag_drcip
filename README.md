# DRCIP RAG API

Standalone retrieval-augmented question-answering API for a disaster-management
chatbot. FastAPI + Qdrant Cloud (vector search and embeddings) + Groq (answer
generation), sized to run on Render's free tier.

**Phase 1 of 6 is complete.** Implemented so far: project skeleton, validated
configuration, secret hygiene, request correlation, uniform error envelope, health
endpoint. The `/v1/ask` contract below is frozen but not yet served.

---

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Skeleton, config, `.gitignore`, health endpoint, tests | ✅ done |
| 2 | Ingestion CLI (PDF → section-aware chunks → Qdrant) | ⬜ |
| 3 | Retrieval + `POST /v1/ask` skeleton, auth, rate limiting | ⬜ |
| 4 | Groq answer generation, citation validation, abstention | ⬜ |
| 5 | Retrieval evaluation + tuning (`MIN_SCORE`, `top_k`) | ⬜ |
| 6 | README/API docs, Render deployment, smoke test | ⬜ |

---

## Corpus caveat (read before ingesting)

`corpus/disaster.pdf` is the **DRCIP Knowledge Base v1.0**, compiled 25 Oct 2026.
Its own cover states: *"Not an official government publication."* Sections carry
`STATUS` labels (`VERIFIED` / `CURATED`) and a `SOURCE` line naming the document
they were compiled from. The API reproduces those labels verbatim and never
upgrades them: an answer is described as *sourced from the corpus*, not as
official government guidance.

The document contains **no copyright, licence or redistribution statement**, and
its metadata credits the "DRCIP project". The PDF is therefore **git-ignored and
not published with this repository** until the corpus owner confirms rights in
writing. See [`corpus/README.md`](corpus/README.md) for the full inspection
(pages, structure, metadata, rights findings).

`STATUS: VERIFIED` in this corpus means *verified by the corpus compiler*, not
verified by a government authority. Response wording must never imply otherwise.

---

## Quick start (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

Copy-Item .env.example .env      # then fill in the blanks
python -m uvicorn app.main:app --reload
```

* API: <http://127.0.0.1:8000>
* Interactive docs: <http://127.0.0.1:8000/docs>
* Health: <http://127.0.0.1:8000/health>

Generate an API key for `API_KEYS`:

```powershell
python -c "import secrets; print('drcip_' + secrets.token_urlsafe(32))"
```

## Tests and checks

```powershell
pytest                      # unit + contract tests, no network, no real credentials
ruff check . && ruff format --check .
python scripts/check_no_secrets.py   # must print PASS before every commit
```

Tests never contact Qdrant or Groq. Tests marked `live` are skipped by default
(`pytest -m live` when you deliberately want a real end-to-end check).

## Verified provider capabilities

Checked against this project's own accounts on 25 Oct 2026:

| Model | Chat | Strict JSON schema | Probe latency | Free-tier limits |
|---|---|---|---|---|
| `qwen/qwen3.8-27b` (default) | ✅ | ✅ | 209–248 ms | 1,000 RPD · 8,000 TPM |
| `openai/gpt-oss-20b` (fallback) | ✅ | ✅ | 404–406 ms | 1,000 RPD · 8,000 TPM |
| `openai/gpt-oss-120b` | ✅ | ✅ | 701–776 ms | 1,000 RPD · 8,000 TPM |
| `allam-2-7b` | ✅ | ❌ | 169 ms | 7,000 RPD · 6,000 TPM |

`llama-3.3-70b-versatile` is not available to this account's free tier; it is not
used. `GROQ_MODEL` / `GROQ_MODEL_FALLBACK` are environment variables, so switching
models requires no code change. The API treats an unavailable model as a
recoverable upstream error and returns `502 upstream_error`, never a broken
answer.

---

## Configuration

All settings come from environment variables; see `.env.example` for the full
annotated list. Required: `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`,
`API_KEYS`. Anything missing fails at startup with a clear validation error
instead of failing on the first request.

Credentials are held as `SecretStr`, excluded from `repr()`, and never appear in
startup logs (`Settings.safe_summary()` reports counts and hostnames only).

---

## Security

* Git-ignored `.env`; committed template only. `python scripts/check_no_secrets.py`
  scans tracked files for tracked secret files and for the literal values of the
  secrets in your local `.env`.
* `X-API-Key` authentication for `/v1/*` (shared key, rotation supported by
  listing two keys in `API_KEYS`). `/health` stays public for uptime monitors.
* The key is a **server-to-server** credential. Never ship it in frontend
  JavaScript: any browser caller must proxy through the integrator's backend.
* No CORS headers are sent unless `ALLOWED_ORIGINS` is set.
* Every response carries `X-Request-ID` (echoed if the caller supplies a sane one),
  and every error uses one envelope: `{"error": {"code", "message", "request_id"}}`.

## Render deployment notes (for the integrating engineer)

* Free instance: 512 MB RAM, 0.1 CPU, sleeps after 15 minutes idle, ~1 minute cold
  start. Use a **60 s client timeout with one retry**; the first request after
  idle can take a minute.
* `GET /health` is cheap and makes no external calls — suitable for both Render's
  health check and external uptime pingers (pinging it every ~10 minutes keeps the
  service awake).
* Groq free tier is the throughput ceiling: 8,000 tokens/minute and 1,000
  requests/day per model. Budget roughly 70–90 questions/day at default settings.
  Exceeding it returns `429 rate_limited` with `Retry-After`; quota exhaustion
  surfaces as `502 upstream_error`.
* Free Qdrant clusters suspend after 7 idle days and are deleted after 4 weeks —
  check the cluster monthly.

## Layout

```
app/            deployed API (no PDF parsing, no local ML)
  api/          HTTP routes
  config.py     validated settings
  errors.py     single error envelope
  observability.py  request ids, access logs, safety headers
  schemas.py    public contract models
ingest/         local-only ingestion CLI (phase 2)
corpus/         source PDF(s) + generated manifest
scripts/        operator tooling (secret scan, evaluation, smoke tests)
tests/          unit + contract tests
```
