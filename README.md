# DRCIP Disaster Management RAG API

A **production-ready**, lightweight RAG (Retrieval-Augmented Generation) API that answers questions about disaster-management documents with accurate page citations.

Built with **FastAPI + Qdrant Cloud + Groq** — designed to run on Render's free tier.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               Developer Machine (local)             │
│  disaster.pdf ──► ingest.py ──► Qdrant Cloud        │
│  (run once before deploying)                        │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│         Render Free Web Service (Docker)            │
│                                                     │
│  POST /query                                        │
│    │                                                │
│    ├─ 1. Embed question  (all-MiniLM-L6-v2, CPU)   │
│    ├─ 2. Vector search   (Qdrant Cloud)             │
│    ├─ 3. Score threshold check → abstain if low     │
│    ├─ 4. Generate answer (Groq LLM)                 │
│    └─ 5. Return answer + page citations             │
└─────────────────────────────────────────────────────┘
```

| Component | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| Embedding | `all-MiniLM-L6-v2` (384-dim, CPU) |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Vector DB | Qdrant Cloud (free tier, 1 GB RAM) |
| PDF parsing | pypdf |
| Deployment | Render (Docker, free tier) |

---

## Prerequisites

- Python 3.11+
- A [Qdrant Cloud](https://cloud.qdrant.io/) account (free cluster)
- A [Groq](https://console.groq.com/) API key (free tier)

---

## Local Setup

### 1. Clone and configure

```bash
git clone <your-repo-url>
cd rag_drcip

cp .env.example .env
# Edit .env with your real credentials
```

### 2. Install dependencies

```bash
# CPU-only PyTorch (MUST be first to avoid the 2 GB GPU build)
pip install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Application + dev dependencies
pip install -r requirements.txt -r requirements-dev.txt
```

### 3. Ingest your PDF

Run this **once locally** before deploying. The API on Render is read-only.

```bash
python -m ingest.ingest --pdf disaster.pdf
```

Options:

```bash
# Reset collection and re-ingest (useful after PDF updates)
python -m ingest.ingest --pdf disaster.pdf --reset

# Custom collection name
python -m ingest.ingest --pdf disaster.pdf --collection my_collection

# Verify available Groq models
python -c "from groq import Groq; import os; c = Groq(api_key=os.environ['GROQ_API_KEY']); [print(m.id) for m in c.models.list().data]"
```

### 4. Run the API locally

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## API Reference

### Authentication

Every request to `/query` requires a bearer token:

```
Authorization: Bearer <your_api_key>
```

API keys are set in the `API_KEYS` environment variable (comma-separated for rotation).

---

### `POST /query`

Ask a disaster-management question.

**Request:**

```json
{
  "question": "What are the evacuation procedures for coastal flooding?"
}
```

| Field | Type | Required | Constraints |
|---|---|---|---|
| `question` | string | ✅ | 3–500 characters |

**Response — answer found (HTTP 200):**

```json
{
  "answer": "Coastal flooding evacuation procedures include immediate evacuation when Flood Alert Level 3 is declared. Residents must...",
  "sources": [
    {
      "document": "disaster.pdf",
      "page": 12,
      "excerpt": "Residents must evacuate within 2 hours of a Flood Alert Level 3 declaration..."
    },
    {
      "document": "disaster.pdf",
      "page": 13,
      "excerpt": "Designated evacuation routes are marked with green directional signage..."
    }
  ],
  "model": "openai/gpt-oss-20b",
  "chunks_retrieved": 5
}
```

**Response — out of scope / abstention (HTTP 200):**

```json
{
  "answer": "I could not find sufficient information in the available disaster-management documents to answer this question reliably. Please consult official sources or rephrase your query.",
  "sources": [],
  "model": "openai/gpt-oss-20b",
  "chunks_retrieved": 0
}
```

> **Tip for consuming apps:** check `sources == []` to detect abstention programmatically.

**Error responses:**

| HTTP | Condition |
|---|---|
| 401 | Invalid or missing bearer token |
| 422 | Request body fails validation (e.g. question > 500 chars) |
| 503 | Qdrant or Groq unreachable |

---

### `GET /health`

Public liveness probe — no authentication required.

```json
{
  "status": "ok",
  "qdrant": "connected",
  "collection": "disaster_docs",
  "model": "all-MiniLM-L6-v2",
  "model_loaded": true
}
```

`qdrant` values: `"connected"` | `"collection_missing"` | `"error"`

---

### `GET /`

API discovery.

```json
{
  "name": "DRCIP Disaster Management RAG API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `QDRANT_URL` | ✅ | — | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | ✅ | — | Qdrant Cloud API key |
| `GROQ_API_KEY` | ✅ | — | Groq API key |
| `API_KEYS` | ✅ | — | Comma-separated bearer tokens |
| `QDRANT_COLLECTION` | — | `disaster_docs` | Collection name |
| `GROQ_MODEL` | — | `openai/gpt-oss-20b` | Groq model ID |
| `TOP_K` | — | `5` | Chunks retrieved per query |
| `SCORE_THRESHOLD` | — | `0.30` | Min cosine similarity for abstention |
| `DATABASE_URL` | — | — | Neon PostgreSQL (future use) |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests run fully offline — Qdrant and Groq are mocked.

---

## Render Deployment

### Option A — One-click blueprint (recommended)

1. Push repo to GitHub
2. Render Dashboard → **New → Blueprint**
3. Connect your repo — Render reads `render.yaml` automatically
4. Set the four secret env vars in the Render dashboard:
   - `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`, `API_KEYS`
5. Click **Deploy**

### Option B — Manual

1. Render Dashboard → **New → Web Service**
2. Connect your GitHub repo
3. Set **Language: Docker**, **Plan: Free**
4. Add all environment variables from the table above
5. Click **Create Web Service**

### Verify deployment

```bash
curl https://your-service.onrender.com/health
```

### Test the deployed API

```bash
curl -X POST https://your-service.onrender.com/query \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the emergency response protocol for earthquakes?"}'
```

---

## Keep-Alive Setup (Important!)

Render's free tier sleeps after **15 minutes of inactivity** (60-second cold start).  
Qdrant's free cluster suspends after **7 days of inactivity**.

**Fix both for free using [cron-job.org](https://cron-job.org):**

1. Sign up at cron-job.org (free)
2. Create a cron job:
   - URL: `https://your-service.onrender.com/health`
   - Schedule: Every **13 minutes** (prevents Render sleep)
3. This also pings Qdrant's client, keeping the cluster alive

---

## Adding More Documents

To add another PDF to the knowledge base:

```bash
python -m ingest.ingest --pdf new_document.pdf
```

This upserts new chunks alongside existing ones. Existing chunks are not affected.

To fully replace the knowledge base:

```bash
python -m ingest.ingest --pdf new_document.pdf --reset
```

---

## Security Notes

- Bearer tokens are validated server-side against `API_KEYS` env var
- To rotate keys: add the new key to `API_KEYS` (comma-separated), deploy, then remove the old key and redeploy
- The LLM system prompt contains injection-resistance instructions
- HTTPS is terminated by Render's proxy — all traffic is encrypted in transit
- Never commit `.env` — it is in `.gitignore`

---

## Latency Expectations

| Condition | Typical latency |
|---|---|
| Warm request (keep-alive active) | 700 ms – 1.8 s |
| Cold start (no keep-alive) | + 50–60 seconds |
| Upgrade to Render Starter ($7/mo) | No cold starts |
