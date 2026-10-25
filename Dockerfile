# ── Stage: runtime ────────────────────────────────────────────────────────────
# python:3.11-slim keeps the base image under 150 MB.
FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf / sentence-transformers (no build-essential needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install CPU-only PyTorch FIRST ───────────────────────────────────────────
# Must happen before sentence-transformers so pip doesn't pull the GPU build.
# GPU PyTorch ≈ 2 GB; CPU-only ≈ 180 MB — critical for Render's 512 MB limit.
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Install application dependencies ─────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model into the image ───────────────────────────────
# Baking the model into the image layer means:
#   • No network download at startup (no timeout risk on cold start)
#   • Consistent behaviour across deployments
# The model weighs ~22 MB on disk.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('all-MiniLM-L6-v2'); \
               print('Model cached successfully.')"

# ── Copy application code ─────────────────────────────────────────────────────
# Copy only the app/ directory — ingest/ and tests/ are not needed at runtime.
COPY app/ ./app/

# ── Runtime config ────────────────────────────────────────────────────────────
# Render injects $PORT (default 10000). Uvicorn binds to it.
ENV PORT=10000
EXPOSE 10000

# Single worker is correct for Render's free single-vCPU instance.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "10000", \
     "--workers", "1", \
     "--log-level", "info"]
