FROM python:3.11-slim

WORKDIR /app

# Build tools needed for sentence-transformers and numpy compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so Docker layer-caches them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source after deps to maximise cache hits on code-only changes
COPY . .

# HF Spaces requires port 7860; override with PORT env var for local use
ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn rag.api:app --host 0.0.0.0 --port ${PORT}"]
