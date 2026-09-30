"""Centralized configuration loaded from environment variables."""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # None for local
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMS = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
VISION_MODEL="gemini-3.8-flash"
# Phase 3 — hybrid search
RRF_K = int(os.getenv("RRF_K", "60"))               # RRF smoothing constant (paper default)
HYBRID_FETCH_K = int(os.getenv("HYBRID_FETCH_K", "20"))  # candidates per retriever before fusion

# Phase 4 — reranking + citation tracking
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_FETCH_N = int(os.getenv("RERANK_FETCH_N", "20"))  # candidates fetched before reranking
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))       # final results returned after reranking
