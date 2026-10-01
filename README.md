---
title: DocuRetrieve API
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<!--
  ^ The YAML block above is REQUIRED by HuggingFace Spaces — it configures the
  Docker Space and MUST stay at the very top (line 1) of this file. Do NOT
  remove it when editing this README, or the Space fails with
  "Missing configuration in README".
-->

# RAG Framework

> Production-grade Retrieval Augmented Generation - built from scratch, $0/month forever.

[![CI](https://github.com/ashimalaiya/rag-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/ashimalaiya/rag-framework/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=nextdotjs&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-200%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Cost](https://img.shields.io/badge/cost-%240%2Fmonth-brightgreen)

A full-stack RAG framework with hybrid BM25 + vector search, cross-encoder reranking, inline citations, multi-modal vision queries, streaming responses, and a built-in RAGAS evaluation engine. The dashboard supports Bring Your Own Key (BYOK), so users can enter their own Groq and Gemini API keys from the Settings page.

---

## What makes this different

Most RAG projects are thin wrappers around an LLM API. This one implements the complete retrieval stack from scratch.

| Capability | Basic RAG | LangChain | This project |
|---|---|---|---|
| Search strategy | Vector only | Vector only | **BM25 + Vector + RRF fusion** |
| Reranking | ❌ | Paid add-on | **Cross-encoder (free, local)** |
| Inline citations | ❌ | ❌ | **[1][2] markers with source tracking** |
| Hallucination detection | ❌ | External | **Built-in RAGAS (LLM-as-judge)** |
| Vision / image queries | ❌ | Optional | **Gemini Vision + RAG** |
| Evaluation metrics | ❌ | External | **Faithfulness · Relevance · Recall** |
| Monthly cost | $30–80 | $30–80 | **$0** |
| Explainability | Black box | Black box | **Full citation + chunk trace** |

---

## Live Demo
https://rag-a51i0ly1a-vetriselvan-s-projects.vercel.app/

## Architecture

```
  Documents                       Indexing
  ─────────                       ────────
  PDF · HTML · TXT                load → chunk (512 tok, 64 tok overlap)
  MD · PNG · JPG  ──────────►            │
  GIF · WebP                             │ embed (all-MiniLM-L6-v2, 384-dim)
                                         ▼
                                  ┌─────────────┐
                                  │   Qdrant    │
                                  │ Vector DB   │
                                  └─────────────┘

  Query (text / image)            Retrieval
  ────────────────────            ─────────
                         ┌────────────────┐   ┌───────────────────┐
                    ────►│  BM25 Search   │   │   Vector Search   │
                         │  (rank-bm25)   │   │   (Qdrant ANN)    │
                         └───────┬────────┘   └────────┬──────────┘
                                 └──────────┬───────────┘
                                      RRF Fusion  (top 20)
                                            │
                                 ┌──────────▼──────────┐
                                 │  Cross-Encoder       │
                                 │  Reranker            │  top 5
                                 │  (ms-marco-MiniLM)   │
                                 └──────────┬───────────┘

                                        Generation
                                        ──────────
                              Gemini 2.0 Flash
                              context chunks + optional image
                                            │
                                            ▼
                              Answer with inline [1][2][3] citations
                              + RAGAS quality score on demand
```

---

## Features

### Phase 1 — Document Ingestion
- Loads PDF, HTML, plain text, Markdown, and images
- Token-aware semantic chunking (tiktoken, 512 token windows, configurable overlap)
- Metadata preserved: page numbers (PDF), title (HTML), source filename
- Graceful handling of corrupt files — failures reported, pipeline continues

### Phase 2 — Embeddings + Vector Storage
- Local bi-encoder: `all-MiniLM-L6-v2` (384-dim, runs on CPU, no API needed)
- Qdrant vector database (local Docker or Qdrant Cloud free tier)
- Upsert, search, and count operations with error handling

### Phase 3 — Hybrid Search
- BM25 keyword index built from stored chunks at query time
- Dense vector search via Qdrant approximate nearest neighbours
- Reciprocal Rank Fusion (RRF, k=60) merges both ranked lists
- Hybrid consistently outperforms vector-only retrieval on tail queries

### Phase 4 — Cross-Encoder Reranking + Citations
- `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores top 20 candidates
- Returns top 5 by true query–chunk relevance (not embedding similarity)
- Answer generation with inline `[N]` citation markers
- `CitedAnswer` dataclass tracks which chunks support which claims

### Phase 5 — RAGAS Evaluation (LLM-as-Judge)
- **Faithfulness** — are answer claims grounded in retrieved context?
- **Context Relevance** — are retrieved chunks relevant to the query?
- **Context Recall** — does the context cover the ground-truth answer?
- **Answer Relevance** — does the answer address what was asked?
- Each metric uses Gemini to score; no external RAGAS package needed

### Phase 6 — REST API + Dashboard
- FastAPI with Swagger UI (`/docs`) and ReDoc (`/redoc`)
- Endpoints: `/health`, `/stats`, `/upload`, `/search`, `/ask`, `/stream`, `/ask-image`
- Next.js 14 dashboard (TypeScript, Tailwind CSS, App Router)
- Real-time token streaming through Server-Sent Events (`/stream`)
- Docker + docker-compose for one-command local deployment
- Dockerfile configured for Hugging Face Spaces (port 7860)
- Settings page for user-provided Groq and Gemini API keys

### Phase 7 — Multi-Modal RAG (Vision + Text)
- **Image ingestion**: upload PNG/JPEG/GIF/WebP → Gemini describes it → stored as searchable text
- **Image queries**: `POST /ask-image` sends image pixels + retrieved text context to Gemini Vision
- Charts, diagrams, screenshots, and scanned documents become fully searchable
- Graceful fallback: answers from image alone when knowledge base is empty
- Gemini API keys for image queries are supplied by the user through the dashboard Settings page

---

## Bring Your Own Key (BYOK)

The dashboard is designed so public users can provide their own LLM API keys instead of using the application owner's keys for normal LLM requests.

### API key usage

| Feature | API key |
|---|---|
| Ask (`/ask`) | User's **Groq** API key |
| Stream (`/stream`) | User's **Groq** API key |
| Ask Image (`/ask-image`) | User's **Gemini** API key |
| Search (`/search`) | No LLM API key |
| Upload (`/upload`) | No LLM API key in the dashboard request |
| Stats (`/stats`) | No LLM API key |

Users can open **Settings** from the sidebar, enter their Groq and/or Gemini key, and save it for the current browser session. The frontend sends the selected key to the backend through the corresponding request header:

- `X-Groq-API-Key`
- `X-Gemini-API-Key`

API keys are kept in browser `sessionStorage` and are not intended to be stored as application data on the backend. Use HTTPS in production.

## Quick start

### Prerequisites

- Python 3.10+
- Docker Desktop (for Qdrant)
- Node.js 18+ (for the dashboard)
- A Groq API key for text generation and streaming
- A Gemini API key for image queries

### 1 — Clone and install

```bash
git clone https://github.com/YOUR-USERNAME/rag-framework.git
cd rag-framework
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
```

For the dashboard, LLM API keys are entered through **Settings** rather than being placed in the frontend. Add your Groq key for text/streaming and your Gemini key for image queries in the Settings page.

The backend still uses `.env` for infrastructure configuration such as Qdrant. Keep `.env` private and never commit real API keys.


### 3 — Start Qdrant

```bash
docker compose up qdrant -d
```

Qdrant UI is available at `http://localhost:6333/dashboard`.

### 4 — Start the API server

```bash
uvicorn rag.api:app --reload --port 8000
```

Swagger UI: `http://localhost:8000/docs`

### 5 — Start the dashboard

```bash
cd dashboard
npm install
npm run dev
```

Dashboard: `http://localhost:3000`

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — no external deps |
| `GET` | `/stats` | Qdrant collection statistics |
| `POST` | `/upload` | Ingest a file (PDF · TXT · MD · HTML · PNG · JPEG · GIF · WebP) |
| `POST` | `/search` | Hybrid BM25 + vector search with cross-encoder reranking |
| `POST` | `/ask` | Full RAG pipeline — retrieval → reranking → cited answer; requires `X-Groq-API-Key` |
| `POST` | `/stream` | Streaming RAG response using Server-Sent Events; requires `X-Groq-API-Key` |
| `POST` | `/ask-image` | Upload an image and ask a question about it; requires `X-Gemini-API-Key` |
| `GET` | `/docs` | Swagger UI (interactive) |
| `GET` | `/redoc` | ReDoc documentation |

### Example: ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-Groq-API-Key: YOUR_GROQ_API_KEY" \
  -d '{"query": "What are the main findings?", "top_k": 5}'
```

```json
{
  "query": "What are the main findings?",
  "answer": "The study found three key results [1][2]. First, ... [1]. Second, ... [2].",
  "citations": [{"source": "report.pdf", "text": "..."}],
  "cited_sources": ["report.pdf"]
}
```

### Example: upload a document

```bash
# Linux / macOS / Windows (curl.exe)
curl -X POST http://localhost:8000/upload -F "file=@report.pdf"
```

```json
{"filename": "report.pdf", "chunks_stored": 42, "message": "Successfully ingested 42 chunks from 'report.pdf'."}
```

### Example: ask about an image

```bash
curl -X POST http://localhost:8000/ask-image \
  -H "X-Gemini-API-Key: YOUR_GEMINI_API_KEY" \
  -F "file=@chart.png" \
  -F "query=What trend does this chart show?" \
  -F "top_k=5"
```

---

## RAGAS evaluation

Run the built-in evaluation suite on your knowledge base:

```bash
# Run evaluation tests (requires Qdrant + Gemini key)
pytest tests/test_phase5.py -v

# Or use the evaluator directly in Python
python - <<'EOF'
from rag.evaluator import Evaluator
from rag.models import EvalSample

samples = [
    EvalSample(
        query="What is the refund policy?",
        ground_truth="Refunds are processed within 5 business days.",
        answer="...",       # your pipeline's answer
        contexts=["..."],   # retrieved chunk texts
    )
]

evaluator = Evaluator()
report = evaluator.evaluate(samples)
print(f"Faithfulness:       {report.faithfulness:.2f}")
print(f"Context Relevance:  {report.context_relevance:.2f}")
print(f"Context Recall:     {report.context_recall:.2f}")
print(f"Answer Relevance:   {report.answer_relevance:.2f}")
print(f"Overall:            {report.overall:.2f}")
EOF
```

Example output:

```
Faithfulness:       0.89
Context Relevance:  0.84
Context Recall:     0.81
Answer Relevance:   0.86
Overall:            0.85
```

> Run this on your own documents and replace these numbers in your README.

---

## Running tests

```bash
# All tests (200 total)
pytest tests/ -v

# Only tests that need no external services
pytest tests/ -v -k "not requires_qdrant and not requires_pipeline and not requires_gemini"

# Individual phase
pytest tests/test_phase3.py -v   # hybrid search
pytest tests/test_vision.py -v   # multi-modal
```

Test breakdown across phases:

| File | Tests | Requires |
|---|---|---|
| `test_phase1.py` | 32 | Nothing |
| `test_phase2.py` | 26 | Qdrant |
| `test_phase3.py` | 24 | Qdrant |
| `test_phase4.py` | 28 | Qdrant + Gemini |
| `test_phase5.py` | 28 | Qdrant + Gemini |
| `test_phase6.py` | 32 | Qdrant (most) |
| `test_vision.py` | 31 | Gemini (most) |

---

## Project structure

```
rag-framework/
├── rag/                        # Core Python package
│   ├── config.py               # Env-based configuration
│   ├── models.py               # Document, Chunk, CitedAnswer, EvalSample
│   ├── loaders.py              # PDF · HTML · TXT · MD · image loaders
│   ├── chunker.py              # Token-aware semantic chunker (tiktoken)
│   ├── ingestion.py            # High-level ingestion pipeline
│   ├── embeddings.py           # Bi-encoder (all-MiniLM-L6-v2)
│   ├── vectorstore.py          # Qdrant wrapper (upsert, search, count)
│   ├── bm25_retriever.py       # BM25 keyword index (rank-bm25)
│   ├── hybrid_retriever.py     # BM25 + vector + RRF fusion
│   ├── reranker.py             # Cross-encoder reranker (ms-marco)
│   ├── generator.py            # Gemini / Groq answer generation
│   ├── vision.py               # Gemini 2.0 Flash vision analysis
│   ├── evaluator.py            # RAGAS-style LLM-as-judge evaluation
│   └── api.py                  # FastAPI REST API
├── dashboard/                  # Next.js 14 web interface
│   ├── app/
│   │   ├── page.tsx            # Text query page
│   │   ├── stream/             # Streaming query page
│   │   ├── ask-image/          # Vision query page
│   │   ├── upload/             # Document upload page
│   │   └── settings/           # BYOK API key settings
│   ├── components/
│   │   ├── AnswerCard.tsx      # Cited answer display
│   │   ├── UploadForm.tsx      # Drag-and-drop uploader
│   │   ├── ImageQueryForm.tsx  # Image upload + question form
│   │   └── Sidebar.tsx         # Main navigation + Settings link
│   ├── context/
│   │   └── APIKeyContext.tsx   # Browser-session BYOK key state
│   └── lib/api.ts              # Typed API client + BYOK headers
├── tests/                      # 200 tests across 7 modules
├── eval/                       # Sample evaluation datasets
├── Dockerfile                  # Optimised for Hugging Face Spaces
├── docker-compose.yml          # Qdrant + API, one command
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Optional server-side Gemini key; dashboard image requests use the user's `X-Gemini-API-Key` |
| `LLM_PROVIDER` | `gemini` | `gemini` (default, supports vision) or `groq` |
| `LLM_MODEL` | `gemini-2.0-flash` | Model name for the selected provider |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint (local or cloud) |
| `QDRANT_API_KEY` | — | Only needed for Qdrant Cloud |
| `QDRANT_COLLECTION` | `rag_chunks` | Collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `GROQ_API_KEY` | — | Optional server-side Groq key; dashboard text/stream requests use the user's `X-Groq-API-Key` |
| `RERANK_FETCH_N` | `20` | Candidates fetched before reranking |
| `RERANK_TOP_N` | `5` | Final results returned after reranking |

---

## Deployment — Hugging Face Spaces (free)

HF Spaces gives you a public URL, 16 GB RAM, 2 vCPUs, and auto-deploys from git push — at no cost.

### 1. Create a Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space):
- **SDK**: Docker
- **Hardware**: CPU basic (free)
- **Visibility**: Public

### 2. Add secrets

In your Space → **Settings → Variables and secrets**:

```
GEMINI_API_KEY     your-gemini-key
QDRANT_URL         https://your-cluster.qdrant.io
QDRANT_API_KEY     your-qdrant-cloud-key
```

Get a free Qdrant Cloud cluster (1 GB) at [cloud.qdrant.io](https://cloud.qdrant.io).

### 3. Deploy

```bash
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/rag-framework
git push hf main
```

Your live API will be at `https://YOUR-USERNAME-rag-framework.hf.space`.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| API framework | FastAPI + Uvicorn |
| Vector database | Qdrant |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Keyword search | rank-bm25 |
| Reranker | sentence-transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| LLM | Groq (text + streaming) · Gemini (vision) |
| Vision | Gemini Vision |
| Frontend | Next.js 14 · TypeScript · Tailwind CSS |
| Containerisation | Docker · docker-compose |
| Testing | pytest (200 tests) |
| Token counting | tiktoken |

---

## License

MIT — see [LICENSE](LICENSE) for details.
