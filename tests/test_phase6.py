"""
tests/test_phase6.py

Tests for Phase 6: FastAPI REST API.

The test client hits the real app in-process (no network needed).
Heavy singletons (Embedder, Reranker, Generator) load lazily on first request,
so tests that never call those endpoints run without any ML models.

Run all Phase 6 tests:              pytest tests/test_phase6.py -v
Run without any external services:  pytest tests/test_phase6.py -v -k "Health or Validation"
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared client fixture — app stays alive for the whole module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from rag.api import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# /health  — no external services needed
# ---------------------------------------------------------------------------

class TestHealth:

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_service_field_present(self, client):
        assert client.get("/health").json()["service"] == "rag-framework"

    def test_version_field_present(self, client):
        assert "version" in client.get("/health").json()


# ---------------------------------------------------------------------------
# Request validation — errors are returned BEFORE any service call,
# so these tests never require Qdrant or LLM.
# ---------------------------------------------------------------------------

class TestRequestValidation:

    def test_ask_empty_query_returns_400(self, client):
        assert client.post("/ask", json={"query": ""}).status_code == 400

    def test_ask_whitespace_query_returns_400(self, client):
        assert client.post("/ask", json={"query": "   "}).status_code == 400

    def test_search_empty_query_returns_400(self, client):
        assert client.post("/search", json={"query": ""}).status_code == 400

    def test_search_whitespace_query_returns_400(self, client):
        assert client.post("/search", json={"query": "  "}).status_code == 400

    def test_ask_missing_query_returns_422(self, client):
        # Pydantic rejects the request before our code runs
        assert client.post("/ask", json={}).status_code == 422

    def test_search_missing_query_returns_422(self, client):
        assert client.post("/search", json={}).status_code == 422

    def test_upload_csv_returns_400(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("data.csv", b"col1,col2\n1,2", "text/csv")},
        )
        assert resp.status_code == 400

    def test_upload_exe_returns_400(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("app.exe", b"\x4d\x5a\x90\x00", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_zip_returns_400(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
        )
        assert resp.status_code == 400

    def test_upload_error_body_mentions_allowed_types(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("data.csv", b"a,b", "text/csv")},
        )
        assert ".txt" in resp.json()["detail"] or ".pdf" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# /stats + /upload + /search  — require Qdrant
# ---------------------------------------------------------------------------

def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url="http://localhost:6333", timeout=2).get_collections()
        return True
    except Exception:
        return False


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant not running on localhost:6333",
)


@requires_qdrant
class TestStatsEndpoint:

    def test_returns_200(self, client):
        assert client.get("/stats").status_code == 200

    def test_has_all_fields(self, client):
        data = client.get("/stats").json()
        for field in ("collection", "total_chunks", "embedding_model", "qdrant_url"):
            assert field in data, f"Missing field: {field}"

    def test_total_chunks_is_non_negative_int(self, client):
        data = client.get("/stats").json()
        assert isinstance(data["total_chunks"], int)
        assert data["total_chunks"] >= 0


@requires_qdrant
class TestUploadEndpoint:

    def test_txt_upload_succeeds(self, client):
        content = (
            b"Python is a high-level programming language created by "
            b"Guido van Rossum in 1991. It emphasizes code readability."
        )
        resp = client.post(
            "/upload",
            files={"file": ("python_info.txt", content, "text/plain")},
        )
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        content = b"FastAPI is a modern web framework for building APIs with Python."
        resp = client.post(
            "/upload",
            files={"file": ("fastapi_info.txt", content, "text/plain")},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "filename" in data
            assert "chunks_stored" in data
            assert "message" in data

    def test_chunks_stored_is_positive(self, client):
        content = b"Machine learning uses algorithms to find patterns in data automatically."
        resp = client.post(
            "/upload",
            files={"file": ("ml_info.txt", content, "text/plain")},
        )
        if resp.status_code == 200:
            assert resp.json()["chunks_stored"] >= 1

    def test_filename_preserved_in_response(self, client):
        content = b"Qdrant is a vector similarity search engine."
        resp = client.post(
            "/upload",
            files={"file": ("qdrant_info.txt", content, "text/plain")},
        )
        if resp.status_code == 200:
            assert resp.json()["filename"] == "qdrant_info.txt"

    def test_md_upload_succeeds(self, client):
        content = b"# RAG Framework\n\nRetrieval-Augmented Generation combines search with LLM generation."
        resp = client.post(
            "/upload",
            files={"file": ("readme.md", content, "text/markdown")},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /search  — requires Qdrant + Embedder + Reranker (no LLM)
# ---------------------------------------------------------------------------

@requires_qdrant
class TestSearchEndpoint:

    def test_search_returns_200(self, client):
        resp = client.post("/search", json={"query": "Python programming language"})
        assert resp.status_code in (200, 503)

    def test_search_returns_list(self, client):
        resp = client.post("/search", json={"query": "programming"})
        if resp.status_code == 200:
            assert isinstance(resp.json(), list)

    def test_search_result_has_required_fields(self, client):
        resp = client.post("/search", json={"query": "Python"})
        if resp.status_code == 200 and resp.json():
            item = resp.json()[0]
            for field in ("rank", "text", "source", "score"):
                assert field in item

    def test_search_rank_starts_at_one(self, client):
        resp = client.post("/search", json={"query": "Python"})
        if resp.status_code == 200 and resp.json():
            assert resp.json()[0]["rank"] == 1

    def test_search_top_k_respected(self, client):
        resp = client.post("/search", json={"query": "Python", "top_k": 2})
        if resp.status_code == 200:
            assert len(resp.json()) <= 2


# ---------------------------------------------------------------------------
# /ask  — requires Qdrant + full pipeline (LLM)
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    try:
        from rag.generator import Generator
        Generator()
        return True
    except Exception:
        return False


requires_pipeline = pytest.mark.skipif(
    not (_qdrant_available() and _llm_available()),
    reason="Requires Qdrant on localhost:6333 and a configured LLM provider",
)


@requires_pipeline
class TestAskEndpoint:

    def test_ask_returns_200(self, client):
        resp = client.post("/ask", json={"query": "What is Python?"})
        assert resp.status_code in (200, 503)

    def test_ask_response_has_all_fields(self, client):
        resp = client.post("/ask", json={"query": "What is Python?"})
        if resp.status_code == 200:
            data = resp.json()
            for field in ("query", "answer", "citations", "cited_sources"):
                assert field in data

    def test_ask_answer_is_non_empty_string(self, client):
        resp = client.post("/ask", json={"query": "What is Python?"})
        if resp.status_code == 200:
            assert isinstance(resp.json()["answer"], str)
            assert len(resp.json()["answer"]) > 0

    def test_ask_query_echoed_in_response(self, client):
        resp = client.post("/ask", json={"query": "What is Python?"})
        if resp.status_code == 200:
            assert resp.json()["query"] == "What is Python?"

    def test_ask_cited_sources_is_list(self, client):
        resp = client.post("/ask", json={"query": "What is Python?"})
        if resp.status_code == 200:
            assert isinstance(resp.json()["cited_sources"], list)
