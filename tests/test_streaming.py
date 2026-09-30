"""
tests/test_streaming.py

Tests for POST /stream — Server-Sent Events streaming endpoint.

Structure:
  TestStreamValidation   — no external services (runs in CI)
  TestStreamSSEFormat    — verifies SSE wire format with mocked generator
  TestStreamPipeline     — @requires_pipeline (Qdrant + LLM)

Run only the free tests:
  pytest tests/test_streaming.py -v -k "Validation or SSEFormat"

Run everything:
  pytest tests/test_streaming.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url="http://localhost:6333", timeout=2).get_collections()
        return True
    except Exception:
        return False


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


# ---------------------------------------------------------------------------
# Shared client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from rag.api import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse(text: str) -> list[dict | str]:
    """Parse SSE body into a list of parsed data payloads."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                events.append(payload)
    return events


# ---------------------------------------------------------------------------
# TestStreamValidation — no external services needed
# ---------------------------------------------------------------------------

class TestStreamValidation:

    def test_empty_query_returns_400(self, client):
        resp = client.post("/stream", json={"query": ""})
        assert resp.status_code == 400

    def test_whitespace_query_returns_400(self, client):
        resp = client.post("/stream", json={"query": "   "})
        assert resp.status_code == 400

    def test_valid_request_returns_200(self, client):
        # 200 even when Gemini/Qdrant are absent — errors surface as SSE events
        resp = client.post("/stream", json={"query": "What is RAG?"})
        assert resp.status_code == 200

    def test_content_type_is_event_stream(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_always_ends_with_done(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        events = _parse_sse(resp.text)
        assert "[DONE]" in events

    def test_done_is_last_event(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        events = _parse_sse(resp.text)
        assert events[-1] == "[DONE]"

    def test_cache_control_header_present(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        assert resp.headers.get("cache-control") == "no-cache"

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/stream", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TestStreamSSEFormat — mocked LLM, verifies the SSE event structure
# ---------------------------------------------------------------------------

class TestStreamSSEFormat:

    def _make_mock_generator(self, tokens: list[str]):
        mock_gen = MagicMock()
        mock_gen.generate_stream.return_value = iter(tokens)
        return mock_gen

    def test_token_events_have_type_and_text(self, client):
        tokens = ["Hello", " world", "!"]
        mock_gen = self._make_mock_generator(tokens)

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        events = _parse_sse(resp.text)
        token_events = [e for e in events if isinstance(e, dict) and e.get("type") == "token"]
        assert len(token_events) == 3
        assert all("text" in e for e in token_events)

    def test_token_texts_match_yielded_tokens(self, client):
        tokens = ["The", " answer", " is", " 42"]
        mock_gen = self._make_mock_generator(tokens)

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        events = _parse_sse(resp.text)
        texts = [e["text"] for e in events if isinstance(e, dict) and e.get("type") == "token"]
        assert texts == tokens

    def test_citations_event_is_emitted(self, client):
        mock_gen = self._make_mock_generator(["answer"])

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        events = _parse_sse(resp.text)
        citation_events = [e for e in events if isinstance(e, dict) and e.get("type") == "citations"]
        assert len(citation_events) == 1

    def test_citations_event_has_required_fields(self, client):
        mock_gen = self._make_mock_generator(["ok"])

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        events = _parse_sse(resp.text)
        citation_event = next(e for e in events if isinstance(e, dict) and e.get("type") == "citations")
        assert "citations" in citation_event
        assert "cited_sources" in citation_event

    def test_event_order_tokens_then_citations_then_done(self, client):
        mock_gen = self._make_mock_generator(["tok"])

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        events = _parse_sse(resp.text)
        types = [
            (e if e == "[DONE]" else e.get("type", "unknown"))
            for e in events if e
        ]
        # tokens must come before citations, citations before [DONE]
        assert types.index("token") < types.index("citations")
        assert types.index("citations") < types.index("[DONE]")

    def test_error_event_on_generator_failure(self, client):
        mock_gen = MagicMock()
        mock_gen.generate_stream.side_effect = RuntimeError("boom")

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        events = _parse_sse(resp.text)
        error_events = [e for e in events if isinstance(e, dict) and e.get("type") == "error"]
        assert len(error_events) == 1
        assert "message" in error_events[0]

    def test_done_still_sent_after_error(self, client):
        mock_gen = MagicMock()
        mock_gen.generate_stream.side_effect = RuntimeError("boom")

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        events = _parse_sse(resp.text)
        assert "[DONE]" in events

    def test_json_in_each_data_line_is_valid(self, client):
        mock_gen = self._make_mock_generator(["a", "b"])

        with patch("rag.api._get_generator", return_value=mock_gen), \
             patch("rag.api._get_retriever", side_effect=Exception("no qdrant")):
            resp = client.post("/stream", json={"query": "test"})

        for line in resp.text.splitlines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                json.loads(line[6:])  # must not raise


# ---------------------------------------------------------------------------
# TestStreamPipeline — requires Qdrant + LLM
# ---------------------------------------------------------------------------

@requires_pipeline
class TestStreamPipeline:

    def test_stream_returns_200(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        assert resp.status_code == 200

    def test_stream_produces_at_least_one_token(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        events = _parse_sse(resp.text)
        token_events = [e for e in events if isinstance(e, dict) and e.get("type") == "token"]
        assert len(token_events) > 0

    def test_concatenated_tokens_form_non_empty_string(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        events = _parse_sse(resp.text)
        answer = "".join(e["text"] for e in events if isinstance(e, dict) and e.get("type") == "token")
        assert len(answer) > 10

    def test_citations_event_present_in_real_stream(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        events = _parse_sse(resp.text)
        assert any(isinstance(e, dict) and e.get("type") == "citations" for e in events)

    def test_cited_sources_is_a_list(self, client):
        resp = client.post("/stream", json={"query": "What is RAG?"})
        events = _parse_sse(resp.text)
        citation_event = next(
            (e for e in events if isinstance(e, dict) and e.get("type") == "citations"), None
        )
        assert citation_event is not None
        assert isinstance(citation_event["cited_sources"], list)
