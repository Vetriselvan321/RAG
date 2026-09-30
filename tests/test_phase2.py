"""
tests/test_phase2.py

Tests for Phase 2: embeddings, vector storage, and LLM generation.

Run all:               pytest tests/test_phase2.py -v
Skip Qdrant tests:     pytest tests/test_phase2.py -v -k "not VectorStore"
Skip Gemini tests:     pytest tests/test_phase2.py -v -k "not Generator"
"""

import sys
import math
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.embeddings import Embedder
from rag.models import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url="http://localhost:6333", timeout=2).get_collections()
        return True
    except Exception:
        return False


def _llm_available() -> bool:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "gemini")
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    return bool(os.getenv("GEMINI_API_KEY"))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (mag_a * mag_b)


requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant not running on localhost:6333 — start with: "
           "docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant",
)

requires_gemini = pytest.mark.skipif(
    not _llm_available(),
    reason="No LLM API key set (GEMINI_API_KEY or GROQ_API_KEY)",
)


# ---------------------------------------------------------------------------
# Embedder — runs locally on CPU, no external services needed
# ---------------------------------------------------------------------------

class TestEmbedder:

    @pytest.fixture(scope="class")
    def embedder(self):
        return Embedder()

    def test_embed_query_returns_384_dims(self, embedder):
        vec = embedder.embed_query("What is the return policy?")
        assert isinstance(vec, list)
        assert len(vec) == 384

    def test_embed_texts_batch(self, embedder):
        texts = ["first sentence", "second sentence", "third sentence"]
        vecs = embedder.embed_texts(texts)
        assert len(vecs) == 3
        assert all(len(v) == 384 for v in vecs)

    def test_embed_query_returns_floats(self, embedder):
        vec = embedder.embed_query("hello world")
        assert all(isinstance(f, float) for f in vec)

    def test_dims_attribute_matches_output(self, embedder):
        vec = embedder.embed_query("test")
        assert embedder.dims == len(vec)

    def test_similar_texts_score_higher_than_unrelated(self, embedder):
        a = embedder.embed_query("How do I cancel my subscription?")
        b = embedder.embed_query("I want to end my membership")
        c = embedder.embed_query("The weather in Paris is sunny today")

        sim_ab = _cosine(a, b)
        sim_ac = _cosine(a, c)
        assert sim_ab > sim_ac, (
            f"Similar pair scored {sim_ab:.3f}, unrelated pair scored {sim_ac:.3f}"
        )

    def test_embed_query_and_embed_texts_are_consistent(self, embedder):
        text = "consistency check sentence"
        single = embedder.embed_query(text)
        batch = embedder.embed_texts([text])[0]
        assert single == batch


# ---------------------------------------------------------------------------
# VectorStore — requires Qdrant running on localhost:6333
# ---------------------------------------------------------------------------

@requires_qdrant
class TestVectorStore:

    _COLLECTION = "test_phase2_temp"

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        try:
            from qdrant_client import QdrantClient
            QdrantClient(url="http://localhost:6333").delete_collection(self._COLLECTION)
        except Exception:
            pass

    @pytest.fixture
    def store(self):
        from rag.vectorstore import VectorStore
        return VectorStore(collection=self._COLLECTION)

    @pytest.fixture(scope="class")
    def embedder(self):
        return Embedder()

    def _chunks(self, n: int = 3) -> list[Chunk]:
        return [
            Chunk(
                text=f"This is test chunk number {i} about product features.",
                doc_id="test-doc-1",
                chunk_index=i,
                source="test.txt",
            )
            for i in range(n)
        ]

    def test_upsert_returns_correct_count(self, store, embedder):
        chunks = self._chunks(3)
        vecs = embedder.embed_texts([c.text for c in chunks])
        assert store.upsert(chunks, vecs) == 3

    def test_count_after_upsert(self, store, embedder):
        chunks = self._chunks(5)
        vecs = embedder.embed_texts([c.text for c in chunks])
        store.upsert(chunks, vecs)
        assert store.count() == 5

    def test_search_returns_top_k_results(self, store, embedder):
        chunks = self._chunks(5)
        vecs = embedder.embed_texts([c.text for c in chunks])
        store.upsert(chunks, vecs)

        query_vec = embedder.embed_query("product features")
        results = store.search(query_vec, top_k=3)
        assert len(results) == 3

    def test_search_result_has_payload_and_score(self, store, embedder):
        chunks = self._chunks(2)
        vecs = embedder.embed_texts([c.text for c in chunks])
        store.upsert(chunks, vecs)

        query_vec = embedder.embed_query("test chunk")
        results = store.search(query_vec, top_k=1)
        payload, score = results[0]

        assert "text" in payload
        assert "source" in payload
        assert "chunk_index" in payload
        assert "doc_id" in payload
        assert 0.0 <= score <= 1.0

    def test_search_ranks_semantically_relevant_chunk_first(self, store, embedder):
        chunks = [
            Chunk(text="The refund policy allows returns within 30 days.",
                  doc_id="d1", chunk_index=0, source="policy.txt"),
            Chunk(text="Our company was founded in 2010 in San Francisco.",
                  doc_id="d1", chunk_index=1, source="about.txt"),
            Chunk(text="Contact our support team at support@example.com.",
                  doc_id="d1", chunk_index=2, source="contact.txt"),
        ]
        vecs = embedder.embed_texts([c.text for c in chunks])
        store.upsert(chunks, vecs)

        query_vec = embedder.embed_query("What is the return or refund policy?")
        results = store.search(query_vec, top_k=3)
        top_text = results[0][0]["text"].lower()
        assert "refund" in top_text or "return" in top_text

    def test_empty_store_returns_no_results(self, store, embedder):
        query_vec = embedder.embed_query("anything")
        results = store.search(query_vec, top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# Generator — requires GEMINI_API_KEY in .env
# ---------------------------------------------------------------------------

def test_generator_raises_without_api_key(monkeypatch):
    import rag.generator as gen_module
    monkeypatch.setattr(gen_module, "GEMINI_API_KEY", None)
    monkeypatch.setattr(gen_module, "GROQ_API_KEY", None)
    monkeypatch.setattr(gen_module, "LLM_PROVIDER", "gemini")
    from rag.generator import Generator
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        Generator()


def _call_or_skip(fn):
    """Call fn(); skip the test if Gemini returns a rate-limit / quota error."""
    try:
        return fn()
    except Exception as exc:
        msg = str(exc)
        if "ResourceExhausted" in type(exc).__name__ or "429" in msg or "quota" in msg.lower():
            pytest.skip(f"Gemini quota/rate-limit: {msg[:120]}")
        raise


@requires_gemini
class TestGenerator:

    @pytest.fixture(scope="class")
    def generator(self):
        from rag.generator import Generator
        return Generator()

    def test_generate_returns_non_empty_string(self, generator):
        context = [{"text": "The refund policy allows returns within 30 days.", "source": "policy.txt"}]
        answer = _call_or_skip(lambda: generator.generate("What is the refund policy?", context))
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0

    def test_generate_uses_context_content(self, generator):
        context = [{"text": "The company headquarters is located in Tokyo, Japan.", "source": "about.txt"}]
        answer = _call_or_skip(lambda: generator.generate("Where is the company located?", context))
        assert "tokyo" in answer.lower() or "japan" in answer.lower()

    def test_generate_with_multiple_context_chunks(self, generator):
        context = [
            {"text": "Returns are accepted within 30 days.", "source": "policy.txt"},
            {"text": "Items must be in original packaging to qualify for a return.", "source": "policy.txt"},
        ]
        answer = _call_or_skip(lambda: generator.generate("What are the return conditions?", context))
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
