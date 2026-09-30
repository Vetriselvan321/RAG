"""
tests/test_phase4.py

Tests for Phase 4: cross-encoder reranking and citation tracking.

Run all Phase 4 tests:               pytest tests/test_phase4.py -v
Run only unit tests (no services):   pytest tests/test_phase4.py -v -k "not LLM and not Pipeline"
Run reranker tests (no API needed):  pytest tests/test_phase4.py -v -k "Reranker or CitedAnswer"
"""

import sys
import re
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.models import CitedAnswer


# ---------------------------------------------------------------------------
# CitedAnswer — pure unit tests, no external services
# ---------------------------------------------------------------------------

class TestCitedAnswer:

    def _ca(self, answer: str, citations: list[dict], query: str = "test query") -> CitedAnswer:
        return CitedAnswer(answer=answer, citations=citations, query=query)

    def test_citation_numbers_extracted(self):
        ca = self._ca("The sky is blue [1] and water is wet [2].", [])
        assert ca.citation_numbers == [1, 2]

    def test_citation_numbers_deduplicated(self):
        ca = self._ca("Topic [1] and also [1] again.", [])
        assert ca.citation_numbers == [1]

    def test_citation_numbers_sorted(self):
        ca = self._ca("See [3] and [1] and [2].", [])
        assert ca.citation_numbers == [1, 2, 3]

    def test_citation_numbers_empty_when_no_markers(self):
        ca = self._ca("No citations here.", [])
        assert ca.citation_numbers == []

    def test_cited_sources_unique_and_ordered(self):
        citations = [
            {"source": "doc_a.pdf", "text": "..."},
            {"source": "doc_b.pdf", "text": "..."},
            {"source": "doc_a.pdf", "text": "..."},
        ]
        ca = self._ca("Answer [1][2][3]", citations)
        assert ca.cited_sources == ["doc_a.pdf", "doc_b.pdf"]

    def test_cited_sources_preserves_first_occurrence_order(self):
        citations = [
            {"source": "z.txt", "text": "..."},
            {"source": "a.txt", "text": "..."},
        ]
        ca = self._ca("Answer [1][2]", citations)
        assert ca.cited_sources == ["z.txt", "a.txt"]

    def test_cited_sources_empty_for_no_citations(self):
        ca = self._ca("Answer", [])
        assert ca.cited_sources == []

    def test_cited_sources_handles_missing_source_key(self):
        citations = [{"text": "some text"}]
        ca = self._ca("Answer [1]", citations)
        assert ca.cited_sources == ["unknown"]

    def test_query_stored_on_dataclass(self):
        ca = self._ca("answer", [], query="my question")
        assert ca.query == "my question"


# ---------------------------------------------------------------------------
# Reranker — downloads model (~80MB on first run), CPU only, no API key
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reranker():
    from rag.reranker import Reranker
    return Reranker()


class TestReranker:

    def test_reranker_loads(self, reranker):
        assert reranker.model is not None

    def test_empty_candidates_returns_empty(self, reranker):
        assert reranker.rerank("query", []) == []

    def test_returns_list_of_tuples(self, reranker):
        candidates = [
            {"text": "The sky is blue", "source": "a.txt", "chunk_index": 0},
            {"text": "Dogs are loyal pets", "source": "b.txt", "chunk_index": 0},
        ]
        result = reranker.rerank("What color is the sky?", candidates)
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    def test_scores_are_floats(self, reranker):
        candidates = [{"text": "some text", "source": "a.txt", "chunk_index": 0}]
        _, score = reranker.rerank("query", candidates)[0]
        assert isinstance(score, float)

    def test_relevant_chunk_ranks_higher(self, reranker):
        candidates = [
            {"text": "Python is a high-level programming language.", "source": "a.txt", "chunk_index": 0},
            {"text": "The Eiffel Tower is located in Paris, France.", "source": "b.txt", "chunk_index": 0},
            {"text": "Java is also a popular programming language.", "source": "c.txt", "chunk_index": 0},
        ]
        results = reranker.rerank("What programming languages are popular?", candidates, top_n=3)
        top_text = results[0][0]["text"]
        assert "programming" in top_text.lower()

    def test_top_n_limits_output(self, reranker):
        candidates = [
            {"text": f"chunk {i}", "source": "a.txt", "chunk_index": i}
            for i in range(10)
        ]
        assert len(reranker.rerank("query", candidates, top_n=3)) == 3

    def test_top_n_larger_than_candidates_returns_all(self, reranker):
        candidates = [{"text": "only chunk", "source": "a.txt", "chunk_index": 0}]
        assert len(reranker.rerank("query", candidates, top_n=10)) == 1

    def test_results_sorted_descending_by_score(self, reranker):
        candidates = [
            {"text": f"text {i}", "source": "a.txt", "chunk_index": i}
            for i in range(5)
        ]
        results = reranker.rerank("query", candidates, top_n=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_payload_keys_preserved(self, reranker):
        candidates = [{"text": "sample", "source": "file.txt", "chunk_index": 0, "doc_id": "abc"}]
        payload, _ = reranker.rerank("query", candidates)[0]
        assert "text" in payload and "source" in payload and "doc_id" in payload

    def test_single_candidate_returns_one_result(self, reranker):
        candidates = [{"text": "exact answer", "source": "s.txt", "chunk_index": 0}]
        result = reranker.rerank("query", candidates, top_n=5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Generator.generate_with_citations() — requires a configured LLM provider
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    try:
        from rag.generator import Generator
        Generator()
        return True
    except Exception:
        return False


requires_llm = pytest.mark.skipif(
    not _llm_available(),
    reason="LLM provider not configured (set LLM_PROVIDER + API key in .env)",
)


def _call_or_skip(fn):
    """Run fn(); skip the test gracefully on quota / auth errors."""
    try:
        return fn()
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("quota", "rate", "limit", "exhausted", "auth", "invalid", "key")):
            pytest.skip(f"LLM quota/auth: {e}")
        raise


@requires_llm
class TestGeneratorWithCitations:

    @pytest.fixture(scope="class")
    def generator(self):
        from rag.generator import Generator
        return Generator()

    @pytest.fixture
    def astronomy_chunks(self):
        return [
            {"text": "The Earth orbits the Sun once every 365.25 days.", "source": "astronomy.txt", "chunk_index": 0},
            {"text": "The Moon orbits the Earth once every 27.3 days.", "source": "astronomy.txt", "chunk_index": 1},
            {"text": "Water freezes at 0 degrees Celsius at standard pressure.", "source": "chemistry.txt", "chunk_index": 0},
        ]

    def test_returns_cited_answer_instance(self, generator, astronomy_chunks):
        result = _call_or_skip(
            lambda: generator.generate_with_citations("How long is a year?", astronomy_chunks)
        )
        assert isinstance(result, CitedAnswer)

    def test_answer_is_non_empty_string(self, generator, astronomy_chunks):
        result = _call_or_skip(
            lambda: generator.generate_with_citations("How long does Earth orbit the Sun?", astronomy_chunks)
        )
        assert isinstance(result.answer, str) and len(result.answer) > 0

    def test_inline_citation_markers_present(self, generator, astronomy_chunks):
        result = _call_or_skip(
            lambda: generator.generate_with_citations("How long is a year?", astronomy_chunks)
        )
        markers = re.findall(r'\[\d+\]', result.answer)
        assert len(markers) >= 1, "Expected at least one [N] citation marker in the answer"

    def test_citations_are_valid_payload_dicts(self, generator, astronomy_chunks):
        result = _call_or_skip(
            lambda: generator.generate_with_citations("Tell me about orbits", astronomy_chunks)
        )
        for c in result.citations:
            assert "text" in c and "source" in c

    def test_citations_reference_valid_chunks(self, generator, astronomy_chunks):
        result = _call_or_skip(
            lambda: generator.generate_with_citations("Tell me about orbits", astronomy_chunks)
        )
        valid_texts = {c["text"] for c in astronomy_chunks}
        for cited in result.citations:
            assert cited["text"] in valid_texts

    def test_query_preserved_in_result(self, generator, astronomy_chunks):
        query = "How does the Moon orbit Earth?"
        result = _call_or_skip(
            lambda: generator.generate_with_citations(query, astronomy_chunks)
        )
        assert result.query == query


# ---------------------------------------------------------------------------
# Full pipeline: hybrid + reranking + citations — requires Qdrant + LLM
# ---------------------------------------------------------------------------

def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url="http://localhost:6333", timeout=2).get_collections()
        return True
    except Exception:
        return False


requires_pipeline = pytest.mark.skipif(
    not (_qdrant_available() and _llm_available()),
    reason="Requires Qdrant on localhost:6333 and a configured LLM provider",
)


@requires_pipeline
class TestFullPipeline:

    _COLLECTION = "test_phase4_temp"

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        try:
            from qdrant_client import QdrantClient
            QdrantClient(url="http://localhost:6333").delete_collection(self._COLLECTION)
        except Exception:
            pass

    @pytest.fixture(scope="class")
    def embedder(self):
        from rag.embeddings import Embedder
        return Embedder()

    @pytest.fixture
    def store_and_chunks(self, embedder):
        from rag.vectorstore import VectorStore
        from rag.models import Chunk

        store = VectorStore(collection=self._COLLECTION)
        chunks = [
            Chunk(text="Python is a high-level language created by Guido van Rossum in 1991.", doc_id="d1", chunk_index=0, source="python.txt"),
            Chunk(text="Machine learning uses statistical algorithms to learn patterns from data.", doc_id="d2", chunk_index=0, source="ml.txt"),
            Chunk(text="The Python language emphasizes code readability and simplicity.", doc_id="d1", chunk_index=1, source="python.txt"),
            Chunk(text="Neural networks are inspired by biological neurons in the brain.", doc_id="d3", chunk_index=0, source="dl.txt"),
            Chunk(text="Python is widely used for data science and machine learning tasks.", doc_id="d1", chunk_index=2, source="python.txt"),
        ]
        vecs = embedder.embed_texts([c.text for c in chunks])
        store.upsert(chunks, vecs)
        return store, chunks

    def test_reranker_produces_relevant_top_result(self, store_and_chunks, embedder):
        from rag.hybrid_retriever import HybridRetriever
        from rag.reranker import Reranker

        store, chunks = store_and_chunks
        retriever = HybridRetriever(vector_store=store, embedder=embedder)
        retriever.build_bm25_index(chunks)

        candidates = [p for p, _ in retriever.search("Python programming language", top_k=5)]
        reranker = Reranker()
        reranked = reranker.rerank("Python programming language", candidates, top_n=3)

        assert len(reranked) <= 3
        top_text = reranked[0][0]["text"].lower()
        assert "python" in top_text

    def test_cited_ask_end_to_end(self, store_and_chunks, embedder):
        from rag.hybrid_retriever import HybridRetriever
        from rag.reranker import Reranker
        from rag.generator import Generator

        store, chunks = store_and_chunks
        retriever = HybridRetriever(vector_store=store, embedder=embedder)
        retriever.build_bm25_index(chunks)

        candidates = [p for p, _ in retriever.search("Who created Python?", top_k=10)]
        reranker = Reranker()
        reranked_chunks = [p for p, _ in reranker.rerank("Who created Python?", candidates, top_n=5)]

        generator = Generator()
        result = _call_or_skip(
            lambda: generator.generate_with_citations("Who created Python?", reranked_chunks)
        )

        assert isinstance(result, CitedAnswer)
        assert len(result.answer) > 0
