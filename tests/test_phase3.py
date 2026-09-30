"""
tests/test_phase3.py

Tests for Phase 3: BM25 retriever, Reciprocal Rank Fusion, and hybrid search.

Run all Phase 3 tests:         pytest tests/test_phase3.py -v
Run only unit tests (no Qdrant): pytest tests/test_phase3.py -v -k "not Hybrid"
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.models import Chunk
from rag.bm25_retriever import BM25Retriever
from rag.hybrid_retriever import reciprocal_rank_fusion, HybridRetriever


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_chunks(texts: list[str], source: str = "test.txt") -> list[Chunk]:
    return [
        Chunk(text=t, doc_id="doc-1", chunk_index=i, source=source)
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# BM25Retriever — pure in-memory, no external services
# ---------------------------------------------------------------------------

class TestBM25Retriever:

    @pytest.fixture
    def corpus(self):
        return _make_chunks([
            "The refund policy allows returns within 30 days of purchase.",
            "Our company was founded in 2010 in San Francisco California.",
            "Contact support at support@example.com for help with your order.",
            "Error code ERR-4021 indicates a failed payment transaction.",
            "Shipping takes 3-5 business days for standard delivery.",
        ])

    @pytest.fixture
    def retriever(self, corpus):
        r = BM25Retriever()
        r.index(corpus)
        return r

    def test_len_after_index(self, retriever):
        assert len(retriever) == 5

    def test_empty_retriever_returns_empty(self):
        r = BM25Retriever()
        assert r.search("anything") == []

    def test_search_returns_list_of_tuples(self, retriever):
        results = retriever.search("refund policy")
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_payload_has_required_keys(self, retriever):
        results = retriever.search("refund")
        payload, score = results[0]
        assert "text" in payload
        assert "source" in payload
        assert "chunk_index" in payload

    def test_exact_keyword_match_ranks_first(self, retriever):
        results = retriever.search("ERR-4021")
        assert len(results) > 0
        assert "ERR-4021" in results[0][0]["text"]

    def test_zero_score_results_excluded(self, retriever):
        results = retriever.search("xyzzy nonexistent token qqqq")
        assert results == []

    def test_top_k_limits_results(self, retriever):
        results = retriever.search("the", top_k=2)
        assert len(results) <= 2

    def test_index_payloads_same_as_index_chunks(self, corpus):
        payloads = [
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "doc_id": c.doc_id,
                "token_count": c.token_count,
            }
            for c in corpus
        ]
        r_chunks = BM25Retriever()
        r_chunks.index(corpus)

        r_payloads = BM25Retriever()
        r_payloads.index_payloads(payloads)

        q = "refund policy"
        res_c = r_chunks.search(q)
        res_p = r_payloads.search(q)

        assert [r[0]["text"] for r in res_c] == [r[0]["text"] for r in res_p]

    def test_keyword_query_beats_semantic_paraphrase(self, retriever):
        # BM25 should rank the chunk with the exact error code higher
        # than a semantically similar but keyword-absent chunk
        results = retriever.search("ERR-4021 payment")
        top_text = results[0][0]["text"]
        assert "ERR-4021" in top_text


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — pure function, no external services
# ---------------------------------------------------------------------------

class TestRRF:

    def _payload(self, name: str, idx: int = 0) -> dict:
        return {"source": name, "chunk_index": idx, "text": f"text from {name}"}

    def test_empty_input_returns_empty(self):
        assert reciprocal_rank_fusion([]) == []

    def test_empty_lists_return_empty(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order(self):
        ranked = [
            (self._payload("a", 0), 0.9),
            (self._payload("b", 0), 0.7),
            (self._payload("c", 0), 0.5),
        ]
        result = reciprocal_rank_fusion([ranked])
        texts = [r[0]["source"] for r in result]
        assert texts == ["a", "b", "c"]

    def test_item_in_both_lists_scores_higher(self):
        shared = self._payload("shared", 0)
        only_in_first = self._payload("first_only", 1)
        only_in_second = self._payload("second_only", 2)

        list1 = [(shared, 0.9), (only_in_first, 0.5)]
        list2 = [(shared, 0.8), (only_in_second, 0.4)]

        result = reciprocal_rank_fusion([list1, list2])
        top_source = result[0][0]["source"]
        assert top_source == "shared"

    def test_all_rrf_scores_are_positive(self):
        ranked = [(self._payload("x", i), float(10 - i)) for i in range(5)]
        result = reciprocal_rank_fusion([ranked])
        assert all(score > 0 for _, score in result)

    def test_deduplication_no_duplicates_in_output(self):
        shared = self._payload("doc", 0)
        list1 = [(shared, 0.9)]
        list2 = [(shared, 0.8)]
        result = reciprocal_rank_fusion([list1, list2])
        keys = [f"{r[0]['source']}::{r[0]['chunk_index']}" for r in result]
        assert len(keys) == len(set(keys))

    def test_higher_k_flattens_rank_differences(self):
        p1 = self._payload("first", 0)
        p2 = self._payload("second", 0)
        ranked = [(p1, 1.0), (p2, 0.5)]

        result_low_k = reciprocal_rank_fusion([ranked], k=1)
        result_high_k = reciprocal_rank_fusion([ranked], k=1000)

        score_diff_low = result_low_k[0][1] - result_low_k[1][1]
        score_diff_high = result_high_k[0][1] - result_high_k[1][1]

        assert score_diff_low > score_diff_high

    def test_merges_three_lists(self):
        a = [(self._payload("a", 0), 1.0)]
        b = [(self._payload("b", 0), 1.0)]
        c = [(self._payload("c", 0), 1.0)]
        result = reciprocal_rank_fusion([a, b, c])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# HybridRetriever — requires Qdrant on localhost:6333
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
class TestHybridRetriever:

    _COLLECTION = "test_phase3_temp"

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
    def store(self):
        from rag.vectorstore import VectorStore
        return VectorStore(collection=self._COLLECTION)

    @pytest.fixture
    def populated_store_and_chunks(self, store, embedder):
        chunks = _make_chunks([
            "The refund policy allows returns within 30 days of purchase.",
            "Our company was founded in 2010 in San Francisco California.",
            "Contact support at support@example.com for help.",
            "Error code ERR-4021 indicates a failed payment transaction.",
            "Shipping takes 3-5 business days for standard delivery.",
        ])
        vecs = embedder.embed_texts([c.text for c in chunks])
        store.upsert(chunks, vecs)
        return store, chunks

    def test_hybrid_search_returns_results(self, populated_store_and_chunks, embedder):
        store, chunks = populated_store_and_chunks
        retriever = HybridRetriever(store, embedder)
        retriever.build_bm25_index(chunks)
        results = retriever.search("refund policy", top_k=3)
        assert len(results) > 0

    def test_result_payload_has_text_and_source(self, populated_store_and_chunks, embedder):
        store, chunks = populated_store_and_chunks
        retriever = HybridRetriever(store, embedder)
        retriever.build_bm25_index(chunks)
        results = retriever.search("return policy")
        payload, score = results[0]
        assert "text" in payload
        assert "source" in payload
        assert score > 0

    def test_exact_keyword_chunk_is_recalled(self, populated_store_and_chunks, embedder):
        store, chunks = populated_store_and_chunks
        retriever = HybridRetriever(store, embedder)
        retriever.build_bm25_index(chunks)
        results = retriever.search("ERR-4021", top_k=5)
        top_texts = [r[0]["text"] for r in results]
        assert any("ERR-4021" in t for t in top_texts)

    def test_build_bm25_from_store_matches_build_from_chunks(
        self, populated_store_and_chunks, embedder
    ):
        store, chunks = populated_store_and_chunks

        r_chunks = HybridRetriever(store, embedder)
        r_chunks.build_bm25_index(chunks)

        r_store = HybridRetriever(store, embedder)
        r_store.build_bm25_index_from_store()

        q = "refund policy"
        res_c = r_chunks.search(q, top_k=3)
        res_s = r_store.search(q, top_k=3)

        assert [r[0]["text"] for r in res_c] == [r[0]["text"] for r in res_s]

    def test_hybrid_recalls_keyword_missed_by_vector_alone(
        self, populated_store_and_chunks, embedder
    ):
        store, chunks = populated_store_and_chunks

        vector_results = store.search(
            embedder.embed_query("ERR-4021"), top_k=5
        )
        hybrid_retriever = HybridRetriever(store, embedder)
        hybrid_retriever.build_bm25_index(chunks)
        hybrid_results = hybrid_retriever.search("ERR-4021", top_k=5)

        vector_top_texts = [r[0]["text"] for r in vector_results]
        hybrid_top_texts = [r[0]["text"] for r in hybrid_results]

        # Hybrid should surface the exact-match chunk; vector alone may not
        hybrid_found = any("ERR-4021" in t for t in hybrid_top_texts)
        assert hybrid_found, "Hybrid search should recall the exact keyword chunk"

        if not any("ERR-4021" in t for t in vector_top_texts):
            # Classic case: BM25 rescued a result vector search missed
            assert hybrid_found, "BM25 component should compensate for vector miss"
