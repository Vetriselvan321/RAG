"""Hybrid retriever: vector search + BM25, fused with Reciprocal Rank Fusion.

Why hybrid?
  Vector search   — finds semantically similar content even with different words.
                    "cancel subscription" matches "end membership".
  BM25            — finds exact keyword matches. Crucial for error codes, names,
                    version strings like "v2.1.3", or product SKUs.
  Together        — they cover each other's blind spots. Research consistently
                    shows 20-30% retrieval accuracy improvement over either alone.

Reciprocal Rank Fusion (RRF):
  Cormack, Clarke & Buettcher (SIGIR 2009).
  score(d) = Σ  1 / (k + rank_i(d))   for each ranked list i that contains d
  k = 60 is the standard default. It controls how much weight rank 1 gets
  over rank 2 vs rank 10 vs rank 100. Higher k → flatter weighting.

  RRF does NOT need score normalization — raw BM25 scores and cosine
  similarities are never compared directly; only their ranks matter.
"""
from .bm25_retriever import BM25Retriever
from .vectorstore import VectorStore
from .embeddings import Embedder
from .models import Chunk
from .config import RRF_K, HYBRID_FETCH_K


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[dict, float]]],
    k: int = RRF_K,
) -> list[tuple[dict, float]]:
    """Merge multiple ranked result lists using RRF.

    Args:
        ranked_lists: Each list is [(payload, raw_score), ...] sorted by
                      descending relevance. raw_score is ignored — only
                      rank position matters.
        k:            Smoothing constant. Default 60 per the original paper.

    Returns:
        Single merged list [(payload, rrf_score), ...] sorted descending.
        rrf_score has no absolute meaning; use it only for relative ranking.
    """
    fused_scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, (payload, _raw) in enumerate(ranked, start=1):
            # Stable unique key: source file + position within that file
            key = f"{payload.get('source', '')}::{payload.get('chunk_index', '')}"
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank)
            payloads[key] = payload

    sorted_keys = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)
    return [(payloads[key], fused_scores[key]) for key in sorted_keys]


class HybridRetriever:
    """Runs vector search and BM25 in parallel, then fuses results with RRF.

    Usage — when you have Chunk objects (right after ingesting a file):

        retriever = HybridRetriever(store, embedder)
        retriever.build_bm25_index(chunks)
        results = retriever.search("your query")

    Usage — CLI mode (chunks already in Qdrant, no Chunk objects available):

        retriever = HybridRetriever(store, embedder)
        retriever.build_bm25_index_from_store()   # scrolls all Qdrant payloads
        results = retriever.search("your query")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        rrf_k: int = RRF_K,
        fetch_k: int = HYBRID_FETCH_K,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k
        self.fetch_k = fetch_k

    def build_bm25_index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from Chunk objects (call immediately after ingest)."""
        self.bm25.index(chunks)

    def build_bm25_index_from_store(self) -> int:
        """Scroll all payloads from Qdrant and build BM25 index in memory.

        Returns the number of chunks indexed.
        Call this in CLI commands where you only have the vector store,
        not the original Chunk objects.
        """
        payloads = self.vector_store.scroll_all()
        self.bm25.index_payloads(payloads)
        return len(payloads)

    def search(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int | None = None,
    ) -> list[tuple[dict, float]]:
        """Hybrid search: vector + BM25, fused with RRF.

        Args:
            query:   Natural language search query.
            top_k:   Number of final results to return after fusion.
            fetch_k: Candidates fetched from each retriever before fusion.
                     Defaults to self.fetch_k (HYBRID_FETCH_K from config).
                     Higher → better recall, slightly slower.

        Returns:
            [(payload_dict, rrf_score), ...] sorted by descending rrf_score.
        """
        k = fetch_k if fetch_k is not None else self.fetch_k

        query_vec = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_vec, top_k=k)
        bm25_results = self.bm25.search(query, top_k=k)

        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results], k=self.rrf_k
        )
        return fused[:top_k]
