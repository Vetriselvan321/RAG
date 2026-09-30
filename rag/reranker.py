"""Cross-encoder reranker — Phase 4.

Reranking is the second-stage retrieval step:

  Stage 1 (cheap, fast):  bi-encoder vector search retrieves top-20 candidates.
                           Query and document are encoded independently, so it
                           scales to millions of chunks with sub-second latency.

  Stage 2 (expensive, accurate): cross-encoder rescores the 20 candidates.
                           The model receives BOTH query and document at once,
                           enabling deep token-level interaction — the same
                           mechanism that makes BERT-style models so accurate.
                           Limited to ~20 comparisons per query (not millions).

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS MARCO passage ranking dataset (510k queries)
  - Downloads once (~80MB), runs on CPU, no API key needed
  - Scores range: unconstrained floats (higher = more relevant)
  - Consistent 20-30% MRR improvement over bi-encoder alone
"""
from sentence_transformers import CrossEncoder
from .config import RERANKER_MODEL, RERANK_TOP_N


class Reranker:
    """Cross-encoder reranker. Slow per call, but highly accurate."""

    def __init__(self, model: str = RERANKER_MODEL) -> None:
        # max_length=512 matches the model's training context window
        self.model = CrossEncoder(model, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_n: int = RERANK_TOP_N,
    ) -> list[tuple[dict, float]]:
        """Re-score candidates with the cross-encoder; return top_n.

        Args:
            query:      The user's search query.
            candidates: List of payload dicts from vector/hybrid search.
                        Each must have a "text" key.
            top_n:      Number of results to return after reranking.

        Returns:
            [(payload, cross_encoder_score), ...] sorted by descending score.
            cross_encoder_score is a raw logit — use it only for ranking,
            not as an absolute relevance threshold.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(payload, float(score)) for payload, score in ranked[:top_n]]
