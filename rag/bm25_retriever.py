"""BM25 keyword retriever using rank-bm25.

BM25 (Best Matching 25) is the algorithm behind Elasticsearch and most
production search engines. It scores documents by exact term frequency,
accounting for document length and corpus-wide term rarity (IDF).

Strength:  exact keyword matches — error codes, names, version numbers,
           product IDs, quoted phrases.
Weakness:  misses synonyms and paraphrases ("cancel" vs "terminate").

Pairs with vector search: vector covers semantics, BM25 covers precision.
"""
from rank_bm25 import BM25Okapi
from .models import Chunk


class BM25Retriever:
    """In-memory BM25 index.

    Accepts either Chunk objects (for fresh ingestion) or payload dicts
    loaded back from Qdrant (for CLI hybrid-search over stored chunks).
    Both code paths produce the same (payload, score) output format as
    VectorStore.search(), making them directly fuseable via RRF.
    """

    def __init__(self) -> None:
        self._payloads: list[dict] = []
        self._index: BM25Okapi | None = None

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Build index from Chunk objects (used right after ingestion)."""
        self._payloads = [
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "doc_id": c.doc_id,
                "token_count": c.token_count,
            }
            for c in chunks
        ]
        self._build()

    def index_payloads(self, payloads: list[dict]) -> None:
        """Build index from Qdrant payload dicts (used by scroll_all path)."""
        self._payloads = payloads
        self._build()

    def _build(self) -> None:
        tokenized = [p["text"].lower().split() for p in self._payloads]
        self._index = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Return top_k results as (payload, bm25_score) pairs.

        Only returns results with score > 0 (i.e. at least one query
        token was found in the document).
        """
        if self._index is None or not self._payloads:
            return []

        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            (self._payloads[i], float(scores[i]))
            for i in ranked_indices
            if scores[i] > 0.0
        ]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._payloads)
