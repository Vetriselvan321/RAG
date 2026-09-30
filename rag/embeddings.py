"""Free local embeddings using sentence-transformers.

No API calls, no rate limits, no costs. Downloads model once (~80MB),
then runs entirely offline on CPU.
"""
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL


class Embedder:
    """Wraps sentence-transformers for batch and single embedding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.dims = self.model.get_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        return self.embed_texts([query])[0]