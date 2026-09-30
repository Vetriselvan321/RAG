"""Qdrant vector database wrapper.

Stores chunks with their embeddings and performs nearest-neighbor search.
Works for both local Qdrant (Docker) and Qdrant Cloud free tier.
"""
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_DIMS
from .models import Chunk


class VectorStore:

    def __init__(self, url: str = QDRANT_URL, collection: str = COLLECTION_NAME):
        # API key is optional - None for local, set for Qdrant Cloud
        self.client = QdrantClient(url=url, api_key=QDRANT_API_KEY)
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMS,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Store chunks with their pre-computed embeddings."""
        points = []
        for chunk, vec in zip(chunks, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "doc_id": chunk.doc_id,
                    "token_count": chunk.token_count,
                },
            ))
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query_vector: list[float], top_k: int = 5):
        """Return top_k most similar chunks with scores."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
        )
        return [(r.payload, r.score) for r in results.points]

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self.client.count(self.collection).count

    def scroll_all(self) -> list[dict]:
        """Fetch every stored payload from Qdrant without vectors.

        Used by the hybrid retriever to build an in-memory BM25 index
        over whatever is currently in the collection.
        """
        all_payloads: list[dict] = []
        offset = None
        while True:
            batch, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_payloads.extend(r.payload for r in batch)
            if next_offset is None:
                break
            offset = next_offset
        return all_payloads
