"""
rag/chunker.py

Document chunking — the most underrated component of a RAG system.
Bad chunking destroys retrieval quality no matter how good your embeddings are.

Our strategy: recursive splitting with semantic boundaries.

Why "recursive"?
  We try splitting on the largest semantic unit first (paragraph),
  then fall back to smaller units (sentence, then word) only if
  a chunk is still too large. This preserves meaning.

Why "semantic boundaries"?
  Splitting mid-sentence breaks meaning. Splitting mid-paragraph
  separates related ideas. Always split on the cleanest boundary
  that fits the size budget.

Token-based sizing, not character-based:
  LLMs see tokens, not characters. "antidisestablishmentarianism" is
  1 word but 6 tokens. Sizing in tokens is the only correct way.
"""

import re
from typing import Optional
from .models import Document, Chunk


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TokenCounter:
    """
    Count tokens the same way OpenAI/Claude models do.

    Uses tiktoken if available (accurate). Falls back to a simple
    heuristic (≈4 chars/token) so tests don't require the dep.
    """

    def __init__(self, model: str = "cl100k_base"):
        self.model = model
        self._encoder = None
        try:
            import tiktoken
            self._encoder = tiktoken.get_encoding(model)
        except Exception:
            # tiktoken may fail to load encoding (no network, no cache).
            # We silently fall back to the heuristic — accuracy is
            # within ~10% which is fine for chunk sizing decisions.
            pass

    def count(self, text: str) -> int:
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        # Fallback: rough heuristic (4 chars ≈ 1 token in English)
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Splitting primitives
# ---------------------------------------------------------------------------

# These regexes split text on common semantic boundaries, in order of
# preference. We try paragraphs first (biggest unit), then sentences,
# then words. Markdown headers get their own treatment.

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_paragraphs(text: str) -> list[str]:
    """Split text on blank lines (paragraph breaks)."""
    parts = _PARAGRAPH_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    """
    Split a paragraph into sentences using a simple heuristic.
    Not perfect — won't handle 'Dr. Smith' correctly — but good
    enough for chunking purposes.
    """
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


def split_words(text: str) -> list[str]:
    """Last-resort splitting when even sentences are too large."""
    return text.split()


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Splits documents into chunks that respect semantic boundaries
    and stay within a target token size.

    Args:
        max_tokens:    Maximum tokens per chunk (target size)
        overlap_tokens: Tokens of overlap between adjacent chunks
                       (helps retrieval when relevant info spans
                       a chunk boundary)
        min_tokens:    Drop chunks smaller than this (usually
                       artifacts of bad splitting)
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_tokens: int = 20,
        token_counter: Optional[TokenCounter] = None,
    ):
        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap ({overlap_tokens}) must be less than "
                f"max_tokens ({max_tokens})"
            )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.counter = token_counter or TokenCounter()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a Document into a list of Chunks."""
        text = document.content
        if not text.strip():
            return []

        # Step 1: Build a list of token-sized chunk text strings
        chunk_texts = self._build_chunks(text)

        # Step 2: Wrap into Chunk objects with metadata
        chunks = []
        cursor = 0
        for i, ct in enumerate(chunk_texts):
            # Find where this chunk lives in the original text
            start = text.find(ct, cursor)
            if start == -1:
                # Overlap or normalized whitespace can break exact find.
                # Fall back to current cursor position.
                start = cursor
            end = start + len(ct)

            chunks.append(Chunk(
                text=ct,
                doc_id=document.doc_id,
                chunk_index=i,
                source=document.source,
                char_start=start,
                char_end=end,
                token_count=self.counter.count(ct),
                metadata={**document.metadata, "doc_source": document.source},
            ))
            cursor = max(cursor, end - 100)  # allow small backward scan for overlap

        return chunks

    def chunk_text(self, text: str, source: str = "raw_text") -> list[Chunk]:
        """Convenience method: chunk raw text without a Document wrapper."""
        if not text or not text.strip():
            return []
        doc = Document(content=text, source=source)
        return self.chunk_document(doc)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _build_chunks(self, text: str) -> list[str]:
        """
        Recursive splitting:
          1. Try to fit the whole text — if it fits, return as one chunk.
          2. Otherwise, split on paragraphs and pack greedily.
          3. If a paragraph is itself too large, split it on sentences.
          4. If a sentence is too large, split on words (rare).
        """
        # If small enough already, return as single chunk
        if self.counter.count(text) <= self.max_tokens:
            return [text]

        # Try paragraph splitting first
        paragraphs = split_paragraphs(text)
        if len(paragraphs) > 1:
            return self._pack_pieces(paragraphs, joiner="\n\n")

        # Single huge paragraph — split on sentences
        sentences = split_sentences(text)
        if len(sentences) > 1:
            return self._pack_pieces(sentences, joiner=" ")

        # Single huge sentence — split on words
        words = split_words(text)
        return self._pack_pieces(words, joiner=" ")

    def _pack_pieces(self, pieces: list[str], joiner: str) -> list[str]:
        """
        Greedily pack pieces into chunks. When a chunk is full, start
        a new one with overlap_tokens of the previous chunk's tail.
        """
        chunks = []
        current_pieces: list[str] = []
        current_tokens = 0

        for piece in pieces:
            piece_tokens = self.counter.count(piece)

            # If a single piece is too large, recursively split it
            if piece_tokens > self.max_tokens:
                # Flush current chunk first
                if current_pieces:
                    chunks.append(joiner.join(current_pieces))
                    current_pieces = []
                    current_tokens = 0

                # Recursively split the oversized piece
                subchunks = self._build_chunks(piece)
                chunks.extend(subchunks)
                continue

            # Will adding this piece overflow?
            if current_tokens + piece_tokens > self.max_tokens and current_pieces:
                # Flush current chunk
                chunks.append(joiner.join(current_pieces))

                # Start new chunk with overlap from end of previous
                overlap_pieces = self._take_overlap(current_pieces, joiner)
                current_pieces = overlap_pieces + [piece]
                current_tokens = (
                    sum(self.counter.count(p) for p in overlap_pieces)
                    + piece_tokens
                )
            else:
                current_pieces.append(piece)
                current_tokens += piece_tokens

        # Flush final chunk
        if current_pieces:
            chunks.append(joiner.join(current_pieces))

        # Filter out chunks that are too small
        return [c for c in chunks if self.counter.count(c) >= self.min_tokens]

    def _take_overlap(self, pieces: list[str], joiner: str) -> list[str]:
        """
        Take pieces from the end of the previous chunk to form an overlap.
        We walk backwards, collecting pieces until we hit overlap_tokens.
        """
        if self.overlap_tokens == 0:
            return []

        overlap = []
        tokens = 0
        for piece in reversed(pieces):
            piece_tokens = self.counter.count(piece)
            if tokens + piece_tokens > self.overlap_tokens:
                break
            overlap.insert(0, piece)
            tokens += piece_tokens
        return overlap
