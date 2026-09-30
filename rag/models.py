"""
rag/models.py

Core data types for the RAG framework.
A Document is raw content + metadata. A Chunk is a piece of a Document
with positional info needed for citation tracking later.
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4
import re


@dataclass
class Document:
    """Raw document loaded from disk or URL, before chunking."""
    content: str                          # full text content
    source: str                           # filename, URL, or identifier
    doc_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict = field(default_factory=dict)  # author, date, page count, etc.

    def __post_init__(self):
        if not self.content:
            raise ValueError(f"Document {self.source} has empty content")


@dataclass
class Chunk:
    """A chunk of a document — the unit that gets embedded and retrieved."""
    text: str                             # the chunk's text content
    doc_id: str                           # which Document this came from
    chunk_index: int                      # position within the document (0, 1, 2...)
    source: str                           # carried over for easy citation
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict = field(default_factory=dict)

    char_start: int = 0                   # offset in original document
    char_end: int = 0
    token_count: int = 0                  # populated by chunker

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError(f"Chunk {self.chunk_id} is empty or whitespace-only")

    def __repr__(self):
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk(idx={self.chunk_index}, src={self.source}, preview='{preview}...')"


@dataclass
class CitedAnswer:
    """LLM answer with tracked inline citations.

    The answer text contains [1], [2] markers that reference specific
    context chunks. citations holds the subset of chunks that were
    actually cited in the answer.
    """
    answer: str           # Answer text with inline [1] [2] citation markers
    citations: list[dict] # Chunks that were cited, in citation-number order
    query: str            # Original query

    @property
    def cited_sources(self) -> list[str]:
        """Unique source filenames that appear in citations."""
        seen: set[str] = set()
        out: list[str] = []
        for c in self.citations:
            s = c.get("source", "unknown")
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @property
    def citation_numbers(self) -> list[int]:
        """Sorted list of citation numbers that appear in the answer text."""
        return sorted(set(
            int(n) for n in re.findall(r'\[(\d+)\]', self.answer)
        ))


# ---------------------------------------------------------------------------
# Phase 5 — evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One evaluation record: query + expected answer + retrieved context.

    Populate answer and contexts either by running the pipeline or by loading
    precomputed values from a JSON eval file. The evaluator accepts either.
    """
    query: str
    ground_truth: str                              # Human-labeled reference answer
    answer: str = ""                               # Generated answer (filled by pipeline)
    contexts: list[str] = field(default_factory=list)  # Retrieved chunk texts


@dataclass
class EvalResult:
    """RAGAS metric scores for a single EvalSample. All values in [0, 1]."""
    query: str
    faithfulness: float        # answer claims grounded in context
    context_relevance: float   # context sentences relevant to query
    context_recall: float      # context covers the ground-truth answer
    answer_relevance: float    # answer addresses the question

    @property
    def mean_score(self) -> float:
        return round(
            (self.faithfulness + self.context_relevance
             + self.context_recall + self.answer_relevance) / 4,
            4,
        )


@dataclass
class EvalReport:
    """Aggregated RAGAS evaluation report over multiple samples."""
    results: list[EvalResult]

    def _avg(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    @property
    def faithfulness(self) -> float:
        return self._avg("faithfulness")

    @property
    def context_relevance(self) -> float:
        return self._avg("context_relevance")

    @property
    def context_recall(self) -> float:
        return self._avg("context_recall")

    @property
    def answer_relevance(self) -> float:
        return self._avg("answer_relevance")

    @property
    def overall(self) -> float:
        return round(
            (self.faithfulness + self.context_relevance
             + self.context_recall + self.answer_relevance) / 4,
            4,
        )
