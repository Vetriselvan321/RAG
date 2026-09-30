"""
rag/ingestion.py

The high-level ingestion pipeline. This is what users of the framework
interact with — they don't have to know about loaders vs chunkers.

Usage:
    pipeline = IngestionPipeline(chunker=SemanticChunker(max_tokens=512))
    chunks = pipeline.ingest_file("contract.pdf")
    chunks = pipeline.ingest_directory("docs/")
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .models import Chunk, Document
from .loaders import load_document
from .chunker import SemanticChunker


@dataclass
class IngestionStats:
    """Returned by ingest_* methods so callers can see what happened."""
    documents_loaded: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    failed_files: list[str] = None

    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []


class IngestionPipeline:
    """
    Combines loading + chunking into a single high-level API.

    The pipeline is deliberately stateless — it returns chunks; storage
    is handled in Phase 2 by the vector store layer.
    """

    SUPPORTED_EXTS = {".pdf", ".html", ".htm", ".txt", ".md",
                      ".markdown", ".rst",
                      ".png", ".jpg", ".jpeg", ".gif", ".webp"}

    def __init__(self, chunker: Optional[SemanticChunker] = None):
        self.chunker = chunker or SemanticChunker()

    # ---------------------------------------------------------------------
    # Single file
    # ---------------------------------------------------------------------

    def ingest_file(self, path: str | Path) -> list[Chunk]:
        """Load and chunk a single file. Returns the chunks."""
        doc = load_document(path)
        return self.chunker.chunk_document(doc)

    # ---------------------------------------------------------------------
    # Directory
    # ---------------------------------------------------------------------

    def ingest_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        verbose: bool = False,
    ) -> tuple[list[Chunk], IngestionStats]:
        """
        Walk a directory and ingest every supported file.

        Returns (all_chunks, stats). Failures are collected, not raised —
        production ingestion shouldn't die on one corrupt PDF.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        all_chunks: list[Chunk] = []
        stats = IngestionStats()

        glob = "**/*" if recursive else "*"
        for path in sorted(directory.glob(glob)):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SUPPORTED_EXTS:
                continue

            try:
                chunks = self.ingest_file(path)
                all_chunks.extend(chunks)
                stats.documents_loaded += 1
                stats.chunks_created += len(chunks)
                if verbose:
                    print(f"  {path.name}: {len(chunks)} chunks")
            except Exception as exc:
                stats.documents_failed += 1
                stats.failed_files.append(f"{path.name}: {exc}")
                if verbose:
                    print(f"  {path.name}: FAILED ({exc})")

        return all_chunks, stats
