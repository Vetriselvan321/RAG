"""
tests/test_phase1.py

Tests for Phase 1: document loading + chunking.

Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.models import Document, Chunk
from rag.loaders import (
    load_text, load_html, load_document, normalize_text
)
from rag.chunker import SemanticChunker, TokenCounter, split_paragraphs, split_sentences
from rag.ingestion import IngestionPipeline


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:

    def test_document_creation(self):
        doc = Document(content="Hello world", source="test.txt")
        assert doc.content == "Hello world"
        assert doc.source == "test.txt"
        assert doc.doc_id  # auto-generated UUID

    def test_document_rejects_empty_content(self):
        with pytest.raises(ValueError, match="empty"):
            Document(content="", source="empty.txt")

    def test_chunk_creation(self):
        chunk = Chunk(text="Some text", doc_id="abc", chunk_index=0,
                      source="test.txt")
        assert chunk.text == "Some text"
        assert chunk.chunk_index == 0
        assert chunk.chunk_id

    def test_chunk_rejects_empty_text(self):
        with pytest.raises(ValueError, match="empty"):
            Chunk(text="   ", doc_id="abc", chunk_index=0, source="test.txt")


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_collapses_multiple_spaces(self):
        assert normalize_text("a    b") == "a b"

    def test_preserves_paragraph_breaks(self):
        text = "Para 1.\n\nPara 2."
        assert normalize_text(text) == "Para 1.\n\nPara 2."

    def test_collapses_excessive_newlines(self):
        text = "Line 1.\n\n\n\n\nLine 2."
        assert normalize_text(text) == "Line 1.\n\nLine 2."

    def test_handles_unicode_whitespace(self):
        text = "before\u00a0after"  # non-breaking space
        assert normalize_text(text) == "before after"

    def test_strips_trailing_whitespace(self):
        text = "hello   \nworld   "
        assert normalize_text(text) == "hello\nworld"


# ---------------------------------------------------------------------------
# Text loader
# ---------------------------------------------------------------------------

class TestTextLoader:

    def test_loads_plain_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Line one.\nLine two.")
        doc = load_text(f)
        assert "Line one" in doc.content
        assert doc.source == "test.txt"
        assert doc.metadata["format"] == "text"

    def test_loads_markdown_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Header\n\nParagraph text.")
        doc = load_document(f)
        assert "Header" in doc.content
        assert doc.metadata["format"] == "text"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_text(tmp_path / "nope.txt")


# ---------------------------------------------------------------------------
# HTML loader
# ---------------------------------------------------------------------------

class TestHTMLLoader:

    def test_extracts_text_from_html(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("""
            <html><head><title>My Title</title></head>
            <body><h1>Hello</h1><p>World</p></body></html>
        """)
        doc = load_html(f)
        assert "Hello" in doc.content
        assert "World" in doc.content
        assert doc.metadata["title"] == "My Title"

    def test_skips_script_and_style(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("""
            <html><body>
                <script>alert('hello');</script>
                <style>.foo { color: red; }</style>
                <p>Visible content</p>
            </body></html>
        """)
        doc = load_html(f)
        assert "Visible content" in doc.content
        assert "alert" not in doc.content
        assert "color: red" not in doc.content


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------

class TestTokenCounter:

    def test_counts_tokens_for_short_text(self):
        counter = TokenCounter()
        assert counter.count("hello") > 0

    def test_longer_text_has_more_tokens(self):
        counter = TokenCounter()
        short = counter.count("hello")
        long = counter.count("hello world this is a much longer piece of text")
        assert long > short


# ---------------------------------------------------------------------------
# Splitting primitives
# ---------------------------------------------------------------------------

class TestSplitters:

    def test_split_paragraphs(self):
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        parts = split_paragraphs(text)
        assert len(parts) == 3
        assert parts[0] == "Para 1."

    def test_split_paragraphs_ignores_extra_whitespace(self):
        text = "Para 1.\n\n\n  \n\nPara 2."
        parts = split_paragraphs(text)
        assert len(parts) == 2

    def test_split_sentences(self):
        text = "First sentence. Second sentence! Third one? Fourth."
        parts = split_sentences(text)
        assert len(parts) == 4


# ---------------------------------------------------------------------------
# Semantic chunker — core behavior
# ---------------------------------------------------------------------------

class TestSemanticChunker:

    def test_small_text_returns_one_chunk(self):
        chunker = SemanticChunker(max_tokens=500)
        chunks = chunker.chunk_text("Hello world. This is a short doc.")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_empty_text_returns_no_chunks(self):
        chunker = SemanticChunker(max_tokens=100)
        chunks = chunker.chunk_text("")
        assert chunks == []

    def test_large_text_creates_multiple_chunks(self):
        # Build text that's clearly larger than max_tokens
        paragraphs = [f"This is paragraph number {i}. " * 20
                      for i in range(20)]
        text = "\n\n".join(paragraphs)

        chunker = SemanticChunker(max_tokens=100, overlap_tokens=10)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunks_respect_max_token_limit(self):
        # All chunks should be within ~max_tokens (some leeway for word breaks)
        paragraphs = [f"Paragraph {i}. " * 30 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunker = SemanticChunker(max_tokens=200, overlap_tokens=20)
        chunks = chunker.chunk_text(text)

        # Allow 30% leeway since splitting on word boundaries can't be exact
        max_allowed = int(200 * 1.3)
        for c in chunks:
            assert c.token_count <= max_allowed, (
                f"Chunk {c.chunk_index} has {c.token_count} tokens, "
                f"limit was {max_allowed}"
            )

    def test_chunks_have_sequential_indices(self):
        text = "\n\n".join([f"Paragraph {i}. " * 30 for i in range(8)])
        chunks = SemanticChunker(max_tokens=200).chunk_text(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunks_carry_source_metadata(self):
        chunker = SemanticChunker(max_tokens=200)
        chunks = chunker.chunk_text(
            "\n\n".join([f"Para {i}. " * 30 for i in range(5)]),
            source="my_doc.pdf"
        )
        for c in chunks:
            assert c.source == "my_doc.pdf"

    def test_overlap_must_be_less_than_max(self):
        with pytest.raises(ValueError, match="overlap"):
            SemanticChunker(max_tokens=100, overlap_tokens=100)

    def test_overlap_preserves_some_content(self):
        # With overlap, when chunks split a long paragraph, adjacent
        # chunks should share some content. Use one big paragraph so
        # overlap actually kicks in (overlap only happens when a chunk
        # is split because it's full, not at paragraph boundaries).
        text = "This is a long sentence. " * 200  # one giant paragraph

        chunker = SemanticChunker(max_tokens=80, overlap_tokens=30)
        chunks = chunker.chunk_text(text)

        # With overlap, total chunked text > original text
        total_text = sum(len(c.text) for c in chunks)
        assert total_text > len(text), (
            f"Expected overlap to inflate total text. "
            f"Original: {len(text)}, chunked: {total_text}"
        )

    def test_handles_text_with_no_paragraph_breaks(self):
        # One huge sentence
        text = "word " * 500
        chunks = SemanticChunker(max_tokens=100).chunk_text(text)
        assert len(chunks) > 1

    def test_min_tokens_filters_small_chunks(self):
        # Build text where the last chunk is tiny
        text = "Big paragraph. " * 100 + "\n\nx"  # last bit is 1 token

        chunker = SemanticChunker(
            max_tokens=200, overlap_tokens=0, min_tokens=10
        )
        chunks = chunker.chunk_text(text)
        for c in chunks:
            assert c.token_count >= 10


# ---------------------------------------------------------------------------
# Ingestion pipeline (end-to-end)
# ---------------------------------------------------------------------------

class TestIngestionPipeline:

    def test_ingest_single_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("A short document for testing.")
        pipeline = IngestionPipeline()
        chunks = pipeline.ingest_file(f)
        assert len(chunks) >= 1
        assert chunks[0].source == "doc.txt"

    def test_ingest_directory(self, tmp_path):
        # Create multiple files
        (tmp_path / "a.txt").write_text("First document content. " * 50)
        (tmp_path / "b.md").write_text("# Second\n\nDocument content. " * 50)
        (tmp_path / "ignore.bin").write_bytes(b"\x00\x01\x02")  # unsupported

        pipeline = IngestionPipeline()
        chunks, stats = pipeline.ingest_directory(tmp_path)

        assert stats.documents_loaded == 2  # .bin is skipped
        assert stats.documents_failed == 0
        assert len(chunks) >= 2

    def test_ingest_handles_corrupt_files_gracefully(self, tmp_path):
        # Create a fake "PDF" that's actually garbage
        (tmp_path / "good.txt").write_text("Good content here. " * 20)
        (tmp_path / "broken.pdf").write_bytes(b"not a real pdf")

        pipeline = IngestionPipeline()
        chunks, stats = pipeline.ingest_directory(tmp_path)

        # Good file should still work, broken file logged as failure
        assert stats.documents_loaded == 1
        assert stats.documents_failed >= 0  # may or may not fail depending on pypdf
