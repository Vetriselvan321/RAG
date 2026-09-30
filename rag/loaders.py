"""
rag/loaders.py

Document loaders for the formats you'll encounter in production:
PDF, HTML, plain text, and markdown. Each loader returns a Document.

Design principles:
1. Loaders only LOAD, they don't chunk. Single responsibility.
2. Plain text output, normalized. Downstream code shouldn't worry
   about HTML entities, weird PDF artifacts, etc.
3. Metadata is preserved (page numbers for PDFs, title for HTML).
"""

import re
from pathlib import Path
from typing import Optional
from html.parser import HTMLParser

from .models import Document
from .vision import SUPPORTED_IMAGE_EXTS


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Clean up whitespace artifacts common in PDF/HTML extraction.

    - Collapse multiple spaces into one
    - Collapse 3+ newlines into 2 (preserves paragraph breaks)
    - Strip trailing whitespace on each line
    - Remove common PDF artifacts (page numbers as standalone lines, etc.)
    """
    # Replace tabs and weird Unicode whitespace with regular spaces
    text = text.replace("\t", " ")
    text = re.sub(r"[\u00a0\u2000-\u200b\ufeff]", " ", text)

    # Collapse runs of spaces (but not newlines)
    text = re.sub(r" {2,}", " ", text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ newlines into 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Plain text loader
# ---------------------------------------------------------------------------

def load_text(path: str | Path) -> Document:
    """Load a plain text or markdown file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")
    content = normalize_text(content)

    return Document(
        content=content,
        source=path.name,
        metadata={
            "format": "text",
            "size_bytes": path.stat().st_size,
            "path": str(path.absolute()),
        }
    )


# ---------------------------------------------------------------------------
# HTML loader (no external dependencies — uses stdlib HTMLParser)
# ---------------------------------------------------------------------------

class _HTMLTextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping scripts/styles."""

    SKIP_TAGS = {"script", "style", "noscript", "head"}
    BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4",
                  "h5", "h6", "blockquote", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0
        self.title = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
            return  # don't increment skip_depth for title
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
            return
        if tag in self.SKIP_TAGS:
            self.skip_depth = max(0, self.skip_depth - 1)
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            if self.title is None:
                self.title = data.strip()
            return  # title text does not go into body parts
        if self.skip_depth > 0:
            return
        self.parts.append(data)

    def get_text(self) -> str:
        return "".join(self.parts)


def load_html(path: str | Path) -> Document:
    """Load an HTML file and extract visible text."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HTML file not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    content = normalize_text(parser.get_text())

    return Document(
        content=content,
        source=path.name,
        metadata={
            "format": "html",
            "title": parser.title,
            "size_bytes": path.stat().st_size,
            "path": str(path.absolute()),
        }
    )


# ---------------------------------------------------------------------------
# PDF loader (uses pypdf — minimal external dep)
# ---------------------------------------------------------------------------

def load_pdf(path: str | Path) -> Document:
    """
    Load a PDF file and extract text from every page.

    Page numbers are preserved in the metadata via a 'pages' list:
        [{'page': 1, 'char_start': 0, 'char_end': 1234}, ...]

    This lets downstream code map chunks back to their source page.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF loading. Install with: pip install pypdf"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    parts = []
    pages_meta = []
    current_pos = 0

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = normalize_text(page_text)

        if not page_text:
            continue

        pages_meta.append({
            "page": page_num,
            "char_start": current_pos,
            "char_end": current_pos + len(page_text),
        })
        parts.append(page_text)
        current_pos += len(page_text) + 2  # +2 for the \n\n join

    full_content = "\n\n".join(parts)

    return Document(
        content=full_content,
        source=path.name,
        metadata={
            "format": "pdf",
            "page_count": len(reader.pages),
            "pages": pages_meta,
            "size_bytes": path.stat().st_size,
            "path": str(path.absolute()),
        }
    )


# ---------------------------------------------------------------------------
# Smart dispatcher — picks the right loader by file extension
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> Document:
    """
    Load an image by sending it to Gemini 2.0 Flash for a text description.

    The returned Document's content is Gemini's detailed description (OCR +
    visual understanding). It flows through the normal chunking → embedding →
    Qdrant pipeline, making the image fully searchable via text queries.
    """
    from .vision import VisionAnalyzer

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    analyzer = VisionAnalyzer()
    description = analyzer.describe_image(path)

    return Document(
        content=description,
        source=path.name,
        metadata={
            "format": "image",
            "image_ext": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "path": str(path.absolute()),
        },
    )


def load_document(path: str | Path) -> Document:
    """
    Load any supported document type, dispatching by file extension.

    Supported: .pdf, .html, .htm, .txt, .md, .png, .jpg, .jpeg, .gif, .webp
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return load_pdf(path)
    if ext in (".html", ".htm"):
        return load_html(path)
    if ext in (".txt", ".md", ".markdown", ".rst"):
        return load_text(path)
    if ext in SUPPORTED_IMAGE_EXTS:
        return load_image(path)

    raise ValueError(f"Unsupported file type: {ext} (file: {path.name})")
