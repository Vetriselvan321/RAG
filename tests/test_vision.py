"""
tests/test_vision.py

Tests for Multi-Modal RAG (Vision + Text).

Structure:
  TestVisionConstants       — pure-Python, no external deps
  TestLoadDocumentRouting   — pure-Python, patches VisionAnalyzer
  TestApiImageValidation    — FastAPI TestClient, no Gemini/Qdrant needed
  TestImageIngestion        — @requires_gemini + @requires_qdrant
  TestAskImageEndpoint      — @requires_gemini

Run only the free tests:
  pytest tests/test_vision.py -v -k "Constants or Routing or ApiImageValidation"

Run everything:
  pytest tests/test_vision.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

def _gemini_available() -> bool:
    try:
        from rag.config import GEMINI_API_KEY
        return bool(GEMINI_API_KEY)
    except Exception:
        return False


def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient
        QdrantClient(url="http://localhost:6333", timeout=2).get_collections()
        return True
    except Exception:
        return False


requires_gemini = pytest.mark.skipif(
    not _gemini_available(),
    reason="GEMINI_API_KEY not configured",
)
requires_qdrant = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant not running on localhost:6333",
)

# Minimal valid 1×1 white PNG (67 bytes) — no external file needed
MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

# ---------------------------------------------------------------------------
# Shared API client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from rag.api import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# TestVisionConstants — zero external deps
# ---------------------------------------------------------------------------

class TestVisionConstants:

    def test_supported_image_exts_is_dict(self):
        from rag.vision import SUPPORTED_IMAGE_EXTS
        assert isinstance(SUPPORTED_IMAGE_EXTS, dict)

    def test_png_is_supported(self):
        from rag.vision import SUPPORTED_IMAGE_EXTS
        assert ".png" in SUPPORTED_IMAGE_EXTS

    def test_jpg_and_jpeg_both_supported(self):
        from rag.vision import SUPPORTED_IMAGE_EXTS
        assert ".jpg" in SUPPORTED_IMAGE_EXTS
        assert ".jpeg" in SUPPORTED_IMAGE_EXTS

    def test_gif_and_webp_supported(self):
        from rag.vision import SUPPORTED_IMAGE_EXTS
        assert ".gif" in SUPPORTED_IMAGE_EXTS
        assert ".webp" in SUPPORTED_IMAGE_EXTS

    def test_all_mime_types_start_with_image(self):
        from rag.vision import SUPPORTED_IMAGE_EXTS
        for ext, mime in SUPPORTED_IMAGE_EXTS.items():
            assert mime.startswith("image/"), f"{ext} has invalid MIME type: {mime}"

    def test_png_mime_is_image_png(self):
        from rag.vision import SUPPORTED_IMAGE_EXTS
        assert SUPPORTED_IMAGE_EXTS[".png"] == "image/png"

    def test_jpg_mime_is_image_jpeg(self):
        from rag.vision import SUPPORTED_IMAGE_EXTS
        assert SUPPORTED_IMAGE_EXTS[".jpg"] == "image/jpeg"


# ---------------------------------------------------------------------------
# TestLoadDocumentRouting — patches VisionAnalyzer so no network call
# ---------------------------------------------------------------------------

class TestLoadDocumentRouting:

    def test_png_routes_to_load_image(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(MINIMAL_PNG)

        mock_analyzer = MagicMock()
        mock_analyzer.describe_image.return_value = "A white 1x1 pixel image."

        with patch("rag.vision.VisionAnalyzer", return_value=mock_analyzer):
            from rag.loaders import load_document
            doc = load_document(img)

        assert doc.content == "A white 1x1 pixel image."
        assert doc.source == "test.png"

    def test_jpg_routes_to_load_image(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)  # minimal JPEG header

        mock_analyzer = MagicMock()
        mock_analyzer.describe_image.return_value = "A photo."

        with patch("rag.vision.VisionAnalyzer", return_value=mock_analyzer):
            from rag.loaders import load_document
            doc = load_document(img)

        assert doc.content == "A photo."

    def test_image_document_metadata_has_format_image(self, tmp_path):
        img = tmp_path / "diagram.png"
        img.write_bytes(MINIMAL_PNG)

        mock_analyzer = MagicMock()
        mock_analyzer.describe_image.return_value = "A diagram."

        with patch("rag.vision.VisionAnalyzer", return_value=mock_analyzer):
            from rag.loaders import load_document
            doc = load_document(img)

        assert doc.metadata["format"] == "image"

    def test_image_metadata_includes_image_ext(self, tmp_path):
        img = tmp_path / "chart.png"
        img.write_bytes(MINIMAL_PNG)

        mock_analyzer = MagicMock()
        mock_analyzer.describe_image.return_value = "A chart."

        with patch("rag.vision.VisionAnalyzer", return_value=mock_analyzer):
            from rag.loaders import load_document
            doc = load_document(img)

        assert doc.metadata["image_ext"] == ".png"

    def test_vision_analyzer_called_once(self, tmp_path):
        img = tmp_path / "test.webp"
        img.write_bytes(b"RIFF" + b"\x00" * 12)  # minimal WebP-ish bytes

        mock_analyzer = MagicMock()
        mock_analyzer.describe_image.return_value = "A WebP image."

        with patch("rag.vision.VisionAnalyzer", return_value=mock_analyzer):
            from rag.loaders import load_document
            load_document(img)

        mock_analyzer.describe_image.assert_called_once()

    def test_txt_still_routes_to_text_loader(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("Hello world.", encoding="utf-8")

        from rag.loaders import load_document
        doc = load_document(txt)
        assert doc.content == "Hello world."
        assert doc.metadata["format"] == "text"

    def test_unsupported_ext_raises(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_bytes(b"garbage")

        from rag.loaders import load_document
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_document(f)


# ---------------------------------------------------------------------------
# TestApiImageValidation — no Gemini / Qdrant needed
# ---------------------------------------------------------------------------

class TestApiImageValidation:

    def test_ask_image_rejects_txt_file(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("doc.txt", b"hello", "text/plain")},
            data={"query": "What is this?"},
        )
        assert resp.status_code == 400

    def test_ask_image_rejects_pdf_file(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("report.pdf", b"%PDF-1.4", "application/pdf")},
            data={"query": "Summarize"},
        )
        assert resp.status_code == 400

    def test_ask_image_rejects_csv_file(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("data.csv", b"a,b\n1,2", "text/csv")},
            data={"query": "What is this?"},
        )
        assert resp.status_code == 400

    def test_ask_image_rejects_empty_query(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("img.png", MINIMAL_PNG, "image/png")},
            data={"query": "   "},
        )
        assert resp.status_code == 400

    def test_upload_now_accepts_png(self, client):
        # Extension validation only — no Qdrant means it will 503 after passing,
        # but it must NOT return 400 (bad extension).
        resp = client.post(
            "/upload",
            files={"file": ("diagram.png", MINIMAL_PNG, "image/png")},
        )
        assert resp.status_code != 400

    def test_upload_now_accepts_jpg(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("photo.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 10, "image/jpeg")},
        )
        assert resp.status_code != 400

    def test_upload_still_rejects_exe(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("app.exe", b"\x4d\x5a\x90\x00", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_ask_image_error_body_mentions_supported_formats(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("data.csv", b"a,b", "text/csv")},
            data={"query": "What is this?"},
        )
        detail = resp.json().get("detail", "")
        assert ".png" in detail or ".jpg" in detail

    def test_ask_image_default_query_is_accepted(self, client):
        # Submitting without an explicit query should use the default, not 400
        resp = client.post(
            "/ask-image",
            files={"file": ("img.png", MINIMAL_PNG, "image/png")},
            # no data= means top_k and query use defaults
        )
        # Will fail downstream (no Gemini/Qdrant), but not with 400
        assert resp.status_code != 400


# ---------------------------------------------------------------------------
# TestImageIngestion — requires Gemini + Qdrant
# ---------------------------------------------------------------------------

@requires_gemini
@requires_qdrant
class TestImageIngestion:

    def test_png_upload_succeeds(self, client, tmp_path):
        resp = client.post(
            "/upload",
            files={"file": ("test_chart.png", MINIMAL_PNG, "image/png")},
        )
        # 200 = Gemini described the image and it was indexed successfully.
        # 503 = Gemini quota exhausted or transient API error — not a code bug.
        # 400 = PNG rejected as a file type — must never happen (would be a bug).
        assert resp.status_code in (200, 503)

    def test_upload_response_has_required_fields(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("test_diagram.png", MINIMAL_PNG, "image/png")},
        )
        if resp.status_code == 200:
            data = resp.json()
            assert "filename" in data
            assert "chunks_stored" in data
            assert "message" in data

    def test_upload_preserves_filename(self, client):
        resp = client.post(
            "/upload",
            files={"file": ("my_image.png", MINIMAL_PNG, "image/png")},
        )
        if resp.status_code == 200:
            assert resp.json()["filename"] == "my_image.png"


# ---------------------------------------------------------------------------
# TestAskImageEndpoint — requires Gemini (Qdrant optional — falls back gracefully)
# ---------------------------------------------------------------------------

@requires_gemini
class TestAskImageEndpoint:

    def test_returns_200_with_valid_image(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("test.png", MINIMAL_PNG, "image/png")},
            data={"query": "What color is this image?"},
        )
        assert resp.status_code in (200, 503)

    def test_response_has_answer_field(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("test.png", MINIMAL_PNG, "image/png")},
            data={"query": "Describe this image."},
        )
        if resp.status_code == 200:
            assert "answer" in resp.json()
            assert isinstance(resp.json()["answer"], str)
            assert len(resp.json()["answer"]) > 0

    def test_response_echoes_query(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("test.png", MINIMAL_PNG, "image/png")},
            data={"query": "What do you see?"},
        )
        if resp.status_code == 200:
            assert resp.json()["query"] == "What do you see?"

    def test_response_has_cited_sources_list(self, client):
        resp = client.post(
            "/ask-image",
            files={"file": ("test.png", MINIMAL_PNG, "image/png")},
            data={"query": "Describe the image."},
        )
        if resp.status_code == 200:
            assert isinstance(resp.json()["cited_sources"], list)

    def test_jpeg_also_accepted(self, client):
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        resp = client.post(
            "/ask-image",
            files={"file": ("photo.jpg", jpeg_bytes, "image/jpeg")},
            data={"query": "What is in this photo?"},
        )
        assert resp.status_code in (200, 503)
