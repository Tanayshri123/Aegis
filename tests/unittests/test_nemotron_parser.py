"""
Unit tests for core/nemotron_parser.py

Testing Strategy main functions:
- find_entity_bbox(): 3 tests (exact match, case-insensitive, not found)
- _parse_bbox_response(): 2 tests (valid response, empty response)
- locate_entities(): 2 tests (bbox populated, API failure graceful)
- convert_page_to_image(): 1 test (mock pdf2image)
- image_to_base64(): 1 test (encoding)
"""

import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import asyncio
import tempfile
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.nemotron_parser import (
    find_entity_bbox,
    _parse_bbox_response,
    locate_entities,
    image_to_base64,
)
from core.document_store import Entity


class TestFindEntityBbox(unittest.TestCase):
    """Tests for find_entity_bbox() - HIGH priority (core matching logic)"""

    def setUp(self):
        """Set up sample Nemotron elements for testing."""
        self.elements = [
            {"text": "CEO: John Smith", "bbox": [100, 50, 250, 70], "type": "text"},
            {"text": "SSN: 123-45-6789", "bbox": [100, 80, 280, 100], "type": "text"},
            {"text": "Email: john@acme.com", "bbox": [100, 110, 300, 130], "type": "text"},
        ]

    def test_exact_match(self):
        """Test 1: Finds bbox when entity value is contained in element text"""
        bbox = find_entity_bbox("123-45-6789", self.elements)
        self.assertEqual(bbox, [100, 80, 280, 100])

    def test_case_insensitive_match(self):
        """Test 2: Falls back to case-insensitive when exact match fails"""
        bbox = find_entity_bbox("john smith", self.elements)
        self.assertEqual(bbox, [100, 50, 250, 70])

    def test_not_found_returns_none(self):
        """Test 3: Returns None when entity value is not in any element"""
        bbox = find_entity_bbox("999-99-9999", self.elements)
        self.assertIsNone(bbox)


class TestParseBboxResponse(unittest.TestCase):
    """Tests for _parse_bbox_response() - HIGH priority (API response parsing)"""

    def test_parses_valid_bbox_lines(self):
        """Test 1: Correctly parses Nemotron bbox format"""
        content = (
            "[bbox: 100,50,250,70] CEO: John Smith\n"
            "[bbox: 100,80,280,100] SSN: 123-45-6789\n"
        )
        elements = _parse_bbox_response(content)

        self.assertEqual(len(elements), 2)
        self.assertEqual(elements[0]["text"], "CEO: John Smith")
        self.assertEqual(elements[0]["bbox"], [100.0, 50.0, 250.0, 70.0])
        self.assertEqual(elements[1]["text"], "SSN: 123-45-6789")

    def test_empty_response(self):
        """Test 2: Returns empty list for empty or no-bbox content"""
        self.assertEqual(_parse_bbox_response(""), [])
        self.assertEqual(_parse_bbox_response("just plain text no bboxes"), [])


class TestLocateEntities(unittest.TestCase):
    """Tests for locate_entities() - HIGH priority (main entry point)"""

    def test_populates_bbox_on_entities(self):
        """Test 1: Entities get bbox filled in after Nemotron parsing"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1, bbox=None),
            Entity(type="EMAIL", value="john@acme.com", page=1, bbox=None),
        ]

        mock_nemotron_response = {
            "choices": [{
                "message": {
                    "content": (
                        "[bbox: 100,80,280,100] SSN: 123-45-6789\n"
                        "[bbox: 100,110,300,130] Email: john@acme.com\n"
                    )
                }
            }]
        }

        with patch("core.nemotron_parser.convert_page_to_image") as mock_convert, \
             patch("core.nemotron_parser.parse_page", new_callable=AsyncMock) as mock_parse:

            mock_convert.return_value = Path("/tmp/fake_page.jpg")
            mock_parse.return_value = [
                {"text": "SSN: 123-45-6789", "bbox": [100, 80, 280, 100], "type": "text"},
                {"text": "Email: john@acme.com", "bbox": [100, 110, 300, 130], "type": "text"},
            ]

            # Mock the temp file cleanup
            with patch.object(Path, "exists", return_value=False):
                result = asyncio.run(locate_entities("/fake/doc.pdf", entities))

        self.assertEqual(result[0].bbox, [100, 80, 280, 100])
        self.assertEqual(result[1].bbox, [100, 110, 300, 130])

    def test_graceful_failure_keeps_bbox_none(self):
        """Test 2: If Nemotron fails, entities keep bbox=None instead of crashing"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1, bbox=None),
        ]

        with patch("core.nemotron_parser.convert_page_to_image", side_effect=Exception("PDF error")):
            with patch.object(Path, "exists", return_value=False):
                result = asyncio.run(locate_entities("/fake/doc.pdf", entities))

        self.assertIsNone(result[0].bbox)


class TestImageToBase64(unittest.TestCase):
    """Tests for image_to_base64() - MEDIUM priority"""

    def test_encodes_file_to_base64(self):
        """Test 1: Correctly encodes a file to base64 string"""
        # Create a small temp file with known content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(b"fake image data")
            temp_path = Path(f.name)

        try:
            result = image_to_base64(temp_path)
            self.assertIsInstance(result, str)
            # Verify it's valid base64 by decoding
            import base64
            decoded = base64.b64decode(result)
            self.assertEqual(decoded, b"fake image data")
        finally:
            os.unlink(temp_path)


class TestConvertPageToImage(unittest.TestCase):
    """Tests for convert_page_to_image() - MEDIUM priority"""

    @patch("core.nemotron_parser.convert_from_path")
    def test_converts_single_page(self, mock_convert):
        """Test 1: Calls pdf2image with correct first_page/last_page for single page"""
        mock_image = MagicMock()
        mock_convert.return_value = [mock_image]

        with patch("tempfile.mkdtemp", return_value="/tmp/aegis_test"):
            with patch.object(Path, "parent", new_callable=lambda: property(lambda self: Path("/tmp/aegis_test"))):
                from core.nemotron_parser import convert_page_to_image
                # Just verify the call args, don't actually save
                mock_image.save = MagicMock()
                result = convert_page_to_image("/fake/doc.pdf", page_number=3, dpi=200)

        # Verify pdf2image was called for page 3 only
        mock_convert.assert_called_once_with(
            "/fake/doc.pdf",
            dpi=200,
            first_page=3,
            last_page=3
        )


if __name__ == "__main__":
    unittest.main()
