"""
Unit tests for core/redactor.py

Testing Strategy main functions:
- _convert_bbox_to_pdf_points(): 3 tests (basic conversion, padding, custom DPI)
- _group_entities_by_page(): 3 tests (filters unselected, filters no-bbox, groups correctly)
- _build_output_path(): 2 tests (default path, collision handling)
- redact_pdf(): 4 tests (happy path, no valid entities, missing PDF, skips out-of-range pages)
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import fitz

from core.redactor import (
    _convert_bbox_to_pdf_points,
    _group_entities_by_page,
    _build_output_path,
    redact_pdf,
)
from core.document_store import Entity


class TestConvertBboxToPdfPoints(unittest.TestCase):
    """Tests for _convert_bbox_to_pdf_points() - HIGH priority (coordinate math)"""

    def test_basic_conversion_at_200_dpi(self):
        """Test 1: Converts pixel coords to PDF points at default 200 DPI with padding"""
        # scale = 72/200 = 0.36, padding = 2.0
        rect = _convert_bbox_to_pdf_points([100, 80, 280, 100], dpi=200, padding=2.0)
        self.assertAlmostEqual(rect.x0, 100 * 0.36 - 2.0)  # 34.0
        self.assertAlmostEqual(rect.y0, 80 * 0.36 - 2.0)   # 26.8
        self.assertAlmostEqual(rect.x1, 280 * 0.36 + 2.0)  # 102.8
        self.assertAlmostEqual(rect.y1, 100 * 0.36 + 2.0)  # 38.0

    def test_no_padding(self):
        """Test 2: Zero padding gives exact conversion"""
        # 200 pixels at 200 DPI = 1 inch = 72 points
        rect = _convert_bbox_to_pdf_points([0, 0, 200, 200], dpi=200, padding=0.0)
        self.assertAlmostEqual(rect.x0, 0.0)
        self.assertAlmostEqual(rect.y0, 0.0)
        self.assertAlmostEqual(rect.x1, 72.0)
        self.assertAlmostEqual(rect.y1, 72.0)

    def test_custom_dpi(self):
        """Test 3: Works with non-default DPI"""
        # scale = 72/100 = 0.72
        rect = _convert_bbox_to_pdf_points([100, 100, 200, 200], dpi=100, padding=0.0)
        self.assertAlmostEqual(rect.x0, 72.0)
        self.assertAlmostEqual(rect.y0, 72.0)
        self.assertAlmostEqual(rect.x1, 144.0)
        self.assertAlmostEqual(rect.y1, 144.0)


class TestGroupEntitiesByPage(unittest.TestCase):
    """Tests for _group_entities_by_page() - HIGH priority (filtering logic)"""

    def test_filters_unselected_entities(self):
        """Test 1: Entities with selected=False are excluded"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1, bbox=[10, 10, 50, 20], selected=True),
            Entity(type="EMAIL", value="a@b.com", page=1, bbox=[10, 30, 50, 40], selected=False),
        ]
        result = _group_entities_by_page(entities)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[1][0].value, "123-45-6789")

    def test_filters_no_bbox_entities(self):
        """Test 2: Entities with bbox=None are excluded with warning"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1, bbox=None),
            Entity(type="EMAIL", value="a@b.com", page=1, bbox=[10, 30, 50, 40]),
        ]
        result = _group_entities_by_page(entities)
        self.assertEqual(len(result[1]), 1)
        self.assertEqual(result[1][0].value, "a@b.com")

    def test_groups_by_page_correctly(self):
        """Test 3: Entities are grouped by their page number"""
        entities = [
            Entity(type="SSN", value="111-11-1111", page=1, bbox=[10, 10, 50, 20]),
            Entity(type="EMAIL", value="a@b.com", page=3, bbox=[10, 30, 50, 40]),
            Entity(type="PHONE", value="555-1234", page=1, bbox=[10, 50, 50, 60]),
        ]
        result = _group_entities_by_page(entities)
        self.assertEqual(len(result), 2)       # 2 pages
        self.assertEqual(len(result[1]), 2)    # 2 entities on page 1
        self.assertEqual(len(result[3]), 1)    # 1 entity on page 3


class TestBuildOutputPath(unittest.TestCase):
    """Tests for _build_output_path() - MEDIUM priority"""

    def test_default_output_path(self):
        """Test 1: Generates correct default path with _redacted suffix"""
        with patch("os.path.exists", return_value=False):
            result = _build_output_path("/some/path/report.pdf", output_dir="/tmp/test_out")
        self.assertEqual(result, "/tmp/test_out/report_redacted.pdf")

    def test_collision_handling(self):
        """Test 2: Appends counter when file already exists"""
        def exists_side_effect(path):
            # report_redacted.pdf exists, report_redacted_1.pdf does not
            return path.endswith("report_redacted.pdf")

        with patch("os.path.exists", side_effect=exists_side_effect):
            result = _build_output_path("/some/path/report.pdf", output_dir="/tmp/test_out")
        self.assertEqual(result, "/tmp/test_out/report_redacted_1.pdf")


class TestRedactPdf(unittest.TestCase):
    """Tests for redact_pdf() - HIGH priority (main entry point)"""

    def test_raises_file_not_found(self):
        """Test 1: Raises FileNotFoundError when PDF doesn't exist"""
        entities = [Entity(type="SSN", value="123-45-6789", page=1, bbox=[10, 10, 50, 20])]
        with self.assertRaises(FileNotFoundError):
            redact_pdf("/nonexistent/file.pdf", entities)

    def test_raises_value_error_no_valid_entities(self):
        """Test 2: Raises ValueError when no entities have bboxes"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1, bbox=None),
        ]
        # Need the file to "exist" so we get past the first check
        with patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                redact_pdf("/fake/file.pdf", entities)

    def test_happy_path_calls_fitz_correctly(self):
        """Test 3: Verifies add_redact_annot and apply_redactions are called"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=1, bbox=[100, 80, 280, 100]),
            Entity(type="EMAIL", value="a@b.com", page=1, bbox=[100, 110, 300, 130]),
        ]

        # Mock the fitz document and page
        mock_page = MagicMock()
        mock_page.rect = fitz.Rect(0, 0, 612, 792)  # Standard US letter

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("os.path.exists", return_value=True), \
             patch("core.redactor.fitz.open", return_value=mock_doc), \
             patch("core.redactor._build_output_path", return_value="/tmp/out.pdf"):

            result = redact_pdf("/fake/file.pdf", entities)

        # Verify redaction annotations were added (2 entities)
        self.assertEqual(mock_page.add_redact_annot.call_count, 2)
        # Verify apply_redactions was called once (both entities on same page)
        mock_page.apply_redactions.assert_called_once()
        # Verify document was saved
        mock_doc.save.assert_called_once()
        self.assertEqual(result, "/tmp/out.pdf")

    def test_skips_out_of_range_pages(self):
        """Test 4: Warns and skips entities on pages beyond document length"""
        entities = [
            Entity(type="SSN", value="123-45-6789", page=99, bbox=[10, 10, 50, 20]),
        ]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)  # Only 5 pages

        with patch("os.path.exists", return_value=True), \
             patch("core.redactor.fitz.open", return_value=mock_doc), \
             patch("core.redactor._build_output_path", return_value="/tmp/out.pdf"):

            result = redact_pdf("/fake/file.pdf", entities)

        # Page was never accessed since page 99 > 5
        mock_doc.__getitem__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
