"""
Unit tests for core/pdf_processor.py

Testing Strategy main functions:
- extract_text_from_pdf(): 5 tests (happy path, empty PDF, corrupted, page filtering)
- chunk_text(): 4 tests (normal, small text, empty, custom size)
- get_full_text(): 3 tests (with markers, without, empty)
- extract_text_from_path(): 2 tests (valid path, invalid path)
"""

import unittest
from unittest.mock import Mock, patch, mock_open
from io import BytesIO
import PyPDF2

# Import functions to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.pdf_processor import (
    extract_text_from_pdf,
    chunk_text,
    get_full_text,
    extract_text_from_path
)


class TestExtractTextFromPDF(unittest.TestCase):
    """
    Tests for extract_text_from_pdf() function.

    This is HIGH PRIORITY because:
    - Core functionality
    - File I/O (error-prone)
    - User-facing (directly called)
    """

    def test_extract_text_from_valid_pdf(self):
        """Test 1: Happy path - extract from valid PDF with text."""
        # Arrange: Create mock PDF
        mock_pdf = Mock(spec=PyPDF2.PdfReader)
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_pdf.pages = [mock_page1, mock_page2]

        # Act: Extract text
        with patch('core.pdf_processor.pypdf2.PdfReader', return_value=mock_pdf):
            result = extract_text_from_pdf(BytesIO(b"fake pdf content"))

        # Assert: Check output structure
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["page_number"], 1)
        self.assertEqual(result[0]["text"], "Page 1 content")
        self.assertEqual(result[1]["page_number"], 2)
        self.assertEqual(result[1]["text"], "Page 2 content")

    def test_extract_text_skips_empty_pages(self):
        """Test 2: Edge case - skip pages with no text."""
        # Arrange: PDF with empty page
        mock_pdf = Mock(spec=PyPDF2.PdfReader)
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = ""  # Empty page
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "Page 3 content"
        mock_pdf.pages = [mock_page1, mock_page2, mock_page3]

        # Act
        with patch('core.pdf_processor.pypdf2.PdfReader', return_value=mock_pdf):
            result = extract_text_from_pdf(BytesIO(b"fake pdf"))

        # Assert: Only 2 pages returned (empty page skipped)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["page_number"], 1)
        self.assertEqual(result[1]["page_number"], 3)  # Page 2 was skipped

    def test_extract_text_from_empty_pdf(self):
        """Test 3: Edge case - PDF with no pages."""
        # Arrange: Empty PDF
        mock_pdf = Mock(spec=PyPDF2.PdfReader)
        mock_pdf.pages = []

        # Act
        with patch('core.pdf_processor.pypdf2.PdfReader', return_value=mock_pdf):
            result = extract_text_from_pdf(BytesIO(b"empty pdf"))

        # Assert: Empty list returned
        self.assertEqual(result, [])

    def test_extract_text_handles_whitespace_only_pages(self):
        """Test 4: Edge case - pages with only whitespace."""
        # Arrange: Pages with whitespace
        mock_pdf = Mock(spec=PyPDF2.PdfReader)
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "   \n\t   "  # Only whitespace
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Real content"
        mock_pdf.pages = [mock_page1, mock_page2]

        # Act
        with patch('core.pdf_processor.pypdf2.PdfReader', return_value=mock_pdf):
            result = extract_text_from_pdf(BytesIO(b"pdf"))

        # Assert: Whitespace-only page skipped
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["page_number"], 2)

    def test_extract_text_raises_on_corrupted_pdf(self):
        """Test 5: Error case - corrupted/invalid PDF."""
        # Arrange: Mock PdfReader to raise exception
        with patch('core.pdf_processor.pypdf2.PdfReader', side_effect=Exception("Corrupted PDF")):
            # Act & Assert: Should raise ValueError
            with self.assertRaises(ValueError) as context:
                extract_text_from_pdf(BytesIO(b"corrupted"))

            self.assertIn("Could not read PDF", str(context.exception))


class TestChunkText(unittest.TestCase):
    """
    Tests for chunk_text() function.

    HIGH PRIORITY - Complex logic with edge cases.
    """

    def test_chunk_text_with_large_pages(self):
        """Test 1: Happy path - chunk pages larger than chunk_size."""
        # Arrange: Pages with text larger than chunk_size
        pages = [
            {"page_number": 1, "text": "A" * 5000},  # 5000 chars
            {"page_number": 2, "text": "B" * 4000}   # 4000 chars
        ]

        # Act: Chunk with size 2000
        result = chunk_text(pages, chunk_size=2000)

        # Assert: Should create multiple chunks per page
        # Page 1: 5000/2000 = 3 chunks
        # Page 2: 4000/2000 = 2 chunks
        # Total: 5 chunks
        self.assertEqual(len(result), 5)

        # Check page 1 chunks
        page1_chunks = [c for c in result if c["page_number"] == 1]
        self.assertEqual(len(page1_chunks), 3)
        self.assertEqual(page1_chunks[0]["chunk_index"], 0)
        self.assertEqual(page1_chunks[1]["chunk_index"], 1)
        self.assertEqual(page1_chunks[2]["chunk_index"], 2)

    def test_chunk_text_with_small_pages(self):
        """Test 2: Edge case - pages smaller than chunk_size."""
        # Arrange: Small pages
        pages = [
            {"page_number": 1, "text": "Short text"},
            {"page_number": 2, "text": "Another short text"}
        ]

        # Act: Chunk with large size
        result = chunk_text(pages, chunk_size=3000)

        # Assert: Each page becomes single chunk
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["chunk_index"], 0)
        self.assertEqual(result[1]["chunk_index"], 0)
        self.assertEqual(result[0]["text"], "Short text")

    def test_chunk_text_with_empty_pages(self):
        """Test 3: Edge case - empty pages list."""
        # Arrange: Empty input
        pages = []

        # Act
        result = chunk_text(pages, chunk_size=3000)

        # Assert: Empty output
        self.assertEqual(result, [])

    def test_chunk_text_with_custom_chunk_size(self):
        """Test 4: Special behavior - custom chunk_size parameter."""
        # Arrange: Page with 1000 chars
        pages = [{"page_number": 1, "text": "X" * 1000}]

        # Act: Chunk with size 300
        result = chunk_text(pages, chunk_size=300)

        # Assert: 1000/300 = 4 chunks
        self.assertEqual(len(result), 4)
        # First 3 chunks should be 300 chars
        self.assertEqual(len(result[0]["text"]), 300)
        self.assertEqual(len(result[1]["text"]), 300)
        self.assertEqual(len(result[2]["text"]), 300)
        # Last chunk should be 100 chars (remainder)
        self.assertEqual(len(result[3]["text"]), 100)


class TestGetFullText(unittest.TestCase):
    """
    Tests for get_full_text() function.

    MEDIUM PRIORITY - Simple but frequently used.
    """

    def test_get_full_text_with_page_markers(self):
        """Test 1: Happy path - concatenate with page markers."""
        # Arrange
        pages = [
            {"page_number": 1, "text": "First page"},
            {"page_number": 2, "text": "Second page"}
        ]

        # Act
        result = get_full_text(pages, include_page_markers=True)

        # Assert: Should have page markers
        self.assertIn("[Page 1]", result)
        self.assertIn("First page", result)
        self.assertIn("[Page 2]", result)
        self.assertIn("Second page", result)

    def test_get_full_text_without_page_markers(self):
        """Test 2: Special behavior - no page markers."""
        # Arrange
        pages = [
            {"page_number": 1, "text": "First page"},
            {"page_number": 2, "text": "Second page"}
        ]

        # Act
        result = get_full_text(pages, include_page_markers=False)

        # Assert: No page markers
        self.assertNotIn("[Page", result)
        self.assertIn("First page", result)
        self.assertIn("Second page", result)

    def test_get_full_text_with_empty_pages(self):
        """Test 3: Edge case - empty pages list."""
        # Arrange
        pages = []

        # Act
        result = get_full_text(pages)

        # Assert: Empty string
        self.assertEqual(result, "")


class TestExtractTextFromPath(unittest.TestCase):
    """
    Tests for extract_text_from_path() function.

    MEDIUM PRIORITY - Wrapper function, but handles file paths.
    """

    def test_extract_text_from_valid_path(self):
        """Test 1: Happy path - extract from valid file path."""
        # Arrange: Mock file system
        mock_pdf_content = b"fake pdf content"

        with patch('builtins.open', mock_open(read_data=mock_pdf_content)):
            with patch('core.pdf_processor.extract_text_from_pdf') as mock_extract:
                mock_extract.return_value = [{"page_number": 1, "text": "Content"}]

                # Act
                result = extract_text_from_path("/fake/path.pdf")

                # Assert
                self.assertEqual(len(result), 1)
                mock_extract.assert_called_once()

    def test_extract_text_from_invalid_path(self):
        """Test 2: Error case - file not found."""
        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            extract_text_from_path("/nonexistent/file.pdf")


# Test runner
if __name__ == '__main__':
    unittest.main(verbosity=2)
