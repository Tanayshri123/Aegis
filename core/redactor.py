"""
PDF Redactor - Apply redactions to PDF documents based on detected PII entities.

Takes Entity objects (with bbox filled in by Nemotron Parse) and produces a new PDF
with those areas permanently removed using PyMuPDF's redaction API.

The underlying text is TRULY deleted, not just visually covered - copy-paste and
text extraction will not recover redacted content.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from core.document_store import Entity


def _convert_bbox_to_pdf_points(
    bbox: List[float],
    dpi: int = 200,
    padding: float = 2.0
) -> fitz.Rect:
    """
    Convert Nemotron pixel-space bounding box to PyMuPDF PDF-point-space Rect. - Here is where the true redaction logic happens.

    Nemotron Parse renders pages at 200 DPI and returns coordinates in that pixel space.
    PyMuPDF works in PDF points (1 point = 1/72 inch).
    Conversion: pdf_coord = pixel_coord * (72 / dpi)

    A small padding is added to ensure full coverage, since OCR bounding boxes can be slightly tight around text.

    Args:
        bbox: [x1, y1, x2, y2] in pixel coordinates from Nemotron Parse
        dpi: The DPI used when rendering the page image (default: 200, matching NemotronConfig.DPI)
        padding: Points of padding to add around the rect (default: 2.0)

    Returns:
        fitz.Rect in PDF point coordinates, padded
    """
    scale = 72.0 / dpi
    x1 = bbox[0] * scale - padding
    y1 = bbox[1] * scale - padding
    x2 = bbox[2] * scale + padding
    y2 = bbox[3] * scale + padding
    return fitz.Rect(x1, y1, x2, y2)


def _group_entities_by_page(entities: List[Entity]) -> Dict[int, List[Entity]]:
    """
    Group entities by page number for efficient batch processing.
    Args:
        entities: List of Entity objects

    Returns:
        Dict mapping page_number (1-indexed) to list of entities on that page.
    """
    pages: Dict[int, List[Entity]] = defaultdict(list)

    for entity in entities:
        if not entity.selected:
            continue
        if entity.bbox is None:
            print(f"  Warning: Skipping entity '{entity.value}' (page {entity.page}) - no bounding box")
            continue
        if len(entity.bbox) != 4:
            print(f"  Warning: Skipping entity '{entity.value}' (page {entity.page}) - invalid bbox format")
            continue
        pages[entity.page].append(entity)

    return dict(pages)


def _build_output_path(original_path: str, output_dir: Optional[str] = None) -> str:
    """
    Construct the output file path for the redacted PDF.

    Never overwrites the original. Places the redacted file in output_dir (defaults to uploads/redacted/) with a '_redacted' suffix.
    Temporary procedure until proper file management and cleanup is implemented.

    Args:
        original_path: Path to the original PDF file
        output_dir: Directory for output. If None, uses uploads/redacted/.

    Returns:
        Full path to the output file (does not yet exist on disk)
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = str(project_root / "uploads" / "redacted")

    os.makedirs(output_dir, exist_ok=True)

    original_name = Path(original_path).stem
    ext = Path(original_path).suffix or ".pdf"
    output_path = os.path.join(output_dir, f"{original_name}_redacted{ext}")

    # Handle collisions (same pattern as pdf_processor.save_uploaded_pdf)
    counter = 1
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{original_name}_redacted_{counter}{ext}")
        counter += 1

    return output_path


def redact_pdf(
    pdf_path: str,
    entities: List[Entity],
    output_dir: Optional[str] = None,
    dpi: int = 200,
    padding: float = 2.0,
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> str:
    """
    Main entry point. Apply redactions to a PDF based on detected PII entities.

    Opens the original PDF, draws black redaction rectangles over each entity's bounding box (converting from Nemotron pixel space to PDF points), applies
    the redactions to permanently remove underlying text, and saves to a new file.

    Uses PyMuPDF's add_redact_annot() + apply_redactions() which truly removes
    the text content underneath (not just a visual overlay).

    Args:
        pdf_path: Path to the original PDF file
        entities: List of Entity objects with bbox filled in by nemotron_parser.
                  Only entities with selected=True and valid bbox are processed.
        output_dir: Directory for redacted output. If None, uses uploads/redacted/.
        dpi: DPI used when Nemotron rendered page images (default: 200)
        padding: Points of padding around each redaction box (default: 2.0)
        fill_color: RGB tuple for the redaction fill (default: black)

    Returns:
        Path to the newly created redacted PDF file

    Raises:
        FileNotFoundError: If pdf_path does not exist
        ValueError: If no entities have valid bounding boxes
        RuntimeError: If PyMuPDF fails to open the PDF
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages_to_redact = _group_entities_by_page(entities)

    if not pages_to_redact:
        raise ValueError(
            "No entities with valid bounding boxes to redact. "
            "Ensure Nemotron parser has run and populated entity.bbox fields."
        )

    output_path = _build_output_path(pdf_path, output_dir)

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF '{pdf_path}': {e}")

    redaction_count = 0

    try:
        for page_num, page_entities in sorted(pages_to_redact.items()):
            # Entity.page is 1-indexed, fitz pages are 0-indexed
            fitz_page_index = page_num - 1

            if fitz_page_index < 0 or fitz_page_index >= len(doc):
                print(f"  Warning: Page {page_num} out of range (PDF has {len(doc)} pages), skipping")
                continue

            page = doc[fitz_page_index]

            for entity in page_entities:
                rect = _convert_bbox_to_pdf_points(entity.bbox, dpi=dpi, padding=padding)

                # Clamp rect to page boundaries
                rect = rect & page.rect

                if rect.is_empty or rect.is_infinite:
                    print(f"  Warning: Invalid rect for '{entity.value}' on page {page_num}, skipping")
                    continue

                page.add_redact_annot(rect, fill=fill_color)
                redaction_count += 1

            # Apply all redactions for this page at once - this removes the underlying text
            page.apply_redactions()

    finally:
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()

    print(f"  Redacted {redaction_count} entities across {len(pages_to_redact)} pages -> {output_path}")
    return output_path
