"""
PDF Processor - Extract and process text from PDF documents.
"""

import os
from typing import List, Dict, BinaryIO
import PyPDF2 as pypdf2
import aiofiles


def extract_text_from_pdf(pdf_file: BinaryIO) -> List[Dict]:
    """
    Extracts text from a PDF file page by page specifically

    Args:
        pdf_file: File object or path to PDF

    Returns:
        LIST of dicts: [{"page_number": 1, "text": "..."}, ...]
    """
    try:
        pdf_reader = pypdf2.PdfReader(pdf_file) # pdf_reader is a PdfReader object
        pages = []
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append({
                    "page_number": i + 1,
                    "text": page_text
                })
        return pages
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")


def chunk_text(pages: List[Dict], chunk_size: int = 3000) -> List[Dict]:
    """
    Just chunking each page's text into smaller pieces for LLM processing
    3000 chunk size be default

    Args:
        pages: List of dicts from extract_text_from_pdf()
        chunk_size: Maximum characters per chunk

    Returns:
        List of dicts: [{"page_number": 1, "chunk_index": 0, "text": "..."}, ...]
    """

    chunks = []

    for page in pages:
        page_num = page["page_number"]
        text = page["text"]
        if len(text) <= chunk_size:
            chunks.append({
                "page_number": page_num,
                "chunk_index": 0,
                "text": text
            })
        else:
            idx = 0
            for start in range(0, len(text), chunk_size): # iterating by one chunk size
                chunks.append({
                    "page_number": page_num,
                    "chunk_index": idx,
                    "text": text[start:start + chunk_size],
                })
                idx += 1
    return chunks


def get_full_text(pages: List[Dict], include_page_markers: bool = True) -> str:
    """
    Concatenate all page text for LLM context.
    Args:
        pages: List of page dicts from extract_text_from_pdf()
        include_page_markers: Whether to include [Page X] markers

    Returns:
        Single string with all document text
    """
    if include_page_markers:
        return "\n\n".join(
            f"[Page {p['page_number']}]\n{p['text']}"
            for p in pages
        )
    return "\n\n".join(p['text'] for p in pages)


def get_text_preview(pages: List[Dict], max_chars: int = 2000) -> str:
    """
    Get a preview of the document text, limited to max_chars.

    Args:
        pages: List of page dicts
        max_chars: Maximum characters to include

    Returns:
        Truncated preview string
    """
    full_text = get_full_text(pages, include_page_markers=True)
    if len(full_text) <= max_chars:
        return full_text
    return full_text[:max_chars] + "..."


async def save_uploaded_pdf(file, upload_dir: str) -> str:
    """
    Save an uploaded PDF file to disk.

    Args:
        file: UploadFile from FastAPI
        upload_dir: Directory to save to

    Returns:
        Full path to saved file
    """
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize filename
    safe_filename = "".join(
        c for c in file.filename
        if c.isalnum() or c in ('_', '-', '.')
    )
    file_path = os.path.join(upload_dir, safe_filename)

    # Handle duplicate filenames
    base, ext = os.path.splitext(file_path)
    counter = 1
    while os.path.exists(file_path):
        file_path = f"{base}_{counter}{ext}"
        counter += 1

    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    return file_path


def extract_text_from_path(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF file path.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of page dicts using extract_text_from_pdf() from before
    """
    with open(pdf_path, 'rb') as f:
        return extract_text_from_pdf(f)
