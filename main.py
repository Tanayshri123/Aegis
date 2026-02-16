"""
Aegis - PDF PII Redaction Pipeline

USAGE INSTRUCTIONS:
    python main.py document.pdf                    # Docker Nemotron (default)
    python main.py doc1.pdf doc2.pdf --cloud       # Cloud Nemotron
"""

import argparse
import asyncio
import concurrent.futures
import os
import threading
import time
from importlib import import_module

from core.pdf_processor import extract_text_from_pdf
from core.pii_detector import scan_document
from core.nemotron_parser import locate_entities
from core.redactor import redact_pdf

# ── Cached LLM instance (avoids re-validating against NVIDIA /models API) ──
_cached_pipeline_llm = None
_pipeline_llm_lock = threading.Lock()
LLM_LOAD_TIMEOUT = 60


def _get_pipeline_llm():
    """Get or create the cached pipeline LLM (temperature=0, deterministic)."""
    global _cached_pipeline_llm
    if _cached_pipeline_llm is None:
        with _pipeline_llm_lock:
            if _cached_pipeline_llm is None:
                llm_utils = import_module("llm-utils")
                _cached_pipeline_llm = llm_utils.get_model()
    return _cached_pipeline_llm


async def process_pdf(pdf_path: str, use_docker: bool = True, return_entities: bool = False,
                      progress_callback=None):
    """
    Run the full redaction pipeline on a single PDF.

    Steps:
        1. Extract text from each page
        2. Detect PII (regex + LLM)
        3. Locate PII bounding boxes via Nemotron Parse
        4. Redact and save clean PDF

    THIS is without the chat interface, to isolate the core pipeline for testing and benchmarking
    This will ideally return the core PII redacted prior to any further user prompting or interaction

    Args:
        pdf_path: Path to the input PDF
        use_docker: True for local Docker Nemotron, False for NVIDIA cloud
        return_entities: If True, return (output_path, entities_list, pages) tuple
        progress_callback: Optional callable(step, detail) for live progress updates

    Returns:
        Path to the redacted output PDF, or None if nothing to redact.
        If return_entities=True, returns (path, entities_list, pages) tuple.
    """
    def _progress(step, detail=""):
        """Report progress to callback and stdout."""
        if progress_callback:
            progress_callback(step, detail)

    pdf_path = os.path.abspath(pdf_path)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {pdf_path}")

    filename = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")

    # Step 1: Extract text
    t0 = time.time()
    print(f"\n[1/4] Extracting text...")
    _progress(1, "Extracting text from PDF...")
    with open(pdf_path, "rb") as f:
        pages = extract_text_from_pdf(f)
    total_chars = sum(len(p.get("text", "")) for p in pages)
    print(f"  Extracted {len(pages)} page(s), {total_chars} total chars  ({time.time()-t0:.1f}s)")

    if not pages:
        raise ValueError(f"No text found in {filename}")

    for p in pages:
        snippet = p.get("text", "").strip()[:120].replace("\n", " ")
        print(f"  Page {p['page_number']} preview: {snippet!r}")

    # Step 2: Detect PII
    t0 = time.time()
    print(f"\n[2/4] Detecting PII...")
    _progress(2, "Detecting PII (regex + LLM)...")
    llm = None
    try:
        print(f"  Loading LLM (nvidia/llama-3.3-nemotron-super-49b-v1.5)... ", end="", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_get_pipeline_llm)
            llm = future.result(timeout=LLM_LOAD_TIMEOUT)
        print(f"ready  ({time.time()-t0:.1f}s)")
        print(f"  Using LLM + regex detection.")
    except concurrent.futures.TimeoutError:
        print(f"timed out after {LLM_LOAD_TIMEOUT}s")
        print(f"\n  *** WARNING: LLM UNAVAILABLE — FALLING BACK TO REGEX-ONLY ***")
        print(f"  *** NAME, ADDRESS, and EMPLOYER_NAME will NOT be detected.  ***")
        print(f"  *** The redacted output will be INCOMPLETE.                 ***")
        print(f"  Tip: check your NVIDIA_API_KEY and network, then re-run.")
        print()
    except Exception as e:
        print(f"failed ({e})")
        print(f"\n  *** WARNING: LLM UNAVAILABLE — FALLING BACK TO REGEX-ONLY ***")
        print(f"  *** NAME, ADDRESS, and EMPLOYER_NAME will NOT be detected.  ***")
        print(f"  *** The redacted output will be INCOMPLETE.                 ***")
        print()

    entities = scan_document(pages, llm=llm)
    print(f"  Found {len(entities)} PII entities  ({time.time()-t0:.1f}s)")

    if entities:
        for e in entities:
            print(f"    [{e.type}] {e.value!r}  (page {e.page}, conf={e.confidence:.2f})")
    else:
        print(f"  No PII detected in {filename}. Nothing to redact.")
        return (None, [], pages) if return_entities else None

    # Step 3: Locate bounding boxes
    t0 = time.time()
    mode = "Docker" if use_docker else "Cloud"
    print(f"\n[3/4] Locating bounding boxes ({mode} Nemotron)...")
    _progress(3, f"Locating bounding boxes ({mode} Nemotron)...")
    entities = await locate_entities(pdf_path, entities, use_docker=use_docker)

    entities_with_bbox = [e for e in entities if e.bbox is not None]
    entities_missing_bbox = [e for e in entities if e.bbox is None]
    print(f"  Located {len(entities_with_bbox)}/{len(entities)} entity bounding boxes  ({time.time()-t0:.1f}s)")

    if entities_missing_bbox:
        print(f"  MISSED {len(entities_missing_bbox)} entities (no bbox found):")
        for e in entities_missing_bbox:
            print(f"    [{e.type}] {e.value!r}  (page {e.page})")

    if not entities_with_bbox:
        print(f"  Could not locate any bounding boxes. Cannot redact.")
        return (None, entities, pages) if return_entities else None

    # Step 4: Redact
    t0 = time.time()
    print(f"\n[4/4] Redacting {len(entities_with_bbox)} regions...")
    _progress(4, f"Redacting {len(entities_with_bbox)} regions...")
    output_path = redact_pdf(pdf_path, entities)
    print(f"  Done  ({time.time()-t0:.1f}s)")
    print(f"\n  Output: {output_path}")

    if return_entities:
        return output_path, entities, pages
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Aegis - Detect and redact PII from PDF documents"
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        help="PDF file(s) to process"
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use NVIDIA cloud API for Nemotron Parse (default: local Docker)"
    )

    args = parser.parse_args()
    use_docker = not args.cloud

    results = []
    for pdf_path in args.pdfs:
        try:
            output = asyncio.run(process_pdf(pdf_path, use_docker=use_docker))
            results.append((pdf_path, output))
        except Exception as e:
            print(f"\n  Error processing {pdf_path}: {e}")
            results.append((pdf_path, None))

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    for pdf_path, output in results:
        name = os.path.basename(pdf_path)
        if output:
            print(f"  {name} -> {output}")
        else:
            print(f"  {name} -> FAILED")


if __name__ == "__main__":
    main()
