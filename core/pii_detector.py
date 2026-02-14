"""
PII Detector - Identify personally identifiable information in document text.

Two detection channels:
1. Automatic scan (regex + LLM) - runs on upload, finds standard PII
2. User-driven (find_custom_entities) - user says "redact Company X" via chat

Output: List of Entity objects (without bbox - coordinates come later from Nemotron parser)
"""

import re
import json
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Optional

from importlib import import_module

from core.document_store import Entity

# Add project root for llm-utils import (filename has hyphen)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Prompt config path
PROMPTS_FILE = Path(__file__).parent.parent / "config" / "prompts.yml"

# regex based recognition of structured PII (SSN, email, phone, credit card, DOB, account numbers, EIN, etc.)
PII_PATTERNS = {
    "SSN": {
        "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
        "context_required": False,
    },
    "EIN": {
        "pattern": r'\b\d{2}-\d{7}\b',
        "context_required": True,
        "context_keywords": [
            "employer", "ein", "fed", "federal", "fein", "tax id",
            "identification number", "employer's fed", "employer's federal",
        ],
    },
    "EMAIL": {
        "pattern": r'\b[\w.-]+@[\w.-]+\.\w{2,}\b',
        "context_required": False,
    },
    "PHONE": {
        "pattern": r'(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
        "context_required": False,
    },
    "CREDIT_CARD": {
        "pattern": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "context_required": False,
    },
    "DATE_OF_BIRTH": {
        "pattern": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        "context_required": True,
        "context_keywords": ["dob", "born", "birth", "date of birth", "birthday"],
    },
    "ACCOUNT_NUMBER": {
        "pattern": r'\b\d{8,17}\b',
        "context_required": True,
        "context_keywords": ["account", "acct", "routing", "bank", "checking", "savings"],
    },
    "STATE_ID": {
        "pattern": r'\b\d{2}-\d{2,3}-\d{3,4}\b',
        "context_required": True,
        "context_keywords": [
            "state", "employer state", "state id", "employee state",
        ],
    },
    "CONTROL_NUMBER": {
        "pattern": r'\b\d{5,9}\b',
        "context_required": True,
        "context_keywords": ["control number"],
        "context_window": 40,
    },
}

def _load_prompt(task_name: str) -> str:
    """FROM the config/prompts.yml file, thsi function loads the specific prompt with the name as input"""
    with open(PROMPTS_FILE, "r") as f:
        config = yaml.safe_load(f)

    for prompt in config.get("prompts", []):
        if prompt.get("task") == task_name:
            return prompt["content"]

    raise ValueError(f"Prompt '{task_name}' not found in {PROMPTS_FILE}")


def _extract_context(text: str, match_start: int, match_end: int, window: int = 50) -> str:
    """Extract surrounding text around a match for context."""
    ctx_start = max(0, match_start - window)
    ctx_end = min(len(text), match_end + window)
    return text[ctx_start:ctx_end].strip()


def regex_scan(text: str, page_number: int) -> List[Entity]:
    """
    Fast pattern-matching scan for structured PII (SSN, email, phone, etc.).

    Args:
        text: Page text to scan
        page_number: Which page this text is from (1-indexed)

    Returns:
        List of Entity objects found via regex (bbox=None, filled later by parser)
    """
    entities = []

    if not text or not text.strip():
        return entities

    text_lower = text.lower()

    for pii_type, config in PII_PATTERNS.items():
        pattern = config["pattern"]
        matches = re.finditer(pattern, text)

        for match in matches:
            value = match.group(0)

            # Context-dependent patterns need nearby keywords to avoid false positives
            if config["context_required"]:
                window = config.get("context_window", 100)
                context_window = text_lower[max(0, match.start() - window):match.end() + window]
                keywords = config["context_keywords"]
                if not any(kw in context_window for kw in keywords):
                    continue

            context = _extract_context(text, match.start(), match.end())

            entities.append(Entity(
                type=pii_type,
                value=value,
                page=page_number,
                confidence=0.95,
                selected=True,
                context=context,
                bbox=None,
                semantic_type=None,
            ))

    return entities


def llm_scan(text: str, page_number: int, llm: Optional[object]) -> List[Entity]:
    """
    Context-aware PII detection using LLM. Catches names, addresses, and other PII that regex cannot detect.
    More advanced checking after the regex scanning

    Goes page by page, sending the text to the LLM with a prompt that asks it to identify any PII and return in structured JSON format.

    Args:
        text: Page text to scan
        page_number: Which page this text is from (1-indexed)
        llm: LangChain-compatible LLM object (from get_model() or any other)

    Returns:
        List of Entity objects found by the LLM
    """
    if not text or not text.strip():
        return []

    # Load prompt from config/prompts.yml (single source of truth)
    template = _load_prompt("entity_extraction")
    prompt = template.replace("{{ page_number }}", str(page_number)).replace("{{ document_text }}", text)

    print(f"    [LLM] Page {page_number}: sending {len(text)} chars (~{len(prompt)//4} tokens) to model...")
    t0 = time.time()
    try:
        response = llm.invoke(prompt)
        print(f"    [LLM] Page {page_number}: response received in {time.time()-t0:.1f}s")
        content = response.content if hasattr(response, 'content') else str(response)
        try:
            
            llm_utils = import_module("llm-utils")
            content = llm_utils.output_cleaner(content)
        except (ImportError, ModuleNotFoundError):
            # Inline fallback: strip <think> blocks if llm-utils not available
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return []

        parsed = json.loads(json_match.group(0))
        raw_entities = parsed.get("entities", [])

        entities = []
        for item in raw_entities:
            entity_type = item.get("type", "UNKNOWN")
            value = item.get("value", "")
            context = item.get("context", "")
            confidence = item.get("confidence", 0.85)

            if not value:
                continue

            entities.append(Entity(
                type=entity_type,
                value=value,
                page=page_number,
                confidence=confidence,
                selected=True,
                context=context,
                bbox=None,
                semantic_type=None,
            ))

        return entities

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        print(f"  Warning: LLM scan failed for page {page_number}: {e}")
        return []
    except Exception as e:
        print(f"  Warning: LLM scan error for page {page_number}: {e}")
        return []


def deduplicate_entities(entities: List[Entity]) -> List[Entity]:
    """
    Merge results from regex + LLM scans, removing duplicates.

    Two-pass deduplication:
    1. By (type, value, page) - keeps highest-confidence when both regex and LLM
       detect the exact same entity.
    2. By (value.lower(), page) - collapses entities that carry the same text
       but differ only in type label (e.g. NAME vs PERSON, or SSN detected by
       both channels under slightly different type strings after an LLM change).
       This is safe because two detections of the identical text string on the
       same page will always produce the same bounding box lookup regardless of
       their type label, so collapsing them is correct.
    """
    # Pass 1: deduplicate by (type, value, page)
    by_type_key: dict = {}
    for entity in entities:
        key = (entity.type, entity.value.strip(), entity.page)
        if key in by_type_key:
            if entity.confidence > by_type_key[key].confidence:
                by_type_key[key] = entity
        else:
            by_type_key[key] = entity

    # Pass 2: deduplicate by (value.lower(), page) — catches same value returned
    # under different type names (e.g. LLM returns NAME one run, PERSON another)
    by_value_key: dict = {}
    for entity in by_type_key.values():
        value_key = (entity.value.strip().lower(), entity.page)
        if value_key in by_value_key:
            if entity.confidence > by_value_key[value_key].confidence:
                by_value_key[value_key] = entity
        else:
            by_value_key[value_key] = entity

    return list(by_value_key.values())


def scan_document(pages: List[Dict], llm=None) -> List[Entity]:
    """
    Main entry point for AUTOMATIC PII detection (Channel A).

    Runs regex scan (fast) + LLM scan (thorough) on every page, then deduplicates.
    If no LLM is provided, falls back to regex-only mode.

    Args:
        pages: Output from pdf_processor.extract_text_from_pdf()
               [{"page_number": 1, "text": "..."}, ...]
        llm: Optional LangChain-compatible LLM. If None, tries to create via get_model().

    Returns:
        List of Entity objects with all detected PII
    """
    # Try to get LLM if not provided
    # ChatNVIDIA constructor blocks on NVIDIA's /models API — use a thread with timeout
    if llm is None:
        try:
            import concurrent.futures
            from importlib import import_module
            llm_utils = import_module("llm-utils")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(llm_utils.get_model)
                llm = future.result(timeout=30)
        except concurrent.futures.TimeoutError:
            print(f"  Warning: LLM load timed out. Using regex-only mode.")
            llm = None
        except Exception as e:
            print(f"  Warning: Could not initialize LLM ({e}). Using regex-only mode.")
            llm = None

    all_entities = []

    for page in pages:
        page_num = page["page_number"]
        text = page["text"]

        if not text or not text.strip():
            print(f"  Page {page_num}: empty, skipping")
            continue

        print(f"  Page {page_num}: scanning {len(text)} chars...")

        # Regex scan (always runs, fast)
        t0 = time.time()
        regex_results = regex_scan(text, page_num)
        print(f"    [regex] {len(regex_results)} entities found  ({time.time()-t0:.2f}s)")
        for e in regex_results:
            print(f"      [{e.type}] {e.value!r}")

        # LLM scan (runs if LLM available)
        llm_results = []
        if llm is not None:
            llm_results = llm_scan(text, page_num, llm)
            print(f"    [LLM]   {len(llm_results)} entities found")
            for e in llm_results:
                print(f"      [{e.type}] {e.value!r}")
        else:
            print(f"    [LLM]   skipped (no model)")

        # Combine and deduplicate for this page
        page_entities = deduplicate_entities(regex_results + llm_results)
        print(f"  Page {page_num}: {len(page_entities)} unique entities after deduplication")
        all_entities.extend(page_entities)

    return all_entities


def find_custom_entities(pages: List[Dict], search_terms: List[str]) -> List[Entity]:
    """
    User-driven PII detection (Channel B).

    When a user says "redact all mentions of Company X" via chat, this function finds every occurrence across all pages.

    Args:
        pages: Output from pdf_processor.extract_text_from_pdf()
        search_terms: List of terms to find (e.g., ["Acme Corp", "John Smith"])

    Returns:
        List of Entity objects with type="CUSTOM" and confidence=1.0
    """
    entities = []

    for term in search_terms:
        if not term or not term.strip():
            continue

        pattern = r'\b' + re.escape(term.strip()) + r'\b'

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]

            if not text:
                continue

            # Find all matches using the case-insensitive flag
            for match in re.finditer(pattern, text, re.IGNORECASE):
                actual_value = match.group(0)  # This preserves the original casing from the document
                context = _extract_context(text, match.start(), match.end())

                entities.append(Entity(
                    type="CUSTOM",
                    value=actual_value,
                    page=page_num,
                    confidence=1.0,
                    selected=True,
                    context=context,
                    bbox=None,
                    semantic_type=None,
                ))

    return entities


def get_affected_pages(entities: List[Entity]) -> List[int]:
    """
    Get sorted list of unique page numbers that contain PII.

    This tells the Nemotron parser which pages need coordinate extraction, avoiding processing all 500 pages when only 20 have PII.
    """
    return sorted(set(entity.page for entity in entities))
