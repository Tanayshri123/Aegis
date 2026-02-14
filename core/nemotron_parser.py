"""
Nemotron Parser - Find pixel coordinates for PII text in PDF pages.

Takes Entity objects (with value and page, but bbox=None) from the PII detector
and returns them with bbox filled in by sending page images to Nemotron Parse.

Supports two modes:
1. Docker (offline) - Nemotron Parse running locally on localhost:8000
2. Cloud (NVIDIA API) - Hosted Nemotron Parse service
"""

import base64
import httpx
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

from pdf2image import convert_from_path
from PIL import Image
from core.document_store import Entity
from core.pii_detector import get_affected_pages


class NemotronConfig:
    """Configuration for Nemotron Parse API."""
    DOCKER_URL = "http://localhost:8000/v1/chat/completions" # Local Docker endpoint
    CLOUD_URL = "https://integrate.api.nvidia.com/v1/chat/completions" # NVIDIA cloud endpoint
    MODEL = "nvidia/nemotron-parse"
    DPI = 200
    IMAGE_FORMAT = "PNG"  # lossless — JPEG was introducing pixel-level noise across runs
    TIMEOUT = 120.0


def convert_page_to_image(pdf_path: str, page_number: int, dpi: int = NemotronConfig.DPI) -> Path:
    """
    Pdf to PNG image - needed because nemotron parse is vision-based.
    PNG (lossless) instead of JPEG to ensure bit-identical images across runs.
    """
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_number,
        last_page=page_number
    )

    temp_dir = tempfile.mkdtemp(prefix="aegis_")
    img_path = Path(temp_dir) / f"page_{page_number}.png"
    images[0].save(img_path, NemotronConfig.IMAGE_FORMAT)

    return img_path


def image_to_base64(image_path: Path) -> str:
    """
    Encode an image file as a base64 string for API transmission.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


async def parse_page(
    image_path: Path,
    use_docker: bool = True,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Send a page image to Nemotron Parse and returns what text elements it finds with their bounding boxes coordinates.

    Args:
        image_path: Path to the page image (JPEG)
        use_docker: True for local Docker, False for NVIDIA cloud
        api_key: NVIDIA API key (required for cloud mode)

    Returns:
        List of elements: [{"text": "...", "bbox": [x1,y1,x2,y2], "type": "..."}, ...]
    """
    if use_docker:
        endpoint = NemotronConfig.DOCKER_URL
        headers = {"Content-Type": "application/json"}
    else:
        if not api_key:
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY required for cloud mode")
        endpoint = NemotronConfig.CLOUD_URL
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    b64_image = image_to_base64(image_path)

    # Nemotron Parse expects markdown input with an <img> tag, and returns results in a markdown format with [bbox: x1,y1,x2,y2] markers. 
    # We wrap the image in a simple prompt to trigger the parsing behavior.
    # MAIN nemotron parse call - sends the page image and gets back the text elements with bounding boxes in a markdown format for EVERY word not just the PII, we will filter for the PII matches later. 
    payload = {
        "model": NemotronConfig.MODEL, # what model to use: specialized parsing model not general model
        "messages": [{
            "role": "user",
            "content": f'<img src="data:image/png;base64,{b64_image}" />'
        }], # image is wrapped in markdown with an <img> tag to trigger Nemotron's parsing, acting as a "user message" in the chat
        "tools": [{
            "type": "function",
            "function": {"name": "markdown_bbox"}
        }], # markdown_bbox is a special tool in Nemotron that triggers the bounding box parsing behavior, it returns the text in a markdown format with [bbox: x1,y1,x2,y2] markers for EVERY word on the page
        "tool_choice": {
            "type": "function",
            "function": {"name": "markdown_bbox"}
        }, # Force the model to use the bbox tool. Without this, it may choose not to, resulting in an empty response.
        "temperature": 0,  # deterministic: same image always produces same elements
        "seed": 42,        # additional reproducibility hint for the NVIDIA backend
        "max_tokens": 8192
    }

    async with httpx.AsyncClient(timeout=NemotronConfig.TIMEOUT) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    choice = result["choices"][0]

    # Detect response truncation — if the model hit max_tokens, some text elements
    # are silently missing. Different runs may truncate at different token boundaries,
    # which is a major source of non-determinism on dense pages.
    finish_reason = choice.get("finish_reason", "")
    if finish_reason == "length":
        print(f"  *** WARNING: Nemotron response was TRUNCATED (hit max_tokens={payload['max_tokens']}). ***")
        print(f"  *** Some text elements may be missing — redaction may be incomplete. ***")

    message = choice["message"]

    # Nemotron can return data in `tool_calls` or `content`.
    # We prioritize `tool_calls` as it's more structured.
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        fn = tc.get("function", {})
        if fn.get("name") != "markdown_bbox":
            continue

        args = fn.get("arguments")
        if not args:
            continue

        # --- NEW: Handle structured JSON response from Nemotron Parse ---
        # This format returns normalized coordinates which we convert to pixels.
        try:
            parsed_args = json.loads(args)
            # Heuristic for new format: list -> list -> dict with 'bbox'
            if (isinstance(parsed_args, list) and parsed_args and
                isinstance(parsed_args[0], list) and parsed_args[0] and
                isinstance(parsed_args[0][0], dict) and 'bbox' in parsed_args[0][0]):

                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                elements = []
                for item in parsed_args[0]:
                    bbox_norm = item.get('bbox')
                    if not isinstance(bbox_norm, dict): continue

                    # Convert normalized bbox (0-1) to pixel bbox
                    x1 = bbox_norm.get('xmin', 0) * img_width
                    y1 = bbox_norm.get('ymin', 0) * img_height
                    x2 = bbox_norm.get('xmax', 0) * img_width
                    y2 = bbox_norm.get('ymax', 0) * img_height

                    elements.append({
                        "text": item.get("text", "").strip(),
                        "bbox": [x1, y1, x2, y2],
                        "type": item.get("type", "text")
                    })

                if elements:
                    return elements
        except (json.JSONDecodeError, TypeError, AttributeError, IndexError):
            # Not the new format, fall through to markdown parsing.
            pass

        # --- FALLBACK: Handle markdown content in 'arguments' ---
        # This format returns pixel coordinates directly in a markdown string.
        content_str = None
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    for value in parsed.values():
                        if isinstance(value, str):
                            content_str = value
                            break
                elif isinstance(parsed, str):
                    content_str = parsed
            except json.JSONDecodeError:
                content_str = args
        elif isinstance(args, dict):
            for value in args.values():
                if isinstance(value, str):
                    content_str = value
                    break

        if content_str:
            return _parse_bbox_response(content_str)

    # --- FINAL FALLBACK: Check top-level 'content' for markdown ---
    content = message.get("content")
    if content:
        return _parse_bbox_response(content)

    # If we've reached here, we found nothing.
    print(f"  Warning: Nemotron returned no parsable content or tool_calls")
    return []


def _parse_bbox_response(content: str) -> List[Dict]:
    """
    Parse Nemotron's markdown_bbox response into structured elements
    Nemotron returns lines like: [bbox: x1,y1,x2,y2] text content
    We are simply extracting messy format into clean list of dicts with text and bbox coordinates for each element on the page
    Args:
        content: Raw response text from Nemotron

    Returns:
        List of element dicts with text, bbox, and type
    """
    elements = []

    for line in content.split("\n"):
        # Strip common markdown prefixes that Nemotron may add (e.g. "- [bbox:", "| [bbox:")
        # so they don't prevent the regex from matching and silently drop elements.
        line = line.lstrip(" \t-|*>#")
        bbox_match = re.match(
            r'\[bbox:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*(.*)',
            line
        )

        if bbox_match:
            x1, y1, x2, y2, text = bbox_match.groups()
            elements.append({
                "text": text.strip(),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "type": "text"
            })

    return elements


def _normalize(text: str) -> str:
    """Strip dashes, spaces, and punctuation for fuzzy comparison."""
    return re.sub(r'[-\s.,()\u2010-\u2015]+', '', text).lower()


def _merge_bboxes(bboxes: List[List[float]]) -> List[float]:
    """Merge multiple bounding boxes into one encompassing box."""
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def find_entity_bbox(entity_value: str, elements: List[Dict]) -> Optional[List[float]]:
    """
    Search parsed page elements for a specific PII value and return its bounding box.
    Returns only the first match. Use find_all_entity_bboxes for multi-occurrence documents.

    Args:
        entity_value: The PII text to find (e.g., "123-45-6789")
        elements: List of elements from parse_page()

    Returns:
        Bounding box [x1, y1, x2, y2] or None if not found
    """
    results = find_all_entity_bboxes(entity_value, elements)
    return results[0] if results else None


def find_all_entity_bboxes(entity_value: str, elements: List[Dict]) -> List[List[float]]:
    """
    Search parsed page elements for ALL occurrences of a PII value and return their bounding boxes.
    Nemotron returns bbox coordinates for EVERY word on the page, here we filter for PII.

    Critical for documents like W-2s where the same name/SSN/address appears
    multiple times across different sections (earnings summary, W-4 profile,
    federal/state/local copies).

    Matching strategy (tries each in order, collecting all hits):
      1. Exact substring match within single elements
      2. Case-insensitive substring match within single elements
      3. Normalized match (ignore dashes/spaces) within single elements
      4. Multi-element spanning match - combines adjacent elements to find
         values that are split across Nemotron text blocks

    Args:
        entity_value: The PII text to find (e.g., "123-45-6789")
        elements: List of elements from parse_page()

    Returns:
        List of bounding boxes [[x1,y1,x2,y2], ...] for every occurrence found
    """
    if not elements:
        return []

    # Sort elements by spatial position (top-to-bottom, then left-to-right).
    # Nemotron may return elements in a different order across runs; sorting
    # makes the sliding-window span matching order-independent.
    elements = sorted(elements, key=lambda e: (
        e.get("bbox", [0, 0])[1],  # y1: top-to-bottom
        e.get("bbox", [0, 0])[0],  # x1: left-to-right
    ))

    found_bboxes = []
    matched_indices = set()  # Track which elements have been matched to avoid double-counting

    value_lower = entity_value.lower()
    normalized_value = _normalize(entity_value)

    # Pass 1: Single-element matches (exact, case-insensitive, normalized)
    for i, element in enumerate(elements):
        text = element.get("text", "")
        bbox = element.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        if (entity_value in text
                or value_lower in text.lower()
                or (normalized_value and normalized_value in _normalize(text))):
            found_bboxes.append(bbox)
            matched_indices.add(i)

    # Pass 2: Multi-element spanning - combine up to 8 adjacent elements.
    # Window is 8 (not 5) to handle long multi-token values like full addresses.
    for i in range(len(elements)):
        if i in matched_indices:
            continue

        combined_text = elements[i].get("text", "")
        span_indices = [i]

        for j in range(i + 1, min(i + 8, len(elements))):
            # Stop extending the span as soon as we hit an already-matched element.
            # Previously this was `continue` (scoped to the j-loop), which kept
            # accumulating the matched index into span_indices on every iteration,
            # permanently poisoning the combined_text and preventing any match.
            if j in matched_indices:
                break

            combined_text += " " + elements[j].get("text", "")
            span_indices.append(j)

            if (entity_value in combined_text
                    or value_lower in combined_text.lower()
                    or (normalized_value and normalized_value in _normalize(combined_text))):
                bboxes = [elements[idx].get("bbox") for idx in span_indices]
                valid = [b for b in bboxes if b and len(b) == 4]
                if valid:
                    found_bboxes.append(_merge_bboxes(valid))
                    matched_indices.update(span_indices)
                break  # Found a span match starting at i, move on

    return found_bboxes


async def locate_entities(
    pdf_path: str,
    entities: List[Entity],
    use_docker: bool = True,
    api_key: Optional[str] = None
) -> List[Entity]:
    """
    Main entry point. Takes entities with bbox=None, returns them with bbox filled in.

    Only processes pages that actually contain PII (from get_affected_pages).
    Converts each affected page to an image, sends to Nemotron Parse,
    and matches entity values to bounding boxes in the response.

    Args:
        pdf_path: Path to the original PDF file
        entities: List of Entity objects from pii_detector (bbox=None)
        use_docker: True for local Docker, False for NVIDIA cloud
        api_key: NVIDIA API key (for cloud mode)

    Returns:
        Same entities but with bbox filled in where found
    """
    affected = get_affected_pages(entities)

    if not affected:
        return entities

    # Cache parsed elements per page so that if multiple entities are on the same page we only call Nemotron Parse once per page, improving efficiency significantly
    page_elements = {}

    for page_num in affected:
        img_path = None
        try:
            # convert just this page to image
            img_path = convert_page_to_image(pdf_path, page_num)

            # sending to Nemotron Parse
            elements = await parse_page(img_path, use_docker=use_docker, api_key=api_key)
            page_elements[page_num] = elements
            print(f"  Page {page_num}: Nemotron returned {len(elements)} text elements")
            if elements:
                print(f"  Sample element: {elements[0].get('text', '')[:80]!r}")
            else:
                print(f"  Warning: No elements parsed - check Nemotron response format")

        except Exception as e:
            print(f"  Warning: Could not parse page {page_num}: {e}")
            page_elements[page_num] = []

        finally:
            # Clean up temp image
            if img_path and img_path.exists():
                img_path.unlink(missing_ok=True)
                try:
                    img_path.parent.rmdir()
                except OSError:
                    pass

    # Match each entity to its bounding box(es)
    # On documents like W-2s, the same PII value appears multiple times on a page
    # (e.g. name in earnings summary, W-4 profile, and 3 W-2 copies).
    # We create a separate entity for each occurrence so all get redacted.
    result_entities = []

    for entity in entities:
        if entity.bbox is not None:
            result_entities.append(entity)
            continue  # Already has coordinates

        elements = page_elements.get(entity.page, [])
        bboxes = find_all_entity_bboxes(entity.value, elements)

        if not bboxes:
            print(f"  Warning: Could not find bbox for '{entity.value}' on page {entity.page}")
            result_entities.append(entity)  # Keep it even without bbox
            continue

        # First bbox goes on the original entity
        entity.bbox = bboxes[0]
        result_entities.append(entity)

        # Additional occurrences get cloned entities
        for bbox in bboxes[1:]:
            clone = Entity(
                type=entity.type,
                value=entity.value,
                page=entity.page,
                confidence=entity.confidence,
                selected=entity.selected,
                context=entity.context,
                bbox=bbox,
                semantic_type=entity.semantic_type,
            )
            result_entities.append(clone)

        if len(bboxes) > 1:
            print(f"  Found {len(bboxes)} occurrences of '{entity.value}' on page {entity.page}")

    return result_entities
