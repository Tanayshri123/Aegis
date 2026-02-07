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
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

from pdf2image import convert_from_path
from core.document_store import Entity
from core.pii_detector import get_affected_pages


class NemotronConfig:
    """Configuration for Nemotron Parse API."""
    DOCKER_URL = "http://localhost:8000/v1/chat/completions" # Local Docker endpoint
    CLOUD_URL = "https://integrate.api.nvidia.com/v1/chat/completions" # NVIDIA cloud endpoint
    MODEL = "nvidia/nemotron-parse"
    DPI = 200
    IMAGE_QUALITY = 95
    TIMEOUT = 120.0


def convert_page_to_image(pdf_path: str, page_number: int, dpi: int = NemotronConfig.DPI) -> Path:
    """
    Pdf to jpeg image - needed because nemotron parse is vision-based
    """
    images = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_number,
        last_page=page_number
    )

    temp_dir = tempfile.mkdtemp(prefix="aegis_")
    img_path = Path(temp_dir) / f"page_{page_number}.jpg"
    images[0].save(img_path, "JPEG", quality=NemotronConfig.IMAGE_QUALITY)

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
            "content": f'<img src="data:image/jpeg;base64,{b64_image}" />'
        }], # image is wrapped in markdown with an <img> tag to trigger Nemotron's parsing, acting as a "user message" in the chat
        "tools": [{
            "type": "function",
            "function": {"name": "markdown_bbox"}
        }], #markdown_bbox is a special tool in Nemotron that triggers the bounding box parsing behavior, it returns the text in a markdown format with [bbox: x1,y1,x2,y2] markers for EVERY word on the page 
        "max_tokens": 8192
    }

    async with httpx.AsyncClient(timeout=NemotronConfig.TIMEOUT) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    # Parse the response into structured elements
    content = result["choices"][0]["message"]["content"]
    return _parse_bbox_response(content)


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
        bbox_match = re.match(
            r'\[bbox:\s*([\d.]+),([\d.]+),([\d.]+),([\d.]+)\]\s*(.*)',
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


def find_entity_bbox(entity_value: str, elements: List[Dict]) -> Optional[List[float]]:
    """
    Search parsed page elements for a specific PII value and return its bounding box.
    Nemotron as a model returns bbox coordinates for EVERY word on the page, in this we filter for PII

    First tries an exact match, then falls back to a case-insensitive search if needed.

    Args:
        entity_value: The PII text to find (e.g., "123-45-6789")
        elements: List of elements from parse_page()

    Returns:
        Bounding box [x1, y1, x2, y2] or None if not found
    """
    # Exact match first
    for element in elements:
        if entity_value in element.get("text", ""):
            return element.get("bbox")

    # Case-insensitive fallback
    value_lower = entity_value.lower()
    for element in elements:
        if value_lower in element.get("text", "").lower():
            return element.get("bbox")

    return None


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

    # Match each entity to its bounding box
    for entity in entities:
        if entity.bbox is not None:
            continue  # Alr has coordinates

        elements = page_elements.get(entity.page, [])
        bbox = find_entity_bbox(entity.value, elements)

        if bbox:
            entity.bbox = bbox
        else:
            print(f"  Warning: Could not find bbox for '{entity.value}' on page {entity.page}")

    return entities
