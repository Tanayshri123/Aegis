"""
Aegis FastAPI Server - Bridges the frontend to the PDF redaction pipeline.

Endpoints:
    POST /api/v1/redact              Upload PDF, kick off pipeline in background
    GET  /api/v1/jobs/{id}/progress   Poll pipeline progress (step 1-4, done, error)
    GET  /api/v1/jobs/{id}/download   Download the redacted PDF
    GET  /api/v1/jobs/{id}/pages/{n}  Get a page image (original or redacted)
    POST /api/v1/jobs/{id}/chat       Chat about document / request additional redactions

Run:
    uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import os
import re
import shutil
import threading
import traceback
import uuid
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from main import process_pdf

app = FastAPI(title="Aegis", version="1.0.0")

# CORS for Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (maps job_id -> job metadata)
_jobs: dict = {}

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
(UPLOAD_DIR / "redacted").mkdir(exist_ok=True)

# Pipeline step labels for the frontend
STEP_LABELS = {
    1: "Extracting text from PDF...",
    2: "Detecting PII (regex + LLM)...",
    3: "Locating bounding boxes via Nemotron...",
    4: "Applying redactions...",
}

# ── LLM cache (thread-safe lazy init) ──────────────────────────────────────
_cached_llm = None
_llm_lock = threading.Lock()


def _get_chat_llm():
    """Get or create the cached LLM instance for chat (temperature=0.1)."""
    global _cached_llm
    if _cached_llm is None:
        with _llm_lock:
            if _cached_llm is None:
                llm_utils = import_module("llm-utils")
                _cached_llm = llm_utils.get_chat_model()
    return _cached_llm


# ── Chat prompt template ───────────────────────────────────────────────────
_PROMPTS_PATH = Path(__file__).parent / "config" / "prompts.yml"
_chat_prompt_template: Optional[str] = None


def _get_chat_prompt_template() -> str:
    """Load and cache the chat_with_document prompt from config/prompts.yml."""
    global _chat_prompt_template
    if _chat_prompt_template is None:
        with open(_PROMPTS_PATH, "r") as f:
            prompts_config = yaml.safe_load(f)
        for prompt in prompts_config.get("prompts", []):
            if prompt.get("task") == "chat_with_document":
                _chat_prompt_template = prompt["content"]
                break
        if _chat_prompt_template is None:
            raise RuntimeError("chat_with_document prompt not found in config/prompts.yml")
    return _chat_prompt_template


# ── Broad vs. focused request detection ───────────────────────────────────

# Signals that the user wants a sweeping/comprehensive action across the
# entire document, not a focused lookup.  When detected, the chat endpoint
# feeds the FULL document text to the LLM instead of the top-5 RAG chunks
# so nothing gets missed.
_BROAD_REQUEST_SIGNALS = [
    "all ", "every ", "any ", "each ",
    "all mentions", "every mention", "any mention",
    "all references", "every reference", "any reference",
    "all occurrences", "every occurrence", "any occurrence",
    "everything", "everywhere", "across the document",
    "throughout", "whole document", "entire document",
]


def _is_broad_request(message: str) -> bool:
    """Return True if the message implies a document-wide sweep."""
    msg_lower = message.lower()
    return any(signal in msg_lower for signal in _BROAD_REQUEST_SIGNALS)


# ── Category-based redaction detection ────────────────────────────────────

# Maps natural-language PII category names to the entity type strings used
# internally by pii_detector.  When a user says "redact all SSNs", we can
# instantly filter the already-detected entity list by type — no LLM needed.
_CATEGORY_ALIASES: Dict[str, List[str]] = {
    "SSN": [
        "ssn", "ssns", "social security", "social security number",
        "social security numbers",
    ],
    "EIN": [
        "ein", "eins", "employer identification", "employer id",
        "employer ids", "federal ein", "federal id", "tax id", "tax ids",
    ],
    "EMAIL": [
        "email", "emails", "email address", "email addresses",
        "e-mail", "e-mails",
    ],
    "PHONE": [
        "phone", "phones", "phone number", "phone numbers",
        "telephone", "telephone number", "cell number",
    ],
    "CREDIT_CARD": [
        "credit card", "credit cards", "card number", "card numbers",
        "cc number",
    ],
    "NAME": [
        "name", "names", "person name", "person names",
        "employee name", "employee names", "personal name",
        "full name", "full names",
    ],
    "EMPLOYER_NAME": [
        "employer name", "employer names", "company name", "company names",
        "employer", "employers", "company", "companies",
        "business name", "organization", "organizations",
    ],
    "ADDRESS": [
        "address", "addresses", "physical address", "mailing address",
        "street address", "employer address", "employer addresses",
        "employee address", "employee addresses", "home address",
    ],
    "DATE_OF_BIRTH": [
        "date of birth", "dob", "birthday", "birthdate", "birth date",
    ],
    "ACCOUNT_NUMBER": [
        "account number", "account numbers", "bank account",
        "routing number", "account", "accounts",
    ],
    "STATE_ID": [
        "state id", "state ids", "state identification",
        "state employer id", "state employer ids",
    ],
    "CONTROL_NUMBER": [
        "control number", "control numbers", "reference number",
    ],
}

# Flatten to a lookup: alias_lower -> [entity_type, ...]
_ALIAS_TO_TYPES: Dict[str, List[str]] = {}
for _etype, _aliases in _CATEGORY_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_TYPES.setdefault(_alias, []).append(_etype)


def _detect_category_redaction(message: str, entities: list) -> Optional[dict]:
    """
    Detect requests like "redact all SSNs" or "remove every phone number".

    If the user is asking to redact an entire PII category, we resolve it
    instantly from the already-detected entity list — no LLM round-trip.

    Returns {"terms": [...], "pages": [...], "reply": "..."} or None.
    """
    msg_lower = message.lower()

    # Must contain a redaction verb
    redaction_verbs = [
        "redact", "remove", "hide", "delete", "mask", "censor", "black out",
        "block out", "take out", "get rid of", "cover", "erase", "wipe", "scrub",
    ]
    if not any(verb in msg_lower for verb in redaction_verbs):
        return None

    # Find which categories the user is referring to — match longest alias
    # first to avoid "employer" shadowing "employer address".
    matched_types = set()
    sorted_aliases = sorted(_ALIAS_TO_TYPES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if alias in msg_lower:
            matched_types.update(_ALIAS_TO_TYPES[alias])

    if not matched_types:
        return None

    # Filter the already-detected entities by those types
    matching = [e for e in entities if e["type"] in matched_types]

    if not matching:
        type_labels = ", ".join(sorted(matched_types))
        return {
            "terms": [],
            "pages": [],
            "reply": (
                f"No {type_labels} entities were detected in this document during "
                f"the initial scan. If you believe some were missed, try describing "
                f"the specific value you see (e.g. 'redact 555-1234')."
            ),
        }

    terms = sorted(set(e["value"] for e in matching))
    pages = sorted(set(e["page"] for e in matching))
    type_labels = ", ".join(sorted(matched_types))
    count = len(matching)

    reply = (
        f"Found {count} {type_labels} entit{'y' if count == 1 else 'ies'} "
        f"across page(s) {', '.join(str(p) for p in pages)}: "
        f"{', '.join(repr(t) for t in terms[:10])}"
        f"{'...' if len(terms) > 10 else ''}. "
        f"Click Confirm below to redact."
    )

    return {"terms": terms, "pages": pages, "reply": reply}


# ── Page-scoped request detection ─────────────────────────────────────────


def _detect_page_scope(message: str) -> Optional[List[int]]:
    """
    Detect when the user is asking about or targeting specific pages.

    Handles:
      - "what's on page 3"
      - "redact everything on page 1 and 2"
      - "pages 1, 2, and 3"
      - "pages 1-3", "pages 1 through 5"
      - "on page 1"

    Returns list of 1-indexed page numbers, or None if no page scope detected.
    """
    msg_lower = message.lower()

    # Only trigger if the message mentions "page" at all
    if "page" not in msg_lower:
        return None

    page_nums: List[int] = []

    # Step 1: expand ranges — "1-3", "1 through 3", "1 to 3"
    expanded = re.sub(
        r'(\d+)\s*(?:-|through|to)\s*(\d+)',
        lambda m: " ".join(str(n) for n in range(int(m.group(1)), int(m.group(2)) + 1)),
        msg_lower,
    )

    # Step 2: grab every number that follows "page" or "pages" anywhere
    # in the sentence (possibly separated by commas, "and", spaces)
    # Strategy: find "page(s)" then collect all nearby numbers
    for m in re.finditer(r'pages?\s+', expanded):
        # From the end of "page(s) ", consume the number-list tail
        tail = expanded[m.end():]
        # Collect digits from a run of "number + separator" tokens
        for token in re.split(r'[^0-9]+', tail):
            if token.isdigit():
                page_nums.append(int(token))
            elif token == "":
                continue
            else:
                break  # hit a non-number, non-separator token

    return sorted(set(page_nums)) if page_nums else None


# ── Remaining / audit request detection ───────────────────────────────────

_AUDIT_SIGNALS = [
    "what's left", "whats left", "what is left",
    "what's still", "whats still", "what is still",
    "what's remaining", "whats remaining", "what remains",
    "still visible", "still showing", "still there",
    "what did you miss", "what was missed", "anything missed",
    "what hasn't been", "what hasnt been", "what has not been",
    "still unredacted", "not redacted", "not yet redacted",
    "remaining pii", "remaining sensitive", "leftover",
    "did you get everything", "did you catch everything",
    "how thorough", "how complete", "coverage",
    "redaction summary", "what was redacted",
    "what did you redact", "show me what", "list what",
]


def _is_audit_request(message: str) -> bool:
    """Return True if the user is asking about redaction coverage/completeness."""
    msg_lower = message.lower()
    return any(signal in msg_lower for signal in _AUDIT_SIGNALS)


def _build_audit_reply(job: dict) -> dict:
    """
    Build a summary of what has been redacted and what might remain.

    Looks at the entity list and groups by type/page so the user can
    quickly assess coverage.
    """
    entities = job.get("entities", [])
    pages_data = job.get("pages", [])

    if not entities:
        return {
            "reply": (
                "No entities have been redacted yet. The initial scan didn't "
                "find any PII, or redaction hasn't been applied. You can ask "
                "me to look for specific items (e.g. 'are there any phone "
                "numbers on page 2?')."
            ),
            "pages_referenced": [],
            "redaction_request": None,
            "pdf_updated": False,
        }

    # Group by type
    by_type: Dict[str, list] = {}
    for e in entities:
        by_type.setdefault(e["type"], []).append(e)

    lines = ["**Redaction summary:**\n"]
    all_pages = set()
    for etype, ents in sorted(by_type.items()):
        values = sorted(set(e["value"] for e in ents))
        pages = sorted(set(e["page"] for e in ents))
        all_pages.update(pages)
        preview = ", ".join(repr(v) for v in values[:5])
        if len(values) > 5:
            preview += f"... (+{len(values) - 5} more)"
        lines.append(f"- **{etype}** ({len(ents)} occurrence(s)): {preview}")

    total_pages = len(pages_data)
    covered_pages = sorted(all_pages)
    uncovered = sorted(set(range(1, total_pages + 1)) - all_pages)

    lines.append(f"\n**Pages with redactions:** {', '.join(str(p) for p in covered_pages)}")
    if uncovered:
        lines.append(
            f"**Pages with no redactions:** {', '.join(str(p) for p in uncovered)} "
            f"— these may have no PII, or the initial scan may have missed something. "
            f"Ask me to check a specific page if you're unsure."
        )
    else:
        lines.append("**All pages** have at least one redaction applied.")

    lines.append(
        f"\nTotal: **{len(entities)} redaction(s)** across "
        f"**{len(covered_pages)}/{total_pages} page(s)**."
    )

    return {
        "reply": "\n".join(lines),
        "pages_referenced": covered_pages,
        "redaction_request": None,
        "pdf_updated": False,
    }


# ── Direct redaction request detection ────────────────────────────────────


def _detect_direct_redaction_request(message: str) -> Optional[list]:
    """
    Parse user messages that directly request redaction of specific terms.

    Handles patterns like:
      - "redact Mary"
      - "remove all mentions of Mary"
      - "also redact 'Acme Corp' and 'John Smith'"
      - "hide the name Mary Wilson"
      - "redact 'Project Alpha'"

    Returns list of terms to redact, or None if not a redaction request.
    """
    msg = message.strip()
    msg_lower = msg.lower()

    # Check if this is a redaction request at all
    redaction_verbs = [
        "redact", "remove", "hide", "delete", "mask", "censor", "black out",
        "block out", "take out", "get rid of",
        "cover up", "cover", "blank out", "obscure", "scrub", "erase", "wipe",
    ]
    if not any(verb in msg_lower for verb in redaction_verbs):
        return None

    terms = []

    # Extract quoted terms: "term" or 'term'
    quoted = re.findall(r"""['"]([^'"]+)['"]""", msg)
    terms.extend(quoted)

    # If no quoted terms, try to extract the target after common patterns
    if not terms:
        # Patterns like "redact Mary", "remove mentions of Mary Wilson",
        # "redact the name John", "hide all references to Acme Corp"
        patterns = [
            r'(?:redact|remove|hide|delete|mask|censor)\s+(?:all\s+)?(?:mentions?\s+(?:of\s+)?|references?\s+(?:to\s+)?|the\s+(?:name|term|word|phrase)\s+)?(.+)',
        ]
        for pattern in patterns:
            m = re.search(pattern, msg, re.IGNORECASE)
            if m:
                raw = m.group(1).strip().rstrip('.!?')
                # Split on " and " to support "redact Mary and John"
                parts = re.split(r'\s+and\s+', raw, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip().strip("'\"")
                    if part and len(part) > 1:
                        terms.append(part)
                break

    if not terms:
        return None

    # If the extracted terms look like natural language instructions rather
    # than actual document values, fall through to the LLM path which has
    # document context and can resolve descriptions like "the state income
    # tax" to actual values like "6612.65".
    instruction_signals = [
        "any", "every", "all", "specifically", "only", "especially",
        "its ", "their ", "the amount", "the value", "the number",
        "please", "i think", "i noticed", "i see", "i found",
        "on page", "from page", "on the", "from the",
        "figure", "should", "shouldn't", "needs to", "need to",
    ]
    for term in terms:
        term_lower = term.lower()
        # If the term contains instruction-like language, it's probably
        # a description, not a literal value to search for.
        if any(signal in term_lower for signal in instruction_signals):
            return None

    return terms


# ── Redaction intent parsing ───────────────────────────────────────────────


def _extract_redaction_intent(llm_response: str) -> Optional[dict]:
    """
    Extract redaction modification JSON from LLM response, if present.

    Looks for a ```json ... ``` code block containing {"add": [...]}.
    Returns the parsed dict or None if no redaction intent found.
    """
    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_response)
    if json_match:
        try:
            intent = json.loads(json_match.group(1))
            if "add" in intent and isinstance(intent["add"], list) and intent["add"]:
                return intent
        except json.JSONDecodeError:
            pass
    return None


# ── Chat request model ─────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    confirmed_terms: Optional[list] = None


def _render_page_to_png(pdf_path: str, page_num: int, dpi: int = 150) -> bytes:
    """Render a single PDF page to PNG bytes using PyMuPDF."""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


def _run_pipeline_thread(job_id: str, upload_path: str):
    """
    Run the redaction pipeline in a separate thread with its own event loop.

    process_pdf contains blocking calls (LLM loading, regex scanning) that would
    freeze the main server event loop if run via asyncio.create_task. Running in
    a dedicated thread keeps the server responsive for progress polling.
    """
    job = _jobs[job_id]

    def on_progress(step, detail):
        job["step"] = step
        job["step_detail"] = detail or STEP_LABELS.get(step, "")

    # Create a fresh event loop for this thread (process_pdf is async due to Nemotron)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        output_path, entities, pages = loop.run_until_complete(
            process_pdf(
                upload_path,
                use_docker=False,
                return_entities=True,
                progress_callback=on_progress,
            )
        )

        if output_path is None:
            job["status"] = "error"
            job["error"] = "No PII detected in the document. Nothing to redact."
            return

        # Count pages
        doc = fitz.open(output_path)
        page_count = len(doc)
        doc.close()

        original_doc = fitz.open(upload_path)
        original_page_count = len(original_doc)
        original_doc.close()

        # Build entity summary
        entity_summary = []
        for e in entities:
            if e.bbox is not None:
                entity_summary.append({
                    "type": e.type,
                    "value": e.value,
                    "page": e.page,
                    "confidence": round(e.confidence, 2),
                })

        job.update({
            "status": "done",
            "redacted_path": output_path,
            "page_count": page_count,
            "original_page_count": original_page_count,
            "entities": entity_summary,
            "entities_full": entities,
            "pages": pages,
            "chat_history": [],
            "chat_redacting": False,
            "pending_redaction": None,
            "redaction_history": [],  # stack of previous states for undo
        })

        # Build RAG knowledge base for chat (uses original extracted text)
        try:
            from core.rag_engine import create_rag_engine
            rag = create_rag_engine(job_id, prefer_local=True)
            rag.build_knowledge_base(pages)
            job["rag_engine"] = rag
            job["rag_ready"] = True
            print(f"  [chat] RAG knowledge base built for job {job_id}")
        except Exception as rag_err:
            print(f"  [chat] Warning: Failed to build RAG knowledge base: {rag_err}")
            job["rag_engine"] = None
            job["rag_ready"] = False

    except Exception as e:
        traceback.print_exc()
        job["status"] = "error"
        job["error"] = str(e)
    finally:
        loop.close()


@app.post("/api/v1/redact")
async def redact_endpoint(pdf_file: UploadFile = File(...)):
    """
    Upload a PDF and start the redaction pipeline.

    Returns immediately with a job_id. Poll /api/v1/jobs/{job_id}/progress
    to track which step the pipeline is on.
    """
    if not pdf_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    job_id = uuid.uuid4().hex[:12]

    # Save uploaded file to disk
    upload_path = str(UPLOAD_DIR / f"{job_id}_{pdf_file.filename}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)

    # Initialize job state
    _jobs[job_id] = {
        "status": "processing",
        "step": 0,
        "step_detail": "Uploading...",
        "original_path": upload_path,
        "original_name": pdf_file.filename,
        "redacted_path": None,
        "page_count": 0,
        "original_page_count": 0,
        "entities": [],
        "error": None,
    }

    # Run pipeline in a separate thread so blocking calls don't freeze the server
    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, upload_path),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}/progress")
async def get_progress(job_id: str):
    """
    Poll pipeline progress.

    Returns:
        status: "processing" | "done" | "error"
        step: current step number (1-4), 0 if not started
        step_detail: human-readable description of current step
        total_steps: always 4
        entities: populated when status is "done"
        page_count: populated when status is "done"
        error: populated when status is "error"
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "step": job["step"],
        "step_detail": job["step_detail"],
        "total_steps": 4,
        "original_name": job["original_name"],
        "page_count": job.get("page_count", 0),
        "entities": job.get("entities", []),
        "error": job.get("error"),
    }


@app.get("/api/v1/jobs/{job_id}/download")
async def download_redacted(job_id: str):
    """Download the redacted PDF."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    redacted_path = job.get("redacted_path")
    if not redacted_path or not os.path.exists(redacted_path):
        raise HTTPException(status_code=404, detail="Redacted file not found")

    return FileResponse(
        path=redacted_path,
        media_type="application/pdf",
        filename=f"redacted_{job['original_name']}",
    )


@app.get("/api/v1/jobs/{job_id}/pages/{page_num}")
async def get_page_image(job_id: str, page_num: int, type: str = "redacted"):
    """
    Get a PNG rendering of a specific page.

    Query params:
        type: "original" or "redacted" (default: "redacted")
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if type == "original":
        pdf_path = job["original_path"]
        total = job.get("original_page_count", 0)
    else:
        pdf_path = job.get("redacted_path")
        total = job.get("page_count", 0)

    if not pdf_path or not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")

    if page_num < 0 or page_num >= total:
        raise HTTPException(status_code=400, detail=f"Page {page_num} out of range (0-{total-1})")

    png_bytes = _render_page_to_png(pdf_path, page_num)
    return Response(content=png_bytes, media_type="image/png")


# ── Chat endpoint ──────────────────────────────────────────────────────────


@app.post("/api/v1/jobs/{job_id}/chat")
async def chat_endpoint(job_id: str, body: ChatRequest):
    """
    Chat about a processed document or request additional redactions.

    Request body:
        {"message": "What SSNs are in this doc?"}
        {"message": "confirm_redaction", "confirmed_terms": ["Acme Corp"]}

    Returns:
        reply: The assistant's response text
        pages_referenced: List of page numbers mentioned in context
        redaction_request: null or {"terms": [...], "pages": [...]} when confirmation needed
        pdf_updated: true if the PDF was re-redacted in this turn
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail="Document is still being processed. Please wait for redaction to complete.",
        )

    # ── Handle redaction confirmation ──────────────────────────────────
    if body.message == "confirm_redaction" and body.confirmed_terms:
        if job.get("chat_redacting"):
            return {
                "reply": "A redaction is currently in progress. Please wait.",
                "pages_referenced": [],
                "redaction_request": None,
                "pdf_updated": False,
            }

        result = await _execute_chat_redaction(job_id, body.confirmed_terms)
        return result

    # ── Audit / "what's left?" request ───────────────────────────────
    if _is_audit_request(body.message):
        result = _build_audit_reply(job)
        job["chat_history"].append({"role": "user", "content": body.message})
        job["chat_history"].append({"role": "assistant", "content": result["reply"]})
        return result

    # ── Category-based redaction ("redact all SSNs") ──────────────────
    # Resolves instantly from the already-detected entity list — no LLM.
    category_result = _detect_category_redaction(
        body.message, job.get("entities", [])
    )
    if category_result is not None:
        reply = category_result["reply"]
        terms = category_result["terms"]
        pages = category_result["pages"]
        job["chat_history"].append({"role": "user", "content": body.message})
        job["chat_history"].append({"role": "assistant", "content": reply})

        if terms:
            job["pending_redaction"] = {"terms": terms}
            return {
                "reply": reply,
                "pages_referenced": pages,
                "redaction_request": {"terms": terms, "pages": pages},
                "pdf_updated": False,
            }
        else:
            # No matching entities found for that category
            return {
                "reply": reply,
                "pages_referenced": [],
                "redaction_request": None,
                "pdf_updated": False,
            }

    # ── Direct redaction request (bypass LLM entirely) ──────────────
    direct_terms = _detect_direct_redaction_request(body.message)
    if direct_terms:
        # Verify terms exist in the document before asking for confirmation
        from core.pii_detector import find_custom_entities
        pages = job.get("pages", [])
        found = find_custom_entities(pages, direct_terms)

        if found:
            # Group found terms and pages for the confirmation UI
            found_terms = sorted(set(e.value for e in found))
            found_pages = sorted(set(e.page for e in found))
            count = len(found)

            reply = (
                f"Found {count} occurrence(s) of "
                f"{', '.join(repr(t) for t in found_terms)} "
                f"on page(s) {', '.join(str(p) for p in found_pages)}. "
                f"Click Confirm below to redact."
            )
            job["chat_history"].append({"role": "user", "content": body.message})
            job["chat_history"].append({"role": "assistant", "content": reply})
            job["pending_redaction"] = {"terms": direct_terms}

            return {
                "reply": reply,
                "pages_referenced": found_pages,
                "redaction_request": {"terms": direct_terms, "pages": found_pages},
                "pdf_updated": False,
            }
        else:
            reply = (
                f"I couldn't find any occurrences of "
                f"{', '.join(repr(t) for t in direct_terms)} in the document. "
                f"Check the exact spelling and try again."
            )
            job["chat_history"].append({"role": "user", "content": body.message})
            job["chat_history"].append({"role": "assistant", "content": reply})
            return {
                "reply": reply,
                "pages_referenced": [],
                "redaction_request": None,
                "pdf_updated": False,
            }

    # ── Check RAG readiness ────────────────────────────────────────────
    if not job.get("rag_ready"):
        return {
            "reply": (
                "Document search is not available for this document. "
                "I can still try to help — what would you like to know?"
            ),
            "pages_referenced": [],
            "redaction_request": None,
            "pdf_updated": False,
        }

    # ── Guardrails input check ─────────────────────────────────────────
    try:
        from core.guardrails import get_guardrails
        guardrails = get_guardrails()
        allowed, refusal = await guardrails.check_input(body.message)
        if not allowed:
            job["chat_history"].append({"role": "user", "content": body.message})
            job["chat_history"].append({"role": "assistant", "content": refusal})
            return {
                "reply": refusal,
                "pages_referenced": [],
                "redaction_request": None,
                "pdf_updated": False,
            }
    except Exception as guard_err:
        # Fail-open: if guardrails can't load, skip the check
        print(f"  [chat] Guardrails check skipped: {guard_err}")

    # ── Build chat prompt with RAG context ─────────────────────────────
    rag_engine = job["rag_engine"]
    llm = _get_chat_llm()

    # Determine context strategy:
    #   1. Page-scoped  — user targets specific pages ("on page 2")
    #   2. Broad        — sweeping request ("all", "every") → full document
    #   3. Focused      — normal question → RAG top-5
    page_scope = _detect_page_scope(body.message)
    broad = _is_broad_request(body.message)
    pages_data = job.get("pages", [])
    context_text = ""
    pages_referenced = []

    if page_scope:
        scoped = [p for p in pages_data if p["page_number"] in page_scope]
        if scoped:
            context_text = "\n\n".join(
                f"[Page {p['page_number']}]\n{p['text']}" for p in scoped
            )
            pages_referenced = sorted(p["page_number"] for p in scoped)
            print(f"  [chat] Page-scoped request — sending page(s) {page_scope}")
        else:
            # Requested pages don't exist; fall through to broad/RAG
            page_scope = None

    if not page_scope and broad:
        context_text = "\n\n".join(
            f"[Page {p['page_number']}]\n{p['text']}" for p in pages_data
        )
        pages_referenced = sorted(p["page_number"] for p in pages_data)
        print(f"  [chat] Broad request detected — sending full document ({len(pages_data)} pages)")

    if not page_scope and not broad:
        chunks = rag_engine.retrieve_context(body.message, top_k=5)
        context_text = rag_engine.format_context_for_llm(chunks, include_scores=False)
        pages_referenced = sorted(set(c["page_number"] for c in chunks))

    # Format entity list for the prompt
    entities_text = "\n".join(
        f"  - [{e['type']}] {e['value']} (page {e['page']})"
        for e in job.get("entities", [])
    )

    # Format conversation history (last 6 turns)
    history = job.get("chat_history", [])[-6:]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Aegis'}: {m['content']}"
        for m in history
    )

    # Build the prompt from template
    prompt_template = _get_chat_prompt_template()
    prompt = prompt_template.replace("{{ document_text }}", context_text)
    prompt = prompt.replace("{{ entities_list }}", entities_text or "No entities detected.")
    prompt = prompt.replace("{{ conversation_history }}", history_text or "No previous messages.")
    prompt = prompt.replace("{{ user_message }}", body.message)

    # ── Call LLM ───────────────────────────────────────────────────────
    try:
        response = llm.invoke(prompt)
        reply_text = response.content if hasattr(response, "content") else str(response)

        # Clean <think> blocks from reasoning models
        llm_utils = import_module("llm-utils")
        reply_text = llm_utils.output_cleaner(reply_text)
    except Exception as llm_err:
        reply_text = f"Sorry, I encountered an error generating a response. Please try again. ({llm_err})"

    # ── Guardrails output check ────────────────────────────────────────
    try:
        from core.guardrails import get_guardrails
        guardrails = get_guardrails()
        safe, sanitized = await guardrails.check_output(reply_text)
        if not safe:
            reply_text = sanitized
    except Exception:
        pass  # Fail-open

    # ── Parse redaction intent ─────────────────────────────────────────
    redaction_intent = _extract_redaction_intent(reply_text)
    redaction_request = None

    if redaction_intent:
        terms = redaction_intent["add"]
        job["pending_redaction"] = {"terms": terms}
        redaction_request = {"terms": terms, "pages": pages_referenced}

        # Strip the JSON block from the displayed reply
        reply_text = re.sub(r"```json\s*\{[\s\S]*?\}\s*```", "", reply_text).strip()

    # ── Update conversation history ────────────────────────────────────
    job["chat_history"].append({"role": "user", "content": body.message})
    job["chat_history"].append({"role": "assistant", "content": reply_text})

    return {
        "reply": reply_text,
        "pages_referenced": pages_referenced,
        "redaction_request": redaction_request,
        "pdf_updated": False,
    }


@app.post("/api/v1/jobs/{job_id}/undo")
async def undo_redaction(job_id: str):
    """
    Undo the most recent chat-driven redaction.

    Saves the current state to a redo stack, then restores the previous
    state from the undo (redaction_history) stack.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    history = job.get("redaction_history", [])
    if not history:
        raise HTTPException(status_code=400, detail="Nothing to undo")

    # Save current state to redo stack before reverting
    job.setdefault("redo_stack", []).append({
        "redacted_path": job["redacted_path"],
        "entities": list(job["entities"]),
        "page_count": job["page_count"],
    })

    prev = history.pop()
    job["redacted_path"] = prev["redacted_path"]
    job["entities"] = prev["entities"]
    job["page_count"] = prev["page_count"]

    return {
        "success": True,
        "entities": prev["entities"],
        "page_count": prev["page_count"],
    }


@app.post("/api/v1/jobs/{job_id}/redo")
async def redo_redaction(job_id: str):
    """
    Redo a previously undone redaction.

    Pops from the redo stack and pushes the current state back onto
    the undo (redaction_history) stack.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    redo_stack = job.get("redo_stack", [])
    if not redo_stack:
        raise HTTPException(status_code=400, detail="Nothing to redo")

    # Save current state back to undo stack
    job.setdefault("redaction_history", []).append({
        "redacted_path": job["redacted_path"],
        "entities": list(job["entities"]),
        "page_count": job["page_count"],
    })

    redo = redo_stack.pop()
    job["redacted_path"] = redo["redacted_path"]
    job["entities"] = redo["entities"]
    job["page_count"] = redo["page_count"]

    return {
        "success": True,
        "entities": redo["entities"],
        "page_count": redo["page_count"],
    }


def _normalize_for_match(text: str) -> str:
    """Strip all dashes, hyphens (ASCII + unicode), spaces, and punctuation for fuzzy comparison."""
    return re.sub(r'[-\s.,()\/\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]+', '', text).lower()


def _fitz_fuzzy_search(page, value: str) -> list:
    """
    Find all bounding boxes for `value` on a fitz page using word-level
    sliding-window matching.

    PyMuPDF's page.search_for() only works when the target string exists as
    a contiguous run in the PDF text layer. Many PDFs split values like
    "KS070-45-878" across multiple internal text spans, so search_for() fails.

    This function:
      1. Tries page.search_for() (fast path for contiguous text)
      2. Falls back to extracting every word with its bbox via get_text("words"),
         then slides a window across adjacent words, normalizing away dashes/
         spaces/punctuation, to find where the value lives — even if it's spread
         across 1-8 separate word boxes.

    Returns a list of fitz.Rect objects covering every occurrence found.
    """
    # Fast path: contiguous text match
    rects = page.search_for(value)
    if rects:
        return rects

    # Word-level fallback: get every word with its bbox
    # get_text("words") returns list of (x0, y0, x1, y1, "word", block, line, word_no)
    words = page.get_text("words")
    if not words:
        return []

    # Sort by reading order (top-to-bottom, left-to-right)
    words = sorted(words, key=lambda w: (w[1], w[0]))

    norm_value = _normalize_for_match(value)
    if not norm_value:
        return []

    found_rects = []

    # Sliding window: combine up to 10 adjacent words and check if the
    # normalized target appears in the normalized concatenation.
    for i in range(len(words)):
        combined = ""
        for j in range(i, min(i + 10, len(words))):
            combined += words[j][4]  # raw word text
            norm_combined = _normalize_for_match(combined)

            if norm_value in norm_combined:
                # Merge bboxes of words[i..j]
                merged = fitz.Rect(
                    min(words[k][0] for k in range(i, j + 1)),
                    min(words[k][1] for k in range(i, j + 1)),
                    max(words[k][2] for k in range(i, j + 1)),
                    max(words[k][3] for k in range(i, j + 1)),
                )
                # Avoid duplicate rects for overlapping windows
                is_dup = any(
                    abs(existing.x0 - merged.x0) < 1 and abs(existing.y0 - merged.y0) < 1
                    for existing in found_rects
                )
                if not is_dup:
                    found_rects.append(merged)
                break  # Move to next starting position

            # If the normalized combined text is already longer than the target,
            # no point extending the window further.
            if len(norm_combined) > len(norm_value) * 2:
                break

    return found_rects


async def _execute_chat_redaction(job_id: str, terms: list) -> dict:
    """
    Execute a chat-driven redaction: find custom entities, locate bboxes, redact.

    Runs the Nemotron bbox detection in a thread (it's async + blocking).
    """
    job = _jobs[job_id]
    job["chat_redacting"] = True

    # Save current state for undo; clear redo stack (new action invalidates it)
    job.setdefault("redaction_history", []).append({
        "redacted_path": job["redacted_path"],
        "entities": list(job["entities"]),
        "page_count": job["page_count"],
    })
    job["redo_stack"] = []

    try:
        from core.pii_detector import find_custom_entities
        from core.nemotron_parser import locate_entities
        from core.redactor import redact_pdf

        pages = job["pages"]

        # 1. Find all occurrences of the requested terms
        new_entities = find_custom_entities(pages, terms)

        if not new_entities:
            job["chat_redacting"] = False
            # No redaction applied — remove the undo snapshot
            if job.get("redaction_history"):
                job["redaction_history"].pop()
            reply = (
                f"I couldn't find any occurrences of {', '.join(repr(t) for t in terms)} "
                "in the document. Could you check the exact spelling?"
            )
            job["chat_history"].append({"role": "assistant", "content": reply})
            return {
                "reply": reply,
                "pages_referenced": [],
                "redaction_request": None,
                "pdf_updated": False,
            }

        # 2. Locate bounding boxes (uses ORIGINAL PDF for accurate coords)
        loop = asyncio.get_event_loop()
        new_entities = await locate_entities(
            job["original_path"], new_entities, use_docker=False
        )

        entities_with_bbox = [e for e in new_entities if e.bbox is not None]

        # 2b. Fallback: if Nemotron couldn't locate bboxes, use PyMuPDF's
        #     native text search on the PDF's text layer. This handles cases
        #     where Nemotron returns zero elements or tokenizes the value in
        #     a way the fuzzy matcher can't reassemble.
        used_fitz_fallback = False
        if not entities_with_bbox:
            print(f"  Nemotron bbox lookup failed for {terms!r}, trying PyMuPDF word-level fallback...")
            current_redacted = job["redacted_path"]
            doc = fitz.open(current_redacted)
            fallback_count = 0
            fallback_pages = set()

            for entity in new_entities:
                if entity.bbox is not None:
                    continue
                fitz_page_idx = entity.page - 1  # fitz is 0-indexed
                if fitz_page_idx < 0 or fitz_page_idx >= len(doc):
                    continue
                page = doc[fitz_page_idx]
                rects = _fitz_fuzzy_search(page, entity.value)

                if rects:
                    print(f"  Fallback: found {len(rects)} rect(s) for {entity.value!r} on page {entity.page}")
                    for rect in rects:
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                        fallback_count += 1
                    fallback_pages.add(entity.page)
                else:
                    print(f"  Fallback: no match for {entity.value!r} on page {entity.page}")

            if fallback_count > 0:
                # Apply redactions on all affected pages
                for page_num in fallback_pages:
                    doc[page_num - 1].apply_redactions()

                # Save to a new file (same collision-safe pattern as redact_pdf)
                from core.redactor import _build_output_path
                new_redacted_path = _build_output_path(current_redacted)
                doc.save(new_redacted_path, garbage=4, deflate=True)
                doc.close()

                print(f"  PyMuPDF fallback: redacted {fallback_count} rect(s) on page(s) {sorted(fallback_pages)}")
                used_fitz_fallback = True

                # Update job state
                job["redacted_path"] = new_redacted_path
                verify_doc = fitz.open(new_redacted_path)
                job["page_count"] = len(verify_doc)
                verify_doc.close()

                for entity in new_entities:
                    job["entities"].append({
                        "type": entity.type,
                        "value": entity.value,
                        "page": entity.page,
                        "confidence": round(entity.confidence, 2),
                    })

                affected_pages = sorted(fallback_pages)
                reply = (
                    f"Done! Redacted {fallback_count} occurrence(s) of "
                    f"{', '.join(repr(t) for t in terms)} on page(s) {', '.join(str(p) for p in affected_pages)}."
                )
                job["chat_history"].append({"role": "assistant", "content": reply})
                job["pending_redaction"] = None

                return {
                    "reply": reply,
                    "pages_referenced": affected_pages,
                    "redaction_request": None,
                    "pdf_updated": True,
                }
            else:
                doc.close()

        if not entities_with_bbox and not used_fitz_fallback:
            job["chat_redacting"] = False
            # No redaction applied — remove the undo snapshot
            if job.get("redaction_history"):
                job["redaction_history"].pop()
            reply = (
                f"Found text matching {', '.join(repr(t) for t in terms)} but couldn't "
                "determine exact positions for redaction. Try more specific terms."
            )
            job["chat_history"].append({"role": "assistant", "content": reply})
            return {
                "reply": reply,
                "pages_referenced": [],
                "redaction_request": None,
                "pdf_updated": False,
            }

        # 3. Apply redaction to the CURRENT redacted PDF (additive)
        current_redacted = job["redacted_path"]
        new_redacted_path = redact_pdf(current_redacted, entities_with_bbox)

        # 4. Update job state
        job["redacted_path"] = new_redacted_path

        doc = fitz.open(new_redacted_path)
        job["page_count"] = len(doc)
        doc.close()

        # Append new entities to the summary
        for e in entities_with_bbox:
            job["entities"].append({
                "type": e.type,
                "value": e.value,
                "page": e.page,
                "confidence": round(e.confidence, 2),
            })

        affected_pages = sorted(set(e.page for e in entities_with_bbox))
        reply = (
            f"Done! Redacted {len(entities_with_bbox)} occurrence(s) of "
            f"{', '.join(repr(t) for t in terms)} on page(s) {', '.join(str(p) for p in affected_pages)}."
        )
        job["chat_history"].append({"role": "assistant", "content": reply})
        job["pending_redaction"] = None

        return {
            "reply": reply,
            "pages_referenced": affected_pages,
            "redaction_request": None,
            "pdf_updated": True,
        }

    except Exception as e:
        traceback.print_exc()
        # Redaction failed — remove the undo snapshot
        if job.get("redaction_history"):
            job["redaction_history"].pop()
        reply = f"Redaction failed: {e}"
        job["chat_history"].append({"role": "assistant", "content": reply})
        return {
            "reply": reply,
            "pages_referenced": [],
            "redaction_request": None,
            "pdf_updated": False,
        }
    finally:
        job["chat_redacting"] = False
