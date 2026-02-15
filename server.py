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
from typing import Optional

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

    return terms if terms else None


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

    # Retrieve relevant document chunks
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


async def _execute_chat_redaction(job_id: str, terms: list) -> dict:
    """
    Execute a chat-driven redaction: find custom entities, locate bboxes, redact.

    Runs the Nemotron bbox detection in a thread (it's async + blocking).
    """
    job = _jobs[job_id]
    job["chat_redacting"] = True

    try:
        from core.pii_detector import find_custom_entities
        from core.nemotron_parser import locate_entities
        from core.redactor import redact_pdf

        pages = job["pages"]

        # 1. Find all occurrences of the requested terms
        new_entities = find_custom_entities(pages, terms)

        if not new_entities:
            job["chat_redacting"] = False
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

        if not entities_with_bbox:
            job["chat_redacting"] = False
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
