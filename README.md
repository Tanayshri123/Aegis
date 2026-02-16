# Aegis — AI-Powered PDF PII Redaction

Aegis automatically detects and permanently removes personally identifiable information (PII) from PDF documents. It combines regex pattern matching, LLM-based entity extraction, and vision-based bounding box detection to find sensitive data — then applies **true redaction** where the underlying text is physically deleted from the PDF, not just visually covered.

Aegis offers two complementary modes: **automatic redaction** that catches all standard PII categories on upload, and an **interactive chat interface** where users can ask questions about the document and request additional redactions through natural conversation.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Project Structure](#project-structure)

---

## Features

- **True Redaction** — Text is permanently removed from the PDF content stream using PyMuPDF. Copy-paste cannot recover redacted content.
- **Multi-Layer PII Detection** — Regex patterns catch structured PII (SSN, EIN, phone, email, credit card). An LLM scan catches unstructured PII (names, addresses, employer names).
- **Vision-Based Bounding Boxes** — NVIDIA Nemotron Parse renders each page as a 200 DPI image and returns pixel-level coordinates for every text element, enabling precise spatial redaction.
- **Deterministic Output** — `temperature=0`, `seed=42`, lossless PNG rendering, and spatial sorting ensure the same input always produces the same redacted output.
- **W-2 / Multi-Copy Handling** — Documents with repeated form copies (W-2s have 4+ copies per page) are handled correctly — every occurrence of each PII value is found and redacted.
- **Interactive Chat** — A RAG-powered chatbot lets users ask questions about their document and request additional redactions beyond the automatic pass (e.g., "also redact every mention of the project codename").
- **Undo / Redo** — Chat-driven redactions can be undone and redone, with full state snapshots maintained per operation.
- **Before/After Comparison** — The web UI renders original and redacted pages side-by-side for visual verification.
- **NeMo Guardrails** — Input and output safety rails keep the chat assistant on-topic and prevent hallucinated or unprofessional responses.

---

## Architecture

```
Upload PDF
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│  Step 1: Text Extraction (pdf_processor.py)              │
│  PyPDF2 extracts text page-by-page                       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Step 2: PII Detection (pii_detector.py)                 │
│  Regex scan (SSN, EIN, phone, email, credit card, etc.)  │
│  + LLM scan (names, addresses, employer names)           │
│  → List of Entity objects with bbox=None                 │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Step 3: Bounding Box Location (nemotron_parser.py)      │
│  Renders each page as 200 DPI PNG                        │
│  → Sends to Nemotron Parse vision model                  │
│  → Returns text elements with [x0, y0, x1, y1] coords   │
│  → 4-tier fuzzy matching maps PII values to bboxes       │
│  → Clones entities for multi-copy documents (W-2s)       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Step 4: Redaction (redactor.py)                         │
│  Converts pixel coords → PDF points (scale = 72/200)    │
│  → add_redact_annot() + apply_redactions() via PyMuPDF   │
│  → Saves to uploads/redacted/<name>_redacted.pdf         │
└──────────────────────────────────────────────────────────┘
```

After automatic redaction completes, the document is vectorized into a ChromaDB knowledge base, enabling the RAG-powered chat interface for Q&A and additional chat-driven redactions.

---

## Tech Stack

### AI / ML

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Primary LLM** | [NVIDIA Llama 3.3 Nemotron Super 49B v1.5](https://build.nvidia.com/) | PII entity extraction and conversational chat via NVIDIA AI Endpoints |
| **Document Parser** | [NVIDIA Nemotron Parse](https://build.nvidia.com/) | Vision model that returns text elements with pixel-level bounding box coordinates |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local) or `nvidia/nv-embed-v1` (cloud) | Document chunk embeddings for RAG vector search |
| **Safety Rails** | [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) + [Colang](https://docs.nvidia.com/nemo/guardrails/) | Input/output content filtering — blocks off-topic, political, and illegal queries; prevents hallucinated responses |

### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | FastAPI + Uvicorn | REST API with async support |
| **PDF Reading** | PyPDF2 | Text extraction from uploaded PDFs |
| **PDF Redaction** | PyMuPDF (fitz) | True redaction — physically removes text from the PDF content stream |
| **Page Rendering** | pdf2image + Pillow | Converts PDF pages to PNG images for Nemotron Parse and UI display |
| **Vector Store** | ChromaDB | Local vector database for RAG document search |
| **LLM Integration** | LangChain + `langchain-nvidia-ai-endpoints` | LLM orchestration and NVIDIA model access |
| **HTTP Client** | httpx (async) | Async API calls to Nemotron Parse |
| **Language** | Python 3.9+ | |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | React 18 | Single-page application |
| **Build Tool** | Vite 7 | Dev server with API proxy + production builds |
| **Styling** | Custom CSS (Manrope + Sora fonts) | Clean, modern interface |

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **CI/CD** | GitHub Actions | Flake8 linting → unit tests on every push/PR to `main` |
| **Nemotron Parse Hosting** | Docker (local) or NVIDIA Cloud | Vision model can run locally at `localhost:8000` or via NVIDIA's hosted endpoint |

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+ (for the frontend)
- [Poppler](https://poppler.freedesktop.org/) (required by `pdf2image` for PDF-to-PNG conversion)
- An [NVIDIA API Key](https://build.nvidia.com/) for the LLM and Nemotron Parse cloud endpoints

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Aegis.git
cd Aegis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### Environment Variables

Create a `.env` file in the project root:

```
NVIDIA_API_KEY=nvapi-...
```

This key is required for the LLM (entity extraction + chat) and Nemotron Parse (bounding box detection) when using cloud mode.

---

## Usage

### Web Application (Recommended)

Start the backend and frontend:

```bash
# Terminal 1 — Backend API server
uvicorn server:app --reload --port 8000

# Terminal 2 — Frontend dev server
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser. Upload a PDF, watch the 4-step pipeline process it, then use the chat interface for Q&A or additional redactions.

### CLI

```bash
# Redact a single document (uses Docker Nemotron Parse by default)
python main.py document.pdf

# Redact multiple documents using NVIDIA cloud endpoints
python main.py doc1.pdf doc2.pdf --cloud
```

Output is saved to `uploads/redacted/<name>_redacted.pdf`.

---

## API Reference

All endpoints are under `/api/v1/`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/redact` | Upload a PDF (multipart form). Returns `{job_id}`. Pipeline runs in background. |
| `GET` | `/api/v1/jobs/{job_id}/progress` | Poll pipeline status. Returns step (1-4), detected entities, page count, and status (`processing`, `done`, `error`). |
| `GET` | `/api/v1/jobs/{job_id}/download` | Download the redacted PDF. |
| `GET` | `/api/v1/jobs/{job_id}/pages/{page_num}?type=original\|redacted` | Get a PNG rendering of a specific page. |
| `POST` | `/api/v1/jobs/{job_id}/chat` | Send a message to the chat interface. Supports Q&A, redaction requests, and audit queries. |
| `POST` | `/api/v1/jobs/{job_id}/undo` | Undo the most recent chat-driven redaction. |
| `POST` | `/api/v1/jobs/{job_id}/redo` | Redo a previously undone redaction. |

### Chat Endpoint Details

The chat endpoint handles multiple interaction types:

- **Q&A** — Ask questions about the document content (powered by RAG)
- **Redaction requests** — "Redact all mentions of Acme Corp" triggers entity search + bbox location
- **Category redactions** — "Redact all SSNs" maps directly to known entity types without an LLM call
- **Confirmation flow** — Redaction requests require explicit confirmation before being applied
- **Audit queries** — "What has been redacted?" returns a summary of all applied redactions

---

## Testing

```bash
# Unit tests (mocked, no API calls required)
python -m unittest discover -s tests/unittests -p "test_*.py"

# Integration tests (creates real PDFs with PyMuPDF, verifies true text removal)
python -m unittest discover -s tests/integration -p "test_*.py"
```

Unit tests follow the AAA pattern (Arrange-Act-Assert). Integration tests verify that redacted text is physically removed from the PDF — not just visually hidden.

**Test coverage includes:**
- PDF text extraction and chunking
- Entity/Document/DocumentStore state management
- Regex and LLM PII detection with deduplication
- Nemotron Parse bbox parsing, fuzzy matching, and multi-span matching
- Redaction coordinate conversion and output path logic

---

## CI/CD

GitHub Actions runs on every push to `main` and on pull requests:

1. **Quality Gate** — `flake8` checks for syntax errors, undefined names, and invalid f-strings
2. **Unit Tests** — Runs the full unit test suite (only if the quality gate passes)

---

## Project Structure

```
Aegis/
├── server.py                  # FastAPI server — all REST endpoints
├── main.py                    # CLI entry point — 4-step async pipeline
├── llm-utils.py               # LLM initialization + output cleaning
├── core/
│   ├── pdf_processor.py       # PyPDF2 text extraction + chunking
│   ├── pii_detector.py        # Regex + LLM PII detection
│   ├── nemotron_parser.py     # Nemotron Parse vision API + bbox matching
│   ├── redactor.py            # PyMuPDF true redaction
│   ├── document_store.py      # In-memory state (Entity, Document, DocumentStore)
│   ├── rag_engine.py          # ChromaDB + embeddings for document Q&A
│   └── guardrails.py          # NeMo Guardrails wrapper (input/output safety)
├── config/
│   ├── config.yml             # NeMo Guardrails model + rails configuration
│   ├── prompts.yml            # LLM prompt templates
│   └── rails.co               # Colang dialog flow rules
├── frontend/
│   ├── App.jsx                # React SPA (upload, progress, chat, compare)
│   ├── App.css                # Styling
│   ├── index.html             # HTML shell
│   ├── vite.config.js         # Vite config with API proxy
│   └── package.json           # React + Vite dependencies
├── tests/
│   ├── unittests/             # Mocked unit tests (no API calls)
│   └── integration/           # End-to-end tests with real PDFs
├── .github/workflows/ci.yml  # GitHub Actions CI pipeline
└── requirements.txt           # Python dependencies
```

---

## How It Works

### Automatic Redaction

On upload, Aegis runs a 4-step pipeline:

1. **Text Extraction** — PyPDF2 reads every page of the PDF and extracts raw text.
2. **PII Detection** — Two complementary methods run in sequence:
   - *Regex patterns* catch structured PII: SSNs (`XXX-XX-XXXX`), EINs, phone numbers, email addresses, credit card numbers, dates of birth, account numbers, state IDs, and control numbers.
   - *LLM scan* sends each page's text to Nemotron Super 49B with a structured prompt, catching unstructured PII: personal names, street addresses, and employer names.
3. **Bounding Box Location** — For each detected entity, Nemotron Parse (a vision model) renders the page as a 200 DPI PNG and returns every text element with pixel coordinates. A 4-tier matching system (exact, normalized, fuzzy, multi-element span) maps each PII value to its bounding box(es). Multi-copy documents like W-2s get every occurrence matched.
4. **True Redaction** — PyMuPDF converts pixel coordinates to PDF points, applies redaction annotations, and writes a new PDF with the sensitive text physically removed.

### Chat-Driven Redaction

After the automatic pass, the document is chunked and vectorized into ChromaDB. Users can then:

- Ask questions about the document ("What company issued this W-2?")
- Request additional redactions ("Also redact every mention of Acme Corp")
- Run category-based redactions ("Remove all phone numbers")
- Audit what was redacted ("Show me everything that was redacted on page 2")
- Undo/redo any chat-driven redaction

The chat endpoint routes requests through NeMo Guardrails for safety filtering, uses RAG for context retrieval, and falls back to PyMuPDF word-level fuzzy search when Nemotron Parse doesn't find a match.

### NeMo Guardrails

Aegis uses NVIDIA NeMo Guardrails with Colang dialog rules to keep the chat assistant safe and on-topic:

- **Input rails** block off-topic queries, political questions, illegal requests, and bypass attempts
- **Output rails** prevent hallucinated content, unprofessional responses, and information outside the document scope
- **Colang flows** define structured dialog patterns for greetings, document analysis, entity inquiries, redaction requests, and process explanations
- **Fail-open design** — if guardrails fail to load, the chat still works (safety degrades gracefully rather than blocking functionality)

---

## License

This project is currently unlicensed. Contact the maintainers for usage terms.
