import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import re
from dotenv import load_dotenv
import PyPDF2 as pypdf2


# Load environment variables from .env file
load_dotenv()

def get_model():
    """
    Intializes the model from NVIDIA Cloud. - nvidia/llama-3.3-nemotron-super-49b-v1.5
    """

    API_KEY = os.getenv("NVIDIA_API_KEY")
    if not API_KEY:
        raise ValueError("NVIDIA_API_KEY not found. Please create a .env file and add your key.")

    return ChatNVIDIA(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5", 
        temperature=0.1, # deterministic output
        max_tokens=512, # keep output minimal
        api_key=API_KEY
    )


def output_cleaner(response_text):
    """
    Cleans the model's response by removing any <think>...</think> blocks.
    """
    clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    return clean_text.strip()

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.
    Returns a list of dicts: [{"page_number": 1, "text": "..."}, ...]
    """
    try:
        pdf_reader = pypdf2.PdfReader(pdf_file)
        pages = []
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                pages.append({"page_number": i + 1, "text": page_text})
        return pages
    except Exception as e:
        raise ValueError(f"Could not read PDF, error: {e}")

def chunk_text(pages, chunk_size=3000):
    """
    Chunks structured page data into smaller segments, preserving page metadata.
    Accepts the list of dicts from extract_text_from_pdf().
    Returns: [{"page_number": 1, "chunk_index": 0, "text": "..."}, ...]
    """
    chunks = []
    for page in pages:
        page_num = page["page_number"]
        text = page["text"]
        if len(text) <= chunk_size:
            chunks.append({"page_number": page_num, "chunk_index": 0, "text": text})
        else:
            idx = 0
            for start in range(0, len(text), chunk_size):
                chunks.append({
                    "page_number": page_num,
                    "chunk_index": idx,
                    "text": text[start:start + chunk_size],
                })
                idx += 1
    return chunks