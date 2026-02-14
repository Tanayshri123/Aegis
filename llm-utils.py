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
        temperature=0,    # fully deterministic: same input always produces same output
        max_tokens=4096,  # reasoning models emit <think> tokens before JSON; 512 was
                          # too low and silently truncated the JSON on complex documents
        api_key=API_KEY
    )


def output_cleaner(response_text):
    """
    Cleans the model's response by removing any <think>...</think> blocks.
    """
    clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    return clean_text.strip()
