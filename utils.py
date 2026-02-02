import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Securely fetch the API key.
if not os.getenv("NVIDIA_API_KEY"):
    raise ValueError("NVIDIA_API_KEY not found. Please create a .env file and add your key.")
    
print("1. Connecting to NVIDIA Cloud...")

try:

    def clean_response(response_text):
        """
        Removes the <think>...</think> block from the model's response 
        to get only the final answer.
        """
        # Regex to find everything between <think> and </think> (including newlines)
        clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        
        # Trim whitespace from the start/end
        return clean_text.strip()

    llm = ChatNVIDIA(model="nvidia/llama-3.3-nemotron-super-49b-v1.5")
    
    print("2. Sending request...")
    result = llm.invoke("respond saying 'Hello World!' and thats it")
    
    print(f"3. SUCCESS! Model replied: {clean_response(result.content)}")

except Exception as e:
    print(f"ERROR: {e}")