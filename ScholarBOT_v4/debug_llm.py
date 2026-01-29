import os
import sys

# Mock environment
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Add current dir
sys.path.append(os.getcwd())

from rag_pipeline_aligned import _generate_with_prompt

print("Testing LLM Generation...")
try:
    response = _generate_with_prompt("What is 2+2?", max_new_tokens=50)
    print(f"Response: {ascii(response)}")
except Exception as e:
    print(f"Error: {e}")
