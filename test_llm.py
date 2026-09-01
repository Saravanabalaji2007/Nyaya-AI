import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traceback
try:
    from llm_engine import chat_with_llm
    print("Testing LLM...")
    response = chat_with_llm("What is the capital of France?")
    print(f"Response: {response}")
except Exception:
    traceback.print_exc()
