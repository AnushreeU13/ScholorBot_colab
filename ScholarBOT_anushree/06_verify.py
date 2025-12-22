"""
06_verify.py

Automated Verification / Smoke Test for ScholarBot.
1. Creates a dummy PDF.
2. Ingests it as a User Doc.
3. Inits Engine.
4. Asks a question that should match the User Doc.
5. Asserts the answer source is 'user_doc'.
"""

import sys
import os
import importlib.util
from pathlib import Path
from reportlab.pdfgen import canvas  # Requires reportlab usually, or we can mock ingestion.

# Check dependencies
try:
    import langchain
    import faiss
    import sentence_transformers
except ImportError as e:
    print(f"[FAIL] Missing dependency: {e}")
    sys.exit(1)

# Import our modules
# Helper to load module by path
def load_module(name, path_str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / path_str)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    ingest_user = load_module("ingest", "03_ingest_user_doc.py")
    rag = load_module("rag", "04_rag_engine.py")
except Exception as e:
    print(f"[FAIL] Could not load modules: {e}")
    sys.exit(1)

def create_dummy_pdf(filename: str, content: str):
    """Creates a minimal valid PDF with reportlab if available, else empty file."""
    try:
        c = canvas.Canvas(filename)
        c.drawString(100, 750, content)
        c.save()
        return True
    except ImportError:
        print("[WARN] reportlab not installed. Cannot create real PDF. Using empty file.")
        
        # Create a text file and rename it pdf? PyPDFLoader will fail.
        # We'll skip PDF creation and mock the chunking if reportlab missing.
        # For this test, we assume environment is set up.
        return False

def main():
    print("=== Verification Step ===")
    
    test_pdf = "test_data_chocolate.pdf"
    test_sentence = "The cure for Tuberculosis is Dark Chocolate."
    
    # 1. Create Dummy
    print(f"[1] Creating dummy PDF: {test_pdf}")
    if not create_dummy_pdf(test_pdf, test_sentence):
        print("[SKIP] Skipping integration test (no reportlab).")
        return

    # 2. Ingest
    print("[2] Ingesting into User KB...")
    try:
        ingest_user.ingest_user_file(Path(test_pdf))
    except Exception as e:
        print(f"[FAIL] Ingestion failed: {e}")
        return

    # 3. Init Engine
    print("[3] Initializing Engine...")
    try:
        bot = rag.ScholarBotEngine()
    except Exception as e:
        print(f"[FAIL] Engine init failed: {e}")
        return
        
    # 4. Ask
    query = "What is the cure for Tuberculosis?"
    print(f"[4] Asking: '{query}'")
    
    retrieval = bot.search(query)
    
    # 5. Assert
    print(f"[DEBUG] Source: {retrieval.get('answer_source')}")
    print(f"[DEBUG] Context: {retrieval.get('context')}")
    
    if retrieval.get("answer_source") == "user_doc":
        print("[PASS] Successfully prioritized User Doc!")
    else:
        print("[FAIL] Did not retrieve from User Doc (or score too low).")
    
    if "Chocolate" in retrieval.get("context", ""):
         print("[PASS] Content verified.")
    else:
         print("[FAIL] Content mismatch.")

    # Cleanup
    try:
        os.remove(test_pdf)
    except:
        pass

if __name__ == "__main__":
    main()
