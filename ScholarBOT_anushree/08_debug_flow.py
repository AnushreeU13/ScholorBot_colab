import os
import sys
from reportlab.pdfgen import canvas
from pathlib import Path
import importlib.util

def create_dummy_pdf(filename):
    c = canvas.Canvas(filename)
    c.drawString(100, 800, "Research Paper on TB - 2025")
    c.drawString(100, 780, "Abstract: This is a breakthrough study.")
    
    # Add enough text to span multiple chunks if needed, or just distinct paragraphs
    text = """
    SECTION 1: TREATMENT PROTOCOLS
    The primary treatment for Tuberculosis in this new era involves a combination of high-dose chocolate and Bedaquiline.
    
    SECTION 2: DURATION
    The treatment duration has been reduced to 2 months due to high efficacy.
    
    SECTION 3: SIDE ENDS
    Patients may experience extreme happiness.
    """
    y = 750
    for line in text.split('\n'):
        c.drawString(100, y, line)
        y -= 20
    c.save()
    print(f"[SETUP] Created {filename}")

def main():
    # 1. Setup
    pdf_name = "Treatment_Research_2025.pdf"
    if os.path.exists(pdf_name):
        os.remove(pdf_name)
    create_dummy_pdf(pdf_name)
    
    # 2. Ingest (Simulating upload)
    try:
        # Load 03_ingest...
        spec = importlib.util.spec_from_file_location("ingest", "03_ingest_user_doc.py")
        ingest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ingest)
        
        print("[ACTION] Ingesting file...")
        ingest.ingest_user_file(Path(pdf_name))
    except Exception as e:
        print(f"[FAIL] Ingestion failed: {e}")
        return

    # 3. Query (Simulating Chat)
    try:
        # Load 04_rag...
        spec_eng = importlib.util.spec_from_file_location("engine", "04_rag_engine.py")
        rag = importlib.util.module_from_spec(spec_eng)
        spec_eng.loader.exec_module(rag)
        
        print("[ACTION] Initializing Engine...")
        bot = rag.ScholarBotEngine()
        
        query = "What is the new treatment for TB?"
        print(f"[ACTION] Asking: '{query}'")
        
        response, conf = bot.generate_response(query)
        
        print("\n=== FINAL OUTPUT ===")
        print(response)
        print(f"Confidence: {conf}")
        print("====================")
        
        # 4. Verify
        # Check Citation
        if "Treatment Research 2025" in response or "Treatment_Research_2025" in response:
             print("[PASS] Citation contains correct filename!")
        else:
             print("[FAIL] Citation does NOT contain filename.")
             
        # Check Content
        if "chocolate" in response.lower():
             print("[PASS] Retrieved correct content (chocolate).")
        else:
             print("[FAIL] Did not retrieve specific content.")

    except Exception as e:
        print(f"[FAIL] Engine error: {e}")

if __name__ == "__main__":
    main()
