"""
03_ingest_user_doc.py

Ingests a single user-uploaded document into the 'User' Knowledge Base.
This allows the bot to prioritize user facts.
Usage: python 03_ingest_user_doc.py <path_to_pdf>
"""

import sys
import argparse
import importlib.util
from pathlib import Path

# Load Utils & Config
utils_spec = importlib.util.spec_from_file_location("utils", Path(__file__).parent / "00_utils.py")
utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils)

CFG = utils.CFG

def ingest_user_file(pdf_path: Path):
    if not pdf_path.exists():
        print(f"[ERR] File not found: {pdf_path}")
        return

    print(f"[INFO] Ingesting User Doc: {pdf_path.name}")
    
    # 1. Chunk
    chunks = utils.load_and_chunk_pdf(pdf_path)
    if not chunks:
        print("[WARN] No text extracted.")
        return
        
    # Mark strictly as user upload
    for c in chunks:
        c.metadata["source_type"] = "user_upload"

    # 2. Embed
    embeddings = utils.get_embedding_model()
    
    # 3. Load existing or create new User Index
    # We want to APPEND to the user index if it exists, or create new.
    # FAISS.load_local ... .merge_from? Or just add_documents?
    
    if CFG.KB_USER_INDEX_DIR.exists():
        print("[INFO] Updating existing User Index...")
        vector_db = utils.load_vector_store(CFG.KB_USER_INDEX_DIR, embeddings)
        vector_db.add_documents(chunks)
    else:
        print("[INFO] Creating new User Index...")
        vector_db = utils.FAISS.from_documents(chunks, embeddings)
    
    # 4. Save
    utils.save_vector_store(vector_db, CFG.KB_USER_INDEX_DIR)
    print(f"[DONE] User document added to {CFG.KB_USER_INDEX_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest User Document")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()
    
    ingest_user_file(Path(args.pdf_path))
