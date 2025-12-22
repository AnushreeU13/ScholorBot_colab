"""
02_ingest_static.py

Ingests the 'Static' Knowledge Base (Guidelines, Drug Labels).
Reads from DATA_SOURCES in 01_config.py
Saves to KB_STATIC_INDEX_DIR
"""

import sys
import importlib.util
from pathlib import Path

# Load Utils & Config
utils_spec = importlib.util.spec_from_file_location("utils", Path(__file__).parent / "00_utils.py")
utils = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(utils)

# Load Config via Utils (or reuse logic)
CFG = utils.CFG

def main():
    print("=== ScholarBot: Static KB Ingestion ===")
    
    # 1. Gather all PDFs
    pdf_files = []
    for source_dir in CFG.DATA_SOURCES:
        if not source_dir.exists():
            print(f"[WARN] Source directory not found: {source_dir}")
            # Ensure we create it locally for testing if 'assume' mode
            # source_dir.mkdir(parents=True, exist_ok=True) 
            continue
            
        print(f"[INFO] Scanning {source_dir}...")
        pdf_files.extend(list(source_dir.glob("*.pdf")))
    
    if not pdf_files:
        print("[ERR] No PDFs found in data sources! Aborting.")
        # For demonstration, allowing clean exit.
        return

    print(f"[INFO] Found {len(pdf_files)} PDFs.")

    # 2. Process in Batches
    BATCH_SIZE = 5
    embeddings = utils.get_embedding_model()
    vector_db = None
    
    total_docs = len(pdf_files)
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch_files = pdf_files[i : i + BATCH_SIZE]
        print(f"\n[INFO] Processing Batch {i // BATCH_SIZE + 1} / {(total_docs + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch_files)} files)")
        
        batch_chunks = []
        for pdf in batch_files:
            print(f"  - Chunking: {pdf.name}")
            try:
                chunks = utils.load_and_chunk_pdf(pdf)
                batch_chunks.extend(chunks)
            except Exception as e:
                print(f"    [ERR] Failed to chunk {pdf.name}: {e}")

        if not batch_chunks:
            continue

        print(f"  -> Embedding {len(batch_chunks)} chunks from this batch...")
        
        if vector_db is None:
            vector_db = utils.FAISS.from_documents(batch_chunks, embeddings)
        else:
            vector_db.add_documents(batch_chunks)
            
        # Incremental Save
        utils.save_vector_store(vector_db, CFG.KB_STATIC_INDEX_DIR)
        print(f"  -> Saved progress to {CFG.KB_STATIC_INDEX_DIR}")

    print("=== Ingestion Complete ===")

if __name__ == "__main__":
    main()
