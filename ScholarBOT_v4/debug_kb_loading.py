import sys
import os
from pathlib import Path
from config import FAISS_INDICES_DIR, KB_GUIDELINES, KB_DRUGLABELS
from storage_utils import create_faiss_store

print(f"Current Working Directory: {os.getcwd()}")
print(f"FAISS_INDICES_DIR (from config): {FAISS_INDICES_DIR}")
print(f"Exists? {FAISS_INDICES_DIR.exists()}")

def check_kb(name):
    print(f"\n--- Checking KB: {name} ---")
    kb_path = FAISS_INDICES_DIR / name
    print(f"Target Path: {kb_path}")
    print(f"Exists? {kb_path.exists()}")
    
    index_file = kb_path / "index.faiss"
    pkl_file = kb_path / "index.pkl"
    print(f"index.faiss exists? {index_file.exists()} (Size: {index_file.stat().st_size if index_file.exists() else 0} bytes)")
    print(f"index.pkl exists? {pkl_file.exists()} (Size: {pkl_file.stat().st_size if pkl_file.exists() else 0} bytes)")

    try:
        store = create_faiss_store(store_name=name, dimension=768, base_dir=str(FAISS_INDICES_DIR))
        print(f"Successfully created store object.")
        try:
            n = store.index.ntotal
            print(f"Index ntotal: {n}")
        except Exception as e:
            print(f"Error accessing .index.ntotal: {e}")
    except Exception as e:
        print(f"FAILED to load store: {e}")

check_kb(KB_GUIDELINES)
check_kb(KB_DRUGLABELS)
