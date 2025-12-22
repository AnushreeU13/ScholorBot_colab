"""
08_citation_refiner.py

Purpose:
To "read the doc again" and find the header where Title and Authors are located.
This runs post-retrieval to ensure the "Source" metadata is accurate,
rather than relying on potentially messy filenames.
"""

from pathlib import Path
import re
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader

def refine_citation(file_path_str: str, fallback_meta: Dict) -> Dict:
    """
    Opens the original PDF, reads the first page, and uses heuristics
    to find the likely Title and Author block.
    """
    path = Path(file_path_str)
    if not path.exists():
        return fallback_meta

    try:
        # Load only the first page
        loader = PyPDFLoader(str(path))
        pages = loader.load_and_split()
        if not pages:
            return fallback_meta
            
        first_page_text = pages[0].page_content
        lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
        
        # Heuristics for Title:
        # 1. Usually the first non-empty line closest to the top (or 2nd if 1st is "Journal Name")
        # 2. Usually all caps or Title Case.
        
        extracted_title = fallback_meta.get("title", path.name)
        extracted_author = fallback_meta.get("author", "Unknown")
        
        # Candidate 1: First 3 meaningful lines
        candidates = lines[:3]
        
        # Filter out common junk
        blocklist = ["volume", "issue", "doi", "http", "www", "received", "accepted"]
        
        final_title = extracted_title
        
        for cand in candidates:
            if len(cand) < 5: continue # Too short
            if any(b in cand.lower() for b in blocklist): continue
            
            # If it looks like a title (longer, no numbers usually)
            final_title = cand
            break
            
        return {
            "title": final_title,
            "author": extracted_author, # Author extraction is harder without NER, keeping fallback for now
            "year": fallback_meta.get("year", "n.d."),
            "source": path.name
        }

    except Exception as e:
        print(f"[WARN] Citation Refiner failed for {path.name}: {e}")
        return fallback_meta
