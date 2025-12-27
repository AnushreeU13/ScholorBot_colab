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
        # OPTIMIZATION: If we already have rich metadata from ingestion, use it!
        # This makes the refined look-up instant (O(1)).
        if "references" in fallback_meta and len(fallback_meta["references"]) > 0:
            print(f"[INFO] Using pre-calculated metadata for {path.name} (Zero Latency)")
            return fallback_meta
            
        print(f"[WARN] No pre-calc metadata for {path.name}. Falling back to slow read.")
        
        # --- LEGACY SLOW READ (Backwards Compatibility) ---
        # Load only the first page
        loader = PyPDFLoader(str(path))
        pages = loader.load_and_split()
        if not pages:
            return fallback_meta
            
        first_page_text = pages[0].page_content
        lines = [line.strip() for line in first_page_text.split('\n') if line.strip()]
        
        # --- ENHANCED HEURISTICS ---
        
        # 1. Expanded Blocklist (Stuff top of pages usually has)
        junk_triggers = [
            "volume", "issue", "doi", "http", "www", "received", "accepted", 
            "journal", "elsevier", "springer", "issn", "impact factor", 
            "original article", "review article", "correspondence",
            "available online", "copyright", "downloaded from"
        ]
        
        final_title = fallback_meta.get("title", path.name)
        final_author = fallback_meta.get("author", "Unknown")
        
        # Scan first 20 lines (Headers can be large)
        title_idx = -1
        
        print(f"\n[DEBUG] Refine Citation for: {path.name}")
        for i, line in enumerate(lines[:20]):
            clean_line = line.strip()
            print(f"  [L{i}] '{clean_line}'")
            
            # Filter junk
            if len(clean_line) < 5: continue 
            if any(trigger in clean_line.lower() for trigger in junk_triggers): continue
            
            # Specific suppression for the JBP logo or similar
            if clean_line.upper() in ["JBP", "REVIEW ARTICLE", "ORIGINAL ARTICLE"]: continue
            
            # If line is mostly numbers (dates/IPs), skip
            digit_ratio = sum(c.isdigit() for c in clean_line) / len(clean_line)
            if digit_ratio > 0.4: continue
            
            # Found potential Title!
            final_title = clean_line
            title_idx = i
            print(f"  -> MATCH TITLE: {final_title}")
            break
            
        # 2. Author Extraction (Naive: Line immediately after title)
        # If we found a title, check the next line for authors
        if title_idx != -1 and (title_idx + 1) < len(lines):
            next_line = lines[title_idx + 1].strip()
            
            # Heuristic: Authors usually have names (capitalized) and commas or 'and'
            is_junk = any(trigger in next_line.lower() for trigger in junk_triggers)
            digit_ratio = sum(c.isdigit() for c in next_line) / (len(next_line)+1)
            
            if not is_junk and len(next_line) > 3 and digit_ratio < 0.2:
                # If it's too long, it might be abstract. Authors are usually < 200 chars
                if len(next_line) < 200:
                    final_author = next_line
                    print(f"  -> MATCH AUTHOR: {final_author}")

        # --- 3. BIBLIOGRAPHY EXTRACTION ---
        references = []
        try:
            # Check last 3 pages for "References"
            num_pages = len(pages)
            start_search = max(0, num_pages - 3)
            
            ref_text = ""
            for p in pages[start_search:]:
                ref_text += p.page_content + "\n"
            
            # Find "References" or "Bibliography"
            # We look for a line that is JUST "References" or very close to it
            match = re.search(r'(?i)^\s*(References|Bibliography|LITERATURA CITADA)\s*$', ref_text, re.MULTILINE)
            
            if match:
                print(f"  -> FOUND REFERENCES SECTION")
                # Get text after match
                post_ref = ref_text[match.end():]
                ref_lines = [l.strip() for l in post_ref.split('\n') if l.strip()]
                
                # Grab top 5 that look like citations (start with [1] or Author Name)
                count = 0
                for rl in ref_lines:
                    if len(rl) < 10: continue
                    references.append(rl)
                    count += 1
                    if count >= 5: break
        except Exception as e:
            print(f"[WARN] Reference extraction failed: {e}")

        return {
            "title": final_title,
            "author": final_author,
            "year": fallback_meta.get("year", "n.d."),
            "source": path.name,
            "references": references
        }

    except Exception as e:
        print(f"[WARN] Citation Refiner failed for {path.name}: {e}")
        return fallback_meta
