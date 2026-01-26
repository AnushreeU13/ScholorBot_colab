"""
Download CDC TB clinical guidance pages in bulk and export:
1) Raw HTML (for reproducibility)
2) Cleaned text (for ingestion)
3) JSONL records (text + metadata) for downstream chunking/indexing

All paths are centralized via config.py.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from config import FAISS_INDICES_DIR  # not used here but keeps conventions consistent

# Optional: use KB_RAW_DIR / KB_PROCESSED_DIR if you added them
try:
    from config import KB_RAW_DIR, KB_PROCESSED_DIR
except Exception:
    from config import BOX_FOLDER
    KB_RAW_DIR = str(Path(BOX_FOLDER) / "KB_raw")
    KB_PROCESSED_DIR = str(Path(BOX_FOLDER) / "KB_processed")


# -----------------------------
# Config: output directories
# -----------------------------
KB_RAW = Path(KB_RAW_DIR)
KB_PROCESSED = Path(KB_PROCESSED_DIR)

CDC_RAW_HTML_DIR = KB_RAW / "guidelines" / "CDC_TB_pages" / "html"
CDC_RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

CDC_TXT_DIR = KB_PROCESSED / "guidelines_text" / "cdc_tb_txt"
CDC_TXT_DIR.mkdir(parents=True, exist_ok=True)

CDC_JSONL_PATH = KB_PROCESSED / "guidelines_text" / "cdc_tb_pages.jsonl"


# -----------------------------
# CDC page list (Tier-1 set)
# -----------------------------
CDC_TB_URLS = [
    {
        "url": "https://www.cdc.gov/tb/hcp/clinical-guidance/index.html",
        "title": "CDC TB Clinical Guidance Hub",
        "topic": "clinical_guidance_hub",
    },
    {
        "url": "https://www.cdc.gov/tb/hcp/treatment/index.html",
        "title": "Clinical Treatment of Tuberculosis",
        "topic": "treatment_active_tb",
    },
    {
        "url": "https://www.cdc.gov/tb/hcp/treatment/latent-tuberculosis-infection.html",
        "title": "Treatment for Latent Tuberculosis Infection",
        "topic": "treatment_latent_tb",
    },
    {
        "url": "https://www.cdc.gov/tb/hcp/clinical-overview/latent-tuberculosis-infection.html",
        "title": "Clinical Overview of Latent TB Infection",
        "topic": "overview_latent_tb",
    },
    {
        "url": "https://www.cdc.gov/tb/hcp/treatment/drug-resistant-tuberculosis-disease.html",
        "title": "Treatment for Drug-Resistant Tuberculosis Disease",
        "topic": "treatment_drug_resistant_tb",
    },
    {
        "url": "https://www.cdc.gov/tb-healthcare-settings/hcp/infection-control/index.html",
        "title": "TB Infection Control (Healthcare Settings)",
        "topic": "infection_control_healthcare",
    },
]

# A respectful rate limit (CDC pages are public but don't hammer them)
REQUEST_SLEEP_SECONDS = 1.0
USER_AGENT = "ScholarBOT-RAG/1.0 (Academic research; contact: your_email@example.com)"


# -----------------------------
# Helpers
# -----------------------------
def slugify(s: str) -> str:
    """Convert a title/topic into a safe filename slug."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def fetch_html(url: str) -> str:
    """Fetch HTML with a stable user-agent and basic error handling."""
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_main_text(html: str) -> str:
    """
    Extract main readable text from a CDC page.
    This uses heuristic DOM selection and removes nav/footer/script/style.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    # Try common "main content" containers
    main = soup.find("main")
    if main is None:
        # fallback: use body
        main = soup.body if soup.body else soup

    # Drop likely navigation/boilerplate blocks if present
    for sel in ["nav", "header", "footer", "aside"]:
        for t in main.find_all(sel):
            t.decompose()

    text = main.get_text(separator="\n")
    # Normalize whitespace and remove empty lines
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]

    # Remove common CDC boilerplate fragments (lightweight)
    cleaned = []
    for ln in lines:
        if "Centers for Disease Control and Prevention" in ln:
            continue
        if ln.startswith("Español"):
            continue
        cleaned.append(ln)

    return "\n".join(cleaned).strip()


def write_jsonl_record(path: Path, record: Dict) -> None:
    """Append a JSONL record."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# Main
# -----------------------------
def main():
    # Reset JSONL each run to avoid mixing runs
    if CDC_JSONL_PATH.exists():
        CDC_JSONL_PATH.unlink()

    for item in CDC_TB_URLS:
        url = item["url"]
        title = item["title"]
        topic = item["topic"]

        print(f"[INFO] Fetching: {title}")
        html = fetch_html(url)

        # Save raw HTML for reproducibility
        html_name = f"{slugify(topic)}.html"
        html_path = CDC_RAW_HTML_DIR / html_name
        html_path.write_text(html, encoding="utf-8")

        # Extract and save clean text
        text = extract_main_text(html)
        txt_name = f"{slugify(topic)}.txt"
        txt_path = CDC_TXT_DIR / txt_name
        txt_path.write_text(text, encoding="utf-8")

        # Write JSONL record for ingestion
        record = {
            "text": text,
            "metadata": {
                "source_type": "clinical_guideline",
                "organization": "CDC",
                "disease": "tuberculosis",
                "document": "CDC_TB_WebGuidance",
                "topic": topic,
                "title": title,
                "url": url,
                "audience": "clinician",
                "evidence_level": "guideline_web",
            },
        }
        write_jsonl_record(CDC_JSONL_PATH, record)

        time.sleep(REQUEST_SLEEP_SECONDS)

    print("\n[DONE] CDC TB pages saved.")
    print(f"  - Raw HTML dir: {CDC_RAW_HTML_DIR}")
    print(f"  - Clean TXT dir: {CDC_TXT_DIR}")
    print(f"  - JSONL output : {CDC_JSONL_PATH}")


if __name__ == "__main__":
    main()
