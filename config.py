"""
config.py
Local-first configuration for ScholarBOT Tier-1 RAG (TB & pneumonia/CAP + drug labels).

Goals:
- Portable paths (no hard-coded Windows drive letters)
- One place to control "zero hallucination mode" vs "helpful mode"
"""

from __future__ import annotations
from pathlib import Path
import os

# =============================
# Project roots (portable)
# =============================
PROJECT_ROOT = Path(os.getenv("SCHOLARBOT_ROOT", Path(__file__).resolve().parent)).resolve()
DATA_DIR = Path(os.getenv("SCHOLARBOT_DATA_DIR", PROJECT_ROOT / "datasets")).resolve()

# Raw / processed KB data (you can keep everything local)
KB_RAW_DIR = Path(os.getenv("SCHOLARBOT_KB_RAW_DIR", DATA_DIR / "KB_raw")).resolve()
KB_PROCESSED_DIR = Path(os.getenv("SCHOLARBOT_KB_PROCESSED_DIR", DATA_DIR / "KB_processed")).resolve()

# FAISS storage (index + metadata json live together)
FAISS_INDICES_DIR = Path(os.getenv("SCHOLARBOT_FAISS_DIR", PROJECT_ROOT / "faiss_indices")).resolve()

# Optional corpora folders (only used if you run those ingesters)
PMC_FOLDER = Path(os.getenv("SCHOLARBOT_PMC_DIR", DATA_DIR / "PMC")).resolve()
PNEUMONIA_FOLDER = Path(os.getenv("SCHOLARBOT_PNEUMONIA_DIR", DATA_DIR / "pneumonia")).resolve()
TUBERCULOSIS_FOLDER = Path(os.getenv("SCHOLARBOT_TB_DIR", DATA_DIR / "tuberculosis")).resolve()
XRAY_FOLDER = Path(os.getenv("SCHOLARBOT_XRAY_DIR", DATA_DIR / "xray")).resolve()

# =============================
# KB store names (single source of truth)
# =============================
KB_DRUGLABELS = "kb_druglabels_medcpt"
KB_GUIDELINES = "kb_guidelines_medcpt"
KB_USER_FACT = "user_fact_kb_medcpt"

# =============================
# Chunking
# =============================
CHUNK_SIZE = 400
OVERLAP = 50
# Guidelines benefit from smaller chunks (precision-first)
# Guidelines KB chunking
GUIDELINE_CHUNK_SIZE = 240
GUIDELINE_CHUNK_OVERLAP = 50
# =============================
# Retrieval
# =============================
TOP_K = 6

DEFAULT_SIM_THRESHOLD = 0.70
KB_SIM_THRESHOLD = {
    KB_DRUGLABELS: 0.70,
    KB_GUIDELINES: 0.60,
    KB_USER_FACT: 0.70,
}

# =============================
# Hallucination control
# =============================
# True  => extractive-only (no clinician LLM generation), more ABSTAIN, safest.
# False => allow clinician LLM (still evidence-gated), more helpful but higher risk.
ZERO_HALLUCINATION_MODE = os.getenv("SCHOLARBOT_ZERO_HALLUCINATION", "1") == "1"

# If you want finer control:
USE_CLINICIAN_LLM = (not ZERO_HALLUCINATION_MODE) and (os.getenv("SCHOLARBOT_USE_CLINICIAN_LLM", "0") == "1")

# =============================
# Local LLM (only used if USE_CLINICIAN_LLM == True)
# =============================
LOCAL_QA_MODEL_NAME = os.getenv("LOCAL_QA_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("SCHOLARBOT_MAX_NEW_TOKENS", "260"))

# Bedrock settings kept for compatibility, but you said local-only.
LLM_BACKEND = os.getenv("SCHOLARBOT_LLM_BACKEND", "local")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
