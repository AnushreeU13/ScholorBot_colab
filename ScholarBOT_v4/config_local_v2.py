
# config_local_v2.py
# Override defaults to match local folder structure (main_kb)

KB_DRUGLABELS = "main_kb"
KB_GUIDELINES = "main_kb"
KB_USER_FACT = "user_kb"

# Keep high recall settings
TOP_K = 8

KB_SIM_THRESHOLD = {
    "main_kb": 0.35,
    "user_kb": 0.35,
    "user_fact_kb_medcpt": 0.35,
    "kb_druglabels_medcpt": 0.35,
    "kb_guidelines_medcpt": 0.35,
}

DEFAULT_SIM_THRESHOLD = 0.35
ZERO_HALLUCINATION_MODE = False
USE_CLINICIAN_LLM = True
