"""
rag_pipeline_aligned.py

Tier-1 RAG pipeline (Hybrid + Gates, stable v4)

Major fixes:
A) Router task_hints separation (no more preferred pollution like "diagnosis" inside section groups)
B) Guideline Diagnosis Gate (hard filter for diagnosis/testing evidence when asked)
C) Drug Anchor Filter (unlocked) to prevent wrong-drug retrieval drift
D) Entailment Gate for clinician bullets (lightweight lexical entailment + high-risk term blocking)
E) Patient Safety Gate (medical-entity-based) + stronger patient prompt to avoid "clinician copy / background explanations"

Dependencies:
- config.py / config_local_v2.py
- router.py
- llm_utils.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Set
import re
import time
import numpy as np
import torch

# ------------------------------------------------------------
# Config Import
# ------------------------------------------------------------
try:
    from config_local_v2 import (
        KB_SIM_THRESHOLD, DEFAULT_SIM_THRESHOLD, TOP_K,
        ZERO_HALLUCINATION_MODE, USE_CLINICIAN_LLM,
        KB_DRUGLABELS, KB_GUIDELINES, KB_USER_FACT,
    )
except Exception:
    KB_SIM_THRESHOLD = {
        "kb_druglabels_medcpt": 0.60,
        "kb_guidelines_medcpt": 0.60,
        "user_fact_kb_medcpt": 0.70,
    }
    DEFAULT_SIM_THRESHOLD = 0.65
    TOP_K = 6
    ZERO_HALLUCINATION_MODE = False
    USE_CLINICIAN_LLM = True
    KB_DRUGLABELS = "kb_druglabels_medcpt"
    KB_GUIDELINES = "kb_guidelines_medcpt"
    KB_USER_FACT = "user_fact_kb_medcpt"

# ------------------------------------------------------------
# Router Import / Mock
# ------------------------------------------------------------
try:
    from router import route_query, RouteDecision
except Exception:
    @dataclass
    class RouteDecision:
        intent: str
        target_kbs: List[str]
        preferred_section_groups: List[str]
        reason: str
        task_hints: List[str] = None

    def route_query(query: str, user_uploaded_available: bool = False) -> RouteDecision:
        targets = ["kb_druglabels_medcpt", "kb_guidelines_medcpt"]
        if user_uploaded_available:
            targets.append("user_fact_kb_medcpt")
        return RouteDecision("mixed", targets, [], "Fallback router.", task_hints=[])

# ------------------------------------------------------------
# Data Structures
# ------------------------------------------------------------
@dataclass
class RAGResult:
    answer: str
    clinician_answer: str
    patient_answer: str
    citations: List[str]
    confidence: float
    status: str
    source_kbs: List[str]
    route: Dict[str, Any]
    debug_info: Dict[str, Any]
    consistency: Dict[str, Any]

# ------------------------------------------------------------
# LLM Helpers
# ------------------------------------------------------------
def _clean_generated_text(raw_text: str, marker: str = "OUTPUT:") -> str:
    if marker in raw_text:
        raw_text = raw_text.split(marker, 1)[-1]
    raw_text = re.sub(r"(?im)^\s*(system|user|assistant)\s*:?\s*", "", raw_text)
    raw_text = re.sub(r"[\u4e00-\u9fff]", "", raw_text)  # strip Chinese artifacts
    return raw_text.strip()

def _generate_with_prompt(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Hybrid generator:
    1) If OPENAI_API_KEY exists and OpenAI SDK is installed -> use OpenAI (default: gpt-4o-mini)
    2) Otherwise -> fallback to local transformers model (llm_utils: tokenizer/model/clean_llm_answer)

    Returns raw text (caller will clean/parse).
    Fail-closed behavior: if BOTH fail, returns a sentinel string "GENERATOR_ERROR: ..."
    """
    import os
    import re

    system_msg = "You are a clinical assistant. Answer in English only. Use ONLY provided evidence."

    # -------------------------
    # 1) Try OpenAI (preferred)
    # -------------------------
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    if use_openai:
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

            resp = client.responses.create(
                model=model_name,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_output_tokens=max_new_tokens,
            )

            text = getattr(resp, "output_text", None)
            if text and text.strip():
                return text.strip()

            # Defensive fallback if output_text is missing
            out = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) == "output_text":
                            out.append(getattr(c, "text", ""))
            joined = ("\n".join(out)).strip()
            if joined:
                return joined

            # If OpenAI returned empty, fall back to local
        except Exception as e:
            # OpenAI failed -> fall back to local
            pass

    # -------------------------
    # 2) Local fallback (Qwen via llm_utils)
    # -------------------------
    try:
        from llm_utils import tokenizer, model, clean_llm_answer  # your local loader

        # If chat template exists, format like chat
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system_msg}\n\n{prompt}"

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        import torch
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = out[0][input_len:]
        raw = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        raw = clean_llm_answer(raw)

        # Basic cleanup (avoid odd role artifacts)
        raw = re.sub(r"(?im)^\s*(system|user|assistant)\s*:?\s*", "", raw).strip()
        return raw if raw else "GENERATOR_ERROR: local_empty"

    except Exception as e:
        return f"GENERATOR_ERROR: {type(e).__name__}"



# ------------------------------------------------------------
# Constants (anchors / domain)
# ------------------------------------------------------------
_DRUG_ANCHORS = {
    "isoniazid","rifampin","rifampicin","pyrazinamide","ethambutol",
    "azithromycin","levofloxacin","moxifloxacin","amoxicillin","linezolid"
}
_DOMAIN_TERMS = {"tuberculosis","tb","pneumonia","cap","community","acquired","ats","idsa","who","cdc"}

# ------------------------------------------------------------
# Utility: cleaning & citations
# ------------------------------------------------------------
def _clean_pdf_text(s: str) -> str:
    if not isinstance(s, str): return ""
    t = s.replace("\u00ad", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _stable_citation(meta: Dict[str, Any]) -> str:
    if not meta: return "Unknown source"
    if meta.get("source") == "DailyMed" or meta.get("doc_type") == "druglabel_spl":
        title = meta.get("title") or meta.get("source_title") or "DrugLabel"
        section = meta.get("section_title") or meta.get("section_group") or "Section"
        date = meta.get("source_date") or meta.get("upload_date") or "n/a"
        return f"DailyMed | {title} | {section} | {date}"
    doc = meta.get("document") or meta.get("title") or "Guideline"
    sec = meta.get("section") or meta.get("section_title") or ""
    year = meta.get("year") or "n/a"
    return f"{doc} | {sec} | {year}"

def _collect_citations(chunks: List[Dict], max_items: int = 5) -> List[str]:
    seen = set()
    out = []
    for c in chunks:
        cit = _stable_citation(c.get("metadata", {}))
        if cit not in seen:
            seen.add(cit)
            out.append(cit)
    return out[:max_items]

def _section_group_from_meta(meta: Dict) -> str:
    sec = (meta.get("section") or meta.get("section_title") or "").lower()
    grp = (meta.get("section_group") or "").lower()
    if grp:
        return grp

    # --- Drug label style groups (existing) ---
    if any(k in sec for k in ["dose", "dosage", "admin", "administration"]): 
        return "dosage"
    if any(k in sec for k in ["contraindication"]): 
        return "contraindications"
    if any(k in sec for k in ["warn", "precaution", "boxed", "black box"]): 
        return "warnings"
    if any(k in sec for k in ["adverse", "side effect", "reaction", "toxicity"]): 
        return "adverse"
    if "interact" in sec or "cyp" in sec: 
        return "interactions"
    if any(k in sec for k in ["indication", "indications", "use in"]): 
        return "indications"

    # --- Guideline-oriented groups (NEW) ---
    # Treatment / regimen / recommendation sections
    if any(k in sec for k in [
        "recommendation", "recommendations",
        "treatment", "therapy", "management",
        "regimen", "regimens", "duration", "course",
        "follow-up", "monitoring", "rationale"
    ]):
        return "g_treatment"

    # Diagnosis / testing / evaluation sections
    if any(k in sec for k in [
        "diagnos", "diagnosis", "testing", "test", "evaluation", "work-up",
        "radiograph", "x-ray", "imaging", "culture", "sputum", "pcr"
    ]):
        return "g_diagnosis"

    # Prevention / infection control
    if any(k in sec for k in [
        "prevention", "prevent", "prophylaxis",
        "infection control", "vaccin"
    ]):
        return "g_prevention"

    return "other"


def _apply_section_bias(sims: np.ndarray, metas: List[Dict], preferred_groups: List[str], texts: List[str] = None, boost: float = 0.12) -> np.ndarray:
    if len(sims) == 0: return sims
    boosted = sims.astype(np.float32).copy()
    pref_set = {p.lower() for p in preferred_groups}

    for i, meta in enumerate(metas):
        add = 0.0
        if _section_group_from_meta(meta) in pref_set:
            add += boost

        if texts:
            txt = (texts[i] or "").lower()[:800]
            # downrank obvious references
            if "references" in txt[:60] or "bibliography" in txt[:60] or "doi:" in txt:
                add -= 0.5

        boosted[i] += add

    return np.clip(boosted, 0.0, 1.0)

# ------------------------------------------------------------
# Gate C: Drug Anchor Filtering (Unlocked)
# ------------------------------------------------------------
def _filter_candidates_by_drug_anchor(candidates: List[Dict], query: str) -> List[Dict]:
    q = (query or "").lower()
    found = [d for d in _DRUG_ANCHORS if d in q]
    if not found:
        return candidates

    filtered = []
    for c in candidates:
        if c["store"] != KB_DRUGLABELS:
            filtered.append(c)
            continue
        title = (c["metadata"].get("title") or c["metadata"].get("source_title") or "").lower()
        text_full = (c["text"] or "").lower()  # unlocked: no truncation
        if any(d in title for d in found) or any(d in text_full for d in found):
            filtered.append(c)
    return filtered

# ------------------------------------------------------------
# Gate B: Guideline Diagnosis Gate
# ------------------------------------------------------------
_DIAG_KEYS = ["diagnos", "testing", "test", "workup", "radiograph", "x-ray", "imaging", "culture", "sputum", "blood", "procalcitonin", "pcr"]

def _guideline_diagnosis_gate(chunks: List[Dict]) -> List[Dict]:
    out = []
    for c in chunks:
        meta = c.get("metadata", {})
        sec = (meta.get("section_title") or meta.get("section") or "").lower()
        txt = (c.get("text") or "").lower()[:900]
        if any(k in sec for k in _DIAG_KEYS) or any(k in txt for k in _DIAG_KEYS):
            out.append(c)
    return out

# ------------------------------------------------------------
# Regex extractors
# ------------------------------------------------------------
def _extract_regex_items(text: str, mode: str) -> List[str]:
    text = _clean_pdf_text(text)
    patterns = []
    if mode == "adr":
        patterns = [
            r"include[^:]{0,80}:\s*(.+?)(?:\.|$)",
            r"following:\s*(.+?)(?:\.|$)",
        ]
    elif mode == "dosage":
        patterns = [r"(\b\d+(?:\.\d+)?\s*mg(?:/kg)?\b.*?(?:\.|$))"]

    items: List[str] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            content = (m.group(1) or "").strip()
            parts = re.split(r",|;|\band\b", content)
            items.extend([p.strip() for p in parts if len(p.strip()) > 3])

    # de-dup
    seen = set()
    out = []
    for it in items:
        key = re.sub(r"\W+", "", it.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out[:12]

def _parse_relaxed_bullets(text: str, max_items: int = 12) -> List[str]:
    if not isinstance(text, str): return []
    lines: List[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln: continue
        if re.match(r"^([-•\*]|\(?\d+\)?[.)])\s+\S", ln):
            item = re.sub(r"^([-•\*]|\(?\d+\)?[.)])\s+", "", ln).strip()
            if item:
                lines.append(item)

    if not lines and len(text) < 700:
        parts = re.split(r",|;|\band\b| or ", text)
        lines = [p.strip() for p in parts if len(p.strip()) > 3]

    seen = set()
    out = []
    for it in lines:
        key = re.sub(r"\W+", "", it.lower())
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out[:max_items]

# ------------------------------------------------------------
# Gate D: Entailment Gate (tuned)
# ------------------------------------------------------------
_STOP = {"the","a","an","of","to","in","for","with","on","at","by","is","are","was","were","be","has","have","had","and","or"}
_HIGH_RISK = ["carbapenem", "ciprofloxacin", "protease", "hiv", "hepatitis c", "hepatitis b", "mri"]

def _verify_entailment(bullet: str, evidence_text: str, threshold: float = 0.25) -> bool:
    if not bullet:
        return False
    b = bullet.lower().strip()
    e = (evidence_text or "").lower()

    # hard block high-risk additions
    for w in _HIGH_RISK:
        if w in b and w not in e:
            return False

    words = [w.lower() for w in re.findall(r"\w+", bullet) if w.lower() not in _STOP and len(w) > 2]
    if not words:
        return True

    hits = sum(1 for w in words if w in e)
    overlap = hits / max(1, len(words))
    return overlap >= threshold

# ------------------------------------------------------------
# Gate E: Patient Safety Gate (medical-entity-based)
# ------------------------------------------------------------
def _extract_critical_tokens(text: str) -> Set[str]:
    toks = set()
    for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/%]*", text or ""):
        lw = w.lower()

        # dosage-like (numbers)
        if any(ch.isdigit() for ch in w):
            toks.add(lw); continue

        # all-caps abbreviation (HIV, CAP)
        if len(w) >= 2 and w.isupper():
            toks.add(lw); continue

        # anchors
        if lw in _DRUG_ANCHORS or lw in _DOMAIN_TERMS:
            toks.add(lw); continue

    return toks

def _check_patient_safety(clinician_text: str, patient_text: str) -> bool:
    c = _extract_critical_tokens(clinician_text)
    p = _extract_critical_tokens(patient_text)

    # allow generic words only if they appear as "critical" (rare)
    whitelist = {
        "patient","doctor","medicine","symptoms","treatment","diagnosis","common","severe","mild",
        "acute","chronic","oral","tablet","capsule","daily","weekly"
    }

    new_tokens = [t for t in p if (t not in c and t not in whitelist)]
    if new_tokens:
        print(f"[RAG] Patient Guard flagged new entities: {new_tokens}")
        return False
    return True

def _patient_rewrite_deterministic(clinician_answer: str) -> str:
    if "ABSTAIN" in clinician_answer:
        return "I could not find an explicit answer in the retrieved documents."
    clean = clinician_answer.replace("FINAL:", "").strip()
    return f"From the documents:\n{clean}"

# ------------------------------------------------------------
# Main Pipeline
# ------------------------------------------------------------
class RAGPipeline:
    def __init__(self, query_embedder, kb_guidelines_store, kb_druglabels_store, user_kb_store=None, top_k=TOP_K, verbose=True, logger=None):
        self.query_embedder = query_embedder
        self.kb_guidelines = kb_guidelines_store
        self.kb_druglabels = kb_druglabels_store
        self.user_kb = user_kb_store
        self.top_k = top_k
        self.verbose = verbose
        self.logger = logger

    def _log(self, msg: str):
        if self.verbose:
            (self.logger if self.logger else print)(msg)

    def _get_store_by_name(self, name: str):
        if name == KB_DRUGLABELS: return self.kb_druglabels
        if name == KB_GUIDELINES: return self.kb_guidelines
        if name == KB_USER_FACT: return self.user_kb
        return None

    def _check_consistency(self, clinician_text: str, patient_text: str) -> Dict[str, Any]:
        if "ABSTAIN" in clinician_text or "could not find" in patient_text.lower():
            return {"score": 1.0, "status": "pass (abstain)"}

        vec_c = self.query_embedder.embed_query(clinician_text)
        vec_p = self.query_embedder.embed_query(patient_text)

        # normalize cosine-ish
        nc = np.linalg.norm(vec_c)
        npv = np.linalg.norm(vec_p)
        if nc > 0: vec_c = vec_c / nc
        if npv > 0: vec_p = vec_p / npv

        sim = float(np.dot(vec_c, vec_p.T))
        return {"score": sim, "status": "pass" if sim >= 0.72 else "fail"}

    def retrieve_and_answer(self, query: str) -> RAGResult:
        self._log(f"\n=== TIER-1 RAG QUERY: {query} ===")
        t0 = time.time()

        q_vec = self.query_embedder.embed_query(query)
        decision = route_query(query, user_uploaded_available=(self.user_kb is not None))

        # task_hints compatible
        task_hints = getattr(decision, "task_hints", None) or []
        self._log(f"[ROUTER] Intent: {decision.intent} | Target: {decision.target_kbs} | Pref: {decision.preferred_section_groups} | Hints: {task_hints}")

        candidates: List[Dict] = []
        for kb_name in decision.target_kbs:
            store = self._get_store_by_name(kb_name)
            if not store: continue

            search_k = 500 if kb_name == KB_DRUGLABELS else max(40, self.top_k * 5)
            sims, idxs, metas = store.search(q_vec, k=search_k)
            texts = [metas[i].get("text", "") for i in range(len(sims))]

            boosted = _apply_section_bias(sims, metas, decision.preferred_section_groups, texts=texts)
            for i in range(len(sims)):
                candidates.append({
                    "score": float(boosted[i]),
                    "raw_sim": float(sims[i]),
                    "text": metas[i].get("text", ""),
                    "metadata": metas[i],
                    "store": kb_name
                })

        # Drug anchor filter
        before = len(candidates)
        candidates = _filter_candidates_by_drug_anchor(candidates, query)
        after = len(candidates)
        if before != after:
            self._log(f"[RAG] Anchor filter kept {after} candidates (dropped {before-after}).")

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_chunks = candidates[:self.top_k]

        # Section isolation for drugs
        final_chunks = top_chunks
        if decision.intent == "drug":
            if "adverse" in decision.preferred_section_groups:
                adverse_only = [c for c in top_chunks if c["metadata"].get("section_group") == "adverse"]
                if adverse_only:
                    final_chunks = adverse_only
                    self._log(f"[RAG] Section Isolation: {len(top_chunks)} -> {len(final_chunks)} (adverse)")
            if "interactions" in decision.preferred_section_groups:
                inter_only = [c for c in top_chunks if c["metadata"].get("section_group") == "interactions"]
                if inter_only:
                    final_chunks = inter_only
                    self._log(f"[RAG] Section Isolation: {len(top_chunks)} -> {len(final_chunks)} (interactions)")

        # Guideline diagnosis gate (key)
        if decision.intent in ("guideline", "mixed") and ("diagnosis" in task_hints):
            gated = _guideline_diagnosis_gate(final_chunks)
            if gated:
                self._log(f"[RAG] Guideline Diagnosis Gate: {len(final_chunks)} -> {len(gated)}")
                final_chunks = gated[:self.top_k]
            else:
                self._log("[RAG] Guideline Diagnosis Gate: no matching evidence -> ABSTAIN.")
                return self._build_abstain_result(query, decision)

        best_score = final_chunks[0]["score"] if final_chunks else 0.0
        required = KB_SIM_THRESHOLD.get(final_chunks[0]["store"], 0.65) if final_chunks else 0.7
        if not final_chunks or best_score < required:
            self._log(f"[RAG] Low confidence ({best_score:.3f} < {required}) -> ABSTAIN.")
            return self._build_abstain_result(query, decision)

        evidence_text = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(final_chunks)])
        citations = _collect_citations(final_chunks)
        primary_kb = final_chunks[0]["store"]

        clinician_answer = "ABSTAIN"
        mode = "unknown"

        # --- Drug logic ---
        if decision.intent == "drug" and primary_kb == KB_DRUGLABELS:
            fast_mode = None
            if "adverse" in decision.preferred_section_groups:
                fast_mode = "adr"
            elif "dosage" in decision.preferred_section_groups:
                fast_mode = "dosage"

            if fast_mode:
                items = _extract_regex_items(evidence_text, fast_mode)
                if items:
                    clinician_answer = "FINAL:\n" + "\n".join([f"- {x}" for x in items])
                    mode = "regex_deterministic"
                else:
                    self._log("[RAG] Regex empty -> Strict LLM fallback.")
                    clinician_answer = self._generate_clinician_answer(query, evidence_text, intent="drug")
                    mode = "fallback_llm"
            else:
                clinician_answer = self._generate_clinician_answer(query, evidence_text, intent="drug_general")
                mode = "llm_gen"

        # --- Guideline logic ---
        else:
            if USE_CLINICIAN_LLM:
                clinician_answer = self._generate_clinician_answer(query, evidence_text, intent="guideline")
                mode = "guideline_synthesis"
            else:
                clinician_answer = "FINAL:\n" + evidence_text[:700] + "..."
                mode = "legacy_extract"

        if "ABSTAIN" in clinician_answer:
            return self._build_abstain_result(query, decision)

        # Patient rewrite
        self._log("[RAG] Generating Patient Answer...")
        patient_answer = self._generate_patient_answer(clinician_answer)

        consistency = self._check_consistency(clinician_answer, patient_answer)
        consistency["mode"] = mode
        self._log(f"[CONSISTENCY] Score: {consistency['score']:.3f} | Status: {consistency['status']}")

        evidence_chunks = []
        for c in final_chunks:
            evidence_chunks.append({
                "text": (c.get("text") or "")[:1600],   # chunk text (truncate for UI)
                "citation": _stable_citation(c.get("metadata", {})),
                "store": c.get("store"),
            })


        return RAGResult(
            answer=clinician_answer,
            clinician_answer=clinician_answer,
            patient_answer=patient_answer,
            citations=citations,
            confidence=best_score,
            status="answer",
            source_kbs=[c["store"] for c in final_chunks],
            route={"intent": decision.intent, "task_hints": task_hints},
            debug_info={
                "top_sim_raw": final_chunks[0]["raw_sim"],
                "elapsed_s": round(time.time() - t0, 3),
                "evidence_chunks": evidence_chunks,   # NEW
            },
            consistency=consistency,
        )

    # ---------------------------------------------------------
    # Generation
    # ---------------------------------------------------------
    def _generate_clinician_answer(self, query: str, context: str, intent: str = "general") -> str:
        # strong negatives to reduce drift
        negative = ""
        if intent.startswith("drug"):
            negative = """
Rules additions (drug):
- ONLY extract items explicitly present in EVIDENCE.
- Do NOT add mechanism, background explanations, or new drug names.
- If question is adverse reactions: list reactions/symptoms only (not risk factors/monitoring).
- If question is interactions: list interacting drugs/classes explicitly mentioned.
""".strip()
        else:
            negative = """
Rules additions (guideline):
- ONLY use EVIDENCE.
- Do NOT add symptoms/tests not mentioned.
- Prefer short complete sentences (not fragments).
""".strip()

        prompt = f"""
Task: Answer the QUESTION using ONLY the EVIDENCE.

Output rules (strict):
1) Output ONLY bullet points.
2) Each bullet must start with "- " (dash + space).
3) 3 to 8 bullets preferred. If insufficient evidence: output "ABSTAIN".

{negative}

QUESTION:
{query}

EVIDENCE:
{context[:2600]}

CLINICIAN OUTPUT:
""".strip()

        raw = _generate_with_prompt(prompt, max_new_tokens=240)

        # NEW: fail fast if generator returned an error sentinel
        if raw.startswith("OPENAI_") or raw in ["OPENAI_API_KEY_MISSING", "OPENAI_EMPTY_OUTPUT"]:
            return "ABSTAIN"

        clean = _clean_generated_text(raw, marker="CLINICIAN OUTPUT:")
        if "ABSTAIN" in clean or len(clean) < 5:
            return "ABSTAIN"

        lines = _parse_relaxed_bullets(clean, max_items=12)

        verified: List[str] = []
        for ln in lines:
            if _verify_entailment(ln, context, threshold=0.25):
                verified.append(ln)
            else:
                print(f"[RAG] Dropped hallucinated bullet: {ln}")

        if not verified:
            return "ABSTAIN"

        # Make bullets more sentence-like for clinicians (no weird fragments)
        final = []
        for b in verified[:10]:
            b2 = b.strip()
            # ensure ends with period if it looks like a sentence
            if len(b2) > 20 and not b2.endswith((".", ";")):
                b2 += "."
            final.append(f"- {b2}")

        return "FINAL:\n" + "\n".join(final)

    def _generate_patient_answer(self, clinician_text: str) -> str:
        clean = clinician_text.replace("FINAL:", "").strip()

        prompt = f"""
Task: Rewrite for a patient (6th-grade reading level).

Rules (strict):
1) Output ONLY bullet points, each starting with "- ".
2) Do NOT add background explanations or definitions (no "X is a ...").
3) Do NOT introduce new drugs, diseases, tests, or examples.
4) Do NOT write a patient story (do NOT say "the patient has...").
5) Keep meaning same as source; shorter is better.

SOURCE BULLETS:
{clean}

PATIENT OUTPUT:
""".strip()

        raw = _generate_with_prompt(prompt, max_new_tokens=200)
        patient = _clean_generated_text(raw, marker="PATIENT OUTPUT:")

        # must be bullets; if not, fallback
        if "- " not in patient:
            self._log("[RAG] Patient output not bullet-only -> deterministic fallback.")
            return _patient_rewrite_deterministic(clinician_text)

        if not _check_patient_safety(clean, patient):
            self._log("[RAG] Patient safety gate failed -> deterministic fallback.")
            return _patient_rewrite_deterministic(clinician_text)

        return patient

    def _build_abstain_result(self, query: str, decision: Any) -> RAGResult:
        return RAGResult(
            answer="ABSTAIN",
            clinician_answer="ABSTAIN",
            patient_answer="I could not find an explicit answer in the retrieved documents.",
            citations=[],
            confidence=0.0,
            status="abstain",
            source_kbs=[],
            route={"intent": getattr(decision, "intent", "unknown"), "task_hints": getattr(decision, "task_hints", [])},
            debug_info={},
            consistency={"mode": "abstain"},
        )
