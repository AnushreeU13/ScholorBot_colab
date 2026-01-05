import streamlit as st
from pathlib import Path
import tempfile

from aligned_backend import AlignedScholarBotEngine, ingest_user_file


st.set_page_config(page_title="ScholarBOT (Aligned Clinical RAG)", layout="wide")

st.title("ScholarBOT (Aligned Clinical RAG)")
st.caption("Readable summaries + paper-style evidence tracing (Claim → Supporting snippet → Citation).")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    force_user_kb = st.checkbox("Use ONLY uploaded file (no fallback)", value=False)
    verbose = st.checkbox("Verbose backend logs", value=False)

    st.divider()
    st.subheader("Upload PDF (User Context KB)")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    ingest_clicked = st.button("Ingest into User KB")


@st.cache_resource
def get_engine(verbose_flag: bool):
    return AlignedScholarBotEngine(verbose=verbose_flag, print_kb_stats=True)


engine = get_engine(verbose)

# --- Ingestion flow ---
if ingest_clicked:
    if not uploaded:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Ingesting uploaded PDF into user KB..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / uploaded.name
                tmp_path.write_bytes(uploaded.getbuffer())
                ingest_user_file(tmp_path)
        st.success("Ingestion completed. You can now ask questions.")


if "history" not in st.session_state:
    st.session_state.history = []

st.divider()
query = st.text_input("Ask a clinical question (TB / pneumonia / drug labels supported):", value="")
ask = st.button("Ask")

if ask and query.strip():
    with st.spinner("Retrieving evidence and generating answer..."):
        response_text, confidence, meta = engine.generate_response(
            query=query.strip(),
            model_name="",
            force_user_kb=force_user_kb,
            history=st.session_state.history,
        )

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # Main product-friendly output (paragraphs)
        st.markdown(response_text)

        # NEW: Claim -> snippet alignment (paper-style)
        with st.expander("Claim → Supporting snippet (paper-style trace)", expanded=True):
            items = meta.get("claim_snippets", []) or []
            if not items:
                st.info("No claim-snippet alignment available (likely abstain or missing evidence chunks).")
            else:
                for i, it in enumerate(items, 1):
                    claim = it.get("claim", "").strip()
                    snippet = it.get("snippet", "").strip()
                    citation = it.get("citation", "").strip()
                    score = it.get("score", 0.0)

                    st.markdown(f"**Claim {i}:** {claim}")
                    if snippet:
                        st.markdown(f"> {snippet}")
                    if citation:
                        st.caption(f"Source: {citation}   |   overlap score={score:.3f}")
                    st.divider()

        with st.expander("Show original claim bullets (for debugging)", expanded=False):
            cb = meta.get("clinician_bullets", "").strip()
            pb = meta.get("patient_bullets", "").strip()
            if cb:
                st.markdown("**Clinician claim bullets**")
                st.code(cb)
            if pb:
                st.markdown("**Patient claim bullets**")
                st.code(pb)

        with st.expander("Show evidence (citations)", expanded=False):
            refs = meta.get("references", []) or []
            if not refs:
                st.info("No citations returned.")
            else:
                for r in refs:
                    st.write("-", r)

        with st.expander("Show routing + system metadata", expanded=False):
            st.json(
                {
                    "status": meta.get("status"),
                    "source": meta.get("source"),
                    "route": meta.get("route", {}),
                    "zero_hallucination_mode": meta.get("zero_hallucination_mode"),
                    "reason": meta.get("reason", None),
                }
            )

    with col2:
        st.metric("Confidence (retrieval)", f"{confidence:.3f}")
        st.write("**Status:**", meta.get("status", "unknown"))
        st.write("**Source KBs:**", meta.get("source", "Unknown"))

        if meta.get("status") == "abstain":
            st.warning("Abstained due to insufficient evidence under current safety settings.")

    # Save history
    st.session_state.history.append({"role": "user", "content": query.strip()})
    st.session_state.history.append({"role": "assistant", "content": response_text})
