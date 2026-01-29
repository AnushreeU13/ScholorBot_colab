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

# --- Chatbot Interface ---
st.divider()

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a clinical question (TB / pneumonia / drug labels supported)"):
    # Add user message
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating answer..."):
            response_text, confidence, meta = engine.generate_response(
                query=query.strip(),
                model_name="",
                force_user_kb=force_user_kb,
                history=st.session_state.history,
            )
        
        # Display Answer
        st.markdown(response_text)
        
        # Add citations cleanly (optional, since debug expanders are removed)
        refs = meta.get("references", [])
        if refs:
            st.caption(f"Sources: {', '.join(refs[:3])}" + ("..." if len(refs)>3 else ""))
        
        # Add metadata footer
        if meta.get("status") == "abstain":
            st.warning("Abstained due to insufficient evidence.")
        else:
            st.caption(f"Confidence: {confidence:.2f} | Status: {meta.get('status')} | Source: {meta.get('source')}")

    # Add assistant message
    st.session_state.history.append({"role": "assistant", "content": response_text})
