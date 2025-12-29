import streamlit as st
import importlib.util
from pathlib import Path
import os
import shutil

# Page Configuration
st.set_page_config(
    page_title="ScholarBot (Clinical RAG)",
    page_icon=None,
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "has_upload" not in st.session_state:
    st.session_state["has_upload"] = False

# Load Backend (RAG Engine)
engine_spec = importlib.util.spec_from_file_location("rag", Path(__file__).parent / "04_engine.py")
rag = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(rag)

# Load Ingestion Logic
ingest_spec = importlib.util.spec_from_file_location("ingest", Path(__file__).parent / "03_ingest_user.py")
ingest = importlib.util.module_from_spec(ingest_spec)
ingest_spec.loader.exec_module(ingest)

# Initialize Engine
if "engine" not in st.session_state:
    with st.spinner("Initializing RAG Engine..."):
        st.session_state.engine = rag.ScholarBotEngine()

# Sidebar - Configuration & Upload
with st.sidebar:
    st.title("Control Panel")
    
    # 1. Upload Document (Primary Action)
    st.markdown("### 1. Upload Document")
    uploaded_file = st.file_uploader("Add to Knowledge Base (PDF)", type=["pdf"])
    
    if uploaded_file:
        if st.button("Ingest Uploaded Document"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                # Save to temp
                safe_name = "".join([c for c in uploaded_file.name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
                temp_path = Path(safe_name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                ingest.ingest_user_file(temp_path)
                
                # Reload Engine to pick up new index
                st.session_state.engine = rag.ScholarBotEngine()
                
                # Set Upload Flag for Context-Aware Search
                st.session_state["has_upload"] = True
                
                # Cleanup
                if temp_path.exists():
                    os.remove(temp_path)
                    
                st.success(f"Ingested {uploaded_file.name}.")

    st.markdown("---")
    
    # 2. Metadata (Secondary)
    st.caption("**System Info**")
    model_name = st.text_input("Model", value="llama3")
    st.caption(f"Running on: Local Ollama\n\n**Data Sources**:\n- WHO Guidelines\n- User Uploads")

# Main Chat Interface
st.title("ScholarBot Assistant")
st.caption("Expert Clinical AI for TB & Pneumonia")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "confidence" in msg:
            st.caption(f"Confidence: {msg['confidence']:.2f}")

# Chat Input
if prompt := st.chat_input("Ask a clinical question..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass the model_name
            # expects: content, confidence, meta
            force_user_kb = st.session_state.get("has_upload", False)
            # Pass history (excluding current user prompt which is already in query, but logic handles it)
            # We want history of *previous* turns. 
            # Current st.session_state.messages includes the new user prompt at the end (appended at line 68).
            # So pass all of it.
            response, confidence, meta = st.session_state.engine.generate_response(prompt, model_name=model_name, force_user_kb=force_user_kb, history=st.session_state.messages[:-1])
            
            st.markdown(response)
            
            # Show Refined Citation (Flat Layout)
            st.markdown("**Source Document**") # Subtle header
            st.markdown(f"**Title**: {meta.get('title', 'Unknown')}")
            st.markdown(f"**File**: `{meta.get('source', 'Unknown')}`")
            
            refs = meta.get("references", [])
            if refs:
                st.markdown("**References:**")
                for i, r in enumerate(refs):
                    st.markdown(f"{i+1}. {r}")
            
            st.caption(f"Confidence: {confidence:.3f} (Generative: {model_name})")
            
    # Add Assistant Message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "confidence": confidence,
        "meta": meta
    })

# --- ILLUSTRATOR SECTION ---
# Only show if we recently answered something
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]
    
    st.markdown("---")
    st.subheader("🎨 Patient Illustrator (Beta)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Generate Visual Explanation"):
            # Dynamic Load
            illustrator_spec = importlib.util.spec_from_file_location("art", Path(__file__).parent / "07_illustrator.py")
            illustrator = importlib.util.module_from_spec(illustrator_spec)
            illustrator_spec.loader.exec_module(illustrator)
            
            with st.spinner("1. Drafting Visual Concept (Ollama)..."):
                # Use the last answer as context
                visual_prompt = illustrator.generate_visual_prompt(last_msg["content"], model_name=model_name)
                st.info(f"**Prompt**: {visual_prompt}")
            
            with st.spinner("2. Painting Image (Cloud API)..."):
                img, status = illustrator.generate_image(visual_prompt)
                
                if img:
                    st.image(img, caption="Patient Visual (Pollinations.ai)", use_container_width=True)
                else:
                    st.error(f"Generation Failed: {status}")
