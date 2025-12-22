import streamlit as st
import importlib.util
from pathlib import Path
import os
import shutil

# Page Configuration
st.set_page_config(
    page_title="ScholarBot (TB & Pneumonia)",
    page_icon="🩺",
    layout="wide"
)

# Load Backend (RAG Engine)
# Dynamic import to handle the "04_" prefix
engine_spec = importlib.util.spec_from_file_location("rag", Path(__file__).parent / "04_engine.py")
rag = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(rag)

# Load Ingestion Logic for User Uploads
ingest_spec = importlib.util.spec_from_file_location("ingest", Path(__file__).parent / "03_ingest_user.py")
ingest = importlib.util.module_from_spec(ingest_spec)
ingest_spec.loader.exec_module(ingest)

# Initialize Session State
if "engine" not in st.session_state:
    with st.spinner("Initializing RAG Engine..."):
        st.session_state.engine = rag.ScholarBotEngine()
        
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar - Configuration & Upload
with st.sidebar:
    st.title("🩺 Control Panel")
    st.markdown("---")
    
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Add to Knowledge Base (PDF)", type=["pdf"])
    
    st.markdown("---")
    st.header("🧠 AI Brain (Local)")
    model_name = st.text_input("Ollama Model Name", value="llama3")
    st.caption("Ensure Ollama is running locally!")

    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                # Save to temp
                # Save to temp with original name (sanitized)
                safe_name = "".join([c for c in uploaded_file.name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
                temp_path = Path(safe_name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest
                ingest.ingest_user_file(temp_path)
                
                # Reload Engine to pick up new index
                st.session_state.engine = rag.ScholarBotEngine()
                
                # Cleanup
                if temp_path.exists():
                    os.remove(temp_path)
                    
                st.success(f"Ingested {uploaded_file.name}!")

    st.markdown("---")
    st.info("**Scope**: Tuberculosis & Pneumonia\n\n**Data Sources**:\n- WHO Guidelines\n- User Uploads")

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
            response, confidence, meta = st.session_state.engine.generate_response(prompt, model_name=model_name)
            
            st.markdown(response)
            
            # Show Refined Citation
            with st.expander("📚 Source Citation (Refined)", expanded=True):
                st.markdown(f"**Title**: {meta.get('title')}")
                st.markdown(f"**Author**: {meta.get('author')}")
                st.markdown(f"**File**: `{meta.get('source')}`")
            
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
            
            with st.spinner("2. Painting Image (Local GPU)..."):
                # Simple overlay text
                overlay_text = "ScholarBot Visual"
                img, status = illustrator.generate_image(visual_prompt, text_overlay=overlay_text)
                
                if img:
                    st.image(img, caption="Patient Visual (Generated by Diffusers)", use_container_width=True)
                else:
                    st.error(f"Generation Failed: {status}")
