"""
07_illustrator.py

Purpose:
Acts as an "Art Director" for the ScholarBot.
1.  Reads the technical medical answer.
2.  Uses Ollama to write a "Visual Prompt" (e.g. "Cartoon style diagram of...").
3.  Sends this prompt to a Local Stable Diffusion API (Automatic1111).
"""

import requests
import base64
import io
from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Configuration for Local SD
# Standard A1111 URL
SD_URL = "http://127.0.0.1:7860/sdapi/v1/txt2img"

def generate_visual_prompt(medical_answer: str, model_name: str = "llama3") -> str:
    """
    Uses Ollama to convert text -> visual description.
    """
    try:
        llm = ChatOllama(model=model_name, temperature=0.7)
        
        sys_prompt = """You are an expert Medical Illustrator.
        Task: Create a Stable Diffusion text prompt to visualize the provided medical information for a patient.
        
        Guidelines:
        1. VISUAL STYLE: "Vibrant, flat vector art, digital illustration, bright colors, friendly, clean lines, high quality".
        2. Content: Simplify the medical concept into a clear metaphor or diagram (e.g., "lungs glowing with health", "colorful pills").
        3. AVOID: Text, words, letters, labels, gory details, realism.
        4. Focus on making it "Attractive and Colourful".
        5. Output ONLY the prompt string.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "Medical Info: {answer}")
        ])
        
        chain = prompt | llm
        visual_prompt = chain.invoke({"answer": medical_answer[:1000]})
        
        clean_prompt = visual_prompt.content.replace('"', '').strip()
        return clean_prompt

    except Exception as e:
        print(f"[WARN] Failed to generate prompt: {e}")
        return "Medical illustration, vibrant vector art, hospital setting, colorful"

import torch
from diffusers import AutoPipelineForText2Image

# Global Pipeline Cache
_PIPE = None

def _get_pipeline():
    global _PIPE
    if _PIPE is None:
        model_id = "stabilityai/sdxl-turbo"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading Turbo Model: {model_id} on {device}...")
        
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Use AutoPipeline for SDXL Turbo
        _PIPE = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            variant="fp16"
        )
        _PIPE = _PIPE.to(device)
    return _PIPE

def generate_image(visual_prompt: str, text_overlay: str = ""):
    """
    Generates image using SDXL Turbo (1 Step)
    """
    try:
        pipe = _get_pipeline()
        
        # Style Boosters
        full_prompt = f"{visual_prompt}, vibrant colors, vector art, flat design, masterpiece, high resolution, 8k"
        
        # Generate with Turbo Settings
        image = pipe(
            prompt=full_prompt, 
            num_inference_steps=1,    # The "Turbo" magic
            guidance_scale=0.0        # Turbo does not use CFG
        ).images[0]
        
        # Text Overlay
        if text_overlay:
            draw = ImageDraw.Draw(image)
            W, H = image.size
            
            # Font Loading
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Calculate Box
            bbox = draw.textbbox((0, 0), text_overlay, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            padding = 10
            
            # Semi-transparent box at bottom
            overlay = Image.new('RGBA', image.size, (0,0,0,0))
            draw_ov = ImageDraw.Draw(overlay)
            
            rect_h = text_h + (padding * 2)
            rect_y0 = H - rect_h - 20
            rect_y1 = H - 20
            
            # White bubble
            draw_ov.rectangle(
                [(20, rect_y0), (W - 20, rect_y1)], 
                fill=(255, 255, 255, 220), 
                outline=(0, 0, 0, 255)
            )
            
            # Centered Text
            text_x = (W - text_w) / 2
            text_y = rect_y0 + padding
            draw_ov.text((text_x, text_y), text_overlay, fill="black", font=font)
            
            # Composite
            image = Image.alpha_composite(image.convert("RGBA"), overlay)
            
        return image, "Success"

    except ImportError:
        return None, "Missing Deps: pip install torch diffusers transformers accelerate"
    except Exception as e:
        return None, f"Diffusers Error: {str(e)}"
