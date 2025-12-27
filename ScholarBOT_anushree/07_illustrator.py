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
        
        sys_prompt = """You are an expert Medical Textbook Illustrator.
        Task: Create a visual description for a high-precision medical textbook diagram.
        
        Guidelines:
        1. STYLE: "Scientific medical illustration, semi-realistic, textbook quality, clean, detailed, white background".
        2. CONTENT: Focus on anatomical or structural clarity (e.g. cross-section of lung, microscopic view of bacteria).
        3. STRICTLY FORBIDDEN: Any form of text, labels, letters, numbers, or watermarks. The image must be purely visual.
        4. TONE: Serious, clinical, educational.
        5. Output ONLY the visual description string.
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

import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

def generate_image(visual_prompt: str, text_overlay: str = ""):
    """
    Generates image using Pollinations.ai (Cloud API, Free, No GPU needed)
    """
    try:
        # 1. Clean Prompt for URL
        # Pollinations handles spaces, but let's be safe.
        safe_prompt = visual_prompt.replace("\n", " ")
        
        # Style boosters - Textbook Quality & Anti-Text
        full_prompt = f"{safe_prompt}, scientific illustration, highly detailed, anatomical, medical textbook style, 4k, realistic texture, soft lighting, no text, no labels, no watermarks"
        
        # 2. Build URL
        # Docs: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md
        # Format: https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&seed={seed}&nologo=true
        url = f"https://image.pollinations.ai/prompt/{full_prompt}?width=1024&height=1024&nologo=true&state=done"
        
        print(f"[INFO] Fetching from Pollinations: {url}")
        
        # 3. Fetch
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            image = Image.open(image_data).convert("RGBA")
            
            # 4. Text Overlay (Local CPU)
            if text_overlay:
                draw = ImageDraw.Draw(image)
                W, H = image.size
                
                # Font Loading
                try:
                    font = ImageFont.truetype("arial.ttf", 40) # Larger font for 1024px
                except:
                    font = ImageFont.load_default()
                
                # Calculate Box
                bbox = draw.textbbox((0, 0), text_overlay, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                padding = 20
                
                # Semi-transparent box at bottom
                overlay = Image.new('RGBA', image.size, (0,0,0,0))
                draw_ov = ImageDraw.Draw(overlay)
                
                rect_h = text_h + (padding * 2)
                rect_y0 = H - rect_h - 40
                rect_y1 = H - 40
                
                # White bubble
                draw_ov.rectangle(
                    [(40, rect_y0), (W - 40, rect_y1)], 
                    fill=(255, 255, 255, 220), 
                    outline=(0, 0, 0, 255)
                )
                
                # Centered Text
                text_x = (W - text_w) / 2
                text_y = rect_y0 + padding
                draw_ov.text((text_x, text_y), text_overlay, fill="black", font=font)
                
                # Composite
                image = Image.alpha_composite(image, overlay)
                
            return image, "Success"
        else:
            return None, f"Pollinations Error: {response.status_code}"

    except Exception as e:
        return None, f"Error: {str(e)}"
