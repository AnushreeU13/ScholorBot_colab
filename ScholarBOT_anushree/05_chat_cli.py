"""
05_chat_cli.py

Terminal-based Chat Interface for ScholarBot.
Run this to interact with the bot.
"""

import sys
import importlib.util
from pathlib import Path

# Load Engine
engine_spec = importlib.util.spec_from_file_location("rag", Path(__file__).parent / "04_rag_engine.py")
rag = importlib.util.module_from_spec(engine_spec)
engine_spec.loader.exec_module(rag)

def main():
    print("==================================================")
    print("      Welcome to ScholarBot (TB & Pneumonia)")
    print("==================================================")
    print("Initializing RAG Engine... (this loads models)")
    
    try:
        bot = rag.ScholarBotEngine()
    except Exception as e:
        print(f"[FATAL] Could not initialize engine: {e}")
        return

    print("\n[READY] ScholarBot is listening.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            break
            
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
            
        if not user_input:
            continue

        print("ScholarBot: (Thinking...)")
        
        # Get Response
        response, confidence = bot.generate_response(user_input)
        
        print("-" * 50)
        print(response)
        print("-" * 50)
        print(f"[System Info] Confidence Score: {confidence:.3f}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
