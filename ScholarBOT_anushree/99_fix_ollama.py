import requests
import json
import sys

def pull_llama3():
    url = "http://localhost:11434/api/pull"
    payload = {
        "name": "llama3",
        "stream": False
    }
    
    print(f"[INFO] Connecting to Ollama at {url}...")
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            print("[SUCCESS] Model 'llama3' pulled successfully!")
            print(response.json())
        else:
            print(f"[ERROR] Failed to pull model. Status: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"[CRITICAL] Could not connect to Ollama: {e}")
        print("Is the 'ollama' app running in the tray?")

if __name__ == "__main__":
    pull_llama3()
