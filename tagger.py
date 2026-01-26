import os
import csv
import json
import base64
import time
import requests


LIBRARY_ROOT = r"C:\Users\vivek\Desktop\code space\RAG\cad_library" #main cad file path
API_KEY = "AIzaSyAPv06sqvhgfeHoMYeJVlkKnO6jTWkh8u4"
MODEL_NAME = "gemini-2.5-flash"

def analyze_image(image_path, category):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    if category == "Prongs":
        prompt = """
        Analyze this jewelry setting (head). Return JSON:
        {"Shape": "Stone shape fitted", "Style": "Basket/Peg/Bezel", "Count": "Prong count", "Profile": "High/Low", "Description": "Visual summary"}
        """
    elif category == "Stones":
        prompt = """
        Analyze this gemstone. Return JSON:
        {"Shape": "Gem shape", "Cut": "Brilliant/Step/Mixed", "Description": "Visual summary"}
        """
    else: # Shanks
        prompt = """
        Analyze this ring band. Return JSON:
        {"Style": "Split/Solitaire/Pave", "Profile": "Comfort/Flat", "Description": "Visual summary"}
        """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt + "\nReturn raw JSON only. No markdown."},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
            ]
        }]
    }

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
        if response.status_code == 200:
            text = response.json()['candidates'][0]['content']['parts'][0]['text']
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
    except Exception as e:
        print(f"   [Error] {e}")
    return None

def main():
    output_csv = "master_library.csv"
    print(f"--- Scanning Library: {LIBRARY_ROOT} ---")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # We create a unified schema
        writer.writerow(["Category", "File_ID", "Shape", "Style", "Description", "Image_Path", "CAD_Path"])
        
        subfolders = ["Prongs", "Stones", "Shanks"]
        
        for category in subfolders:
            folder_path = os.path.join(LIBRARY_ROOT, category)
            if not os.path.exists(folder_path):
                print(f"Skipping {category} (Folder not found)")
                continue
                
            files = [x for x in os.listdir(folder_path) if x.lower().endswith('.jpg')]
            print(f"\nProcessing {len(files)} items in {category}...")
            
            for filename in files:
                jpg_path = os.path.join(folder_path, filename)
                cad_name = filename.replace(".jpg", ".3dm")
                cad_path = os.path.join(folder_path, cad_name)
                
                if not os.path.exists(cad_path):
                    print(f"   [Warning] Orphan image found: {filename} (No CAD file). Skipping.")
                    continue
                
                print(f"   Tagging: {filename}...")
                data = analyze_image(jpg_path, category)
                
                if data:
                    writer.writerow([
                        category,
                        cad_name,
                        data.get("Shape", "Any"),
                        data.get("Style", data.get("Cut", "-")),
                        data.get("Description", ""),
                        jpg_path, 
                        cad_path
                    ])
                
                time.sleep(1) 

    print(f"\n--- Done! Database saved to {output_csv} ---")

if __name__ == "__main__":
    main()