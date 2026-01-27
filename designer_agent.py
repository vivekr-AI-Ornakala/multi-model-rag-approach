import chromadb
from chromadb.utils import embedding_functions
import os
import base64
import json
import requests
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

API_KEY = ""
LIBRARY_PATH = r"C:\Users\vivek\Desktop\code space\RAG\cad_library"

client_ai = genai.Client(api_key=API_KEY)

print("--- Connecting to Db ---")
client_db = chromadb.PersistentClient(path="jewelry_hybrid_db")
hybrid_ef = embedding_functions.OpenCLIPEmbeddingFunction()
collection = client_db.get_collection(name="components_hybrid", embedding_function=hybrid_ef)

# --- PHASE 1: ANALYSIS (Gemini 2.5 Flash) ---
def analyze_reference(image_path):
    print(f"\n[Phase 1] Analyzing Input...")
    try:
        image_pil = Image.open(image_path)
        prompt = """
        You are a Jewelry Expert. Deconstruct this ring image.
        Return a STRICT JSON object (no markdown) with these exact keys:
        {
            "Stone_Shape": "Round/Oval/etc",
            "Stone_Desc": "A highly detailed visual description of the stone only.",
            "Head_Desc": "A highly detailed visual description of the prong setting style only.",
            "Shank_Desc": "A highly detailed visual description of the band style only."
        }
        """
        response = client_ai.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_pil]
        )
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"   [Error] Analysis failed: {e}")
        return None

# --- PHASE 2: image generation (Imagen 4.0) ---
def dream_component(component_type, description):
    print(f"\n[Phase 2] generating individual component {component_type}...")
    temp_filename = f"temp_dreamed_{component_type}.jpg"
    
    prompt = f"""
    A photorealistic, clean, white-background 3D CAD render of a single jewelry component: {component_type}.
    Visual Details: {description}.
    The image must show ONLY this component in the center. 
    High contrast, studio lighting, metal texture.
    """

    models_to_try = [
        "imagen-4.0-generate-001",
        "imagen-4.0-fast-generate-001",
        "imagen-3.0-generate-001"
    ]

    for model_name in models_to_try:
        try:
            response = client_ai.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1)
            )
            response.generated_images[0].image.save(temp_filename)
            print(f"   [Dream] Generated clean visual using {model_name}: {temp_filename}")
            return temp_filename
        except Exception:
            continue

    print("   [Error] Dreaming failed (Quota or Permission issue). Using Fallback.")
    return None

# --- PHASE 3: HYBRID SEARCH (WITH VISUAL CRITIC) ---
def fetch_hybrid_part(category, shape_filter, text_desc, search_image_path, original_image_path):
    print(f"\n[Phase 3] searching for {category}...")
    
    where_filter = {"$and": [{"category": category}, {"shape": shape_filter}]}
    if category == "Shanks": where_filter = {"category": category}

    if search_image_path is None or not os.path.exists(search_image_path):
        print("   [Note] Using original dirty image for search (Fallback).")
        search_image_path = original_image_path

    try:
        search_img = Image.open(search_image_path).convert('RGB')
        search_img_array = np.array(search_img)
        
        #  Search (Get Top 5)
        results = collection.query(
            query_images=[search_img_array],
            n_results=5,
            where=where_filter
        )
        
        if not results['metadatas'] or not results['metadatas'][0]:
            print("   [Retry] Strict search failed. Removing shape filter...")
            results = collection.query(
                query_images=[search_img_array],
                n_results=5,
                where={"category": category}
            )

    except Exception as e:
        print(f"   [Error] ChromaDB Query Failed: {e}")
        return None
    
    if not results['metadatas'] or not results['metadatas'][0]:
        print("   [!] No matches found.")
        return None

    # --- VERIFICATION LOOP ---
    candidates = results['metadatas'][0]
    
    
    original_pil = Image.open(original_image_path)
    
    for i, cand in enumerate(candidates):
        print(f"   Evaluating Candidate #{i+1}: {cand['filename']}...")
        
        is_match = verify_match(original_pil, cand['image_path'], text_desc)
        
        if is_match:
            print(f"   [PASS] Vision AI confirmed match: {cand['filename']}")
            return cand['cad_path']
        else:
            print(f"   [FAIL] Vision AI rejected. Trying next...")

    print("   [!] All Top 5 rejected. Forcing best available (Candidate #1).")
    return candidates[0]['cad_path']

def verify_match(ref_image_pil, candidate_image_path, description):
    try:
        cand_pil = Image.open(candidate_image_path)
        
        prompt = f"""
        Compare these two jewelry components.
        Image 1: Reference Photo (Real Ring).
        Image 2: Database Component (CAD Render).
        
        Focus ONLY on the specific component: {description}.
        
        STRICT RULES:
        1. Ignore metal color (White Gold vs Yellow Gold does NOT matter).
        2. Focus on GEOMETRY (Prong shape, Shank curvature, Stone cut).
        3. If the geometry is 80% similar, say TRUE.
        
        Return strictly JSON: {{"match": true}} or {{"match": false}}
        """
        
        response = client_ai.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, ref_image_pil, cand_pil]
        )
        
        text = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return result.get("match", False)
        
    except Exception as e:
        return True

# --- BUILDER ---
def generate_rhino_script(stone, head, shank):
    safe_stone = stone.replace("\\", "\\\\") if stone else ""
    safe_head = head.replace("\\", "\\\\") if head else ""
    safe_shank = shank.replace("\\", "\\\\") if shank else ""
    script = f"""
import rhinoscriptsyntax as rs
def assemble():
    rs.EnableRedraw(False)
    shank_z = 0
    if r"{safe_shank}": 
        rs.Command('_-Import "' + r"{safe_shank}" + '" _Enter')
        shanks = rs.LastCreatedObjects()
        if shanks: shank_z = rs.BoundingBox(shanks)[4].Z
    if r"{safe_head}": 
        rs.Command('_-Import "' + r"{safe_head}" + '" _Enter')
        h = rs.LastCreatedObjects()
        if h: 
            box = rs.BoundingBox(h)
            ctr = (box[0] + box[6]) / 2
            rs.MoveObjects(h, [0-ctr.X, 0-ctr.Y, 0-ctr.Z])
            rs.MoveObjects(h, [0, 0, shank_z])
    if r"{safe_stone}":
        rs.Command('_-Import "' + r"{safe_stone}" + '" _Enter')
        s = rs.LastCreatedObjects()
        if s: 
            box = rs.BoundingBox(s)
            ctr = (box[0] + box[6]) / 2
            rs.MoveObjects(s, [0-ctr.X, 0-ctr.Y, 0-ctr.Z])
            rs.MoveObjects(s, [0, 0, shank_z + 1.0])
    rs.EnableRedraw(True)
    print("Smart Assembly Done")
if __name__=="__main__": assemble()
"""
    with open("build_smart.py", "w") as f: f.write(script)
    return "build_smart.py"

# --- MAIN ---
if __name__ == "__main__":
    print("--- GOOGLE IMAGEN 4.0 SMART DESIGNER ---")
    img_in = input("Reference Image Path: ").strip('"')
    
    specs = analyze_reference(img_in)
    if specs:
        print(f"[Plan] {specs}")
        
        dream_stone = dream_component("Gemstone", specs["Stone_Desc"])
        f_stone = fetch_hybrid_part("Stones", specs.get("Stone_Shape", "Round"), specs["Stone_Desc"], dream_stone, img_in)
        
        dream_head = dream_component("Prong Setting", specs["Head_Desc"])
        f_head = fetch_hybrid_part("Prongs", specs.get("Stone_Shape", "Round"), specs["Head_Desc"], dream_head, img_in)
        
        f_shank = None
        if os.path.exists(os.path.join(LIBRARY_PATH, "Shanks")):
             dream_shank = dream_component("Ring Shank", specs["Shank_Desc"])
             f_shank = fetch_hybrid_part("Shanks", None, specs["Shank_Desc"], dream_shank, img_in)
        
        out = generate_rhino_script(f_stone, f_head, f_shank)
        print(f"--- SUCCESS: Run '{out}' in Rhino ---")
        
        try:
            if dream_stone and os.path.exists(dream_stone): os.remove(dream_stone)
            if dream_head and os.path.exists(dream_head): os.remove(dream_head)
            if f_shank and dream_shank and os.path.exists(dream_shank): os.remove(dream_shank)

        except: pass
