import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
from PIL import Image
import numpy as np

def build_hybrid_brain():
    print("--- Initializing Hybrid AI (OpenCLIP) ---")
    
    # 1. Setup ChromaDB
    client = chromadb.PersistentClient(path="jewelry_hybrid_db")
    
    hybrid_ef = embedding_functions.OpenCLIPEmbeddingFunction()# image to vector conversion
    
    try: client.delete_collection("components_hybrid")
    except: pass
    
    collection = client.create_collection(name="components_hybrid", embedding_function=hybrid_ef)
    
    # 2. Load CSV
    csv_file = "master_library.csv"
    if not os.path.exists(csv_file):
        print("Error: master_library.csv not found.")
        return

    df = pd.read_csv(csv_file)
    print(f"--- Indexing {len(df)} items (Images -> Vectors) ---")
    
    ids = []
    image_data_list = []
    metas = []
    
    for _, row in df.iterrows():
        unique_id = f"{row['Category']}_{row['File_ID']}"
        text_content = f"{row['Style']} {row['Category']} for {row['Shape']}. {row['Description']}"
        
        if os.path.exists(row['Image_Path']):
            try:
                # Open Image & Convert to Array
                img = Image.open(row['Image_Path']).convert('RGB') # Ensure RGB
                img_array = np.array(img)
                
                ids.append(unique_id)
                image_data_list.append(img_array)
                
                # MOVE TEXT TO METADATA
                metas.append({
                    "category": row['Category'],
                    "filename": row['File_ID'],
                    "shape": row['Shape'],
                    "image_path": row['Image_Path'],
                    "cad_path": row['CAD_Path'],
                    "description": text_content
                })
            except Exception as e:
                print(f"Warning: Could not read image {row['Image_Path']}: {e}")
        else:
            print(f"Skipping {unique_id} (Image missing)")

    batch_size = 5
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        print(f"   Indexing batch {i} to {end}...")
        
        try:
            collection.add(
                ids=ids[i:end],
                images=image_data_list[i:end], 
                metadatas=metas[i:end]
            )
        except Exception as e:
            print(f"   [Error] Failed to add batch {i}: {e}")
        
    print("--- Hybrid Knowledge Base Built Successfully! ---")

if __name__ == "__main__":
    build_hybrid_brain()