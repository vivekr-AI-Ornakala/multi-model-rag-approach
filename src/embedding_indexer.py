"""
Advanced Embedding Indexer for Multi-Modal RAG
Uses state-of-the-art models:
- SigLIP (Google) - Best image-text alignment
- ChromaDB - Production vector database with persistence
- GPU acceleration when available
- Hybrid search: Image embeddings + Text metadata
"""
import hashlib
import json
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import torch
from transformers import SiglipImageProcessor, SiglipTokenizer, SiglipModel
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

from config import (
    PRONGS_CAD_DIR, STONES_CAD_DIR,
    PRONGS_SCREENSHOTS_DIR, STONES_SCREENSHOTS_DIR,
    CHROMA_DB_DIR, CHROMA_MULTIVIEW_DIR, ACTIVE_VECTOR_STORE,
    EMBEDDING_MODEL, BATCH_SIZE, VECTOR_STORE_DIR
)
from models import CADComponent, ComponentType

# Metadata file paths
PRONGS_METADATA_FILE = VECTOR_STORE_DIR / "prongs_metadata.json"
STONES_METADATA_FILE = VECTOR_STORE_DIR / "stones_metadata.json"

# Multi-view metadata paths
MV_METADATA_DIR = VECTOR_STORE_DIR / "multiview"
PRONGS_MV_METADATA_FILE = MV_METADATA_DIR / "prongs_metadata.json"
STONES_MV_METADATA_FILE = MV_METADATA_DIR / "stones_metadata.json"


class EmbeddingIndexer:
    """
    Creates and manages embeddings using SigLIP and ChromaDB
    
    Features:
    - SigLIP for superior image-text alignment
    - ChromaDB for persistent, production-ready vector storage
    - Batch processing for efficiency
    - GPU acceleration
    """
    
    def __init__(self, device: Optional[str] = None):
        # Set device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("💻 Using CPU (GPU not available)")
        
        # Load SigLIP model components separately
        print(f"📦 Loading model: {EMBEDDING_MODEL}")
        self.image_processor = SiglipImageProcessor.from_pretrained(EMBEDDING_MODEL)
        self.tokenizer = SiglipTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = SiglipModel.from_pretrained(EMBEDDING_MODEL).to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Initialize ChromaDB
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB with persistent storage"""
        # Select directory based on active store
        if ACTIVE_VECTOR_STORE == "multiview":
            db_dir = CHROMA_MULTIVIEW_DIR
            collection_suffix = "_multiview"
        else:
            db_dir = CHROMA_DB_DIR
            collection_suffix = ""
        
        self.chroma_client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collections for each component type
        self.collections = {
            ComponentType.PRONGS: self.chroma_client.get_or_create_collection(
                name=f"prongs{collection_suffix}",
                metadata={"hnsw:space": "cosine"}
            ),
            ComponentType.STONES: self.chroma_client.get_or_create_collection(
                name=f"stones{collection_suffix}",
                metadata={"hnsw:space": "cosine"}
            )
        }
        
        # Load CAD parameter databases
        self.prong_params = {}
        self.stone_params = {}
        self._load_cad_params()
    
    def _load_cad_params(self):
        """Load CAD parameter databases for precise matching"""
        prong_params_file = VECTOR_STORE_DIR / "prong_parameters.json"
        stone_params_file = VECTOR_STORE_DIR / "stone_parameters.json"
        
        if prong_params_file.exists():
            import json
            with open(prong_params_file) as f:
                self.prong_params = json.load(f)
        
        if stone_params_file.exists():
            import json
            with open(stone_params_file) as f:
                self.stone_params = json.load(f)
    
    def _get_prong_opening(self, prong_id: str) -> float:
        """Get prong opening diameter from parameter database"""
        params = self.prong_params.get(prong_id, {})
        return params.get('opening_diameter')
    
    def _get_stone_girdle(self, stone_id: str) -> float:
        """Get stone girdle max dimension from parameter database"""
        params = self.stone_params.get(stone_id, {})
        return params.get('girdle_max')
        
        print(f"📊 ChromaDB initialized at: {db_dir} (mode: {ACTIVE_VECTOR_STORE})")
    
    def _load_metadata(self, component_type: ComponentType) -> dict:
        """Load generated metadata for components"""
        if component_type == ComponentType.PRONGS:
            path = PRONGS_METADATA_FILE
        else:
            path = STONES_METADATA_FILE
        
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}
    
    def _get_rich_description(self, component_id: str, component_type: ComponentType, basic_metadata: dict) -> str:
        """Get rich text description from metadata for embedding"""
        all_metadata = self._load_metadata(component_type)
        
        if component_id in all_metadata:
            m = all_metadata[component_id]
            if component_type == ComponentType.PRONGS:
                parts = [f"{component_type.value}"]
                if "prong_type" in m:
                    parts.append(m["prong_type"])
                if "setting_style" in m:
                    parts.append(m["setting_style"])
                if "design_style" in m:
                    parts.append(m["design_style"])
                if "compatible_stone_shapes" in m and isinstance(m["compatible_stone_shapes"], list):
                    parts.append(f"for {', '.join(m['compatible_stone_shapes'])} stones")
                if "features" in m and isinstance(m["features"], list):
                    parts.append(" ".join(m["features"]))
                if "description" in m:
                    parts.append(m["description"])
                return " ".join(parts)
            else:
                parts = [f"{component_type.value}"]
                if "stone_shape" in m:
                    parts.append(m["stone_shape"])
                if "cut_style" in m:
                    parts.append(m["cut_style"])
                if "proportions" in m:
                    parts.append(m["proportions"])
                if "description" in m:
                    parts.append(m["description"])
                return " ".join(parts)
        
        # Fallback to basic metadata
        return f"{component_type.value} {component_id} {' '.join(str(v) for v in basic_metadata.values())}"
    
    @torch.no_grad()
    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get SigLIP embedding for a single image"""
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.vision_model(**inputs)
        # Get pooled output and normalize
        embedding = outputs.pooler_output
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get SigLIP embedding for text"""
        inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        outputs = self.model.text_model(**inputs)
        # Get pooled output and normalize
        embedding = outputs.pooler_output
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()
    
    def _discover_prongs(self) -> list[CADComponent]:
        """Discover all prong CAD files and their multi-view screenshots"""
        components = []
        
        if not PRONGS_SCREENSHOTS_DIR.exists():
            print(f"⚠️ Prongs screenshots directory not found: {PRONGS_SCREENSHOTS_DIR}")
            return components
        
        # Find unique component IDs from perspective screenshots
        seen_ids = set()
        for screenshot_file in PRONGS_SCREENSHOTS_DIR.glob("*_perspective.png"):
            # Extract component ID (remove _perspective suffix)
            component_id = screenshot_file.stem.replace("_perspective", "")
            if component_id in seen_ids:
                continue
            seen_ids.add(component_id)
            
            cad_filename = f"{component_id}.3dm"
            cad_path = PRONGS_CAD_DIR / cad_filename
            
            parts = component_id.replace("_S", "").split("_")
            metadata = {
                "category_1": parts[0] if len(parts) > 0 else "",
                "category_2": parts[1] if len(parts) > 1 else "",
                "item_id": parts[2] if len(parts) > 2 else "",
                "filename": f"{component_id}_perspective.png"
            }
            
            component = CADComponent(
                component_id=component_id,
                component_type=ComponentType.PRONGS,
                cad_file_path=cad_path,
                screenshot_path=PRONGS_SCREENSHOTS_DIR / f"{component_id}_perspective.png",
                metadata=metadata
            )
            components.append(component)
        
        return components
    
    def _discover_stones(self) -> list[CADComponent]:
        """Discover all stone CAD files and their multi-view screenshots"""
        components = []
        
        if not STONES_CAD_DIR.exists():
            print(f"⚠️ Stones CAD directory not found: {STONES_CAD_DIR}")
            return components
        
        # Check for multi-view screenshots first
        seen_ids = set()
        for screenshot_file in STONES_SCREENSHOTS_DIR.glob("*_perspective.png"):
            component_id = screenshot_file.stem.replace("_perspective", "")
            if component_id in seen_ids:
                continue
            seen_ids.add(component_id)
            
            cad_path = STONES_CAD_DIR / f"{component_id}.3dm"
            
            name_parts = component_id.upper().split()
            stone_type = " ".join(name_parts[:-2]) if len(name_parts) > 2 else name_parts[0] if name_parts else ""
            size = " ".join(name_parts[-2:]) if len(name_parts) >= 2 else ""
            
            metadata = {
                "stone_type": stone_type,
                "size": size,
                "full_name": component_id,
                "filename": f"{component_id}_perspective.png"
            }
            
            component = CADComponent(
                component_id=component_id,
                component_type=ComponentType.STONES,
                cad_file_path=cad_path,
                screenshot_path=STONES_SCREENSHOTS_DIR / f"{component_id}_perspective.png",
                metadata=metadata
            )
            components.append(component)
        
        return components
    
    @torch.no_grad()
    def _get_multiview_embedding(self, component_id: str, screenshot_dir: Path) -> Optional[np.ndarray]:
        """Get weighted embedding from all 4 views of a component"""
        # Perspective shows most detail, orthographic views add context
        view_weights = {
            "perspective": 0.50,
            "front": 0.20,
            "right": 0.15,
            "top": 0.15
        }
        
        weighted_sum = None
        total_weight = 0.0
        
        for view, weight in view_weights.items():
            img_path = screenshot_dir / f"{component_id}_{view}.png"
            if img_path.exists():
                try:
                    image = Image.open(img_path).convert("RGB")
                    emb = self._get_image_embedding(image)
                    if weighted_sum is None:
                        weighted_sum = emb * weight
                    else:
                        weighted_sum += emb * weight
                    total_weight += weight
                except Exception:
                    pass
        
        if weighted_sum is None or total_weight == 0:
            return None
        
        # Normalize by actual weights used and then L2 normalize
        avg_embedding = weighted_sum / total_weight
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        return avg_embedding
    
    def index_components(
        self, 
        component_type: Optional[ComponentType] = None, 
        force_reindex: bool = False
    ):
        """Index all components or specific component type"""
        if component_type is None or component_type == ComponentType.PRONGS:
            self._index_component_type(ComponentType.PRONGS, force_reindex)
        
        if component_type is None or component_type == ComponentType.STONES:
            self._index_component_type(ComponentType.STONES, force_reindex)
    
    def _index_component_type(self, component_type: ComponentType, force_reindex: bool):
        """Index components of a specific type"""
        collection = self.collections[component_type]
        
        if component_type == ComponentType.PRONGS:
            components = self._discover_prongs()
        else:
            components = self._discover_stones()
        
        print(f"\n{'='*50}")
        print(f"Indexing {component_type.value.upper()}")
        print(f"{'='*50}")
        print(f"Found {len(components)} components")
        
        if not force_reindex and collection.count() == len(components):
            print(f"✅ Already indexed {collection.count()} components")
            return
        
        # Get collection name with suffix for multiview mode
        if ACTIVE_VECTOR_STORE == "multiview":
            collection_name = f"{component_type.value}_multiview"
        else:
            collection_name = component_type.value
        
        if force_reindex and collection.count() > 0:
            print("🗑️ Clearing existing index...")
            self.chroma_client.delete_collection(collection_name)
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.collections[component_type] = collection
        
        existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
        new_components = [c for c in components if c.component_id not in existing_ids]
        
        if not new_components:
            print("✅ No new components to index")
            return
        
        print(f"📥 Indexing {len(new_components)} new components...")
        
        # Load metadata for rich descriptions
        all_metadata = self._load_metadata(component_type)
        if all_metadata:
            print(f"📝 Found metadata for {len(all_metadata)} components (hybrid mode)")
        else:
            print("⚠️ No metadata found - run 'python metadata_generator.py' for better accuracy")
        
        batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []
        
        # Determine screenshot directory for multi-view
        if component_type == ComponentType.PRONGS:
            screenshot_dir = PRONGS_SCREENSHOTS_DIR
        else:
            screenshot_dir = STONES_SCREENSHOTS_DIR
        
        for component in tqdm(new_components, desc=f"Embedding {component_type.value}"):
            try:
                # Use multi-view averaged embedding
                embedding = self._get_multiview_embedding(component.component_id, screenshot_dir)
                if embedding is None:
                    # Fallback to single perspective image
                    image = Image.open(component.screenshot_path).convert("RGB")
                    embedding = self._get_image_embedding(image)
                
                # Build metadata dict with LLM-generated info if available
                metadata = {
                    "component_id": component.component_id,
                    "cad_file": str(component.cad_file_path),
                    "screenshot": str(component.screenshot_path),
                    **{k: str(v) for k, v in component.metadata.items()}
                }
                
                # Add rich metadata if available
                if component.component_id in all_metadata:
                    rich_meta = all_metadata[component.component_id]
                    if component_type == ComponentType.PRONGS:
                        if "prong_type" in rich_meta:
                            metadata["prong_type"] = rich_meta["prong_type"]
                        if "setting_style" in rich_meta:
                            metadata["setting_style"] = rich_meta["setting_style"]
                        if "compatible_stone_shapes" in rich_meta:
                            metadata["compatible_shapes"] = ",".join(rich_meta["compatible_stone_shapes"]) if isinstance(rich_meta["compatible_stone_shapes"], list) else str(rich_meta["compatible_stone_shapes"])
                        if "design_style" in rich_meta:
                            metadata["design_style"] = rich_meta["design_style"]
                        if "features" in rich_meta:
                            metadata["features"] = ",".join(rich_meta["features"]) if isinstance(rich_meta["features"], list) else str(rich_meta["features"])
                    else:
                        if "stone_shape" in rich_meta:
                            metadata["stone_shape"] = rich_meta["stone_shape"]
                        if "cut_style" in rich_meta:
                            metadata["cut_style"] = rich_meta["cut_style"]
                    if "description" in rich_meta:
                        metadata["description"] = rich_meta["description"]
                
                # Use rich description for document text
                doc_text = self._get_rich_description(component.component_id, component_type, component.metadata)
                
                batch_ids.append(component.component_id)
                batch_embeddings.append(embedding.tolist())
                batch_metadatas.append(metadata)
                batch_documents.append(doc_text)
                
                if len(batch_ids) >= BATCH_SIZE:
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        documents=batch_documents
                    )
                    batch_ids, batch_embeddings, batch_metadatas, batch_documents = [], [], [], []
                    
            except Exception as e:
                print(f"❌ Error processing {component.component_id}: {e}")
        
        if batch_ids:
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents
            )
        
        print(f"✅ Indexed {len(new_components)} {component_type.value} components")
        print(f"📊 Total in collection: {collection.count()}")
    
    def get_collection(self, component_type: ComponentType) -> chromadb.Collection:
        """Get the ChromaDB collection for a component type"""
        return self.collections[component_type]
    
    def get_component_count(self, component_type: ComponentType) -> int:
        """Get the number of indexed components"""
        return self.collections[component_type].count()
    
    def search_by_text(
        self,
        query: str,
        component_type: ComponentType,
        top_k: int = 5,
        exclude_ids: Optional[list[str]] = None
    ) -> list[dict]:
        """Search for components using text query"""
        exclude_ids = exclude_ids or []
        query_embedding = self._get_text_embedding(query)
        collection = self.collections[component_type]
        
        fetch_k = top_k + len(exclude_ids)
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(fetch_k, collection.count()),
            include=["metadatas", "distances", "documents"]
        )
        
        formatted = []
        for i, id_ in enumerate(results["ids"][0]):
            if id_ in exclude_ids:
                continue
            
            formatted.append({
                "id": id_,
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i] if results["documents"] else None
            })
            
            if len(formatted) >= top_k:
                break
        
        return formatted
    
    def search_by_image(
        self,
        image_path: Path,
        component_type: ComponentType,
        top_k: int = 5,
        exclude_ids: Optional[list[str]] = None
    ) -> list[dict]:
        """Search for similar components using an image"""
        exclude_ids = exclude_ids or []
        image = Image.open(image_path).convert("RGB")
        query_embedding = self._get_image_embedding(image)
        
        collection = self.collections[component_type]
        fetch_k = top_k + len(exclude_ids)
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(fetch_k, collection.count()),
            include=["metadatas", "distances", "documents"]
        )
        
        formatted = []
        for i, id_ in enumerate(results["ids"][0]):
            if id_ in exclude_ids:
                continue
            
            formatted.append({
                "id": id_,
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i]
            })
            
            if len(formatted) >= top_k:
                break
        
        return formatted
    
    def hybrid_search(
        self,
        text_query: str,
        image_path: Optional[Path],
        component_type: ComponentType,
        top_k: int = 5,
        text_weight: float = 0.4,
        image_weight: float = 0.6,
        exclude_ids: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Hybrid search combining text and image embeddings
        
        Args:
            text_query: Natural language description
            image_path: Optional reference image
            component_type: Type of component to search
            top_k: Number of results
            text_weight: Weight for text similarity (0-1)
            image_weight: Weight for image similarity (0-1)
            exclude_ids: IDs to exclude from results
        """
        exclude_ids = exclude_ids or []
        collection = self.collections[component_type]
        
        # Get all embeddings for re-ranking
        all_data = collection.get(include=["embeddings", "metadatas", "documents"])
        
        if not all_data["ids"]:
            return []
        
        # Compute text similarity
        text_embedding = self._get_text_embedding(text_query)
        embeddings = np.array(all_data["embeddings"])
        text_similarities = np.dot(embeddings, text_embedding)
        
        # Compute image similarity if provided
        if image_path:
            image = Image.open(image_path).convert("RGB")
            image_embedding = self._get_image_embedding(image)
            image_similarities = np.dot(embeddings, image_embedding)
        else:
            image_similarities = np.zeros(len(embeddings))
            text_weight = 1.0
            image_weight = 0.0
        
        # Combine scores
        combined_scores = text_weight * text_similarities + image_weight * image_similarities
        
        # Smart keyword boosting based on metadata
        keyword_boost = np.zeros(len(embeddings))
        query_lower = text_query.lower().replace("-", " ")  # Normalize hyphens
        
        # Extract key terms from query
        prong_patterns = ["2 prong", "3 prong", "4 prong", "5 prong", "6 prong", "8 prong", "bezel", "channel", "pave", "halo", "cathedral", "tiffany", "basket"]
        shape_patterns = ["round", "oval", "pear", "marquise", "princess", "cushion", "emerald", "heart", "trillion", "baguette", "radiant", "asscher"]
        style_patterns = ["solitaire", "three stone", "vintage", "modern", "classic", "minimalist"]
        
        # Extract requested prong count from query
        requested_prong_count = None
        for i in range(2, 9):
            if f"{i} prong" in query_lower or f"{i}prong" in query_lower:
                requested_prong_count = i
                break
        
        # Shapes that require rectangular basket
        rectangular_shapes = ["radiant", "emerald", "princess", "baguette", "asscher", "square"]
        # Shapes that require round/oval basket
        round_shapes = ["round", "circular"]
        oval_shapes = ["oval", "ellipse"]
        # Special shapes that need exact matching
        special_shapes = ["heart", "pear", "marquise", "trillion"]
        
        # Determine required basket shape from query
        needs_rectangular = any(s in query_lower for s in rectangular_shapes)
        needs_round = any(s in query_lower for s in round_shapes)
        needs_oval = any(s in query_lower for s in oval_shapes)
        # Check for special shapes
        needs_special_shape = None
        for shape in special_shapes:
            if shape in query_lower:
                needs_special_shape = shape
                break
        
        for i, metadata in enumerate(all_data["metadatas"]):
            boost = 0.0
            
            # Prong type matching (high importance)
            prong_type = str(metadata.get("prong_type", "")).lower().replace("-", " ")
            description = str(metadata.get("description", "")).lower().replace("-", " ")
            
            # CRITICAL: Prong count matching (highest priority)
            if requested_prong_count:
                # Check if this component has the right prong count
                has_correct_count = False
                wrong_count = False
                
                count_str = f"{requested_prong_count} prong"
                count_str_alt = f"{requested_prong_count}prong"
                
                if count_str in prong_type or count_str in description:
                    has_correct_count = True
                    boost += 0.40  # Very strong boost for correct prong count
                else:
                    # Check if it has a different prong count (penalize)
                    for other_count in range(2, 9):
                        if other_count != requested_prong_count:
                            other_str = f"{other_count} prong"
                            if other_str in prong_type or other_str in description:
                                wrong_count = True
                                boost -= 0.50  # Strong penalty for wrong prong count
                                break
            
            # General prong type pattern matching
            for pattern in prong_patterns:
                if pattern in query_lower and pattern in prong_type:
                    boost += 0.20  # Boost for prong type match
            
            # Stone shape matching (high importance for prong compatibility)
            compatible = str(metadata.get("compatible_shapes", "")).lower()
            stone_shape = str(metadata.get("stone_shape", "")).lower()
            
            for pattern in shape_patterns:
                if pattern in query_lower:
                    if pattern in compatible:
                        boost += 0.20  # Strong boost for compatible shape match
                    elif pattern in stone_shape:
                        boost += 0.12  # Moderate boost for stone shape match
            
            # Check for basket shape match based on description
            is_rectangular_basket = any(term in description for term in ["rectangular", "square", "corner"])
            is_round_basket = "round" in description or "circular" in description
            
            # Boost/penalize based on basket shape match
            if needs_rectangular:
                if is_rectangular_basket:
                    boost += 0.15  # Boost for matching basket shape
                elif is_round_basket:
                    boost -= 0.20  # Penalize mismatched basket shape
            elif needs_round:
                if is_round_basket:
                    boost += 0.15
                elif is_rectangular_basket:
                    boost -= 0.20
            
            # Special shape matching (heart, pear, marquise, trillion)
            # These shapes need EXACT matches in compatible_stone_shapes
            if needs_special_shape:
                if needs_special_shape in compatible:
                    boost += 0.40  # Very strong boost for exact special shape match
                elif needs_special_shape in description:
                    boost += 0.25  # Strong boost if mentioned in description
                else:
                    # Penalize if looking for special shape but prong doesn't support it
                    boost -= 0.15
            
            # Setting style matching
            setting_style = str(metadata.get("setting_style", "")).lower()
            for pattern in style_patterns:
                if pattern in query_lower and pattern in setting_style:
                    boost += 0.08
            
            # General description keyword overlap
            doc = all_data["documents"][i] if all_data["documents"] else ""
            if doc:
                query_words = set(query_lower.split())
                doc_words = set(doc.lower().split())
                overlap = len(query_words & doc_words)
                boost += overlap * 0.02  # Small general boost
            
            # SIZE-BASED MATCHING for prongs using CAD parameters
            # Extract requested stone size from query (e.g., "10mm" or "approximately 10mm")
            import re
            size_match = re.search(r'(\d+(?:\.\d+)?)\s*mm', query_lower)
            if size_match:
                requested_stone_size = float(size_match.group(1))
                prong_id = all_data["ids"][i]
                
                # Try to get actual opening diameter from parameter database
                prong_opening = self._get_prong_opening(prong_id)
                
                if prong_opening:
                    # Ideal: prong opening = stone girdle * 1.05 (5% clearance)
                    ideal_opening = requested_stone_size * 1.05
                    
                    if prong_opening >= ideal_opening:
                        # Prong can fit the stone without scaling
                        clearance_ratio = prong_opening / requested_stone_size
                        if 1.02 <= clearance_ratio <= 1.15:
                            boost += 0.40  # Perfect fit - minimal scaling needed
                        elif clearance_ratio < 1.30:
                            boost += 0.25  # Good fit
                        else:
                            boost += 0.10  # Prong is oversized but works
                    else:
                        # Prong too small - will need significant scaling
                        scale_needed = ideal_opening / prong_opening
                        if scale_needed < 1.20:
                            boost += 0.15  # Minor scaling OK
                        elif scale_needed < 1.50:
                            boost -= 0.10  # Moderate scaling
                        else:
                            boost -= 0.30  # Excessive scaling needed
                else:
                    # Fall back to bounding box based matching
                    prong_size = metadata.get("prong_size_mm")
                    if prong_size:
                        prong_size = float(prong_size)
                        ideal_prong_size = requested_stone_size * 1.12
                        size_diff_pct = abs(prong_size - ideal_prong_size) / ideal_prong_size
                        
                        if size_diff_pct < 0.15:
                            boost += 0.35
                        elif size_diff_pct < 0.30:
                            boost += 0.20
                        elif size_diff_pct < 0.50:
                            boost += 0.05
                        else:
                            boost -= 0.20
            
            keyword_boost[i] = boost
        
        combined_scores += keyword_boost
        
        # Rank and filter
        ranked_indices = np.argsort(combined_scores)[::-1]
        
        formatted = []
        for idx in ranked_indices:
            id_ = all_data["ids"][idx]
            if id_ in exclude_ids:
                continue
            
            formatted.append({
                "id": id_,
                "similarity": float(combined_scores[idx]),
                "text_similarity": float(text_similarities[idx]),
                "image_similarity": float(image_similarities[idx]) if image_path else 0.0,
                "metadata": all_data["metadatas"][idx],
                "document": all_data["documents"][idx]
            })
            
            if len(formatted) >= top_k:
                break
        
        return formatted
    
    def search_by_metadata_filter(
        self,
        component_type: ComponentType,
        filters: dict,
        top_k: int = 10
    ) -> list[dict]:
        """Search components by metadata filters (e.g., prong_type, stone_shape)"""
        collection = self.collections[component_type]
        
        # Build ChromaDB where filter
        where_clauses = []
        for key, value in filters.items():
            if isinstance(value, list):
                where_clauses.append({key: {"$in": value}})
            else:
                where_clauses.append({key: value})
        
        where = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0] if where_clauses else None
        
        try:
            results = collection.get(
                where=where,
                limit=top_k,
                include=["metadatas", "documents"]
            )
            
            return [
                {
                    "id": results["ids"][i],
                    "metadata": results["metadatas"][i],
                    "document": results["documents"][i] if results["documents"] else None
                }
                for i in range(len(results["ids"]))
            ]
        except Exception as e:
            print(f"⚠️ Filter search failed: {e}")
            return []


def main():
    """CLI to build indices"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build embedding indices for CAD components")
    parser.add_argument("--component", choices=["prongs", "stones", "all"], default="all",
                       help="Component type to index")
    parser.add_argument("--force", action="store_true", help="Force reindex even if exists")
    
    args = parser.parse_args()
    
    indexer = EmbeddingIndexer()
    
    if args.component == "all":
        indexer.index_components(force_reindex=args.force)
    elif args.component == "prongs":
        indexer.index_components(ComponentType.PRONGS, force_reindex=args.force)
    elif args.component == "stones":
        indexer.index_components(ComponentType.STONES, force_reindex=args.force)
    
    print("\n✅ Indexing complete!")


if __name__ == "__main__":
    main()
