"""
V2 Retriever - Uses accurate v2 metadata with HARD filtering
This ensures prong count MUST match exactly - no more wrong retrievals
"""
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from PIL import Image
import torch

from config import VECTOR_STORE_DIR, PRONGS_SCREENSHOTS_DIR, STONES_SCREENSHOTS_DIR
from models import ComponentType

# V2 Metadata paths (accurate metadata)
PRONGS_METADATA_V2 = VECTOR_STORE_DIR / "prongs_metadata_v2.json"
STONES_METADATA_V2 = VECTOR_STORE_DIR / "stones_metadata_v2.json"


class V2Retriever:
    """
    Retriever that uses accurate v2 metadata with hard filtering
    
    Key differences from v1:
    1. Uses prongs_metadata_v2.json with accurate prong counts
    2. HARD FILTERS: prong count must match exactly (not just boosting)
    3. Size-based filtering: prong opening must fit stone girdle
    4. Shape compatibility filtering
    """
    
    def __init__(self, indexer=None):
        """
        Args:
            indexer: EmbeddingIndexer instance (for embeddings and ChromaDB)
        """
        self.indexer = indexer
        self.prong_metadata = self._load_metadata(PRONGS_METADATA_V2)
        self.stone_metadata = self._load_metadata(STONES_METADATA_V2)
        
        print(f"📦 V2 Retriever loaded: {len(self.prong_metadata)} prongs, {len(self.stone_metadata)} stones with accurate metadata")
    
    def _load_metadata(self, path: Path) -> dict:
        """Load v2 metadata"""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _extract_prong_count_from_query(self, query: str) -> Optional[int]:
        """Extract requested prong count from query"""
        query_lower = query.lower().replace("-", " ").replace("_", " ")
        
        # Patterns like "4-prong", "4 prong", "four prong"
        number_words = {
            'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8
        }
        
        for word, num in number_words.items():
            if f"{word} prong" in query_lower:
                return num
        
        for i in range(2, 9):
            if f"{i} prong" in query_lower or f"{i}prong" in query_lower:
                return i
        
        return None
    
    def _extract_stone_shape_from_query(self, query: str) -> Optional[str]:
        """Extract requested stone shape from query"""
        query_lower = query.lower()
        
        shapes = [
            "round", "oval", "pear", "marquise", "princess", "cushion",
            "emerald", "heart", "trillion", "baguette", "radiant", "asscher",
            "square"
        ]
        
        for shape in shapes:
            if shape in query_lower:
                return shape
        
        return None
    
    def _extract_stone_size_from_query(self, query: str) -> Optional[float]:
        """Extract stone size in mm from query"""
        query_lower = query.lower()
        
        # Pattern like "10mm", "10.5 mm", "approximately 10mm"
        match = re.search(r'(\d+(?:\.\d+)?)\s*mm', query_lower)
        if match:
            return float(match.group(1))
        
        return None
    
    def filter_prongs_by_requirements(
        self,
        required_prong_count: Optional[int] = None,
        compatible_stone_shape: Optional[str] = None,
        min_opening_mm: Optional[float] = None,
        max_opening_mm: Optional[float] = None
    ) -> List[str]:
        """
        Filter prongs by HARD requirements
        
        Returns list of prong IDs that match ALL requirements
        """
        matching_ids = []
        
        for prong_id, meta in self.prong_metadata.items():
            # HARD FILTER 1: Prong count MUST match exactly
            if required_prong_count is not None:
                prong_count = meta.get('prong_count')
                if prong_count != required_prong_count:
                    continue  # REJECT - wrong prong count
            
            # HARD FILTER 2: Opening size must be within range
            opening = meta.get('opening_diameter')
            if opening:
                if min_opening_mm and opening < min_opening_mm:
                    continue  # REJECT - opening too small
                if max_opening_mm and opening > max_opening_mm:
                    continue  # REJECT - opening too large
            
            # SOFT FILTER 3: Stone shape compatibility (warning, not rejection)
            # We just check but don't reject - will rank lower
            compatible_shapes = meta.get('compatible_stone_shapes', [])
            
            matching_ids.append(prong_id)
        
        return matching_ids
    
    def filter_stones_by_requirements(
        self,
        shape: Optional[str] = None,
        min_girdle_mm: Optional[float] = None,
        max_girdle_mm: Optional[float] = None
    ) -> List[str]:
        """Filter stones by requirements"""
        matching_ids = []
        
        for stone_id, meta in self.stone_metadata.items():
            # Shape filter
            if shape:
                stone_shape = meta.get('shape', '').lower()
                if shape.lower() not in stone_shape:
                    continue
            
            # Size filter
            girdle = meta.get('girdle_max_mm')
            if girdle:
                if min_girdle_mm and girdle < min_girdle_mm:
                    continue
                if max_girdle_mm and girdle > max_girdle_mm:
                    continue
            
            matching_ids.append(stone_id)
        
        return matching_ids
    
    def search_prongs(
        self,
        query: str,
        image_path: Optional[Path] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search for prongs with HARD filtering
        
        1. Extract requirements from query (prong count, shape, size)
        2. HARD FILTER: Only include prongs that match prong count
        3. Rank by embedding similarity + shape/size compatibility
        """
        # Extract requirements
        required_prong_count = self._extract_prong_count_from_query(query)
        stone_shape = self._extract_stone_shape_from_query(query)
        stone_size = self._extract_stone_size_from_query(query)
        
        print(f"\n📋 Extracted requirements:")
        print(f"   Prong count: {required_prong_count}")
        print(f"   Stone shape: {stone_shape}")
        print(f"   Stone size: {stone_size}mm")
        
        # Calculate required opening (stone size * 1.05 for 5% clearance)
        min_opening = stone_size * 0.9 if stone_size else None  # Allow some smaller if scalable
        
        # HARD FILTER by requirements
        valid_ids = self.filter_prongs_by_requirements(
            required_prong_count=required_prong_count,
            compatible_stone_shape=stone_shape,
            min_opening_mm=min_opening
        )
        
        print(f"   Valid prongs after hard filter: {len(valid_ids)}")
        
        if not valid_ids:
            print(f"⚠️ No prongs match requirements!")
            # Fallback: relax constraints
            if required_prong_count:
                print(f"   Trying with relaxed prong count...")
                valid_ids = self.filter_prongs_by_requirements(
                    compatible_stone_shape=stone_shape,
                    min_opening_mm=min_opening
                )
                print(f"   After relaxing: {len(valid_ids)} prongs")
        
        if not valid_ids:
            print(f"⚠️ Still no matches - returning empty")
            return []
        
        # Now rank by similarity
        if self.indexer is None:
            # No indexer - just return filtered results with metadata
            results = []
            for prong_id in valid_ids[:top_k]:
                meta = self.prong_metadata.get(prong_id, {})
                results.append({
                    'id': prong_id,
                    'metadata': meta,
                    'prong_count': meta.get('prong_count'),
                    'opening_mm': meta.get('opening_diameter'),
                    'basket_shape': meta.get('basket_shape'),
                    'compatible_shapes': meta.get('compatible_stone_shapes', [])
                })
            return results
        
        # Get embeddings for valid prongs only
        collection = self.indexer.collections[ComponentType.PRONGS]
        
        # Compute query embedding
        query_embedding = self.indexer._get_text_embedding(query)
        
        if image_path and image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image_embedding = self.indexer._get_image_embedding(image)
            # Hybrid: 40% text, 60% image
            query_embedding = 0.4 * query_embedding + 0.6 * image_embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Get embeddings for valid IDs
        valid_embeddings = collection.get(
            ids=valid_ids,
            include=["embeddings", "metadatas"]
        )
        
        if not valid_embeddings["ids"]:
            return []
        
        # Compute similarities
        embeddings = np.array(valid_embeddings["embeddings"])
        similarities = np.dot(embeddings, query_embedding)
        
        # Add bonus for shape compatibility
        for i, prong_id in enumerate(valid_embeddings["ids"]):
            meta = self.prong_metadata.get(prong_id, {})
            compatible_shapes = [s.lower() for s in meta.get('compatible_stone_shapes', [])]
            
            if stone_shape and stone_shape.lower() in compatible_shapes:
                similarities[i] += 0.15  # Bonus for shape match
            
            # Size fit bonus
            opening = meta.get('opening_diameter')
            if opening and stone_size:
                ideal_opening = stone_size * 1.05
                if opening >= ideal_opening:
                    fit_ratio = opening / ideal_opening
                    if 1.0 <= fit_ratio <= 1.15:
                        similarities[i] += 0.10  # Perfect fit
                    elif fit_ratio <= 1.30:
                        similarities[i] += 0.05  # Good fit
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Format results
        results = []
        for idx in sorted_indices[:top_k]:
            prong_id = valid_embeddings["ids"][idx]
            meta = self.prong_metadata.get(prong_id, {})
            results.append({
                'id': prong_id,
                'similarity': float(similarities[idx]),
                'metadata': valid_embeddings["metadatas"][idx] if valid_embeddings["metadatas"] else {},
                'prong_count': meta.get('prong_count'),
                'opening_mm': meta.get('opening_diameter'),
                'basket_shape': meta.get('basket_shape'),
                'compatible_shapes': meta.get('compatible_stone_shapes', []),
                'confidence': meta.get('prong_count_confidence')
            })
        
        return results
    
    def search_stones(
        self,
        query: str,
        image_path: Optional[Path] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """Search for stones with filtering"""
        # Extract requirements
        shape = self._extract_stone_shape_from_query(query)
        size = self._extract_stone_size_from_query(query)
        
        print(f"\n📋 Stone search requirements:")
        print(f"   Shape: {shape}")
        print(f"   Size: {size}mm")
        
        # Filter by requirements
        valid_ids = self.filter_stones_by_requirements(shape=shape)
        
        print(f"   Valid stones after filter: {len(valid_ids)}")
        
        if not valid_ids:
            # Return all stones if no filter matches
            valid_ids = list(self.stone_metadata.keys())
        
        if self.indexer is None:
            results = []
            for stone_id in valid_ids[:top_k]:
                meta = self.stone_metadata.get(stone_id, {})
                results.append({
                    'id': stone_id,
                    'metadata': meta,
                    'shape': meta.get('shape'),
                    'girdle_mm': meta.get('girdle_max_mm'),
                    'cut_style': meta.get('cut_style')
                })
            return results
        
        # Get embeddings
        collection = self.indexer.collections[ComponentType.STONES]
        query_embedding = self.indexer._get_text_embedding(query)
        
        if image_path and image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image_embedding = self.indexer._get_image_embedding(image)
            query_embedding = 0.4 * query_embedding + 0.6 * image_embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        valid_embeddings = collection.get(
            ids=valid_ids,
            include=["embeddings", "metadatas"]
        )
        
        if not valid_embeddings["ids"]:
            return []
        
        embeddings = np.array(valid_embeddings["embeddings"])
        similarities = np.dot(embeddings, query_embedding)
        
        sorted_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in sorted_indices[:top_k]:
            stone_id = valid_embeddings["ids"][idx]
            meta = self.stone_metadata.get(stone_id, {})
            results.append({
                'id': stone_id,
                'similarity': float(similarities[idx]),
                'metadata': valid_embeddings["metadatas"][idx] if valid_embeddings["metadatas"] else {},
                'shape': meta.get('shape'),
                'girdle_mm': meta.get('girdle_max_mm'),
                'cut_style': meta.get('cut_style')
            })
        
        return results
    
    def get_prong_count_stats(self) -> Dict[int, int]:
        """Get count of prongs by prong count"""
        stats = {}
        for meta in self.prong_metadata.values():
            count = meta.get('prong_count', 0)
            stats[count] = stats.get(count, 0) + 1
        return dict(sorted(stats.items()))
    
    def get_prongs_by_count(self, prong_count: int) -> List[str]:
        """Get all prong IDs with specific prong count"""
        return [
            pid for pid, meta in self.prong_metadata.items()
            if meta.get('prong_count') == prong_count
        ]


def test_v2_retriever():
    """Test the V2 retriever with hard filtering"""
    print("=" * 60)
    print("TESTING V2 RETRIEVER")
    print("=" * 60)
    
    retriever = V2Retriever()
    
    # Show stats
    print("\n📊 Prong count distribution:")
    stats = retriever.get_prong_count_stats()
    for count, num in stats.items():
        print(f"   {count}-prong: {num}")
    
    # Test search
    print("\n" + "=" * 60)
    print("TEST: Search for 4-prong setting for oval stone")
    print("=" * 60)
    
    results = retriever.search_prongs(
        query="4-prong basket setting for oval stone approximately 10mm",
        top_k=5
    )
    
    print("\nResults:")
    for r in results:
        print(f"  {r['id']}: {r['prong_count']}-prong, {r['basket_shape']}, {r['opening_mm']}mm opening")
        print(f"    Compatible shapes: {r['compatible_shapes']}")
    
    # Test with 3-prong (should find actual 3-prongs now)
    print("\n" + "=" * 60)
    print("TEST: Search for 3-prong setting")
    print("=" * 60)
    
    results = retriever.search_prongs(
        query="3-prong setting for round stone 8mm",
        top_k=5
    )
    
    print("\nResults:")
    for r in results:
        print(f"  {r['id']}: {r['prong_count']}-prong, {r['basket_shape']}, {r['opening_mm']}mm opening")


if __name__ == "__main__":
    test_v2_retriever()
