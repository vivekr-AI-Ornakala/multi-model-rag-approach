"""
RAG Retrieval Engine with ChromaDB
Multi-modal retrieval with support for:
- Text-to-image search (from LLM descriptions)
- Image-to-image search (similarity matching)
- Exclusion of previously rejected components
- V2: Hard filtering by prong count using accurate metadata
"""
import json
import re
from pathlib import Path
from typing import Optional
from PIL import Image

from config import TOP_K_RESULTS, SIMILARITY_THRESHOLD, VECTOR_STORE_DIR
from models import (
    ComponentType, CADComponent, ComponentRequirement,
    RetrievalResult
)
from embedding_indexer import EmbeddingIndexer

# V2 Metadata paths (accurate metadata with correct prong counts)
PRONGS_METADATA_V2 = VECTOR_STORE_DIR / "prongs_metadata_v2.json"
STONES_METADATA_V2 = VECTOR_STORE_DIR / "stones_metadata_v2.json"


class RAGRetriever:
    """Multi-modal RAG retrieval engine using ChromaDB with V2 hard filtering"""
    
    def __init__(self, indexer: Optional[EmbeddingIndexer] = None):
        self.indexer = indexer or EmbeddingIndexer()
        
        # Load V2 metadata for hard filtering
        self.prong_metadata_v2 = self._load_v2_metadata(PRONGS_METADATA_V2)
        self.stone_metadata_v2 = self._load_v2_metadata(STONES_METADATA_V2)
        
        if self.prong_metadata_v2:
            print(f"📦 Loaded V2 metadata: {len(self.prong_metadata_v2)} prongs with accurate counts")
    
    def _load_v2_metadata(self, path: Path) -> dict:
        """Load V2 metadata for hard filtering"""
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _extract_prong_count(self, text: str) -> Optional[int]:
        """Extract prong count from text query. Returns 0 for bezel settings."""
        text_lower = text.lower().replace("-", " ").replace("_", " ")
        
        # Check for bezel (0-prong)
        if 'bezel' in text_lower:
            return 0
        
        # Word to number mapping
        word_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8}
        for word, num in word_map.items():
            if f"{word} prong" in text_lower:
                return num
        
        # Numeric patterns
        for i in range(2, 9):
            if f"{i} prong" in text_lower or f"{i}prong" in text_lower:
                return i
        
        return None
    
    def _extract_basket_shape(self, text: str) -> Optional[str]:
        """Extract required basket/opening shape from text"""
        text_lower = text.lower()
        
        shapes = ['oval', 'round', 'pear', 'marquise', 'rectangular', 'square', 'cushion', 'heart']
        for shape in shapes:
            if shape in text_lower:
                return shape
        return None
    
    def _get_valid_prong_ids(self, required_prong_count: Optional[int], required_shape: Optional[str] = None) -> set:
        """Get set of prong IDs that match requirements (HARD FILTER)"""
        if required_prong_count is None and required_shape is None:
            return None  # No filtering
        
        if not self.prong_metadata_v2:
            return None
        
        valid_ids = set()
        for prong_id, meta in self.prong_metadata_v2.items():
            # Check prong count if specified
            if required_prong_count is not None:
                if meta.get('prong_count') != required_prong_count:
                    continue
            
            # Check shape compatibility if specified
            if required_shape:
                basket_shape = (meta.get('basket_shape') or '').lower()
                compatible_shapes = [s.lower() for s in meta.get('compatible_stone_shapes', [])]
                
                # Match if basket shape or compatible shapes include required shape
                if required_shape not in basket_shape and required_shape not in ' '.join(compatible_shapes):
                    continue
            
            valid_ids.add(prong_id)
        
        return valid_ids
    
    def _build_search_query(self, requirement: ComponentRequirement) -> str:
        """Build a search query string from component requirement"""
        parts = [requirement.description]
        
        if requirement.shape:
            parts.append(f"{requirement.shape} shape")
        if requirement.size:
            parts.append(f"{requirement.size} size")
        if requirement.style:
            parts.append(f"{requirement.style} style")
        
        if requirement.component_type == ComponentType.PRONGS:
            parts.append("jewelry prong setting metal clasp holder")
        elif requirement.component_type == ComponentType.STONES:
            parts.append("gemstone jewel crystal diamond")
        
        return " ".join(parts)
    
    def _result_to_component(self, result: dict, component_type: ComponentType) -> CADComponent:
        """Convert ChromaDB result to CADComponent"""
        metadata = result["metadata"]
        return CADComponent(
            component_id=result["id"],
            component_type=component_type,
            cad_file_path=Path(metadata["cad_file"]),
            screenshot_path=Path(metadata["screenshot"]),
            metadata={k: v for k, v in metadata.items() 
                     if k not in ["cad_file", "screenshot", "component_id"]}
        )
    
    def search_by_text(
        self,
        requirement: ComponentRequirement,
        top_k: int = TOP_K_RESULTS,
        exclude_ids: Optional[list[str]] = None,
        use_hybrid: bool = True,
        reference_image_path: Optional[Path] = None
    ) -> list[RetrievalResult]:
        """Search for components matching a text description
        
        Args:
            requirement: Component requirements
            top_k: Number of results
            exclude_ids: IDs to exclude
            use_hybrid: If True, uses hybrid search with text+image matching
            reference_image_path: Original jewelry image for image-based matching
        """
        exclude_ids = exclude_ids or []
        
        search_text = self._build_search_query(requirement)
        
        # V2: HARD FILTER by prong count AND shape for prong searches
        hard_filter_ids = None
        if requirement.component_type == ComponentType.PRONGS:
            required_prong_count = self._extract_prong_count(search_text)
            required_shape = self._extract_basket_shape(search_text)
            
            if required_prong_count is not None or required_shape:
                hard_filter_ids = self._get_valid_prong_ids(required_prong_count, required_shape)
                filter_desc = []
                if required_prong_count is not None:
                    filter_desc.append(f"{required_prong_count}-prong" if required_prong_count > 0 else "bezel")
                if required_shape:
                    filter_desc.append(f"{required_shape} shape")
                print(f"🎯 Hard filter: {' + '.join(filter_desc)} ({len(hard_filter_ids) if hard_filter_ids else 0} valid)")
        
        if reference_image_path:
            print(f"🔍 Hybrid search (text + image): {search_text[:40]}...")
        else:
            print(f"🔍 Text search: {search_text[:60]}...")
        
        # Combine hard filter IDs with exclude IDs
        all_exclude_ids = list(exclude_ids)
        
        if use_hybrid:
            # Use hybrid search - combine text AND image if available
            results = self.indexer.hybrid_search(
                text_query=search_text,
                image_path=reference_image_path,  # Now uses reference image!
                component_type=requirement.component_type,
                top_k=top_k * 5 if hard_filter_ids else top_k,  # Fetch more if filtering
                text_weight=0.4 if reference_image_path else 0.6,  # More image weight when available
                image_weight=0.6 if reference_image_path else 0.4,
                exclude_ids=all_exclude_ids
            )
        else:
            results = self.indexer.search_by_text(
                query=search_text,
                component_type=requirement.component_type,
                top_k=top_k * 5 if hard_filter_ids else top_k,
                exclude_ids=all_exclude_ids
            )
        
        # V2: Apply hard filter AFTER retrieval
        if hard_filter_ids:
            filtered_results = [r for r in results if r["id"] in hard_filter_ids]
            if filtered_results:
                results = filtered_results
                print(f"   After hard filter: {len(results)} results")
            else:
                # No exact matches - try relaxed filter (prong count only)
                print(f"   ⚠️ No exact matches, relaxing filter...")
                required_prong_count = self._extract_prong_count(search_text)
                if required_prong_count is not None:
                    relaxed_ids = self._get_valid_prong_ids(required_prong_count, None)
                    if relaxed_ids:
                        results = [r for r in results if r["id"] in relaxed_ids]
                        print(f"   After relaxed filter (count only): {len(results)} results")
        
        retrieval_results = []
        for rank, result in enumerate(results, 1):
            if result["similarity"] < SIMILARITY_THRESHOLD:
                continue
                
            component = self._result_to_component(result, requirement.component_type)
            retrieval_results.append(RetrievalResult(
                component=component,
                similarity_score=result["similarity"],
                rank=rank
            ))
            
            if len(retrieval_results) >= top_k:
                break
        
        return retrieval_results
    
    def search_by_image(
        self,
        image_path: Path,
        component_type: ComponentType,
        top_k: int = TOP_K_RESULTS,
        exclude_ids: Optional[list[str]] = None
    ) -> list[RetrievalResult]:
        """Search for similar components using an image"""
        exclude_ids = exclude_ids or []
        
        results = self.indexer.search_by_image(
            image_path=image_path,
            component_type=component_type,
            top_k=top_k,
            exclude_ids=exclude_ids
        )
        
        retrieval_results = []
        for rank, result in enumerate(results, 1):
            component = self._result_to_component(result, component_type)
            retrieval_results.append(RetrievalResult(
                component=component,
                similarity_score=result["similarity"],
                rank=rank
            ))
        
        return retrieval_results
    
    def get_component_image(self, component: CADComponent) -> Image.Image:
        """Load and return the screenshot image for a component"""
        return Image.open(component.screenshot_path)
    
    def get_available_count(self, component_type: ComponentType) -> int:
        """Get the number of available components of a type"""
        return self.indexer.get_component_count(component_type)
    
    def search_hybrid(
        self,
        requirement: ComponentRequirement,
        reference_image_path: Path,
        top_k: int = TOP_K_RESULTS,
        exclude_ids: Optional[list[str]] = None
    ) -> list[RetrievalResult]:
        """
        Hybrid search using both text description AND reference image.
        This is the preferred method when a generated component image is available.
        
        Args:
            requirement: Component requirements (text description)
            reference_image_path: Generated reference image for visual matching
            top_k: Number of results to return
            exclude_ids: IDs to exclude from results
            
        Returns:
            List of matching components ranked by similarity
        """
        return self.search_by_text(
            requirement=requirement,
            top_k=top_k,
            exclude_ids=exclude_ids,
            use_hybrid=True,
            reference_image_path=reference_image_path
        )


class RetrievalSession:
    """
    Manages a retrieval session with exclusion tracking
    Used for iterative refinement when Vision LLM rejects components
    """
    
    def __init__(self, retriever: RAGRetriever, reference_image_path: Optional[Path] = None):
        self.retriever = retriever
        self.reference_image_path = reference_image_path  # Original jewelry image for similarity
        self.excluded_ids: dict[ComponentType, list[str]] = {
            ComponentType.PRONGS: [],
            ComponentType.STONES: []
        }
        self.retrieval_history: list[list[RetrievalResult]] = []
    
    def search(
        self,
        requirement: ComponentRequirement,
        top_k: int = TOP_K_RESULTS
    ) -> list[RetrievalResult]:
        """Search with automatic exclusion of previously rejected components"""
        exclude_ids = self.excluded_ids.get(requirement.component_type, [])
        
        results = self.retriever.search_by_text(
            requirement=requirement,
            top_k=top_k,
            exclude_ids=exclude_ids,
            reference_image_path=self.reference_image_path  # Use reference image!
        )
        
        self.retrieval_history.append(results)
        return results
    
    def reject_component(self, component: CADComponent):
        """Mark a component as rejected (will be excluded in future searches)"""
        self.excluded_ids[component.component_type].append(component.component_id)
    
    def reject_all_results(self, results: list[RetrievalResult]):
        """Reject all components in a result set"""
        for result in results:
            self.reject_component(result.component)
    
    def get_remaining_count(self, component_type: ComponentType) -> int:
        """Get count of remaining (non-excluded) components"""
        total = self.retriever.get_available_count(component_type)
        excluded = len(self.excluded_ids.get(component_type, []))
        return total - excluded
    
    def reset(self, component_type: Optional[ComponentType] = None):
        """Reset exclusions for a component type or all types"""
        if component_type:
            self.excluded_ids[component_type] = []
        else:
            self.excluded_ids = {
                ComponentType.PRONGS: [],
                ComponentType.STONES: []
            }
        self.retrieval_history = []
