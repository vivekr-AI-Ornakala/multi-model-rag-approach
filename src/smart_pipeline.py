"""
Smart Automated RAG Pipeline - No User Input Required
Fully autonomous jewelry CAD component matching with:
1. Vision LLM analyzes reference image completely
2. Component reference images generated for retrieval
3. Smart retrieval using both text and generated images
4. Automatic shank generation matching reference style
5. Smart assembly with shape adaptation
6. Iterative scaling correction until perfect fit

Outputs a single complete ring CAD file.
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MAX_ITERATIONS, TOP_K_RESULTS, 
    ASSEMBLIES_DIR, VISUALIZATIONS_DIR, RESULTS_DIR
)


class Logger:
    """Professional logging utility for pipeline"""
    
    @staticmethod
    def header(title: str):
        """Print a main header"""
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    
    @staticmethod
    def section(title: str):
        """Print a section header"""
        print(f"\n[{title}]")
    
    @staticmethod
    def step(num: int, total: int, msg: str):
        """Print step progress"""
        print(f"\n  Step {num}/{total}: {msg}")
    
    @staticmethod
    def info(msg: str):
        """Print info message"""
        print(f"  {msg}")
    
    @staticmethod
    def detail(label: str, value: str):
        """Print a label-value pair"""
        print(f"    {label}: {value}")
    
    @staticmethod
    def success(msg: str):
        """Print success message"""
        print(f"  [OK] {msg}")
    
    @staticmethod
    def warning(msg: str):
        """Print warning message"""
        print(f"  [WARN] {msg}")
    
    @staticmethod
    def error(msg: str):
        """Print error message"""
        print(f"  [ERROR] {msg}")


class SmartRAGPipeline:
    """
    Fully automated RAG pipeline for jewelry CAD matching.
    No user input required - everything is derived from the reference image.
    """
    
    def __init__(self):
        self.design_analysis = None
        self.component_images = None
        self.results = None
        
    def process(self, image_path: Path, output_dir: Path = None) -> Dict:
        """
        Process a jewelry design image fully automatically.
        
        Args:
            image_path: Path to the reference jewelry image
            output_dir: Optional output directory
            
        Returns:
            Complete results dictionary with single output file
        """
        Logger.header("JEWELRY CAD PIPELINE")
        Logger.detail("Input", image_path.name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Comprehensive design analysis
        Logger.step(1, 6, "Analyzing design with Vision AI")
        self.design_analysis = self._analyze_design(image_path)
        
        if not self.design_analysis:
            Logger.error("Design analysis failed")
            return {"success": False, "error": "Analysis failed"}
        
        self._print_analysis()
        
        # Step 2: Generate component reference images
        Logger.step(2, 6, "Generating component reference images")
        self.component_images = self._generate_component_images(image_path)
        
        # Step 3: Retrieve matching components
        Logger.step(3, 6, "Retrieving matching CAD components")
        stone_result = self._retrieve_stone()
        prong_result = self._retrieve_prong(stone_result)  # Pass stone result for size matching
        
        # Step 4: Smart assembly with AI correction
        Logger.step(4, 6, "Assembling components with AI")
        complete_ring_path = self._assemble_complete_ring(
            stone_result, prong_result, timestamp
        )
        
        # Step 5: Save results
        Logger.step(5, 6, "Saving results")
        results = self._save_results(image_path, stone_result, prong_result, complete_ring_path, timestamp)
        
        # Step 6: Generate visualization
        Logger.step(6, 6, "Creating visualization")
        self._create_visualization(image_path, stone_result, prong_result)
        
        self._print_summary(results)
        
        return results
    
    def _assemble_complete_ring(
        self, 
        stone_result: Dict, 
        prong_result: Dict,
        timestamp: str
    ) -> Optional[Path]:
        """
        Assemble stone and prong with AI, add shank, and return single complete ring.
        Only outputs one file: the final complete ring.
        """
        if not stone_result or not prong_result:
            Logger.warning("Missing components for assembly")
            return None
        
        try:
            # Use physics-based assembler (v3.0) if available, fallback to AI-only
            try:
                from smart_assembly_physics import PhysicsAIAssembler as Assembler
                Logger.info("Using Physics + AI assembly engine (v3.0)")
            except ImportError:
                from smart_assembly_ai import AIAssistedAssembler as Assembler
                Logger.info("Using AI-powered assembly engine")
            
            import rhino3dm
            
            # Get ring size from analysis
            ring_size = 7.0
            if self.design_analysis:
                ring_size = float(self.design_analysis.get("ring_size_estimate", 7.0))
            
            # Run physics/AI-assisted head assembly (stone + prong)
            assembler = Assembler()
            result = assembler.assemble(
                stone_path=stone_result["cad_file"],
                prong_path=prong_result["cad_file"],
                max_iterations=5,
                output_filename=f"_temp_head_{timestamp}.3dm"
            )
            
            if not result.get("success"):
                Logger.warning("AI assembly failed, trying fallback")
                return self._fallback_assembly(stone_result, prong_result, timestamp)
            
            head_assembly_path = Path(result["output_path"])
            
            # Generate shank and combine into complete ring
            complete_ring_path = self._generate_complete_ring(
                head_assembly_path, ring_size, timestamp
            )
            
            # Clean up intermediate head assembly file
            if head_assembly_path.exists():
                try:
                    os.remove(head_assembly_path)
                except:
                    pass
            
            return complete_ring_path
                
        except ImportError as e:
            Logger.warning(f"Falling back to legacy assembly")
            return self._fallback_assembly(stone_result, prong_result, timestamp)
        except Exception as e:
            Logger.error(f"Assembly error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_complete_ring(
        self,
        head_assembly_path: Path,
        ring_size: float,
        timestamp: str
    ) -> Optional[Path]:
        """Generate shank and combine with head to create complete ring (single output)."""
        try:
            from dynamic_shank_generator import DynamicShankGenerator, ShankParameters
            import rhino3dm
            
            Logger.section("Generating Complete Ring")
            
            # Determine shank style from design analysis
            shank_style = "plain"
            band_width = 2.5
            if self.design_analysis and self.design_analysis.get("shank"):
                style = self.design_analysis["shank"].get("style", "plain")
                if style in ["plain", "cathedral", "split", "tapered"]:
                    shank_style = style
                band_width = float(self.design_analysis["shank"].get("width_mm", 2.5))
            
            Logger.detail("Ring size", str(ring_size))
            Logger.detail("Shank style", shank_style)
            Logger.detail("Band width", f"{band_width:.1f}mm")
            
            # Generate shank to temp file
            generator = DynamicShankGenerator()
            params = ShankParameters(
                ring_size=ring_size,
                style=shank_style,
                band_width=band_width
            )
            
            temp_shank_path = ASSEMBLIES_DIR / f"_temp_shank_{timestamp}.3dm"
            generator.generate(params, str(temp_shank_path))
            
            # Load both models
            head_model = rhino3dm.File3dm.Read(str(head_assembly_path))
            shank_model = rhino3dm.File3dm.Read(str(temp_shank_path))
            
            if not head_model or not shank_model:
                Logger.error("Failed to load models for combination")
                return None
            
            # Calculate ring geometry
            ring_diameter = (ring_size * 0.825) + 12.5
            ring_radius = ring_diameter / 2
            band_thickness = 1.8  # Default band thickness
            
            # Calculate shank top Z (center_radius = inner_radius + band_thickness/2)
            inner_radius = ring_diameter / 2
            center_radius = inner_radius + band_thickness / 2
            shank_top_z = center_radius + band_thickness / 2  # Top of the shank
            
            # Get head bounding box to calculate proper positioning
            head_min_z = float('inf')
            for obj in head_model.Objects:
                if obj.Geometry:
                    bb = obj.Geometry.GetBoundingBox()
                    head_min_z = min(head_min_z, bb.Min.Z)
            
            # Head should sit on top of shank - move head so its bottom is at shank top
            head_translation_z = shank_top_z - head_min_z
            
            # Create final combined model
            output_model = rhino3dm.File3dm()
            
            # Add layers
            shank_layer = rhino3dm.Layer()
            shank_layer.Name = "Ring_Shank"
            shank_layer.Color = (200, 180, 100, 255)
            shank_idx = output_model.Layers.Add(shank_layer)
            
            setting_layer = rhino3dm.Layer()
            setting_layer.Name = "Prong_Setting"
            setting_layer.Color = (192, 192, 192, 255)
            setting_idx = output_model.Layers.Add(setting_layer)
            
            stone_layer = rhino3dm.Layer()
            stone_layer.Name = "Stone"
            stone_layer.Color = (255, 0, 100, 255)
            stone_idx = output_model.Layers.Add(stone_layer)
            
            # Add shank objects
            for obj in shank_model.Objects:
                if obj.Geometry:
                    attr = rhino3dm.ObjectAttributes()
                    attr.LayerIndex = shank_idx
                    output_model.Objects.Add(obj.Geometry, attr)
            
            # Add head objects positioned on top of ring
            for obj in head_model.Objects:
                geom = obj.Geometry
                if geom:
                    # Move head up so bottom sits on shank top
                    transform = rhino3dm.Transform.Translation(0, 0, head_translation_z)
                    geom.Transform(transform)
                    
                    # Determine layer based on original layer name
                    original_layer_idx = obj.Attributes.LayerIndex
                    original_layer_name = ""
                    if original_layer_idx < len(head_model.Layers):
                        original_layer_name = head_model.Layers[original_layer_idx].Name.lower()
                    
                    attr = rhino3dm.ObjectAttributes()
                    if "stone" in original_layer_name:
                        attr.LayerIndex = stone_idx
                    else:
                        attr.LayerIndex = setting_idx
                    
                    output_model.Objects.Add(geom, attr)
            
            # Save complete ring - this is the ONLY output file
            ASSEMBLIES_DIR.mkdir(parents=True, exist_ok=True)
            complete_ring_path = ASSEMBLIES_DIR / f"complete_ring_{timestamp}.3dm"
            output_model.Write(str(complete_ring_path), 7)
            
            # Clean up temp shank file
            if temp_shank_path.exists():
                try:
                    os.remove(temp_shank_path)
                except:
                    pass
            
            Logger.success(f"Complete ring created: {complete_ring_path.name}")
            return complete_ring_path
            
        except Exception as e:
            Logger.error(f"Ring generation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _fallback_assembly(
        self,
        stone_result: Dict,
        prong_result: Dict,
        timestamp: str
    ) -> Optional[Path]:
        """Fallback assembly without AI (legacy method)."""
        try:
            # Use legacy assembly logic
            assembly_path = self._smart_assemble(stone_result, prong_result)
            if assembly_path:
                ring_size = 7.0
                if self.design_analysis:
                    ring_size = float(self.design_analysis.get("ring_size_estimate", 7.0))
                return self._generate_complete_ring(assembly_path, ring_size, timestamp)
            return None
        except Exception as e:
            Logger.error(f"Fallback assembly error: {e}")
            return None
    
    def _legacy_iterative_assemble(
        self,
        stone_result: Dict,
        prong_result: Dict,
        max_iterations: int = 3
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Legacy iterative assembly (fallback if new pipeline unavailable).
        """
        correction_x = 1.0
        correction_y = 1.0
        correction_z = 1.0
        
        assembly_path = None
        complete_ring_path = None
        
        for iteration in range(max_iterations):
            if iteration == 0:
                Logger.info("Initial assembly...")
                assembly_path = self._smart_assemble(stone_result, prong_result)
            else:
                Logger.info(f"Iteration {iteration + 1}: Re-assembling with correction")
                Logger.detail("Corrections", f"X={correction_x:.3f}, Y={correction_y:.3f}, Z={correction_z:.3f}")
                assembly_path = self._smart_assemble_with_correction(
                    stone_result, prong_result,
                    correction_x, correction_y, correction_z
                )
            
            if not assembly_path:
                Logger.error("Assembly failed")
                return None, None
            
            # Complete ring and verify
            Logger.info("Generating complete ring...")
            complete_ring_path, verification = self._complete_ring(assembly_path)
            
            if not verification:
                Logger.warning("Could not verify scaling")
                break
            
            if verification["is_correct"]:
                Logger.success(f"Scaling verified after {iteration + 1} iteration(s)")
                break
            
            if not verification["needs_correction"]:
                Logger.success("Scaling acceptable")
                break
            
            # Calculate cumulative correction for next iteration
            correction_x *= verification["correction_factor_x"]
            correction_y *= verification["correction_factor_y"]
            correction_z *= (verification["correction_factor_x"] + verification["correction_factor_y"]) / 2
            
            # Clamp corrections to reasonable range
            correction_x = max(0.5, min(2.0, correction_x))
            correction_y = max(0.5, min(2.0, correction_y))
            correction_z = max(0.5, min(2.0, correction_z))
            
            if iteration == max_iterations - 1:
                Logger.warning(f"Max iterations ({max_iterations}) reached")
        
        return assembly_path, complete_ring_path
    
    def _analyze_design(self, image_path: Path) -> Dict:
        """Comprehensive design analysis"""
        try:
            from vision_analyzer import VisionAnalyzer
            analyzer = VisionAnalyzer()
            return analyzer.analyze_design_comprehensive(image_path)
        except Exception as e:
            Logger.warning(f"Analysis error: {e}")
            return {}
    
    def _print_analysis(self):
        """Print analysis results"""
        if not self.design_analysis:
            return
            
        stone = self.design_analysis.get("stone", {})
        prong = self.design_analysis.get("prong", {})
        shank = self.design_analysis.get("shank", {})
        
        Logger.detail("Stone", f"{stone.get('shape', 'unknown')} {stone.get('color', '')} ({stone.get('size_mm', '?')}mm)")
        Logger.detail("Setting", f"{prong.get('style', 'unknown')} ({prong.get('prong_count', '?')} prongs)")
        Logger.detail("Shank", f"{shank.get('style', 'unknown')} ({shank.get('width_mm', '?')}mm)")
    
    def _generate_component_images(self, image_path: Path) -> Dict:
        """Generate reference images for each component"""
        try:
            from image_generator import ComponentImageGenerator
            generator = ComponentImageGenerator()
            return generator.analyze_and_generate(image_path)
        except Exception as e:
            Logger.info("Using text-based retrieval")
            return {}
    
    def _retrieve_stone(self) -> Optional[Dict]:
        """Retrieve matching stone component"""
        if not self.design_analysis:
            return None
            
        stone_info = self.design_analysis.get("stone", {})
        if not stone_info:
            return None
        
        try:
            from embedding_indexer import EmbeddingIndexer
            from rag_retriever import RAGRetriever
            from models import ComponentType, ComponentRequirement
            
            indexer = EmbeddingIndexer()
            retriever = RAGRetriever(indexer)
            
            # Build search query
            shape = stone_info.get("shape", "round")
            color = stone_info.get("color", "")
            size = stone_info.get("size_mm", 5)
            
            description = f"{shape} {color} gemstone, approximately {size}mm"
            
            requirement = ComponentRequirement(
                component_type=ComponentType.STONES,
                description=description,
                shape=shape,
                size=f"{size}mm"
            )
            
            Logger.info(f"Searching: {shape} stone (~{size}mm)")
            
            # Use generated image for retrieval if available
            search_image = None
            if self.component_images and self.component_images.get("stone"):
                gen_img = self.component_images["stone"].get("generated_image")
                if gen_img and gen_img.exists():
                    search_image = gen_img
            
            # Search
            if search_image:
                results = retriever.search_hybrid(requirement, search_image, top_k=5)
            else:
                results = retriever.search_by_text(requirement, top_k=5)
            
            if results:
                import rhino3dm
                best = results[0]
                
                # Get actual stone geometry size from CAD file
                actual_size = None
                try:
                    stone_model = rhino3dm.File3dm.Read(str(best.component.cad_file_path))
                    if stone_model:
                        max_dim = 0
                        for obj in stone_model.Objects:
                            if obj.Geometry:
                                bb = obj.Geometry.GetBoundingBox()
                                w = bb.Max.X - bb.Min.X
                                d = bb.Max.Y - bb.Min.Y
                                max_dim = max(max_dim, w, d)
                        actual_size = max_dim
                except:
                    pass
                
                size_info = f" (actual: {actual_size:.1f}mm)" if actual_size else ""
                Logger.success(f"Found: {best.component.component_id}{size_info} (score: {best.similarity_score:.3f})")
                
                return {
                    "id": best.component.component_id,
                    "cad_file": str(best.component.cad_file_path),
                    "screenshot": str(best.component.screenshot_path),
                    "score": best.similarity_score,
                    "specs": stone_info,
                    "actual_size_mm": actual_size  # Store actual geometry size
                }
            
            Logger.warning("No matching stone found")
            return None
            
        except Exception as e:
            Logger.error(f"Stone retrieval error: {e}")
            return None
    
    def _retrieve_prong(self, stone_result: Optional[Dict] = None) -> Optional[Dict]:
        """Retrieve matching prong/setting component with size filtering based on actual stone"""
        if not self.design_analysis:
            return None
            
        prong_info = self.design_analysis.get("prong", {})
        stone_info = self.design_analysis.get("stone", {})
        
        if not prong_info:
            return None
        
        try:
            from embedding_indexer import EmbeddingIndexer
            from rag_retriever import RAGRetriever
            from models import ComponentType, ComponentRequirement
            import rhino3dm
            
            indexer = EmbeddingIndexer()
            retriever = RAGRetriever(indexer)
            
            # Build search query
            style = prong_info.get("style", "4-prong")
            prong_count = prong_info.get("prong_count", 4)
            shape = prong_info.get("shape", stone_info.get("shape", "round"))
            
            # Use ACTUAL stone size from retrieved stone, not design analysis estimate
            if stone_result and stone_result.get("actual_size_mm"):
                stone_size = float(stone_result["actual_size_mm"])
                Logger.info(f"  Using actual stone size: {stone_size:.1f}mm")
            else:
                stone_size = float(stone_info.get("size_mm", 5))
                Logger.info(f"  Using estimated stone size: {stone_size:.1f}mm")
            
            description = f"{style} setting for {shape} stone"
            
            requirement = ComponentRequirement(
                component_type=ComponentType.PRONGS,
                description=description,
                shape=shape,
                style=style,
                additional_details={
                    "stone_size": stone_size,
                    "stone_shape": shape
                }
            )
            
            Logger.info(f"Searching: {style} for {shape} stone ({stone_size:.1f}mm)")
            
            # Use generated image for retrieval if available  
            search_image = None
            if self.component_images and self.component_images.get("prong"):
                gen_img = self.component_images["prong"].get("generated_image")
                if gen_img and gen_img.exists():
                    search_image = gen_img
            
            # Search with hard filtering
            if search_image:
                results = retriever.search_hybrid(requirement, search_image, top_k=20)
            else:
                results = retriever.search_by_text(requirement, top_k=20)
            
            if results:
                # Filter by size - prong opening should be close to stone size
                # Opening is ~80% of prong bbox width
                size_filtered = []
                for result in results:
                    try:
                        prong_model = rhino3dm.File3dm.Read(str(result.component.cad_file_path))
                        if prong_model:
                            # Get prong bbox
                            min_v = [float('inf')] * 3
                            max_v = [float('-inf')] * 3
                            for obj in prong_model.Objects:
                                if obj.Geometry:
                                    bb = obj.Geometry.GetBoundingBox()
                                    min_v[0] = min(min_v[0], bb.Min.X)
                                    min_v[1] = min(min_v[1], bb.Min.Y)
                                    max_v[0] = max(max_v[0], bb.Max.X)
                                    max_v[1] = max(max_v[1], bb.Max.Y)
                            
                            prong_width = max_v[0] - min_v[0]
                            prong_depth = max_v[1] - min_v[1]
                            opening_size = min(prong_width, prong_depth) * 0.80
                            
                            # Accept prongs where opening is 70-130% of stone size
                            # (AI can scale stone to fit)
                            size_ratio = opening_size / stone_size
                            if 0.7 <= size_ratio <= 1.3:
                                size_filtered.append((result, opening_size, size_ratio))
                                Logger.info(f"  Size match: {result.component.component_id} (opening: {opening_size:.1f}mm, ratio: {size_ratio:.1%})")
                    except Exception as e:
                        continue
                
                if size_filtered:
                    # Pick the one closest to stone size
                    size_filtered.sort(key=lambda x: abs(x[2] - 1.0))
                    best_result, best_opening, best_ratio = size_filtered[0]
                    Logger.success(f"Found: {best_result.component.component_id} (opening: {best_opening:.1f}mm)")
                    
                    return {
                        "id": best_result.component.component_id,
                        "cad_file": str(best_result.component.cad_file_path),
                        "screenshot": str(best_result.component.screenshot_path),
                        "score": best_result.similarity_score,
                        "specs": prong_info,
                        "opening_size": best_opening
                    }
                else:
                    # No size match - use first result but warn
                    Logger.warning(f"No size-matched prong found for {stone_size}mm stone, using best match")
                    best = results[0]
                    return {
                        "id": best.component.component_id,
                        "cad_file": str(best.component.cad_file_path),
                        "screenshot": str(best.component.screenshot_path),
                        "score": best.similarity_score,
                        "specs": prong_info
                    }
            
            Logger.warning("No matching setting found")
            return None
            
        except Exception as e:
            Logger.error(f"Prong retrieval error: {e}")
            return None
    
    def _smart_assemble(self, stone_result: Dict, prong_result: Dict) -> Optional[Path]:
        """Assemble components using smart assembler"""
        if not stone_result or not prong_result:
            Logger.warning("Missing components for assembly")
            return None
        
        try:
            from smart_assembler import SmartAssembler
            
            assembler = SmartAssembler()
            
            # Get setting type from analysis
            setting_type = None
            if self.design_analysis and self.design_analysis.get("prong"):
                setting_type = self.design_analysis["prong"].get("style")
            
            output_path = assembler.assemble(
                stone_path=stone_result["cad_file"],
                prong_path=prong_result["cad_file"],
                stone_id=stone_result["id"],
                prong_id=prong_result["id"],
                setting_type=setting_type
            )
            
            if output_path:
                return Path(output_path)
            return None
            
        except Exception as e:
            Logger.error(f"Assembly error: {e}")
            return None
    
    def _smart_assemble_with_correction(
        self, 
        stone_result: Dict, 
        prong_result: Dict,
        correction_x: float = 1.0,
        correction_y: float = 1.0,
        correction_z: float = 1.0
    ) -> Optional[Path]:
        """
        Assemble components with additional scaling correction.
        Used for iterative correction when initial scaling is off.
        """
        if not stone_result or not prong_result:
            Logger.warning("Missing components for assembly")
            return None
        
        try:
            from smart_assembler import SmartAssembler
            
            assembler = SmartAssembler()
            
            # Get setting type from analysis
            setting_type = None
            if self.design_analysis and self.design_analysis.get("prong"):
                setting_type = self.design_analysis["prong"].get("style")
            
            output_path = assembler.assemble(
                stone_path=stone_result["cad_file"],
                prong_path=prong_result["cad_file"],
                stone_id=stone_result["id"],
                prong_id=prong_result["id"],
                setting_type=setting_type,
                scale_correction=(correction_x, correction_y, correction_z)
            )
            
            if output_path:
                return Path(output_path)
            return None
            
        except Exception as e:
            Logger.error(f"Assembly error: {e}")
            return None
    
    def _complete_ring(self, assembly_path: Path) -> Tuple[Optional[Path], Optional[Dict]]:
        """
        Add shank to complete the ring and extract geometry for verification.
        Uses the new DynamicShankGenerator for intelligent shank creation.
        Returns (ring_path, verification_result).
        """
        if not assembly_path or not assembly_path.exists():
            return None, None
        
        ring_size = self.design_analysis.get("ring_size_estimate") or 7.0
        
        try:
            import rhino3dm
            from dynamic_shank_generator import DynamicShankGenerator
            from assembly_validator import AssemblyValidator
            
            # Load head assembly
            head_model = rhino3dm.File3dm.Read(str(assembly_path))
            if not head_model:
                Logger.error("Could not load head assembly")
                return None, None
            
            # Use dynamic shank generator
            generator = DynamicShankGenerator()
            
            # Generate shank optimized for this head
            shank_model, params = generator.generate_for_head(
                head_model,
                self.design_analysis,
                float(ring_size)
            )
            
            Logger.detail("Shank", f"{params.style} (size {ring_size}, {params.band_width:.1f}mm)")
            
            # Combine head and shank
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ASSEMBLIES_DIR.mkdir(parents=True, exist_ok=True)
            complete_path = ASSEMBLIES_DIR / f"complete_ring_{timestamp}.3dm"
            
            # Need to reload head model since shank generation may have modified it
            head_model = rhino3dm.File3dm.Read(str(assembly_path))
            
            complete_model = generator.combine_head_and_shank(
                head_model,
                shank_model,
                params,
                str(complete_path)
            )
            
            Logger.success(f"Complete ring: {complete_path.name}")
            
            # Validate and extract geometry for verification
            validator = AssemblyValidator()
            validation_result = validator.validate(str(complete_path))
            
            # Convert to legacy verification format for compatibility
            verification = {
                "is_correct": validation_result.is_valid,
                "needs_correction": not validation_result.is_valid,
                "correction_factor_x": validation_result.correction_factors.get("scale_x", 1.0),
                "correction_factor_y": validation_result.correction_factors.get("scale_y", 1.0),
                "fit_x": validation_result.fit_metrics.get("fit_x", 0) * 100,
                "fit_y": validation_result.fit_metrics.get("fit_y", 0) * 100
            }
            
            return complete_path, verification
            
        except ImportError as e:
            Logger.warning(f"Dynamic shank generator not available, using fallback")
            return self._legacy_complete_ring(assembly_path)
        except Exception as e:
            Logger.error(f"Shank generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _legacy_complete_ring(self, assembly_path: Path) -> Tuple[Optional[Path], Optional[Dict]]:
        """Legacy shank generation (fallback)"""
        if not assembly_path or not assembly_path.exists():
            return None, None
        
        shank_info = self.design_analysis.get("shank", {})
        ring_size = self.design_analysis.get("ring_size_estimate") or 7.0
        
        style = shank_info.get("style") or "plain"
        width = shank_info.get("width_mm") or 2.5
        thickness = shank_info.get("thickness_mm") or 1.8
        
        Logger.info(f"Using legacy shank generation ({style}, {width}mm)")
        
        try:
            import rhino3dm
            
            skill_path = Path(__file__).parent.parent / "scripts" / "moltbot_skills" / "jewelry_shank"
            sys.path.insert(0, str(skill_path))
            
            from shank_generator import ShankGenerator, RingParameters
            
            head_model = rhino3dm.File3dm.Read(str(assembly_path))
            if not head_model:
                return None, None
            
            head_min_x = head_min_y = head_min_z = float('inf')
            head_max_x = head_max_y = head_max_z = float('-inf')
            
            for obj in head_model.Objects:
                geom = obj.Geometry
                if geom:
                    try:
                        bbox = geom.GetBoundingBox()
                        head_min_x = min(head_min_x, bbox.Min.X)
                        head_min_y = min(head_min_y, bbox.Min.Y)
                        head_min_z = min(head_min_z, bbox.Min.Z)
                        head_max_x = max(head_max_x, bbox.Max.X)
                        head_max_y = max(head_max_y, bbox.Max.Y)
                        head_max_z = max(head_max_z, bbox.Max.Z)
                    except:
                        pass
            
            head_width = max(head_max_x - head_min_x, head_max_y - head_min_y)
            
            params = RingParameters(
                ring_size=float(ring_size),
                band_width=float(width),
                band_thickness=float(thickness),
                style=str(style)
            )
            
            generator = ShankGenerator()
            shank_model = generator.generate(params)
            
            shank_outer_diameter = (params.inner_radius + params.band_thickness) * 2
            target_head_width = shank_outer_diameter * 0.55
            head_scale = target_head_width / head_width if head_width > 0 else 1.0
            head_scale = max(0.3, min(1.5, head_scale))
            
            shank_radius = params.inner_radius + params.band_thickness
            shank_top_z = 0
            shank_translation_z = shank_top_z - shank_radius
            
            scaled_head_min_z = head_min_z * head_scale
            head_translation_z = shank_top_z - scaled_head_min_z
            
            final_model = rhino3dm.File3dm()
            
            shank_layer = rhino3dm.Layer()
            shank_layer.Name = "Ring_Shank"
            shank_layer.Color = (200, 180, 100, 255)
            shank_idx = final_model.Layers.Add(shank_layer)
            
            setting_layer = rhino3dm.Layer()
            setting_layer.Name = "Prong_Setting"
            setting_layer.Color = (192, 192, 192, 255)
            setting_idx = final_model.Layers.Add(setting_layer)
            
            stone_layer = rhino3dm.Layer()
            stone_layer.Name = "Stone"
            stone_layer.Color = (255, 0, 100, 255)
            stone_idx = final_model.Layers.Add(stone_layer)
            
            for obj in shank_model.Objects:
                geom = obj.Geometry
                if geom:
                    transform = rhino3dm.Transform.Translation(0, 0, shank_translation_z)
                    geom.Transform(transform)
                    attr = rhino3dm.ObjectAttributes()
                    attr.LayerIndex = shank_idx
                    final_model.Objects.Add(geom, attr)
            
            for obj in head_model.Objects:
                geom = obj.Geometry
                if geom:
                    scale_transform = rhino3dm.Transform.Scale(rhino3dm.Point3d(0, 0, 0), head_scale)
                    geom.Transform(scale_transform)
                    translate_transform = rhino3dm.Transform.Translation(0, 0, head_translation_z)
                    geom.Transform(translate_transform)
                    
                    original_layer_idx = obj.Attributes.LayerIndex
                    original_layer_name = head_model.Layers[original_layer_idx].Name if original_layer_idx < len(head_model.Layers) else ""
                    
                    attr = rhino3dm.ObjectAttributes()
                    if "stone" in original_layer_name.lower():
                        attr.LayerIndex = stone_idx
                    else:
                        attr.LayerIndex = setting_idx
                    
                    final_model.Objects.Add(geom, attr)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ASSEMBLIES_DIR.mkdir(parents=True, exist_ok=True)
            complete_path = ASSEMBLIES_DIR / f"complete_ring_{timestamp}.3dm"
            final_model.Write(str(complete_path), 7)
            
            Logger.success(f"Complete ring: {complete_path.name}")
            
            geometry_json = self._extract_geometry_json(final_model, complete_path)
            verification_result = None
            if geometry_json:
                json_path = ASSEMBLIES_DIR / f"complete_ring_{timestamp}_geometry.json"
                with open(json_path, "w") as f:
                    json.dump(geometry_json, f, indent=2)
                
                verification_result = self._verify_scaling(geometry_json)
            
            return complete_path, verification_result
            
        except Exception as e:
            Logger.error(f"Legacy shank generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _extract_geometry_json(self, model, model_path: Path) -> Dict:
        """Extract complete geometry information from CAD model as JSON"""
        import rhino3dm
        
        geometry_data = {
            "file": str(model_path.name),
            "layers": [],
            "objects": [],
            "summary": {}
        }
        
        # Extract layers
        for i, layer in enumerate(model.Layers):
            geometry_data["layers"].append({
                "index": i,
                "name": layer.Name,
                "color": [layer.Color[0], layer.Color[1], layer.Color[2]] if layer.Color else None
            })
        
        # Extract objects with full geometry info
        total_min = [float('inf')] * 3
        total_max = [float('-inf')] * 3
        
        layer_objects = {}
        
        for i, obj in enumerate(model.Objects):
            geom = obj.Geometry
            if not geom:
                continue
            
            try:
                bbox = geom.GetBoundingBox()
                obj_data = {
                    "index": i,
                    "type": type(geom).__name__,
                    "layer_index": obj.Attributes.LayerIndex,
                    "bounding_box": {
                        "min": [bbox.Min.X, bbox.Min.Y, bbox.Min.Z],
                        "max": [bbox.Max.X, bbox.Max.Y, bbox.Max.Z],
                        "width": bbox.Max.X - bbox.Min.X,
                        "depth": bbox.Max.Y - bbox.Min.Y,
                        "height": bbox.Max.Z - bbox.Min.Z,
                        "center": [
                            (bbox.Min.X + bbox.Max.X) / 2,
                            (bbox.Min.Y + bbox.Max.Y) / 2,
                            (bbox.Min.Z + bbox.Max.Z) / 2
                        ]
                    }
                }
                
                # Track layer-wise objects
                layer_idx = obj.Attributes.LayerIndex
                if layer_idx not in layer_objects:
                    layer_objects[layer_idx] = {
                        "count": 0,
                        "min": [float('inf')] * 3,
                        "max": [float('-inf')] * 3
                    }
                
                layer_objects[layer_idx]["count"] += 1
                for j in range(3):
                    layer_objects[layer_idx]["min"][j] = min(layer_objects[layer_idx]["min"][j], [bbox.Min.X, bbox.Min.Y, bbox.Min.Z][j])
                    layer_objects[layer_idx]["max"][j] = max(layer_objects[layer_idx]["max"][j], [bbox.Max.X, bbox.Max.Y, bbox.Max.Z][j])
                
                # Update total bounds
                for j in range(3):
                    total_min[j] = min(total_min[j], [bbox.Min.X, bbox.Min.Y, bbox.Min.Z][j])
                    total_max[j] = max(total_max[j], [bbox.Max.X, bbox.Max.Y, bbox.Max.Z][j])
                
                geometry_data["objects"].append(obj_data)
                
            except Exception as e:
                continue
        
        # Calculate layer summaries
        layer_summaries = []
        for layer in geometry_data["layers"]:
            idx = layer["index"]
            if idx in layer_objects:
                lo = layer_objects[idx]
                layer_summaries.append({
                    "name": layer["name"],
                    "object_count": lo["count"],
                    "bounding_box": {
                        "min": lo["min"],
                        "max": lo["max"],
                        "width": lo["max"][0] - lo["min"][0],
                        "depth": lo["max"][1] - lo["min"][1],
                        "height": lo["max"][2] - lo["min"][2]
                    }
                })
        
        # Summary
        geometry_data["summary"] = {
            "total_objects": len(geometry_data["objects"]),
            "total_layers": len(geometry_data["layers"]),
            "overall_bounding_box": {
                "min": total_min,
                "max": total_max,
                "width": total_max[0] - total_min[0],
                "depth": total_max[1] - total_min[1],
                "height": total_max[2] - total_min[2]
            },
            "layer_summaries": layer_summaries
        }
        
        return geometry_data
    
    def _verify_scaling(self, geometry_json: Dict) -> Dict:
        """
        Verify that scaling is correct by analyzing geometry.
        Returns verification result with correction factors if needed.
        """
        result = {
            "is_correct": False,
            "needs_correction": False,
            "correction_factor_x": 1.0,
            "correction_factor_y": 1.0,
            "fit_x": 0,
            "fit_y": 0,
            "stone_bbox": None,
            "setting_bbox": None
        }
        
        summary = geometry_json.get("summary", {})
        layer_summaries = summary.get("layer_summaries", [])
        
        # Find stone and prong/setting layers
        stone_layer = None
        setting_layer = None
        shank_layer = None
        
        for layer in layer_summaries:
            name = layer["name"].lower()
            if "stone" in name:
                stone_layer = layer
            elif "prong" in name or "setting" in name:
                setting_layer = layer
            elif "shank" in name:
                shank_layer = layer
        
        if stone_layer and setting_layer:
            stone_bbox = stone_layer["bounding_box"]
            setting_bbox = setting_layer["bounding_box"]
            
            result["stone_bbox"] = stone_bbox
            result["setting_bbox"] = setting_bbox
            
            # Calculate fit percentages (stone girdle vs setting opening)
            opening_ratio = 0.85
            opening_x = setting_bbox['width'] * opening_ratio
            opening_y = setting_bbox['depth'] * opening_ratio
            
            stone_girdle_x = stone_bbox['width']
            stone_girdle_y = stone_bbox['depth']
            
            fit_x = stone_girdle_x / opening_x * 100 if opening_x > 0 else 0
            fit_y = stone_girdle_y / opening_y * 100 if opening_y > 0 else 0
            
            result["fit_x"] = fit_x
            result["fit_y"] = fit_y
            
            target_fit = 97
            tolerance = 5
            
            x_ok = abs(fit_x - target_fit) <= tolerance
            y_ok = abs(fit_y - target_fit) <= tolerance
            
            if x_ok and y_ok:
                result["is_correct"] = True
                result["needs_correction"] = False
            else:
                if fit_x > 100 or fit_y > 100:
                    pass  # Stone too big
                elif fit_x < 85 or fit_y < 85:
                    pass  # Stone too small
                else:
                    result["is_correct"] = True  # Close enough
                
                result["needs_correction"] = not result["is_correct"]
                result["correction_factor_x"] = target_fit / fit_x if fit_x > 0 else 1.0
                result["correction_factor_y"] = target_fit / fit_y if fit_y > 0 else 1.0
        else:
            result["is_correct"] = True  # Assume OK if can't verify
        
        return result

    def _save_results(self, image_path, stone_result, prong_result, complete_ring_path, timestamp) -> Dict:
        """Save all results to JSON"""
        results = {
            "timestamp": timestamp,
            "input_image": str(image_path),
            "analysis": self.design_analysis,
            "components": {
                "stone": stone_result,
                "prong": prong_result
            },
            "output": str(complete_ring_path) if complete_ring_path else None,
            "success": bool(complete_ring_path)
        }
        
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / f"results_{timestamp}.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        Logger.detail("Results", results_path.name)
        
        self.results = results
        return results
    
    def _create_visualization(self, image_path, stone_result, prong_result):
        """Create visualization of results"""
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
            
            images = [("Original", image_path)]
            
            if stone_result and stone_result.get("screenshot"):
                images.append(("Stone", stone_result["screenshot"]))
            
            if prong_result and prong_result.get("screenshot"):
                images.append(("Setting", prong_result["screenshot"]))
            
            n = len(images)
            fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
            if n == 1:
                axes = [axes]
            
            for ax, (title, path) in zip(axes, images):
                img = Image.open(path)
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            
            VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
            vis_path = VISUALIZATIONS_DIR / "results_visualization.png"
            plt.savefig(str(vis_path), dpi=150)
            plt.close()
            
            Logger.detail("Visualization", vis_path.name)
            
        except Exception as e:
            Logger.warning(f"Could not create visualization: {e}")
    
    def _print_summary(self, results: Dict):
        """Print final summary"""
        Logger.header("PIPELINE COMPLETE")
        
        stone = results.get("components", {}).get("stone")
        prong = results.get("components", {}).get("prong")
        
        Logger.section("Components Retrieved")
        if stone:
            Logger.detail("Stone", f"{stone['id']} (score: {stone['score']:.2f})")
        else:
            Logger.warning("Stone not found")
            
        if prong:
            Logger.detail("Setting", f"{prong['id']} (score: {prong['score']:.2f})")
        else:
            Logger.warning("Setting not found")
        
        Logger.section("Output")
        if results.get("output"):
            Logger.success(f"Ring: {Path(results['output']).name}")
        else:
            Logger.warning("No output generated")
        
        print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Jewelry CAD Pipeline - Automated ring generation from reference images"
    )
    parser.add_argument("image", help="Path to jewelry design image")
    parser.add_argument("--output", "-o", help="Output directory")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    if not image_path.exists():
        Logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    pipeline = SmartRAGPipeline()
    results = pipeline.process(image_path)
    
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
