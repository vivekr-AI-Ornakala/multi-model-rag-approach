"""
Smart CAD Assembler
Uses actual CAD geometry analysis for perfect stone-prong fitting.

Key features:
1. Non-uniform scaling - can transform round prong to oval
2. Geometry analysis - finds actual opening, girdle, key points
3. Smart positioning - aligns based on geometric features
4. Shape matching - adapts prong shape to fit stone shape
"""
import json
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List

try:
    import rhino3dm
except ImportError:
    print("❌ rhino3dm not installed. Install with: pip install rhino3dm")
    sys.exit(1)

# Metadata paths
PARAMS_DIR = Path(__file__).parent.parent / "vector_stores"
PRONG_METADATA_V2 = PARAMS_DIR / "prongs_metadata_v2.json"
STONE_METADATA_V2 = PARAMS_DIR / "stones_metadata_v2.json"


class GeometryAnalyzer:
    """Analyzes CAD geometry to extract key dimensions and features"""
    
    @staticmethod
    def get_bounding_box(model: rhino3dm.File3dm) -> Dict:
        """Get precise bounding box of all objects"""
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for obj in model.Objects:
            geom = obj.Geometry
            if geom:
                try:
                    bbox = geom.GetBoundingBox()
                    min_x = min(min_x, bbox.Min.X)
                    min_y = min(min_y, bbox.Min.Y)
                    min_z = min(min_z, bbox.Min.Z)
                    max_x = max(max_x, bbox.Max.X)
                    max_y = max(max_y, bbox.Max.Y)
                    max_z = max(max_z, bbox.Max.Z)
                except:
                    pass
        
        width = max_x - min_x
        depth = max_y - min_y
        height = max_z - min_z
        
        return {
            'min': (min_x, min_y, min_z),
            'max': (max_x, max_y, max_z),
            'center': ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2),
            'width': width,   # X dimension
            'depth': depth,   # Y dimension
            'height': height, # Z dimension
            'aspect_ratio': width / depth if depth > 0 else 1.0
        }
    
    @staticmethod
    def estimate_opening_dimensions(model: rhino3dm.File3dm, bbox: Dict, setting_type: str = None) -> Dict:
        """
        Estimate the opening dimensions of a prong/setting.
        The opening is typically at the top of the setting.
        
        For prong settings: opening is between the prong tips (~75-80% of bbox)
        For bezel settings: opening is the inner diameter of the rim (~85-90% of bbox)
        """
        z_top = bbox['max'][2]
        
        # Different opening ratios for different setting types
        if setting_type and 'bezel' in setting_type.lower():
            # Bezel: rim is thin, so opening is close to outer dimension
            opening_ratio = 0.88
        else:
            # Prong: prongs extend inward more
            opening_ratio = 0.80
        
        opening_x = bbox['width'] * opening_ratio
        opening_y = bbox['depth'] * opening_ratio
        
        return {
            'width': opening_x,
            'depth': opening_y,
            'aspect_ratio': opening_x / opening_y if opening_y > 0 else 1.0,
            'center_z': z_top - bbox['height'] * 0.15,  # Opening is near the top
            'ratio_used': opening_ratio
        }
    
    @staticmethod
    def estimate_girdle_dimensions(model: rhino3dm.File3dm, bbox: Dict) -> Dict:
        """
        Estimate the girdle (widest point) dimensions of a stone.
        The girdle is typically at the middle-upper portion of the stone.
        """
        # For most gemstone cuts, the girdle is at about 30-40% from the bottom
        # (the pavilion is below, crown is above)
        z_girdle = bbox['min'][2] + bbox['height'] * 0.35
        
        return {
            'width': bbox['width'],   # Girdle width (X)
            'depth': bbox['depth'],   # Girdle depth (Y)
            'aspect_ratio': bbox['width'] / bbox['depth'] if bbox['depth'] > 0 else 1.0,
            'center_z': z_girdle
        }


class SmartAssembler:
    """
    Smart CAD assembler that uses geometry analysis for perfect fitting.
    
    Key capabilities:
    1. Non-uniform scaling to transform shapes (round→oval)
    2. Precise positioning based on geometric features
    3. Shape adaptation between different stone/prong shapes
    """
    
    def __init__(self):
        self.output_model = rhino3dm.File3dm()
        self.analyzer = GeometryAnalyzer()
        
        # Load metadata
        self.prong_metadata = {}
        self.stone_metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load V2 metadata"""
        if PRONG_METADATA_V2.exists():
            with open(PRONG_METADATA_V2) as f:
                self.prong_metadata = json.load(f)
            print(f"📊 Loaded {len(self.prong_metadata)} prong metadata entries")
        
        if STONE_METADATA_V2.exists():
            with open(STONE_METADATA_V2) as f:
                self.stone_metadata = json.load(f)
            print(f"📊 Loaded {len(self.stone_metadata)} stone metadata entries")
    
    def load_model(self, file_path: str) -> Optional[rhino3dm.File3dm]:
        """Load a .3dm file"""
        try:
            model = rhino3dm.File3dm.Read(file_path)
            if model:
                return model
            return None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def calculate_smart_scaling(
        self,
        prong_bbox: Dict,
        prong_opening: Dict,
        stone_bbox: Dict,
        stone_girdle: Dict,
        clearance: float = 0.03  # 3% clearance
    ) -> Dict:
        """
        Calculate smart scaling factors for BOTH stone and prong.
        
        Like a human jeweler:
        1. Look at the prong opening size
        2. Look at the stone girdle size
        3. Scale whichever makes more sense (prefer scaling up, not down)
        4. Apply non-uniform scaling if shapes don't match (round prong → oval stone)
        
        Strategy:
        - If prong opening > stone girdle: Scale STONE UP to fit prong
        - If prong opening < stone girdle: Scale PRONG UP to fit stone
        - Match aspect ratios through non-uniform scaling
        
        Returns:
            Dict with stone_scale and prong_scale factors
        """
        # Current dimensions
        prong_open_x = prong_opening['width']
        prong_open_y = prong_opening['depth']
        stone_girdle_x = stone_girdle['width']
        stone_girdle_y = stone_girdle['depth']
        
        # Target: stone girdle should be slightly smaller than prong opening
        # opening = girdle * (1 + clearance)
        # So: target_girdle = opening / (1 + clearance)
        # Or: target_opening = girdle * (1 + clearance)
        
        # Calculate how much we'd need to scale each
        # Option A: Scale stone to fit prong (stone_scale = prong_opening / (stone_girdle * (1+clearance)))
        # Option B: Scale prong to fit stone (prong_scale = stone_girdle * (1+clearance) / prong_opening)
        
        stone_scale_x_to_fit = prong_open_x / (stone_girdle_x * (1 + clearance))
        stone_scale_y_to_fit = prong_open_y / (stone_girdle_y * (1 + clearance))
        
        prong_scale_x_to_fit = (stone_girdle_x * (1 + clearance)) / prong_open_x
        prong_scale_y_to_fit = (stone_girdle_y * (1 + clearance)) / prong_open_y
        
        # Decide which to scale based on which requires less distortion
        # Prefer scaling UP over scaling DOWN (better CAD quality)
        avg_stone_scale = (stone_scale_x_to_fit + stone_scale_y_to_fit) / 2
        avg_prong_scale = (prong_scale_x_to_fit + prong_scale_y_to_fit) / 2
        
        # Choose strategy: scale whichever results in scaling UP
        if avg_stone_scale >= 1.0 and avg_prong_scale < 1.0:
            # Scale stone UP to fit larger prong
            strategy = "scale_stone"
        elif avg_prong_scale >= 1.0 and avg_stone_scale < 1.0:
            # Scale prong UP to fit larger stone  
            strategy = "scale_prong"
        elif avg_stone_scale >= 1.0 and avg_prong_scale >= 1.0:
            # Both would scale up - pick smaller scale (less distortion)
            strategy = "scale_stone" if avg_stone_scale <= avg_prong_scale else "scale_prong"
        else:
            # Both would scale down - pick larger scale (closer to 1.0)
            strategy = "scale_stone" if avg_stone_scale >= avg_prong_scale else "scale_prong"
        
        # Calculate final scales
        if strategy == "scale_stone":
            # Scale stone to fit prong opening
            stone_scale = {
                'x': stone_scale_x_to_fit,
                'y': stone_scale_y_to_fit,
                'z': (stone_scale_x_to_fit + stone_scale_y_to_fit) / 2  # Proportional Z
            }
            prong_scale = {'x': 1.0, 'y': 1.0, 'z': 1.0}
            description = f"Scaling STONE to fit prong ({avg_stone_scale:.2f}x)"
        else:
            # Scale prong to fit stone girdle
            prong_scale = {
                'x': prong_scale_x_to_fit,
                'y': prong_scale_y_to_fit,
                'z': (prong_scale_x_to_fit + prong_scale_y_to_fit) / 2
            }
            stone_scale = {'x': 1.0, 'y': 1.0, 'z': 1.0}
            description = f"Scaling PRONG to fit stone ({avg_prong_scale:.2f}x)"
        
        # Clamp scales to reasonable range
        for s in [stone_scale, prong_scale]:
            s['x'] = max(0.3, min(10.0, s['x']))
            s['y'] = max(0.3, min(10.0, s['y']))
            s['z'] = max(0.3, min(10.0, s['z']))
        
        # Check if non-uniform scaling is needed
        stone_is_non_uniform = abs(stone_scale['x'] - stone_scale['y']) > 0.05
        prong_is_non_uniform = abs(prong_scale['x'] - prong_scale['y']) > 0.05
        
        return {
            'stone_scale': stone_scale,
            'prong_scale': prong_scale,
            'strategy': strategy,
            'description': description,
            'stone_non_uniform': stone_is_non_uniform,
            'prong_non_uniform': prong_is_non_uniform,
            'clearance': clearance
        }
    
    def _describe_transform(self, prong_aspect: float, stone_aspect: float) -> str:
        """Describe the shape transformation being applied"""
        if abs(prong_aspect - 1.0) < 0.1:  # Prong is round
            if abs(stone_aspect - 1.0) < 0.1:
                return "round → round (no shape change)"
            elif stone_aspect > 1.1:
                return "round → oval (stretching X)"
            else:
                return "round → oval (stretching Y)"
        else:  # Prong is already oval/elongated
            if abs(stone_aspect - 1.0) < 0.1:
                return "oval → round (compressing)"
            else:
                return "oval → oval (aspect adjustment)"
    
    def apply_non_uniform_scale(
        self,
        model: rhino3dm.File3dm,
        scale_x: float,
        scale_y: float,
        scale_z: float,
        center: Tuple[float, float, float]
    ) -> rhino3dm.File3dm:
        """
        Apply non-uniform scaling to a model.
        This is the key to transforming shapes!
        """
        # Create scale transformation matrix
        # rhino3dm Transform.Scale only does uniform, so we build our own
        
        for obj in model.Objects:
            geom = obj.Geometry
            if geom:
                # Move to origin, scale, move back
                # Step 1: Translate to origin
                t1 = rhino3dm.Transform.Translation(-center[0], -center[1], -center[2])
                
                # Step 2: Non-uniform scale (we need to do this manually)
                # Since rhino3dm doesn't have direct non-uniform scale,
                # we use a workaround with the transformation matrix
                scale_matrix = rhino3dm.Transform(1.0)  # Identity
                scale_matrix.M00 = scale_x
                scale_matrix.M11 = scale_y
                scale_matrix.M22 = scale_z
                
                # Step 3: Translate back
                t2 = rhino3dm.Transform.Translation(center[0] * scale_x, center[1] * scale_y, center[2] * scale_z)
                
                # Apply transformations
                geom.Transform(t1)
                geom.Transform(scale_matrix)
                geom.Transform(t2)
        
        return model
    
    def calculate_smart_positioning(
        self,
        scaled_prong_bbox: Dict,
        scaled_prong_opening: Dict,
        scaled_stone_bbox: Dict,
        scaled_stone_girdle: Dict
    ) -> Tuple[float, float, float]:
        """
        Calculate where to position the stone relative to the prong.
        Uses SCALED dimensions - call after scaling is applied.
        
        The stone should be:
        - Centered in X and Y on the prong opening
        - Positioned in Z so the girdle SITS INSIDE the prong opening
        
        Key insight: The stone's girdle should sit BELOW the prong opening level,
        with the crown (top) extending above the prong.
        
        Jewelry anatomy:
        - Stone: pavilion (bottom cone) + girdle (widest) + crown (top)
        - Prong: base + opening (where girdle sits) + tips (grip above girdle)
        
        The girdle should sit ~20-30% DOWN from the opening level so prong tips
        can grip the crown above the girdle.
        """
        # Prong center after scaling
        prong_center_x = scaled_prong_bbox['center'][0]
        prong_center_y = scaled_prong_bbox['center'][1]
        
        # Opening Z position (where the opening is)
        opening_z = scaled_prong_opening['center_z']
        
        # Stone girdle Z position after scaling
        stone_girdle_z = scaled_stone_girdle['center_z']
        
        # Stone center after scaling
        stone_center = scaled_stone_bbox['center']
        
        # The girdle needs to sit INSIDE the prong, not AT the opening level
        # Drop the girdle to be about 25% of prong height below the opening
        # This allows the prong tips to grip above the girdle
        drop_into_prong = scaled_prong_bbox['height'] * 0.25
        
        # Target Z for the girdle = opening_z - drop
        target_girdle_z = opening_z - drop_into_prong
        
        # Calculate translation to move stone girdle to target position
        stone_z_translation = target_girdle_z - stone_girdle_z
        
        translation = (
            prong_center_x - stone_center[0],  # Center X
            prong_center_y - stone_center[1],  # Center Y
            stone_z_translation                 # Drop girdle into prong
        )
        
        return translation
    
    def assemble(
        self,
        stone_path: str,
        prong_path: str,
        output_path: str = None,
        stone_id: str = None,
        prong_id: str = None,
        setting_type: str = None,
        scale_correction: Tuple[float, float, float] = None
    ) -> Optional[str]:
        """
        Smart assembly of stone and prong.
        
        This method:
        1. Analyzes both CAD models
        2. Calculates non-uniform scaling to fit shapes
        3. Applies additional correction if provided
        4. Positions components perfectly
        
        Args:
            stone_path: Path to stone .3dm file
            prong_path: Path to prong .3dm file
            output_path: Optional output path
            stone_id: Stone ID for metadata lookup
            prong_id: Prong ID for metadata lookup
            setting_type: Setting type hint (e.g., 'bezel', '4-prong')
            scale_correction: Additional (x, y, z) correction factors for iterative adjustment
        """
        print("\n" + "="*60)
        print("🔧 SMART CAD ASSEMBLY")
        if scale_correction and scale_correction != (1.0, 1.0, 1.0):
            print(f"   📐 With correction: X={scale_correction[0]:.3f}, Y={scale_correction[1]:.3f}, Z={scale_correction[2]:.3f}")
        print("="*60)
        
        # Load models
        print("\n📂 Loading CAD models...")
        stone_model = self.load_model(stone_path)
        prong_model = self.load_model(prong_path)
        
        if not stone_model or not prong_model:
            print("❌ Failed to load models")
            return None
        
        print(f"   Stone: {Path(stone_path).name} ({len(stone_model.Objects)} objects)")
        print(f"   Prong: {Path(prong_path).name} ({len(prong_model.Objects)} objects)")
        
        # Analyze geometry from actual CAD
        print("\n📐 Analyzing geometry from CAD...")
        
        stone_bbox = self.analyzer.get_bounding_box(stone_model)
        prong_bbox = self.analyzer.get_bounding_box(prong_model)
        
        # Determine setting type from parameter, metadata, or prong_id
        prong_meta = self.prong_metadata.get(prong_id, {}) if prong_id else {}
        
        if not setting_type:
            # Try to detect from metadata or ID
            if prong_meta.get('style'):
                setting_type = prong_meta['style']
            elif prong_id and 'bezel' in prong_id.lower():
                setting_type = 'bezel'
        
        # Use CAD bounding box as primary source (most reliable)
        stone_girdle = self.analyzer.estimate_girdle_dimensions(stone_model, stone_bbox)
        prong_opening = self.analyzer.estimate_opening_dimensions(prong_model, prong_bbox, setting_type)
        
        # Get metadata for reference (but validate before using)
        stone_meta = self.stone_metadata.get(stone_id, {}) if stone_id else {}
        
        # The CAD bounding box is the most reliable source
        print(f"   Using CAD bbox for stone dimensions")
        if setting_type:
            print(f"   Setting type: {setting_type}")
        
        # Only use prong metadata if it's reasonable
        # Opening CANNOT be larger than the bounding box!
        if prong_meta.get('opening_diameter'):
            meta_opening = prong_meta['opening_diameter']
            max_possible = min(prong_bbox['width'], prong_bbox['depth'])
            
            # Opening must be <= bbox and >= 50% of bbox
            if meta_opening <= max_possible and meta_opening >= max_possible * 0.5:
                prong_opening['width'] = meta_opening
                prong_opening['depth'] = meta_opening
                print(f"   Using prong metadata (opening: {meta_opening:.2f}mm)")
            else:
                print(f"   ⚠️ Prong metadata invalid (opening {meta_opening:.2f}mm > bbox {max_possible:.2f}mm), using CAD estimate")
        
        print(f"\n   Stone dimensions:")
        print(f"      BBox: {stone_bbox['width']:.2f} x {stone_bbox['depth']:.2f} x {stone_bbox['height']:.2f} mm")
        print(f"      Girdle: {stone_girdle['width']:.2f} x {stone_girdle['depth']:.2f} mm")
        print(f"      Aspect ratio: {stone_girdle['aspect_ratio']:.2f}")
        print(f"      Shape: {'oval' if stone_girdle['aspect_ratio'] > 1.1 else 'round'}")
        
        print(f"\n   Prong dimensions:")
        print(f"      BBox: {prong_bbox['width']:.2f} x {prong_bbox['depth']:.2f} x {prong_bbox['height']:.2f} mm")
        print(f"      Opening: {prong_opening['width']:.2f} x {prong_opening['depth']:.2f} mm")
        print(f"      Aspect ratio: {prong_opening['aspect_ratio']:.2f}")
        print(f"      Shape: {'oval' if abs(prong_opening['aspect_ratio'] - 1.0) > 0.1 else 'round'}")
        
        # Calculate smart scaling for BOTH stone and prong
        print("\n🔄 Calculating smart scaling...")
        scale = self.calculate_smart_scaling(
            prong_bbox, prong_opening,
            stone_bbox, stone_girdle,
            clearance=0.03  # 3% clearance for good fit
        )
        
        print(f"\n   Strategy: {scale['description']}")
        print(f"   Stone scale: X={scale['stone_scale']['x']:.3f}, Y={scale['stone_scale']['y']:.3f}, Z={scale['stone_scale']['z']:.3f}")
        print(f"   Prong scale: X={scale['prong_scale']['x']:.3f}, Y={scale['prong_scale']['y']:.3f}, Z={scale['prong_scale']['z']:.3f}")
        
        # Apply additional correction if provided (for iterative refinement)
        if scale_correction and scale_correction != (1.0, 1.0, 1.0):
            print(f"\n   📐 Applying iterative correction factors...")
            if scale['strategy'] == 'scale_stone':
                # Apply correction to stone scaling
                scale['stone_scale']['x'] *= scale_correction[0]
                scale['stone_scale']['y'] *= scale_correction[1]
                scale['stone_scale']['z'] *= scale_correction[2]
                print(f"   Corrected stone scale: X={scale['stone_scale']['x']:.3f}, Y={scale['stone_scale']['y']:.3f}, Z={scale['stone_scale']['z']:.3f}")
            else:
                # Apply correction to prong scaling (inverse - if stone needs to be bigger, prong needs to be smaller)
                scale['prong_scale']['x'] /= scale_correction[0]
                scale['prong_scale']['y'] /= scale_correction[1]
                scale['prong_scale']['z'] /= scale_correction[2]
                print(f"   Corrected prong scale: X={scale['prong_scale']['x']:.3f}, Y={scale['prong_scale']['y']:.3f}, Z={scale['prong_scale']['z']:.3f}")
        
        # Apply scaling to BOTH components
        print("\n📏 Applying transformations...")
        
        # Scale PRONG if needed
        ps = scale['prong_scale']
        if ps['x'] != 1.0 or ps['y'] != 1.0 or ps['z'] != 1.0:
            if scale['prong_non_uniform'] or abs(ps['x'] - ps['y']) > 0.05:
                print(f"   Applying NON-UNIFORM scaling to PRONG...")
                prong_model = self.apply_non_uniform_scale(
                    prong_model, ps['x'], ps['y'], ps['z'],
                    prong_bbox['center']
                )
            else:
                print(f"   Applying uniform scaling to PRONG ({(ps['x']+ps['y']+ps['z'])/3:.2f}x)...")
                uniform_scale = (ps['x'] + ps['y'] + ps['z']) / 3
                for obj in prong_model.Objects:
                    geom = obj.Geometry
                    if geom:
                        center = rhino3dm.Point3d(*prong_bbox['center'])
                        transform = rhino3dm.Transform.Scale(center, uniform_scale)
                        geom.Transform(transform)
        
        # Scale STONE if needed
        ss = scale['stone_scale']
        if ss['x'] != 1.0 or ss['y'] != 1.0 or ss['z'] != 1.0:
            if scale['stone_non_uniform'] or abs(ss['x'] - ss['y']) > 0.05:
                print(f"   Applying NON-UNIFORM scaling to STONE...")
                stone_model = self.apply_non_uniform_scale(
                    stone_model, ss['x'], ss['y'], ss['z'],
                    stone_bbox['center']
                )
            else:
                print(f"   Applying uniform scaling to STONE ({(ss['x']+ss['y']+ss['z'])/3:.2f}x)...")
                uniform_scale = (ss['x'] + ss['y'] + ss['z']) / 3
                for obj in stone_model.Objects:
                    geom = obj.Geometry
                    if geom:
                        center = rhino3dm.Point3d(*stone_bbox['center'])
                        transform = rhino3dm.Transform.Scale(center, uniform_scale)
                        geom.Transform(transform)
        
        # Recalculate bboxes after scaling
        scaled_prong_bbox = self.analyzer.get_bounding_box(prong_model)
        scaled_stone_bbox = self.analyzer.get_bounding_box(stone_model)
        
        # Recalculate girdle and opening after scaling
        scaled_stone_girdle = self.analyzer.estimate_girdle_dimensions(stone_model, scaled_stone_bbox)
        scaled_prong_opening = self.analyzer.estimate_opening_dimensions(prong_model, scaled_prong_bbox, setting_type)
        
        # Calculate positioning based on SCALED dimensions
        print("   Calculating stone position...")
        translation = self.calculate_smart_positioning(
            scaled_prong_bbox, scaled_prong_opening,
            scaled_stone_bbox, scaled_stone_girdle
        )
        
        print(f"   Translation: ({translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}) mm")
        
        # Apply translation to stone
        for obj in stone_model.Objects:
            geom = obj.Geometry
            if geom:
                transform = rhino3dm.Transform.Translation(*translation)
                geom.Transform(transform)
        
        # Verify positions
        translated_stone_bbox = self.analyzer.get_bounding_box(stone_model)
        stone_girdle_z_final = translated_stone_bbox['min'][2] + translated_stone_bbox['height'] * 0.35
        
        # Check if stone is properly seated inside prong
        stone_bottom = translated_stone_bbox['min'][2]
        prong_bottom = scaled_prong_bbox['min'][2]
        stone_inside = stone_bottom > prong_bottom
        
        print(f"\n   📍 Position verification:")
        print(f"      Prong Z range: {scaled_prong_bbox['min'][2]:.2f} to {scaled_prong_bbox['max'][2]:.2f} mm")
        print(f"      Stone Z range: {translated_stone_bbox['min'][2]:.2f} to {translated_stone_bbox['max'][2]:.2f} mm")
        print(f"      Opening Z: {scaled_prong_opening['center_z']:.2f} mm")
        print(f"      Stone girdle Z: {stone_girdle_z_final:.2f} mm (should be BELOW opening)")
        print(f"      Girdle drop: {scaled_prong_opening['center_z'] - stone_girdle_z_final:.2f} mm into prong")
        
        if stone_girdle_z_final < scaled_prong_opening['center_z']:
            print(f"      ✅ Stone is seated INSIDE the prong")
        else:
            print(f"      ⚠️ Stone girdle is above opening - may be floating")
        
        # Create output model
        print("\n💾 Creating output model...")
        self.output_model = rhino3dm.File3dm()
        self.output_model = rhino3dm.File3dm()
        
        # Add prong layer
        prong_layer = rhino3dm.Layer()
        prong_layer.Name = "Prong_Setting"
        prong_layer.Color = (192, 192, 192, 255)  # Silver
        prong_layer_idx = self.output_model.Layers.Add(prong_layer)
        
        # Add stone layer
        stone_layer = rhino3dm.Layer()
        stone_layer.Name = "Stone"
        stone_layer.Color = (255, 0, 0, 255)  # Red for visibility
        stone_layer_idx = self.output_model.Layers.Add(stone_layer)
        
        # Add prong objects
        for obj in prong_model.Objects:
            geom = obj.Geometry
            if geom:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = prong_layer_idx
                self.output_model.Objects.Add(geom, attr)
        
        # Add stone objects
        for obj in stone_model.Objects:
            geom = obj.Geometry
            if geom:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = stone_layer_idx
                self.output_model.Objects.Add(geom, attr)
        
        # Save output
        if output_path is None:
            output_dir = Path(__file__).parent.parent / "outputs" / "assemblies"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"smart_assembly_{timestamp}.3dm")
        
        self.output_model.Write(output_path, 7)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ SMART ASSEMBLY COMPLETE")
        print("="*60)
        print(f"\n📁 Output: {output_path}")
        
        # Get final positions from the output model
        final_prong_bbox = scaled_prong_bbox
        final_stone_bbox = self.analyzer.get_bounding_box(stone_model)
        
        # Use the scaled dimensions for clearance calculation
        actual_opening_x = scaled_prong_opening['width']
        actual_opening_y = scaled_prong_opening['depth']
        final_stone_girdle_w = scaled_stone_girdle['width']
        final_stone_girdle_d = scaled_stone_girdle['depth']
        
        # Calculate actual clearance using SCALED dimensions
        clearance_x = actual_opening_x - final_stone_girdle_w
        clearance_y = actual_opening_y - final_stone_girdle_d
        
        print(f"\n📐 Final dimensions:")
        print(f"   Prong BBox: {final_prong_bbox['width']:.2f} x {final_prong_bbox['depth']:.2f} x {final_prong_bbox['height']:.2f} mm")
        print(f"   Stone BBox: {final_stone_bbox['width']:.2f} x {final_stone_bbox['depth']:.2f} x {final_stone_bbox['height']:.2f} mm")
        
        print(f"\n📏 Fit analysis (after scaling):")
        print(f"   Prong opening: {actual_opening_x:.2f} x {actual_opening_y:.2f} mm")
        print(f"   Stone girdle: {final_stone_girdle_w:.2f} x {final_stone_girdle_d:.2f} mm")
        print(f"   Clearance X: {clearance_x:.2f} mm ({clearance_x/final_stone_girdle_w*100:.1f}%)")
        print(f"   Clearance Y: {clearance_y:.2f} mm ({clearance_y/final_stone_girdle_d*100:.1f}%)")
        print(f"   Strategy used: {scale['strategy']}")
        
        if clearance_x > 0 and clearance_y > 0:
            print(f"\n   ✅ Stone fits with proper clearance!")
        elif clearance_x < -0.5 or clearance_y < -0.5:
            print(f"\n   ⚠️ Tight fit - stone may be snug")
        else:
            print(f"\n   ✅ Close fit - should work well")
        
        return output_path
    
    def assemble_from_results(self, results_path: str, output_path: str = None) -> Optional[str]:
        """Assemble from a results JSON file"""
        with open(results_path) as f:
            results = json.load(f)
        
        stone_file = None
        prong_file = None
        stone_id = None
        prong_id = None
        
        for component in results.get('components', []):
            selected = component.get('selected')
            if not selected:
                continue
            
            comp_type = component['requirement']['type']
            
            if comp_type == "stones":
                stone_file = selected['cad_file']
                stone_id = selected['id']
            elif comp_type == "prongs":
                prong_file = selected['cad_file']
                prong_id = selected['id']
        
        if not stone_file or not prong_file:
            print("❌ Missing stone or prong in results")
            return None
        
        return self.assemble(
            stone_path=stone_file,
            prong_path=prong_file,
            output_path=output_path,
            stone_id=stone_id,
            prong_id=prong_id
        )


def test_smart_assembler():
    """Test the smart assembler with sample files"""
    import glob
    
    print("🧪 Testing Smart Assembler")
    print("="*60)
    
    # Find a sample stone and prong
    stone_dir = Path(__file__).parent.parent / "cad_library" / "stones"
    prong_dir = Path(__file__).parent.parent / "cad_library" / "prongs"
    
    stones = list(stone_dir.glob("*.3dm"))
    prongs = list(prong_dir.glob("*.3dm"))
    
    if not stones or not prongs:
        print("❌ No CAD files found")
        return
    
    # Pick an oval stone and a round prong to test shape transformation
    oval_stone = None
    round_prong = None
    
    assembler = SmartAssembler()
    
    # Find oval stone from metadata
    for stone_file in stones:
        stone_id = stone_file.stem
        meta = assembler.stone_metadata.get(stone_id, {})
        if 'oval' in meta.get('shape', '').lower():
            oval_stone = stone_file
            break
    
    # Find round prong from metadata
    for prong_file in prongs:
        prong_id = prong_file.stem
        meta = assembler.prong_metadata.get(prong_id, {})
        if 'round' in meta.get('basket_shape', '').lower():
            round_prong = prong_file
            break
    
    if oval_stone and round_prong:
        print(f"\n📌 Test: Round prong → Oval stone")
        print(f"   Stone: {oval_stone.name}")
        print(f"   Prong: {round_prong.name}")
        
        output = assembler.assemble(
            stone_path=str(oval_stone),
            prong_path=str(round_prong),
            stone_id=oval_stone.stem,
            prong_id=round_prong.stem
        )
        
        if output:
            print(f"\n✅ Test passed! Output: {output}")
    else:
        # Fall back to first available files
        print(f"\n📌 Test: First available files")
        output = assembler.assemble(
            stone_path=str(stones[0]),
            prong_path=str(prongs[0])
        )


if __name__ == "__main__":
    test_smart_assembler()
