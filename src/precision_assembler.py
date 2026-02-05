"""
Precision CAD Assembler
Assembles jewelry CAD components with high precision:
1. Places prong at origin with correct orientation
2. Precisely analyzes prong opening geometry
3. Scales stone to fit perfectly in prong opening
4. Inserts stone at correct depth with proper positioning

Key improvements over previous assembler:
- Uses vertex-level analysis for accurate opening detection
- Separates scaling logic from positioning
- Supports iterative correction based on validator feedback
- Maintains original geometry quality
"""
import json
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import rhino3dm
except ImportError:
    print("❌ rhino3dm not installed. Install with: pip install rhino3dm")
    raise


# Metadata paths
PARAMS_DIR = Path(__file__).parent.parent / "vector_stores"
PRONG_METADATA_V2 = PARAMS_DIR / "prongs_metadata_v2.json"
STONE_METADATA_V2 = PARAMS_DIR / "stones_metadata_v2.json"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "assemblies"


@dataclass
class ComponentGeometry:
    """Analyzed geometry of a CAD component"""
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    center: Tuple[float, float, float]
    width: float   # X dimension
    depth: float   # Y dimension
    height: float  # Z dimension
    aspect_ratio: float
    volume_estimate: float
    
    @property
    def size(self) -> float:
        """Average size across all dimensions"""
        return (self.width + self.depth + self.height) / 3


@dataclass
class ProngGeometry(ComponentGeometry):
    """Prong-specific geometry analysis"""
    opening_width: float
    opening_depth: float
    opening_z: float  # Z height of the opening
    opening_center: Tuple[float, float, float]
    opening_ratio: float  # Opening size vs bbox size
    detected_prong_count: int = 4
    setting_type: str = "prong"


@dataclass
class StoneGeometry(ComponentGeometry):
    """Stone-specific geometry analysis"""
    girdle_width: float
    girdle_depth: float
    girdle_z: float  # Z height of girdle
    girdle_center: Tuple[float, float, float]
    crown_height: float  # Height above girdle
    pavilion_depth: float  # Depth below girdle
    shape: str = "round"


class GeometryAnalyzer:
    """Precise geometry analysis using vertex-level inspection"""
    
    @staticmethod
    def get_all_vertices(model: rhino3dm.File3dm) -> List[Tuple[float, float, float]]:
        """Extract all vertices from a model for precise analysis"""
        vertices = []
        
        for obj in model.Objects:
            geom = obj.Geometry
            if not geom:
                continue
            
            # Handle different geometry types
            geom_type = type(geom).__name__
            
            if geom_type == "Mesh":
                for i in range(geom.Vertices.Count):
                    v = geom.Vertices[i]
                    vertices.append((v.X, v.Y, v.Z))
            
            elif geom_type == "Brep":
                # Get vertices from brep faces
                try:
                    mesh = rhino3dm.Mesh.CreateFromBrep(geom)
                    if mesh:
                        for m in mesh:
                            for i in range(m.Vertices.Count):
                                v = m.Vertices[i]
                                vertices.append((v.X, v.Y, v.Z))
                except:
                    pass
            
            elif geom_type == "Extrusion":
                try:
                    brep = geom.ToBrep()
                    if brep:
                        mesh = rhino3dm.Mesh.CreateFromBrep(brep)
                        if mesh:
                            for m in mesh:
                                for i in range(m.Vertices.Count):
                                    v = m.Vertices[i]
                                    vertices.append((v.X, v.Y, v.Z))
                except:
                    pass
        
        return vertices
    
    @staticmethod
    def analyze_model(model: rhino3dm.File3dm) -> ComponentGeometry:
        """Basic geometry analysis from bounding box"""
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
        
        return ComponentGeometry(
            bbox_min=(min_x, min_y, min_z),
            bbox_max=(max_x, max_y, max_z),
            center=((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2),
            width=width,
            depth=depth,
            height=height,
            aspect_ratio=width / depth if depth > 0 else 1.0,
            volume_estimate=width * depth * height
        )
    
    @staticmethod
    def analyze_prong(model: rhino3dm.File3dm, setting_type: str = "prong") -> ProngGeometry:
        """
        Analyze prong geometry with focus on opening detection.
        
        The opening is detected by analyzing vertices at the top of the prong.
        For prong settings: opening is ~75-85% of bbox (prongs extend inward)
        For bezel settings: opening is ~85-92% of bbox (thin rim)
        """
        base = GeometryAnalyzer.analyze_model(model)
        
        # Get vertices for detailed analysis
        vertices = GeometryAnalyzer.get_all_vertices(model)
        
        # Determine opening ratio based on setting type
        if "bezel" in setting_type.lower():
            opening_ratio = 0.88
        elif "halo" in setting_type.lower():
            opening_ratio = 0.85
        elif "cathedral" in setting_type.lower():
            opening_ratio = 0.82
        else:
            opening_ratio = 0.80
        
        # If we have vertices, try to detect opening more precisely
        if vertices:
            # Get vertices near the top (top 20% of height)
            top_threshold = base.bbox_max[2] - base.height * 0.2
            top_vertices = [v for v in vertices if v[2] > top_threshold]
            
            if len(top_vertices) > 10:
                # Find inner and outer edges at top
                top_xs = [v[0] for v in top_vertices]
                top_ys = [v[1] for v in top_vertices]
                
                # The opening is typically formed by the inner vertices
                # Use the 20th and 80th percentile to estimate opening
                top_xs.sort()
                top_ys.sort()
                
                n = len(top_xs)
                inner_min_x = top_xs[int(n * 0.2)]
                inner_max_x = top_xs[int(n * 0.8)]
                inner_min_y = top_ys[int(n * 0.2)]
                inner_max_y = top_ys[int(n * 0.8)]
                
                detected_opening_w = inner_max_x - inner_min_x
                detected_opening_d = inner_max_y - inner_min_y
                
                # Validate detection (opening should be > 50% and < 95% of bbox)
                if 0.5 < detected_opening_w / base.width < 0.95:
                    opening_ratio = detected_opening_w / base.width
        
        opening_width = base.width * opening_ratio
        opening_depth = base.depth * opening_ratio
        opening_z = base.bbox_max[2] - base.height * 0.12  # Opening is near top
        
        return ProngGeometry(
            bbox_min=base.bbox_min,
            bbox_max=base.bbox_max,
            center=base.center,
            width=base.width,
            depth=base.depth,
            height=base.height,
            aspect_ratio=base.aspect_ratio,
            volume_estimate=base.volume_estimate,
            opening_width=opening_width,
            opening_depth=opening_depth,
            opening_z=opening_z,
            opening_center=(base.center[0], base.center[1], opening_z),
            opening_ratio=opening_ratio,
            setting_type=setting_type
        )
    
    @staticmethod
    def analyze_stone(model: rhino3dm.File3dm, shape: str = "round") -> StoneGeometry:
        """
        Analyze stone geometry with focus on girdle detection.
        
        The girdle is the widest point of the stone, typically at 30-40% height.
        """
        base = GeometryAnalyzer.analyze_model(model)
        
        # Get vertices for detailed analysis
        vertices = GeometryAnalyzer.get_all_vertices(model)
        
        # Default girdle position (35% from bottom)
        girdle_z = base.bbox_min[2] + base.height * 0.35
        girdle_width = base.width
        girdle_depth = base.depth
        
        # If we have vertices, detect girdle more precisely
        if vertices:
            # Sample widths at different heights to find the widest point (girdle)
            height_samples = 10
            max_width = 0
            max_depth = 0
            girdle_height = 0.35
            
            for i in range(height_samples):
                height_ratio = (i + 1) / (height_samples + 1)
                sample_z = base.bbox_min[2] + base.height * height_ratio
                tolerance = base.height * 0.05
                
                # Get vertices near this height
                slice_verts = [v for v in vertices if abs(v[2] - sample_z) < tolerance]
                
                if slice_verts:
                    xs = [v[0] for v in slice_verts]
                    ys = [v[1] for v in slice_verts]
                    slice_width = max(xs) - min(xs)
                    slice_depth = max(ys) - min(ys)
                    
                    if slice_width > max_width:
                        max_width = slice_width
                        max_depth = slice_depth
                        girdle_height = height_ratio
            
            if max_width > 0:
                girdle_width = max_width
                girdle_depth = max_depth
                girdle_z = base.bbox_min[2] + base.height * girdle_height
        
        # Crown is above girdle, pavilion is below
        crown_height = base.bbox_max[2] - girdle_z
        pavilion_depth = girdle_z - base.bbox_min[2]
        
        # Detect shape from aspect ratio
        if base.aspect_ratio > 1.2:
            shape = "oval"
        elif base.aspect_ratio < 0.8:
            shape = "oval"  # Rotated oval
        else:
            shape = "round"
        
        return StoneGeometry(
            bbox_min=base.bbox_min,
            bbox_max=base.bbox_max,
            center=base.center,
            width=base.width,
            depth=base.depth,
            height=base.height,
            aspect_ratio=base.aspect_ratio,
            volume_estimate=base.volume_estimate,
            girdle_width=girdle_width,
            girdle_depth=girdle_depth,
            girdle_z=girdle_z,
            girdle_center=(base.center[0], base.center[1], girdle_z),
            crown_height=crown_height,
            pavilion_depth=pavilion_depth,
            shape=shape
        )


class PrecisionAssembler:
    """
    Precision CAD assembler for jewelry components.
    
    Assembly strategy:
    1. Place prong at origin, centered
    2. Analyze prong opening precisely
    3. Calculate scale factors for stone to fit opening
    4. Apply scaling to stone
    5. Position stone with correct XY center and Z depth
    """
    
    # Target clearance: stone girdle should be 97% of opening (3% gap)
    TARGET_FIT_RATIO = 0.97
    # Depth: girdle should drop 20-30% into prong from opening
    TARGET_DEPTH_RATIO = 0.25
    
    def __init__(self):
        self.analyzer = GeometryAnalyzer()
        self.prong_metadata = {}
        self.stone_metadata = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load V2 metadata for additional info"""
        if PRONG_METADATA_V2.exists():
            with open(PRONG_METADATA_V2) as f:
                self.prong_metadata = json.load(f)
        
        if STONE_METADATA_V2.exists():
            with open(STONE_METADATA_V2) as f:
                self.stone_metadata = json.load(f)
    
    def load_model(self, file_path: str) -> Optional[rhino3dm.File3dm]:
        """Load a .3dm file"""
        try:
            model = rhino3dm.File3dm.Read(file_path)
            return model
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def center_model_at_origin(self, model: rhino3dm.File3dm) -> Tuple[float, float, float]:
        """Move model so its center is at origin. Returns the translation applied."""
        geom = self.analyzer.analyze_model(model)
        translation = (-geom.center[0], -geom.center[1], -geom.center[2])
        
        for obj in model.Objects:
            g = obj.Geometry
            if g:
                transform = rhino3dm.Transform.Translation(*translation)
                g.Transform(transform)
        
        return translation
    
    def calculate_stone_scale(
        self,
        prong_geom: ProngGeometry,
        stone_geom: StoneGeometry,
        clearance: float = 0.03
    ) -> Dict[str, float]:
        """
        Calculate scale factors to make stone fit in prong opening.
        
        Target: stone_girdle * scale = prong_opening * (1 - clearance)
        
        Returns dict with scale_x, scale_y, scale_z factors.
        """
        # Target dimensions: stone girdle should be slightly smaller than opening
        target_girdle_x = prong_geom.opening_width * (1 - clearance)
        target_girdle_y = prong_geom.opening_depth * (1 - clearance)
        
        # Current stone girdle dimensions
        current_girdle_x = stone_geom.girdle_width
        current_girdle_y = stone_geom.girdle_depth
        
        # Calculate scale factors
        scale_x = target_girdle_x / current_girdle_x if current_girdle_x > 0 else 1.0
        scale_y = target_girdle_y / current_girdle_y if current_girdle_y > 0 else 1.0
        
        # Z scale: average of X and Y to maintain proportions
        scale_z = (scale_x + scale_y) / 2
        
        # Determine if non-uniform scaling is needed
        aspect_diff = abs(scale_x - scale_y)
        needs_non_uniform = aspect_diff > 0.05  # More than 5% difference
        
        # If shapes are very different (round prong, oval stone), we might need
        # to reshape, but generally prefer uniform scaling
        if needs_non_uniform:
            print(f"   ⚠️ Non-uniform scaling needed: X={scale_x:.3f}, Y={scale_y:.3f}")
        
        return {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "scale_z": scale_z,
            "uniform_scale": (scale_x + scale_y + scale_z) / 3,
            "needs_non_uniform": needs_non_uniform,
            "target_girdle_x": target_girdle_x,
            "target_girdle_y": target_girdle_y
        }
    
    def calculate_stone_position(
        self,
        prong_geom: ProngGeometry,
        stone_geom: StoneGeometry,
        scaled_stone_height: float
    ) -> Tuple[float, float, float]:
        """
        Calculate where to position the stone center.
        
        The stone girdle should:
        - Be centered on prong opening in XY
        - Drop into prong by ~25% of prong height in Z
        
        Returns (x, y, z) position for stone CENTER.
        """
        # XY: Center of prong opening
        target_x = prong_geom.opening_center[0]
        target_y = prong_geom.opening_center[1]
        
        # Z: The girdle should drop below the opening
        drop_distance = prong_geom.height * self.TARGET_DEPTH_RATIO
        target_girdle_z = prong_geom.opening_z - drop_distance
        
        # Stone geometry:
        # - Girdle is at 35% from BOTTOM of stone
        # - Center is at 50% from BOTTOM of stone
        # - So center is 15% above girdle (50% - 35% = 15%)
        center_above_girdle = scaled_stone_height * 0.15
        
        # Position stone center so girdle lands at target_girdle_z
        target_stone_center_z = target_girdle_z + center_above_girdle
        
        return (target_x, target_y, target_stone_center_z)
    
    def apply_scale(
        self,
        model: rhino3dm.File3dm,
        scale_x: float,
        scale_y: float,
        scale_z: float,
        center: Tuple[float, float, float]
    ):
        """Apply non-uniform scaling to a model around a center point."""
        for obj in model.Objects:
            geom = obj.Geometry
            if geom:
                # Translate to origin
                t1 = rhino3dm.Transform.Translation(-center[0], -center[1], -center[2])
                
                # Scale
                scale_transform = rhino3dm.Transform(1.0)
                scale_transform.M00 = scale_x
                scale_transform.M11 = scale_y
                scale_transform.M22 = scale_z
                
                # Translate back (with scaling applied to position)
                t2 = rhino3dm.Transform.Translation(
                    center[0] * scale_x,
                    center[1] * scale_y,
                    center[2] * scale_z
                )
                
                geom.Transform(t1)
                geom.Transform(scale_transform)
                geom.Transform(t2)
    
    def apply_translation(
        self,
        model: rhino3dm.File3dm,
        translation: Tuple[float, float, float]
    ):
        """Apply translation to all objects in model."""
        for obj in model.Objects:
            geom = obj.Geometry
            if geom:
                transform = rhino3dm.Transform.Translation(*translation)
                geom.Transform(transform)
    
    def assemble(
        self,
        stone_path: str,
        prong_path: str,
        output_path: str = None,
        stone_id: str = None,
        prong_id: str = None,
        setting_type: str = "prong",
        correction_factors: Dict = None
    ) -> Optional[str]:
        """
        Assemble stone and prong with precision fitting.
        
        Args:
            stone_path: Path to stone .3dm file
            prong_path: Path to prong .3dm file
            output_path: Optional output path
            stone_id: Stone ID for metadata lookup
            prong_id: Prong ID for metadata lookup
            setting_type: Type of setting (prong, bezel, etc.)
            correction_factors: Optional correction from previous validation
            
        Returns:
            Path to output file, or None if failed
        """
        print("\n" + "="*60)
        print("🔧 PRECISION CAD ASSEMBLY")
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
        
        # Step 1: Center prong at origin
        print("\n📍 Step 1: Centering prong at origin...")
        prong_translation = self.center_model_at_origin(prong_model)
        print(f"   Translated prong by: ({prong_translation[0]:.2f}, {prong_translation[1]:.2f}, {prong_translation[2]:.2f})")
        
        # Step 2: Analyze geometries
        print("\n📐 Step 2: Analyzing component geometry...")
        
        prong_geom = self.analyzer.analyze_prong(prong_model, setting_type)
        stone_geom = self.analyzer.analyze_stone(stone_model)
        
        print(f"\n   PRONG ({setting_type}):")
        print(f"      BBox: {prong_geom.width:.2f} x {prong_geom.depth:.2f} x {prong_geom.height:.2f} mm")
        print(f"      Opening: {prong_geom.opening_width:.2f} x {prong_geom.opening_depth:.2f} mm")
        print(f"      Opening at Z: {prong_geom.opening_z:.2f} mm")
        
        print(f"\n   STONE ({stone_geom.shape}):")
        print(f"      BBox: {stone_geom.width:.2f} x {stone_geom.depth:.2f} x {stone_geom.height:.2f} mm")
        print(f"      Girdle: {stone_geom.girdle_width:.2f} x {stone_geom.girdle_depth:.2f} mm")
        print(f"      Girdle at Z: {stone_geom.girdle_z:.2f} mm")
        
        # Step 3: Calculate stone scaling
        print("\n📏 Step 3: Calculating stone scale for perfect fit...")
        
        scale_info = self.calculate_stone_scale(prong_geom, stone_geom, clearance=0.03)
        
        # Apply correction factors if provided
        if correction_factors:
            print(f"   Applying correction factors from validation...")
            scale_info["scale_x"] *= correction_factors.get("scale_x", 1.0)
            scale_info["scale_y"] *= correction_factors.get("scale_y", 1.0)
            scale_info["scale_z"] *= correction_factors.get("scale_z", 1.0)
        
        print(f"\n   Scale factors:")
        print(f"      X: {scale_info['scale_x']:.3f}")
        print(f"      Y: {scale_info['scale_y']:.3f}")
        print(f"      Z: {scale_info['scale_z']:.3f}")
        print(f"      Target girdle: {scale_info['target_girdle_x']:.2f} x {scale_info['target_girdle_y']:.2f} mm")
        
        # Step 4: Apply scaling to stone
        print("\n🔄 Step 4: Scaling stone...")
        
        self.apply_scale(
            stone_model,
            scale_info["scale_x"],
            scale_info["scale_y"],
            scale_info["scale_z"],
            stone_geom.center
        )
        
        # Re-analyze stone after scaling
        scaled_stone_geom = self.analyzer.analyze_stone(stone_model)
        print(f"   Scaled stone girdle: {scaled_stone_geom.girdle_width:.2f} x {scaled_stone_geom.girdle_depth:.2f} mm")
        
        # Step 5: Calculate and apply stone position
        print("\n📍 Step 5: Positioning stone in prong...")
        
        # First center the stone at origin
        stone_center_before = scaled_stone_geom.center
        self.center_model_at_origin(stone_model)
        
        # Re-analyze after centering
        centered_stone_geom = self.analyzer.analyze_stone(stone_model)
        
        # Calculate target position
        target_pos = self.calculate_stone_position(
            prong_geom,
            centered_stone_geom,
            scaled_stone_geom.height
        )
        
        print(f"   Target position: ({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f})")
        
        # Apply translation
        self.apply_translation(stone_model, target_pos)
        
        # Verify final position
        final_stone_geom = self.analyzer.analyze_stone(stone_model)
        print(f"   Final stone center: ({final_stone_geom.center[0]:.2f}, {final_stone_geom.center[1]:.2f}, {final_stone_geom.center[2]:.2f})")
        
        # Calculate fit verification
        fit_x = final_stone_geom.girdle_width / prong_geom.opening_width
        fit_y = final_stone_geom.girdle_depth / prong_geom.opening_depth
        clearance_x = prong_geom.opening_width - final_stone_geom.girdle_width
        clearance_y = prong_geom.opening_depth - final_stone_geom.girdle_depth
        
        print(f"\n   ✅ FIT VERIFICATION:")
        print(f"      Fit ratio X: {fit_x:.1%} (clearance: {clearance_x:.2f}mm)")
        print(f"      Fit ratio Y: {fit_y:.1%} (clearance: {clearance_y:.2f}mm)")
        
        # Step 6: Create output model
        print("\n💾 Step 6: Creating output model...")
        
        output_model = rhino3dm.File3dm()
        
        # Add prong layer
        prong_layer = rhino3dm.Layer()
        prong_layer.Name = "Prong_Setting"
        prong_layer.Color = (192, 192, 192, 255)  # Silver
        prong_idx = output_model.Layers.Add(prong_layer)
        
        # Add stone layer
        stone_layer = rhino3dm.Layer()
        stone_layer.Name = "Stone"
        stone_layer.Color = (255, 50, 50, 255)  # Red
        stone_idx = output_model.Layers.Add(stone_layer)
        
        # Add prong objects
        for obj in prong_model.Objects:
            geom = obj.Geometry
            if geom:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = prong_idx
                output_model.Objects.Add(geom, attr)
        
        # Add stone objects
        for obj in stone_model.Objects:
            geom = obj.Geometry
            if geom:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = stone_idx
                output_model.Objects.Add(geom, attr)
        
        # Save output
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(OUTPUT_DIR / f"precision_assembly_{timestamp}.3dm")
        
        output_model.Write(output_path, 7)
        
        print("\n" + "="*60)
        print("✅ PRECISION ASSEMBLY COMPLETE")
        print("="*60)
        print(f"\n📁 Output: {output_path}")
        print(f"\n📐 Summary:")
        print(f"   Stone scaled by: {scale_info['uniform_scale']:.2f}x")
        print(f"   Fit: {(fit_x + fit_y) / 2:.1%} (target: 97%)")
        print(f"   Clearance: {(clearance_x + clearance_y) / 2:.2f}mm")
        
        return output_path


def assemble_precision(
    stone_path: str,
    prong_path: str,
    output_path: str = None,
    setting_type: str = "prong"
) -> Optional[str]:
    """Convenience function for precision assembly"""
    assembler = PrecisionAssembler()
    return assembler.assemble(
        stone_path=stone_path,
        prong_path=prong_path,
        output_path=output_path,
        setting_type=setting_type
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python precision_assembler.py <stone.3dm> <prong.3dm> [output.3dm]")
        sys.exit(1)
    
    stone_path = sys.argv[1]
    prong_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = assemble_precision(stone_path, prong_path, output_path)
    sys.exit(0 if result else 1)
