"""
Dynamic Shank Generator
Generates ring shank geometry dynamically based on design analysis.

Instead of using predefined shank templates, this generator:
1. Analyzes the head assembly to determine optimal shank parameters
2. Generates shank geometry procedurally based on style requirements
3. Automatically sizes and positions shank to match head
4. Supports various styles: plain, cathedral, split, knife-edge, tapered

Key features:
- Automatic head-to-shank fitting
- Style-aware generation
- Proper orientation (finger through Y-axis)
- Smooth transitions to head
"""
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


# Ring size conversion: US size to inner diameter in mm
# Formula: diameter_mm = (size * 0.825) + 12.5
RING_SIZE_TO_MM = {
    4: 14.86, 5: 15.7, 6: 16.51, 7: 17.35, 8: 18.19,
    9: 19.0, 10: 19.84, 11: 20.68, 12: 21.49, 13: 22.33
}


@dataclass
class ShankParameters:
    """Complete parameters for shank generation"""
    # Ring sizing
    ring_size: float = 7.0
    inner_diameter: float = 0  # Calculated from ring_size
    
    # Band dimensions
    band_width: float = 2.5  # mm - width of band (Y direction)
    band_thickness: float = 1.8  # mm - thickness (radial)
    
    # Style
    style: str = "plain"  # plain, cathedral, split, knife, tapered
    profile: str = "comfort"  # comfort, flat, round, knife
    
    # Head integration
    head_width: float = 8.0  # Width of the head assembly
    head_depth: float = 8.0  # Depth of the head assembly
    head_height: float = 6.0  # Height of head from band top
    
    # Cathedral-specific
    cathedral_rise: float = 3.0  # How high cathedral shoulders rise
    cathedral_angle: float = 45.0  # Angle of cathedral rise
    
    # Split shank specific
    split_gap: float = 1.5  # Gap between split sections
    split_start_angle: float = 30.0  # Where split begins (degrees from top)
    
    # Taper specific
    taper_ratio: float = 0.7  # Narrowing at top (1.0 = no taper)
    
    def __post_init__(self):
        """Calculate derived values"""
        if self.inner_diameter == 0:
            self.inner_diameter = (self.ring_size * 0.825) + 12.5
    
    @property
    def inner_radius(self) -> float:
        return self.inner_diameter / 2
    
    @property
    def outer_radius(self) -> float:
        return self.inner_radius + self.band_thickness
    
    @property
    def center_radius(self) -> float:
        return self.inner_radius + self.band_thickness / 2


class DynamicShankGenerator:
    """
    Generates ring shank geometry dynamically.
    
    Orientation:
    - Ring lies in XZ plane
    - Finger goes through Y-axis
    - Stone points up (+Z)
    - Head is at top of ring (+Z direction)
    """
    
    def __init__(self):
        self.model = rhino3dm.File3dm()
    
    def analyze_head_assembly(self, head_model: rhino3dm.File3dm) -> Dict:
        """Analyze head assembly to determine shank parameters"""
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for obj in head_model.Objects:
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
        
        return {
            "min": (min_x, min_y, min_z),
            "max": (max_x, max_y, max_z),
            "width": max_x - min_x,
            "depth": max_y - min_y,
            "height": max_z - min_z,
            "center": ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2),
            "bottom_z": min_z
        }
    
    def calculate_optimal_params(
        self,
        head_info: Dict,
        design_analysis: Dict = None,
        ring_size: float = 7.0
    ) -> ShankParameters:
        """
        Calculate optimal shank parameters based on head assembly and design.
        
        This is where the "intelligence" happens - determining the right
        shank proportions to match the head.
        """
        head_width = head_info.get("width", 8.0)
        head_depth = head_info.get("depth", 8.0)
        head_height = head_info.get("height", 6.0)
        
        # Default style from design analysis
        style = "plain"
        profile = "comfort"
        band_width = 2.5
        band_thickness = 1.8
        
        if design_analysis:
            shank_info = design_analysis.get("shank", {})
            
            # Extract style
            style_raw = shank_info.get("style", "plain").lower()
            if "cathedral" in style_raw:
                style = "cathedral"
            elif "split" in style_raw:
                style = "split"
            elif "knife" in style_raw:
                style = "knife"
                profile = "knife"
            elif "taper" in style_raw:
                style = "tapered"
            else:
                style = "plain"
            
            # Extract dimensions
            band_width = shank_info.get("width_mm", 2.5)
            band_thickness = shank_info.get("thickness_mm", 1.8)
            
            # Get ring size estimate
            if design_analysis.get("ring_size_estimate"):
                ring_size = float(design_analysis["ring_size_estimate"])
        
        # Auto-adjust band width based on head size
        # Larger heads need slightly wider bands for visual balance
        if head_width > 10:
            band_width = max(band_width, 3.0)
        elif head_width > 8:
            band_width = max(band_width, 2.5)
        
        # Cathedral rise based on head height
        cathedral_rise = min(head_height * 0.5, 4.0)
        
        return ShankParameters(
            ring_size=ring_size,
            band_width=band_width,
            band_thickness=band_thickness,
            style=style,
            profile=profile,
            head_width=head_width,
            head_depth=head_depth,
            head_height=head_height,
            cathedral_rise=cathedral_rise
        )
    
    def generate_band_mesh(
        self,
        params: ShankParameters,
        segments: int = 64,
        profile_segments: int = 16
    ) -> rhino3dm.Mesh:
        """
        Generate the base band mesh.
        
        Orientation: Ring in XZ plane, finger through Y
        """
        mesh = rhino3dm.Mesh()
        
        for i in range(segments):
            angle = (i / segments) * 2 * math.pi
            
            # Calculate modifiers based on style and position
            # angle = 0 is at +X, angle = pi/2 is at +Z (top where head goes)
            
            # Get style-specific modifiers
            modifiers = self._get_style_modifiers(params, angle)
            
            # Ring centerline position in XZ plane
            cx = params.center_radius * math.cos(angle)
            cz = params.center_radius * math.sin(angle) + modifiers["rise"]
            
            # Radial direction
            nx = math.cos(angle)
            nz = math.sin(angle)
            
            # Current band dimensions with modifiers
            current_width = params.band_width * modifiers["width_factor"]
            current_thickness = params.band_thickness * modifiers["thickness_factor"]
            
            for j in range(profile_segments):
                profile_angle = (j / profile_segments) * 2 * math.pi
                
                # Profile shape
                pr, py = self._get_profile_point(
                    params.profile,
                    profile_angle,
                    current_thickness,
                    current_width
                )
                
                # Final vertex position
                vx = cx + pr * nx
                vy = py + modifiers["y_offset"]  # Y offset for split shanks
                vz = cz + pr * nz
                
                mesh.Vertices.Add(vx, vy, vz)
        
        # Generate faces
        for i in range(segments):
            i_next = (i + 1) % segments
            for j in range(profile_segments):
                j_next = (j + 1) % profile_segments
                v0 = i * profile_segments + j
                v1 = i * profile_segments + j_next
                v2 = i_next * profile_segments + j_next
                v3 = i_next * profile_segments + j
                mesh.Faces.AddFace(v0, v1, v2, v3)
        
        mesh.Normals.ComputeNormals()
        return mesh
    
    def _get_style_modifiers(self, params: ShankParameters, angle: float) -> Dict:
        """
        Calculate style-specific modifiers for a given angle around the ring.
        
        angle: 0 = +X (right), pi/2 = +Z (top), pi = -X (left), 3pi/2 = -Z (bottom)
        """
        modifiers = {
            "rise": 0.0,
            "width_factor": 1.0,
            "thickness_factor": 1.0,
            "y_offset": 0.0
        }
        
        # Position factor: how close to the top (where head is)
        # 1.0 at top (angle = pi/2), 0.0 at bottom (angle = 3pi/2)
        top_factor = max(0, math.sin(angle))
        
        if params.style == "cathedral":
            # Cathedral: rises smoothly toward the head
            if top_factor > 0.3:
                rise_factor = ((top_factor - 0.3) / 0.7) ** 1.5
                modifiers["rise"] = rise_factor * params.cathedral_rise
                # Taper width as it rises
                modifiers["width_factor"] = 1.0 - rise_factor * 0.3
        
        elif params.style == "tapered":
            # Taper: band gets narrower toward head
            if top_factor > 0:
                taper = 1.0 - (1.0 - params.taper_ratio) * top_factor
                modifiers["width_factor"] = taper
        
        elif params.style == "knife":
            # Knife edge: thickness tapers to edge
            if top_factor > 0.3:
                edge_factor = ((top_factor - 0.3) / 0.7)
                modifiers["thickness_factor"] = 1.0 - edge_factor * 0.4
        
        elif params.style == "split":
            # Split: band splits near head
            # This is handled differently - we generate two half-bands
            pass  # Handled in generate_split_shank
        
        return modifiers
    
    def _get_profile_point(
        self,
        profile: str,
        angle: float,
        thickness: float,
        width: float
    ) -> Tuple[float, float]:
        """
        Get profile point (radial offset, Y offset) for given angle.
        
        Returns (pr, py) where:
        - pr: radial offset from band centerline
        - py: Y offset (along finger direction)
        """
        half_thickness = thickness / 2
        half_width = width / 2
        
        if profile == "comfort":
            # Elliptical profile - comfortable inside
            pr = half_thickness * abs(math.cos(angle))
            py = half_width * math.sin(angle)
        
        elif profile == "flat":
            # Rectangular profile
            pr = half_thickness * math.cos(angle)
            py = half_width * math.sin(angle)
        
        elif profile == "round":
            # Circular profile
            radius = min(half_thickness, half_width)
            pr = radius * math.cos(angle)
            py = radius * math.sin(angle)
        
        elif profile == "knife":
            # Knife edge - thin at edges
            pr = half_thickness * abs(math.cos(angle)) ** 1.5
            py = half_width * math.sin(angle)
        
        else:
            # Default to comfort
            pr = half_thickness * abs(math.cos(angle))
            py = half_width * math.sin(angle)
        
        return pr, py
    
    def generate_split_shank(
        self,
        params: ShankParameters,
        segments: int = 64,
        profile_segments: int = 12
    ) -> List[rhino3dm.Mesh]:
        """
        Generate split shank as two separate band meshes.
        
        Returns list of two meshes (left and right of split).
        """
        meshes = []
        
        for side in [-1, 1]:  # Left and right
            mesh = rhino3dm.Mesh()
            
            for i in range(segments):
                angle = (i / segments) * 2 * math.pi
                top_factor = max(0, math.sin(angle))
                
                # Calculate split offset
                y_offset = 0
                width_factor = 1.0
                
                if top_factor > 0.3:
                    split_factor = ((top_factor - 0.3) / 0.7)
                    y_offset = side * (params.split_gap / 2 + params.band_width * 0.3) * split_factor
                    width_factor = 0.6 + 0.4 * (1 - split_factor)  # Narrow in split zone
                
                # Ring centerline
                cx = params.center_radius * math.cos(angle)
                cz = params.center_radius * math.sin(angle)
                
                nx = math.cos(angle)
                nz = math.sin(angle)
                
                current_width = params.band_width * width_factor
                
                for j in range(profile_segments):
                    profile_angle = (j / profile_segments) * 2 * math.pi
                    pr, py = self._get_profile_point(
                        params.profile, profile_angle,
                        params.band_thickness, current_width
                    )
                    
                    vx = cx + pr * nx
                    vy = py + y_offset
                    vz = cz + pr * nz
                    
                    mesh.Vertices.Add(vx, vy, vz)
            
            # Generate faces
            for i in range(segments):
                i_next = (i + 1) % segments
                for j in range(profile_segments):
                    j_next = (j + 1) % profile_segments
                    v0 = i * profile_segments + j
                    v1 = i * profile_segments + j_next
                    v2 = i_next * profile_segments + j_next
                    v3 = i_next * profile_segments + j
                    mesh.Faces.AddFace(v0, v1, v2, v3)
            
            mesh.Normals.ComputeNormals()
            meshes.append(mesh)
        
        return meshes
    
    def generate(
        self,
        params: ShankParameters,
        output_path: str = None
    ) -> rhino3dm.File3dm:
        """
        Generate complete shank model based on parameters.
        
        Returns the File3dm model (also saves if output_path provided).
        """
        print(f"\n🔧 Generating {params.style} shank...")
        print(f"   Ring size: {params.ring_size} (Ø{params.inner_diameter:.1f}mm)")
        print(f"   Band: {params.band_width:.1f}mm wide, {params.band_thickness:.1f}mm thick")
        print(f"   Profile: {params.profile}")
        
        self.model = rhino3dm.File3dm()
        
        # Create layer
        layer = rhino3dm.Layer()
        layer.Name = "Ring_Shank"
        layer.Color = (200, 180, 100, 255)  # Gold
        layer_idx = self.model.Layers.Add(layer)
        
        attr = rhino3dm.ObjectAttributes()
        attr.LayerIndex = layer_idx
        
        if params.style == "split":
            # Generate split shank (two meshes)
            meshes = self.generate_split_shank(params)
            for mesh in meshes:
                self.model.Objects.AddMesh(mesh, attr)
            print(f"   Created split shank with {len(meshes)} sections")
        else:
            # Generate single band mesh
            mesh = self.generate_band_mesh(params)
            self.model.Objects.AddMesh(mesh, attr)
            print(f"   Created {params.style} band")
        
        if output_path:
            self.model.Write(output_path, 7)
            print(f"   💾 Saved: {output_path}")
        
        return self.model
    
    def generate_for_head(
        self,
        head_model: rhino3dm.File3dm,
        design_analysis: Dict = None,
        ring_size: float = 7.0,
        output_path: str = None
    ) -> Tuple[rhino3dm.File3dm, ShankParameters]:
        """
        Generate shank optimized for a specific head assembly.
        
        Analyzes head dimensions and generates matching shank.
        Returns (shank_model, parameters_used).
        """
        print("\n📐 Analyzing head assembly for shank generation...")
        
        # Analyze head
        head_info = self.analyze_head_assembly(head_model)
        print(f"   Head dimensions: {head_info['width']:.1f} x {head_info['depth']:.1f} x {head_info['height']:.1f} mm")
        
        # Calculate optimal parameters
        params = self.calculate_optimal_params(
            head_info,
            design_analysis,
            ring_size
        )
        
        print(f"   Calculated style: {params.style}")
        print(f"   Band width: {params.band_width:.1f}mm")
        
        # Generate shank
        shank_model = self.generate(params, output_path)
        
        return shank_model, params
    
    def combine_head_and_shank(
        self,
        head_model: rhino3dm.File3dm,
        shank_model: rhino3dm.File3dm,
        params: ShankParameters,
        output_path: str = None
    ) -> rhino3dm.File3dm:
        """
        Combine head and shank into complete ring.
        
        Positions head at top of shank with proper alignment.
        """
        print("\n💍 Combining head and shank...")
        
        # Analyze head for positioning
        head_info = self.analyze_head_assembly(head_model)
        
        # Calculate head scale to fit shank
        # Head should span about 50-60% of shank outer diameter
        shank_outer = params.outer_radius * 2
        target_head_span = shank_outer * 0.55
        head_max_dim = max(head_info["width"], head_info["depth"])
        head_scale = target_head_span / head_max_dim if head_max_dim > 0 else 1.0
        head_scale = max(0.4, min(1.2, head_scale))  # Clamp to reasonable range
        
        # Head position: at top of ring (+Z), sitting on the shank
        # Shank top is at Z = outer_radius
        shank_top_z = params.outer_radius
        
        # Head bottom should sit at shank top
        head_bottom = head_info["min"][2] * head_scale
        head_translation_z = shank_top_z - head_bottom
        
        print(f"   Head scale: {head_scale:.2f}")
        print(f"   Head Z offset: {head_translation_z:.2f}mm")
        
        # Create combined model
        combined = rhino3dm.File3dm()
        
        # Add layers
        shank_layer = rhino3dm.Layer()
        shank_layer.Name = "Ring_Shank"
        shank_layer.Color = (200, 180, 100, 255)
        shank_idx = combined.Layers.Add(shank_layer)
        
        setting_layer = rhino3dm.Layer()
        setting_layer.Name = "Prong_Setting"
        setting_layer.Color = (192, 192, 192, 255)
        setting_idx = combined.Layers.Add(setting_layer)
        
        stone_layer = rhino3dm.Layer()
        stone_layer.Name = "Stone"
        stone_layer.Color = (255, 50, 100, 255)
        stone_idx = combined.Layers.Add(stone_layer)
        
        # Add shank objects
        for obj in shank_model.Objects:
            geom = obj.Geometry
            if geom:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = shank_idx
                combined.Objects.Add(geom, attr)
        
        # Add head objects with scaling and translation
        for obj in head_model.Objects:
            geom = obj.Geometry
            if geom:
                # Apply scale around origin
                scale_transform = rhino3dm.Transform.Scale(
                    rhino3dm.Point3d(0, 0, 0), head_scale
                )
                geom.Transform(scale_transform)
                
                # Apply translation
                translate_transform = rhino3dm.Transform.Translation(0, 0, head_translation_z)
                geom.Transform(translate_transform)
                
                # Determine layer from original
                original_layer_idx = obj.Attributes.LayerIndex
                layer_name = ""
                if original_layer_idx < len(head_model.Layers):
                    layer_name = head_model.Layers[original_layer_idx].Name.lower()
                
                attr = rhino3dm.ObjectAttributes()
                if "stone" in layer_name:
                    attr.LayerIndex = stone_idx
                else:
                    attr.LayerIndex = setting_idx
                
                combined.Objects.Add(geom, attr)
        
        print(f"   Combined: {len(combined.Objects)} total objects")
        
        if output_path:
            combined.Write(output_path, 7)
            print(f"   💾 Saved: {output_path}")
        
        return combined


def generate_dynamic_shank(
    head_path: str = None,
    design_analysis: Dict = None,
    ring_size: float = 7.0,
    style: str = None,
    output_path: str = None
) -> str:
    """
    Convenience function to generate a dynamic shank.
    
    Can work with:
    - head_path: Analyze head and generate matching shank
    - design_analysis: Use design analysis for style/dimensions
    - style: Override style directly
    """
    generator = DynamicShankGenerator()
    
    if head_path:
        head_model = rhino3dm.File3dm.Read(head_path)
        if head_model:
            shank_model, params = generator.generate_for_head(
                head_model,
                design_analysis,
                ring_size,
                output_path
            )
            return output_path
    
    # Generate standalone shank
    params = ShankParameters(
        ring_size=ring_size,
        style=style or "plain"
    )
    
    if design_analysis:
        shank_info = design_analysis.get("shank", {})
        params.band_width = shank_info.get("width_mm", 2.5)
        params.band_thickness = shank_info.get("thickness_mm", 1.8)
    
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "outputs" / "assemblies"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"shank_{params.style}_{timestamp}.3dm")
    
    generator.generate(params, output_path)
    return output_path


if __name__ == "__main__":
    import sys
    
    # Demo: generate different shank styles
    print("\n💍 Dynamic Shank Generator Demo")
    print("="*50)
    
    styles = ["plain", "cathedral", "split", "tapered"]
    
    for style in styles:
        params = ShankParameters(ring_size=7.0, style=style)
        generator = DynamicShankGenerator()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent.parent / "outputs" / "generated_components"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"demo_shank_{style}_{timestamp}.3dm")
        
        generator.generate(params, output_path)
    
    print("\n✅ Generated demo shanks for all styles")
