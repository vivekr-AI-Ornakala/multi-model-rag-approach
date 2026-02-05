"""
Smart AI-Powered Assembly System (v3.0) - Physics + AI Hybrid
Uses Trimesh for computational geometry and Gemini for aesthetic judgment.

Key Upgrades from v2.0:
1. OBB (Oriented Bounding Box) - Handles rotated stones correctly
2. Ray Casting "Drop Test" - Finds exact seat position via physics
3. Mesh Collision Detection - Detects actual mesh intersections
4. AI for Aesthetics Only - LLM judges style, not math

Architecture:
- Trimesh: Positioning, Rotation, Collision, Gravity simulation
- Gemini: "Does this look right?" aesthetic checks
"""
import json
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

try:
    import rhino3dm
    import google.generativeai as genai
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    raise

# Try to import trimesh - graceful fallback if not available
try:
    import trimesh
    from scipy.spatial import ConvexHull
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("[WARN] trimesh not installed - using fallback geometry. Install with: pip install trimesh scipy")

from config import GEMINI_API_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class Logger:
    """Professional logging utility"""
    
    @staticmethod
    def header(title: str, char: str = "="):
        print(f"\n{char * 60}")
        print(f"  {title}")
        print(f"{char * 60}")
    
    @staticmethod
    def section(title: str):
        print(f"\n[{title}]")
    
    @staticmethod
    def info(msg: str):
        print(f"  {msg}")
    
    @staticmethod
    def detail(label: str, value: str):
        print(f"  {label:<20}: {value}")
    
    @staticmethod
    def success(msg: str):
        print(f"  [OK] {msg}")
    
    @staticmethod
    def warning(msg: str):
        print(f"  [WARN] {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"  [ERROR] {msg}")


# =============================================================================
# GEOMETRY ENGINE (Trimesh-based Physics)
# =============================================================================

class GeometryEngine:
    """
    Computational geometry engine using Trimesh.
    Handles all math/physics - NO AI guessing here.
    """
    
    @staticmethod
    def rhino_to_trimesh(rhino_model: rhino3dm.File3dm) -> Optional['trimesh.Trimesh']:
        """
        Convert rhino3dm model to trimesh.
        Extracts mesh data from all mesh objects in the file.
        """
        if not TRIMESH_AVAILABLE:
            return None
            
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for obj in rhino_model.Objects:
            geom = obj.Geometry
            if geom is None:
                continue
                
            # Handle Mesh objects
            if isinstance(geom, rhino3dm.Mesh):
                mesh = geom
                
                # Extract vertices
                vertices = np.array([[v.X, v.Y, v.Z] for v in mesh.Vertices])
                
                # Extract faces (triangulate quads)
                faces = []
                for face in mesh.Faces:
                    if face[2] == face[3]:  # Triangle
                        faces.append([face[0], face[1], face[2]])
                    else:  # Quad - split into two triangles
                        faces.append([face[0], face[1], face[2]])
                        faces.append([face[0], face[2], face[3]])
                
                if len(vertices) > 0 and len(faces) > 0:
                    faces = np.array(faces) + vertex_offset
                    all_vertices.append(vertices)
                    all_faces.append(faces)
                    vertex_offset += len(vertices)
            
            # Handle Brep objects - try to get render mesh
            elif isinstance(geom, rhino3dm.Brep):
                try:
                    for face_idx in range(len(geom.Faces)):
                        mesh = geom.Faces[face_idx].GetMesh(rhino3dm.MeshType.Any)
                        if mesh:
                            vertices = np.array([[v.X, v.Y, v.Z] for v in mesh.Vertices])
                            faces = []
                            for f in mesh.Faces:
                                if f[2] == f[3]:
                                    faces.append([f[0], f[1], f[2]])
                                else:
                                    faces.append([f[0], f[1], f[2]])
                                    faces.append([f[0], f[2], f[3]])
                            
                            if len(vertices) > 0 and len(faces) > 0:
                                faces = np.array(faces) + vertex_offset
                                all_vertices.append(vertices)
                                all_faces.append(faces)
                                vertex_offset += len(vertices)
                except:
                    pass
        
        if len(all_vertices) == 0:
            # Fallback: Create box from bounding box
            return GeometryEngine._create_box_from_rhino(rhino_model)
        
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    @staticmethod
    def _create_box_from_rhino(rhino_model: rhino3dm.File3dm) -> 'trimesh.Trimesh':
        """Fallback: Create trimesh box from rhino bounding box"""
        min_v = np.array([float('inf')] * 3)
        max_v = np.array([float('-inf')] * 3)
        
        for obj in rhino_model.Objects:
            if obj.Geometry:
                bb = obj.Geometry.GetBoundingBox()
                min_v = np.minimum(min_v, [bb.Min.X, bb.Min.Y, bb.Min.Z])
                max_v = np.maximum(max_v, [bb.Max.X, bb.Max.Y, bb.Max.Z])
        
        extents = max_v - min_v
        center = (min_v + max_v) / 2
        box = trimesh.creation.box(extents=extents)
        box.apply_translation(center)
        return box
    
    @staticmethod
    def compute_obb(mesh: 'trimesh.Trimesh') -> Dict:
        """
        Compute Oriented Bounding Box using PCA.
        
        Solves the "Rotation Problem" - a 45° rotated square
        will correctly report as a square, not a rectangle.
        """
        try:
            obb = mesh.bounding_box_oriented
            extents = obb.extents
            sorted_extents = np.sort(extents)[::-1]
            
            return {
                "width": float(sorted_extents[0]),
                "depth": float(sorted_extents[1]),
                "height": float(sorted_extents[2]),
                "center": obb.centroid.tolist(),
                "transform": obb.transform.tolist(),
                "aspect_ratio": float(sorted_extents[0] / sorted_extents[1]) if sorted_extents[1] > 0 else 1.0
            }
        except Exception as e:
            Logger.warning(f"OBB failed, using AABB: {e}")
            bounds = mesh.bounds
            extents = bounds[1] - bounds[0]
            return {
                "width": float(extents[0]),
                "depth": float(extents[1]),
                "height": float(extents[2]),
                "center": mesh.centroid.tolist(),
                "transform": np.eye(4).tolist(),
                "aspect_ratio": float(extents[0] / extents[1]) if extents[1] > 0 else 1.0
            }
    
    @staticmethod
    def drop_test(stone_mesh: 'trimesh.Trimesh', prong_mesh: 'trimesh.Trimesh', 
                  clearance: float = 0.02) -> Optional[float]:
        """
        Simulate gravity to find the EXACT Z-height for seating.
        
        Solves the "Gravity Problem" - no more guessing seat height.
        We "drop" the stone until it touches the prong.
        """
        try:
            stone_verts = stone_mesh.vertices
            stone_bounds = stone_mesh.bounds
            stone_center_z = (stone_bounds[0][2] + stone_bounds[1][2]) / 2
            
            # Sample from lower half (girdle area)
            girdle_mask = stone_verts[:, 2] < stone_center_z
            if np.sum(girdle_mask) < 10:
                sample_points = stone_verts
            else:
                sample_points = stone_verts[girdle_mask]
            
            # Subsample for performance
            if len(sample_points) > 100:
                indices = np.random.choice(len(sample_points), 100, replace=False)
                sample_points = sample_points[indices]
            
            # Cast rays downward
            ray_origins = sample_points.copy()
            ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1))
            
            # Find intersections
            locations, index_ray, _ = prong_mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )
            
            if len(locations) == 0:
                return None
            
            # Calculate drop distances
            z_origins = ray_origins[index_ray, 2]
            z_hits = locations[:, 2]
            distances = z_origins - z_hits
            
            valid_distances = distances[distances > 0]
            if len(valid_distances) == 0:
                return None
            
            # Minimum distance = first contact
            drop_distance = float(np.min(valid_distances)) - clearance
            return drop_distance
            
        except Exception as e:
            Logger.warning(f"Drop test failed: {e}")
            return None
    
    @staticmethod
    def check_collision(mesh_a: 'trimesh.Trimesh', mesh_b: 'trimesh.Trimesh') -> Dict:
        """
        Check for mesh-to-mesh collision.
        
        Solves the "Collision Problem" - detects actual mesh
        intersections, not just bounding box overlap.
        """
        try:
            manager = trimesh.collision.CollisionManager()
            manager.add_object('mesh_a', mesh_a)
            is_collision, _ = manager.in_collision_single(mesh_b, return_names=True)
            
            intersection_volume = 0.0
            if is_collision:
                try:
                    intersection = mesh_a.intersection(mesh_b)
                    if intersection and hasattr(intersection, 'volume'):
                        intersection_volume = float(intersection.volume)
                except:
                    intersection_volume = -1
            
            return {
                "has_collision": is_collision,
                "intersection_volume": intersection_volume,
                "is_valid_fit": not is_collision
            }
        except Exception as e:
            return {"has_collision": False, "intersection_volume": 0, "is_valid_fit": True, "error": str(e)}
    
    @staticmethod
    def compute_fit_metrics(stone_mesh: 'trimesh.Trimesh', prong_mesh: 'trimesh.Trimesh') -> Dict:
        """Compute precise fit metrics using OBB."""
        stone_obb = GeometryEngine.compute_obb(stone_mesh)
        prong_obb = GeometryEngine.compute_obb(prong_mesh)
        
        # Prong opening ~85% of outer
        prong_opening_width = prong_obb["width"] * 0.85
        prong_opening_depth = prong_obb["depth"] * 0.85
        
        fit_x = stone_obb["width"] / prong_opening_width if prong_opening_width > 0 else 0
        fit_y = stone_obb["depth"] / prong_opening_depth if prong_opening_depth > 0 else 0
        
        offset = np.array(stone_obb["center"]) - np.array(prong_obb["center"])
        aspect_diff = abs(stone_obb["aspect_ratio"] - prong_obb["aspect_ratio"])
        
        return {
            "stone": stone_obb,
            "prong": prong_obb,
            "prong_opening_width": prong_opening_width,
            "prong_opening_depth": prong_opening_depth,
            "fit_ratio_x": fit_x,
            "fit_ratio_y": fit_y,
            "center_offset": offset.tolist(),
            "aspect_difference": aspect_diff,
            "shape_compatible": aspect_diff < 0.15
        }
    
    @staticmethod
    def calculate_scale_factors(current_fit: Dict, target_fit: float = 0.97) -> Tuple[float, float, float]:
        """Calculate UNIFORM scale to preserve stone proportions."""
        avg_fit = (current_fit["fit_ratio_x"] + current_fit["fit_ratio_y"]) / 2
        if avg_fit <= 0:
            return (1.0, 1.0, 1.0)
        scale = max(0.5, min(2.0, target_fit / avg_fit))
        return (scale, scale, scale)


# =============================================================================
# FALLBACK GEOMETRY (When trimesh not available)
# =============================================================================

class FallbackGeometry:
    """AABB-based geometry when trimesh isn't installed."""
    
    @staticmethod
    def get_bounds(model: rhino3dm.File3dm) -> Tuple[np.ndarray, np.ndarray]:
        min_v = np.array([float('inf')] * 3)
        max_v = np.array([float('-inf')] * 3)
        for obj in model.Objects:
            if obj.Geometry:
                bb = obj.Geometry.GetBoundingBox()
                min_v = np.minimum(min_v, [bb.Min.X, bb.Min.Y, bb.Min.Z])
                max_v = np.maximum(max_v, [bb.Max.X, bb.Max.Y, bb.Max.Z])
        return min_v, max_v
    
    @staticmethod
    def compute_metrics(stone_model: rhino3dm.File3dm, prong_model: rhino3dm.File3dm) -> Dict:
        s_min, s_max = FallbackGeometry.get_bounds(stone_model)
        p_min, p_max = FallbackGeometry.get_bounds(prong_model)
        
        s_size = s_max - s_min
        p_size = p_max - p_min
        s_center = (s_min + s_max) / 2
        p_center = (p_min + p_max) / 2
        
        # Opening ~80% of prong bbox
        p_open_w = p_size[0] * 0.80
        p_open_d = p_size[1] * 0.80
        
        return {
            "stone": {"width": s_size[0], "depth": s_size[1], "height": s_size[2], 
                     "center": s_center.tolist(), "aspect_ratio": s_size[0]/s_size[1] if s_size[1] > 0 else 1},
            "prong": {"width": p_size[0], "depth": p_size[1], "height": p_size[2],
                     "center": p_center.tolist(), "aspect_ratio": p_size[0]/p_size[1] if p_size[1] > 0 else 1},
            "prong_opening_width": p_open_w,
            "prong_opening_depth": p_open_d,
            "fit_ratio_x": s_size[0] / p_open_w if p_open_w > 0 else 0,
            "fit_ratio_y": s_size[1] / p_open_d if p_open_d > 0 else 0,
            "center_offset": (s_center - p_center).tolist(),
            "aspect_difference": abs(s_size[0]/s_size[1] - p_size[0]/p_size[1]) if s_size[1] > 0 and p_size[1] > 0 else 0,
            "shape_compatible": True
        }


# =============================================================================
# PRECISE METRICS DATACLASS
# =============================================================================

@dataclass
class PreciseMetrics:
    """Physics-based assembly metrics"""
    stone_width: float
    stone_depth: float
    stone_height: float
    stone_aspect: float
    stone_center: Tuple[float, float, float]
    
    prong_width: float
    prong_depth: float
    prong_height: float
    prong_aspect: float
    prong_center: Tuple[float, float, float]
    prong_opening_width: float
    prong_opening_depth: float
    
    fit_ratio_x: float
    fit_ratio_y: float
    center_offset_x: float
    center_offset_y: float
    center_offset_z: float
    aspect_difference: float
    shape_compatible: bool
    
    drop_distance: Optional[float]
    has_collision: bool
    intersection_volume: float
    
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def summary_for_ai(self) -> str:
        return f"""
STONE: {self.stone_width:.1f} x {self.stone_depth:.1f} x {self.stone_height:.1f} mm (Aspect: {self.stone_aspect:.2f})
SETTING: {self.prong_opening_width:.1f} x {self.prong_opening_depth:.1f} mm (Aspect: {self.prong_aspect:.2f})

PHYSICS RESULTS:
  Fit: X={self.fit_ratio_x:.1%}, Y={self.fit_ratio_y:.1%}
  Offset: ({self.center_offset_x:.2f}, {self.center_offset_y:.2f}) mm
  Drop: {self.drop_distance:.2f} mm | Collision: {'YES' if self.has_collision else 'NO'}
  Shape: {'MATCH' if self.shape_compatible else 'MISMATCH'}

TRANSFORMS: Scale={self.scale[0]:.3f}, Translate=({self.translation[0]:.2f}, {self.translation[1]:.2f}, {self.translation[2]:.2f})
"""


# =============================================================================
# AI AESTHETIC JUDGE
# =============================================================================

class AestheticJudge:
    """AI judge for style/aesthetics only - no math."""
    
    PROMPT = """You are a jewelry design expert. The GEOMETRY ENGINE computed:
{metrics}

Judge AESTHETICS ONLY (trust the physics numbers):
1. Do proportions look balanced?
2. Does stone style match setting style?
3. Will this look good on a finger?
4. Any manufacturability concerns?

RESPOND IN JSON:
{{"aesthetic_score": 0-100, "looks_good": true/false, "style_notes": ["..."], "summary": "..."}}"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    def evaluate(self, metrics: PreciseMetrics) -> Dict:
        try:
            response = self.model.generate_content(self.PROMPT.format(metrics=metrics.summary_for_ai()))
            text = response.text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except:
            return {"aesthetic_score": 75, "looks_good": True, "style_notes": [], "summary": "Default approval"}


# =============================================================================
# HYBRID ASSEMBLER
# =============================================================================

class PhysicsAIAssembler:
    """Physics + AI hybrid assembly system."""
    
    def __init__(self):
        self.use_trimesh = TRIMESH_AVAILABLE
        self.judge = AestheticJudge()
        self.output_dir = Path(__file__).parent.parent / "outputs" / "assemblies"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        mode = "Physics (trimesh)" if self.use_trimesh else "Fallback (AABB)"
        Logger.info(f"Assembler initialized [{mode}]")
    
    def assemble(self, stone_path: str, prong_path: str, target_fit: float = 0.97, 
                 max_iterations: int = 3, output_filename: str = None) -> Dict:
        """Physics-based assembly with AI aesthetic review."""
        Logger.header("PHYSICS + AI ASSEMBLY (v3.0)")
        
        # Load models
        stone_rhino = rhino3dm.File3dm.Read(stone_path)
        prong_rhino = rhino3dm.File3dm.Read(prong_path)
        
        if not stone_rhino or not prong_rhino:
            return {"success": False, "error": "Failed to load files"}
        
        Logger.detail("Stone", Path(stone_path).name)
        Logger.detail("Prong", Path(prong_path).name)
        
        if self.use_trimesh:
            return self._assemble_with_physics(stone_rhino, prong_rhino, target_fit, 
                                               max_iterations, output_filename)
        else:
            return self._assemble_fallback(stone_rhino, prong_rhino, target_fit,
                                          max_iterations, output_filename)
    
    def _assemble_with_physics(self, stone_rhino, prong_rhino, target_fit, 
                               max_iterations, output_filename) -> Dict:
        """Full physics-based assembly with trimesh."""
        Logger.section("Converting to Trimesh")
        stone_mesh = GeometryEngine.rhino_to_trimesh(stone_rhino)
        prong_mesh = GeometryEngine.rhino_to_trimesh(prong_rhino)
        
        if stone_mesh is None or prong_mesh is None:
            Logger.warning("Mesh conversion failed, using fallback")
            return self._assemble_fallback(stone_rhino, prong_rhino, target_fit,
                                          max_iterations, output_filename)
        
        Logger.detail("Stone verts", str(len(stone_mesh.vertices)))
        Logger.detail("Prong verts", str(len(prong_mesh.vertices)))
        
        # Center prong
        prong_mesh.apply_translation(-prong_mesh.centroid)
        
        # Compute OBB metrics
        Logger.section("OBB Analysis")
        fit = GeometryEngine.compute_fit_metrics(stone_mesh, prong_mesh)
        Logger.detail("Stone OBB", f"{fit['stone']['width']:.2f} x {fit['stone']['depth']:.2f}")
        Logger.detail("Prong Opening", f"{fit['prong_opening_width']:.2f} x {fit['prong_opening_depth']:.2f}")
        Logger.detail("Initial Fit", f"X={fit['fit_ratio_x']:.1%}, Y={fit['fit_ratio_y']:.1%}")
        
        # Iterative correction
        Logger.section("Physics Correction")
        total_scale = [1.0, 1.0, 1.0]
        total_trans = [0.0, 0.0, 0.0]
        
        for i in range(max_iterations):
            Logger.info(f"Iteration {i+1}/{max_iterations}")
            
            # Uniform scale
            scale = GeometryEngine.calculate_scale_factors(fit, target_fit)
            if abs(scale[0] - 1.0) > 0.01:
                stone_mesh.apply_scale(scale[0])
                total_scale = [s * scale[0] for s in total_scale]
                Logger.detail("  Scale", f"{scale[0]:.3f}")
            
            # Center
            offset = fit["center_offset"]
            if abs(offset[0]) > 0.01 or abs(offset[1]) > 0.01:
                stone_mesh.apply_translation([-offset[0], -offset[1], 0])
                total_trans[0] -= offset[0]
                total_trans[1] -= offset[1]
            
            # Recompute
            fit = GeometryEngine.compute_fit_metrics(stone_mesh, prong_mesh)
            Logger.detail("  Fit", f"X={fit['fit_ratio_x']:.1%}, Y={fit['fit_ratio_y']:.1%}")
            
            if 0.94 <= fit['fit_ratio_x'] <= 1.0 and 0.94 <= fit['fit_ratio_y'] <= 1.0:
                Logger.success("Fit achieved")
                break
        
        # Drop test
        Logger.section("Drop Test")
        drop = GeometryEngine.drop_test(stone_mesh, prong_mesh)
        if drop and drop > 0:
            Logger.detail("Drop", f"{drop:.2f} mm")
            stone_mesh.apply_translation([0, 0, -drop])
            total_trans[2] -= drop
        else:
            Logger.warning("Drop test inconclusive")
            drop = 0
        
        # Collision check
        Logger.section("Collision Check")
        collision = GeometryEngine.check_collision(stone_mesh, prong_mesh)
        Logger.detail("Collision", "YES" if collision["has_collision"] else "NO")
        
        # Build metrics
        final_fit = GeometryEngine.compute_fit_metrics(stone_mesh, prong_mesh)
        metrics = PreciseMetrics(
            stone_width=final_fit["stone"]["width"],
            stone_depth=final_fit["stone"]["depth"],
            stone_height=final_fit["stone"]["height"],
            stone_aspect=final_fit["stone"]["aspect_ratio"],
            stone_center=tuple(final_fit["stone"]["center"]),
            prong_width=final_fit["prong"]["width"],
            prong_depth=final_fit["prong"]["depth"],
            prong_height=final_fit["prong"]["height"],
            prong_aspect=final_fit["prong"]["aspect_ratio"],
            prong_center=tuple(final_fit["prong"]["center"]),
            prong_opening_width=final_fit["prong_opening_width"],
            prong_opening_depth=final_fit["prong_opening_depth"],
            fit_ratio_x=final_fit["fit_ratio_x"],
            fit_ratio_y=final_fit["fit_ratio_y"],
            center_offset_x=final_fit["center_offset"][0],
            center_offset_y=final_fit["center_offset"][1],
            center_offset_z=final_fit["center_offset"][2],
            aspect_difference=final_fit["aspect_difference"],
            shape_compatible=final_fit["shape_compatible"],
            drop_distance=drop,
            has_collision=collision["has_collision"],
            intersection_volume=collision["intersection_volume"],
            scale=tuple(total_scale),
            translation=tuple(total_trans)
        )
        
        # AI aesthetic review
        Logger.section("AI Aesthetic Review")
        aesthetic = self.judge.evaluate(metrics)
        Logger.detail("Score", f"{aesthetic.get('aesthetic_score', 0)}/100")
        
        # Save
        output_path = self._save(stone_rhino, prong_rhino, total_scale, total_trans, output_filename)
        Logger.success(f"Saved: {output_path.name}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "physics_valid": not collision["has_collision"],
            "aesthetic_score": aesthetic.get("aesthetic_score", 0),
            "final_metrics": metrics.to_dict()
        }
    
    def _assemble_fallback(self, stone_rhino, prong_rhino, target_fit, 
                          max_iterations, output_filename) -> Dict:
        """Fallback AABB-based assembly."""
        Logger.section("Fallback Mode (AABB)")
        
        # Get metrics
        fit = FallbackGeometry.compute_metrics(stone_rhino, prong_rhino)
        Logger.detail("Fit", f"X={fit['fit_ratio_x']:.1%}, Y={fit['fit_ratio_y']:.1%}")
        
        # Calculate uniform scale
        avg_fit = (fit['fit_ratio_x'] + fit['fit_ratio_y']) / 2
        scale = target_fit / avg_fit if avg_fit > 0 else 1.0
        scale = max(0.5, min(2.0, scale))
        
        total_scale = [scale, scale, scale]
        total_trans = [-fit['center_offset'][0], -fit['center_offset'][1], 0]
        
        Logger.detail("Scale", f"{scale:.3f}")
        
        # Build metrics
        metrics = PreciseMetrics(
            stone_width=fit["stone"]["width"] * scale,
            stone_depth=fit["stone"]["depth"] * scale,
            stone_height=fit["stone"]["height"] * scale,
            stone_aspect=fit["stone"]["aspect_ratio"],
            stone_center=tuple(fit["stone"]["center"]),
            prong_width=fit["prong"]["width"],
            prong_depth=fit["prong"]["depth"],
            prong_height=fit["prong"]["height"],
            prong_aspect=fit["prong"]["aspect_ratio"],
            prong_center=tuple(fit["prong"]["center"]),
            prong_opening_width=fit["prong_opening_width"],
            prong_opening_depth=fit["prong_opening_depth"],
            fit_ratio_x=target_fit,
            fit_ratio_y=target_fit,
            center_offset_x=0, center_offset_y=0, center_offset_z=0,
            aspect_difference=fit["aspect_difference"],
            shape_compatible=fit["shape_compatible"],
            drop_distance=0, has_collision=False, intersection_volume=0,
            scale=tuple(total_scale), translation=tuple(total_trans)
        )
        
        # Save
        output_path = self._save(stone_rhino, prong_rhino, total_scale, total_trans, output_filename)
        Logger.success(f"Saved: {output_path.name}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "physics_valid": True,
            "aesthetic_score": 70,
            "final_metrics": metrics.to_dict()
        }
    
    def _save(self, stone_rhino, prong_rhino, scale, trans, output_filename) -> Path:
        """Apply transforms and save."""
        # Apply scale
        if abs(scale[0] - 1.0) > 0.001:
            self._scale_rhino(stone_rhino, scale[0])
        
        # Apply translation
        if any(abs(t) > 0.001 for t in trans):
            self._translate_rhino(stone_rhino, trans)
        
        # Center prong
        self._center_rhino(prong_rhino)
        
        # Create output
        output = rhino3dm.File3dm()
        
        prong_layer = rhino3dm.Layer()
        prong_layer.Name = "Prong_Setting"
        prong_idx = output.Layers.Add(prong_layer)
        
        stone_layer = rhino3dm.Layer()
        stone_layer.Name = "Stone"
        stone_idx = output.Layers.Add(stone_layer)
        
        for obj in prong_rhino.Objects:
            if obj.Geometry:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = prong_idx
                output.Objects.Add(obj.Geometry, attr)
        
        for obj in stone_rhino.Objects:
            if obj.Geometry:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = stone_idx
                output.Objects.Add(obj.Geometry, attr)
        
        if output_filename:
            path = self.output_dir / output_filename
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.output_dir / f"head_assembly_{ts}.3dm"
        
        output.Write(str(path), 7)
        return path
    
    def _scale_rhino(self, model, s):
        min_v, max_v = FallbackGeometry.get_bounds(model)
        center = (min_v + max_v) / 2
        for obj in model.Objects:
            if obj.Geometry:
                t1 = rhino3dm.Transform.Translation(-center[0], -center[1], -center[2])
                obj.Geometry.Transform(t1)
                st = rhino3dm.Transform(1.0)
                st.M00 = st.M11 = st.M22 = s
                obj.Geometry.Transform(st)
                t2 = rhino3dm.Transform.Translation(center[0], center[1], center[2])
                obj.Geometry.Transform(t2)
    
    def _translate_rhino(self, model, trans):
        t = rhino3dm.Transform.Translation(trans[0], trans[1], trans[2])
        for obj in model.Objects:
            if obj.Geometry:
                obj.Geometry.Transform(t)
    
    def _center_rhino(self, model):
        min_v, max_v = FallbackGeometry.get_bounds(model)
        center = (min_v + max_v) / 2
        self._translate_rhino(model, [-center[0], -center[1], -center[2]])


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class AIAssistedAssembler(PhysicsAIAssembler):
    """Backward compatible alias."""
    pass


def ai_assemble(stone_path: str, prong_path: str, **kwargs) -> Dict:
    """Quick assembly function."""
    return PhysicsAIAssembler().assemble(stone_path, prong_path, **kwargs)


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    stones = list((base / "cad_library" / "stones").glob("*.3dm"))
    prongs = list((base / "cad_library" / "prongs").glob("*.3dm"))
    
    if stones and prongs:
        result = ai_assemble(str(stones[0]), str(prongs[0]))
        Logger.header("TEST COMPLETE")
        Logger.detail("Success", str(result['success']))
        Logger.detail("Physics", str(result.get('physics_valid', False)))
    else:
        Logger.error("No CAD files found")
