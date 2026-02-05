"""
Smart AI-Powered Assembly System
Uses Gemini to dynamically analyze CAD assemblies and determine corrections.

Key features:
1. AI analyzes geometry metrics (not hardcoded rules)
2. AI reasons about what's wrong and how to fix it
3. AI generates correction factors for each unique assembly
4. Iterative refinement until AI approves

This replaces rule-based thresholds with intelligent reasoning.
"""
import json
import math
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import rhino3dm
    import google.generativeai as genai
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    raise

from config import GEMINI_API_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class Logger:
    """Professional logging utility"""
    
    @staticmethod
    def header(title: str, char: str = "="):
        """Print a header"""
        print(f"\n{char * 60}")
        print(f"  {title}")
        print(f"{char * 60}")
    
    @staticmethod
    def section(title: str):
        """Print a section header"""
        print(f"\n[{title}]")
    
    @staticmethod
    def info(msg: str):
        """Print info message"""
        print(f"  {msg}")
    
    @staticmethod
    def detail(label: str, value: str):
        """Print a label-value pair"""
        print(f"  {label}: {value}")
    
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
    
    @staticmethod
    def step(num: int, total: int, msg: str):
        """Print step progress"""
        print(f"\n  Step {num}/{total}: {msg}")


@dataclass
class AssemblyMetrics:
    """Complete metrics of an assembly for AI analysis"""
    # Stone measurements
    stone_width: float
    stone_depth: float
    stone_height: float
    stone_girdle_width: float
    stone_girdle_depth: float
    stone_girdle_z: float
    stone_center: Tuple[float, float, float]
    stone_min_z: float
    stone_max_z: float
    
    # Prong measurements
    prong_width: float
    prong_depth: float
    prong_height: float
    prong_opening_width: float
    prong_opening_depth: float
    prong_opening_z: float
    prong_center: Tuple[float, float, float]
    prong_min_z: float
    prong_max_z: float
    
    # Derived metrics
    fit_ratio_x: float
    fit_ratio_y: float
    center_offset_x: float
    center_offset_y: float
    girdle_to_opening_z: float
    depth_ratio: float
    
    # Scale applied
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def summary(self) -> str:
        """Human-readable summary for AI"""
        return f"""
STONE:
  Dimensions: {self.stone_width:.2f} x {self.stone_depth:.2f} x {self.stone_height:.2f} mm
  Girdle: {self.stone_girdle_width:.2f} x {self.stone_girdle_depth:.2f} mm at Z={self.stone_girdle_z:.2f}
  Center: ({self.stone_center[0]:.2f}, {self.stone_center[1]:.2f}, {self.stone_center[2]:.2f})

PRONG:
  Dimensions: {self.prong_width:.2f} x {self.prong_depth:.2f} x {self.prong_height:.2f} mm
  Opening: {self.prong_opening_width:.2f} x {self.prong_opening_depth:.2f} mm at Z={self.prong_opening_z:.2f}
  Center: ({self.prong_center[0]:.2f}, {self.prong_center[1]:.2f}, {self.prong_center[2]:.2f})

FIT ANALYSIS:
  Fit ratio X: {self.fit_ratio_x:.1%} (stone_girdle / prong_opening)
  Fit ratio Y: {self.fit_ratio_y:.1%}
  Center offset X: {self.center_offset_x:.3f} mm
  Center offset Y: {self.center_offset_y:.3f} mm
  Girdle to opening Z: {self.girdle_to_opening_z:.2f} mm (positive = stone inside prong)
  Depth ratio: {self.depth_ratio:.1%} (how deep stone sits in prong)

CURRENT SCALE APPLIED:
  Scale X: {self.scale_x:.3f}
  Scale Y: {self.scale_y:.3f}
  Scale Z: {self.scale_z:.3f}
"""


class SmartAssemblyAI:
    """
    AI-powered assembly analyzer and corrector.
    Uses Gemini to reason about assembly quality and generate corrections.
    """
    
    ANALYSIS_PROMPT = """You are an expert jewelry CAD engineer analyzing a stone-prong assembly.

COORDINATE SYSTEM:
- Z axis points UP (positive Z = higher, negative Z = lower)
- The prong opening is near the TOP of the prong (high Z)
- Stone should sit WITH its girdle BELOW the prong opening

ASSEMBLY METRICS:
{metrics}

JEWELRY ASSEMBLY RULES:
1. STONE FIT: The stone's girdle (widest part) should fit inside the prong opening with 2-5% clearance
   - Fit ratio 95-98% is IDEAL (stone slightly smaller than opening for secure setting)
   - Fit ratio > 100% means stone is TOO BIG - scale DOWN
   - Fit ratio < 90% means stone is TOO SMALL - scale UP

2. STONE POSITION: Stone must be centered on prong opening
   - Center offset should be < 0.1mm in X and Y
   
3. STONE DEPTH: Stone's girdle should be BELOW the prong opening
   - Depth ratio = how far girdle drops into prong (as % of prong height)
   - Depth ratio 20-35% is IDEAL
   - Depth ratio < 15% means stone is too HIGH (floating) - move stone DOWN (negative Z)
   - Depth ratio > 50% means stone is too LOW (deep) - move stone UP (positive Z)

4. PROPORTIONS: Try to maintain original stone aspect ratio when possible

ANALYZE AND RESPOND IN JSON:
{{
    "is_valid": true/false,
    "quality_score": 0-100,
    "issues": [
        {{
            "type": "fit_x|fit_y|position_x|position_y|depth|proportion",
            "severity": "critical|warning|minor",
            "description": "what's wrong",
            "current_value": number,
            "ideal_value": number
        }}
    ],
    "corrections": {{
        "scale_x": float (multiply by this, 1.0 = no change, <1 = shrink, >1 = grow),
        "scale_y": float,
        "scale_z": float,
        "translate_x": float (mm),
        "translate_y": float (mm),
        "translate_z": float (POSITIVE = move UP, NEGATIVE = move DOWN)
    }},
    "reasoning": "explain your analysis"
}}

IMPORTANT: If depth ratio is too HIGH (>50%), translate_z should be POSITIVE (move stone UP).
If depth ratio is too LOW (<15%), translate_z should be NEGATIVE (move stone DOWN)."""

    FINAL_CHECK_PROMPT = """You are an expert jewelry CAD engineer doing FINAL QUALITY CHECK on a stone-prong assembly.

ASSEMBLY METRICS AFTER CORRECTIONS:
{metrics}

This is iteration {iteration} of the assembly process. Previous corrections have been applied.

CRITICAL CHECKS:
1. Is the stone properly sized for the prong? (fit ratio 95-98%)
2. Is the stone centered? (offset < 0.15mm)
3. Is the stone at correct depth? (depth ratio 15-40%)
4. Are proportions acceptable?

RESPOND IN JSON:
{{
    "approved": true/false,
    "quality_score": 0-100,
    "remaining_issues": ["list of any remaining problems"],
    "final_adjustments": {{
        "scale_x": float (1.0 if no adjustment needed),
        "scale_y": float,
        "scale_z": float,
        "translate_x": float,
        "translate_y": float,
        "translate_z": float
    }},
    "summary": "final assessment"
}}

Be lenient on minor issues (< 0.1mm offsets, 2-3% fit variance). Focus on structural correctness."""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        Logger.info("AI Assembly Engine initialized (Gemini 2.5 Flash)")
    
    def extract_metrics(
        self,
        stone_model: rhino3dm.File3dm,
        prong_model: rhino3dm.File3dm,
        current_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> AssemblyMetrics:
        """Extract comprehensive metrics from CAD models"""
        
        # Get stone bounding box
        stone_min = [float('inf')] * 3
        stone_max = [float('-inf')] * 3
        for obj in stone_model.Objects:
            if obj.Geometry:
                bb = obj.Geometry.GetBoundingBox()
                stone_min[0] = min(stone_min[0], bb.Min.X)
                stone_min[1] = min(stone_min[1], bb.Min.Y)
                stone_min[2] = min(stone_min[2], bb.Min.Z)
                stone_max[0] = max(stone_max[0], bb.Max.X)
                stone_max[1] = max(stone_max[1], bb.Max.Y)
                stone_max[2] = max(stone_max[2], bb.Max.Z)
        
        stone_width = stone_max[0] - stone_min[0]
        stone_depth = stone_max[1] - stone_min[1]
        stone_height = stone_max[2] - stone_min[2]
        stone_center = (
            (stone_min[0] + stone_max[0]) / 2,
            (stone_min[1] + stone_max[1]) / 2,
            (stone_min[2] + stone_max[2]) / 2
        )
        
        # Stone girdle (widest point, typically at 35% height)
        stone_girdle_z = stone_min[2] + stone_height * 0.35
        stone_girdle_width = stone_width  # Approximate
        stone_girdle_depth = stone_depth
        
        # Get prong bounding box
        prong_min = [float('inf')] * 3
        prong_max = [float('-inf')] * 3
        for obj in prong_model.Objects:
            if obj.Geometry:
                bb = obj.Geometry.GetBoundingBox()
                prong_min[0] = min(prong_min[0], bb.Min.X)
                prong_min[1] = min(prong_min[1], bb.Min.Y)
                prong_min[2] = min(prong_min[2], bb.Min.Z)
                prong_max[0] = max(prong_max[0], bb.Max.X)
                prong_max[1] = max(prong_max[1], bb.Max.Y)
                prong_max[2] = max(prong_max[2], bb.Max.Z)
        
        prong_width = prong_max[0] - prong_min[0]
        prong_depth = prong_max[1] - prong_min[1]
        prong_height = prong_max[2] - prong_min[2]
        prong_center = (
            (prong_min[0] + prong_max[0]) / 2,
            (prong_min[1] + prong_max[1]) / 2,
            (prong_min[2] + prong_max[2]) / 2
        )
        
        # Prong opening (typically 80% of bbox, near top)
        prong_opening_width = prong_width * 0.80
        prong_opening_depth = prong_depth * 0.80
        prong_opening_z = prong_max[2] - prong_height * 0.12
        
        # Calculate derived metrics
        fit_ratio_x = stone_girdle_width / prong_opening_width if prong_opening_width > 0 else 0
        fit_ratio_y = stone_girdle_depth / prong_opening_depth if prong_opening_depth > 0 else 0
        center_offset_x = stone_center[0] - prong_center[0]
        center_offset_y = stone_center[1] - prong_center[1]
        girdle_to_opening_z = prong_opening_z - stone_girdle_z
        depth_ratio = girdle_to_opening_z / prong_height if prong_height > 0 else 0
        
        return AssemblyMetrics(
            stone_width=stone_width,
            stone_depth=stone_depth,
            stone_height=stone_height,
            stone_girdle_width=stone_girdle_width,
            stone_girdle_depth=stone_girdle_depth,
            stone_girdle_z=stone_girdle_z,
            stone_center=stone_center,
            stone_min_z=stone_min[2],
            stone_max_z=stone_max[2],
            prong_width=prong_width,
            prong_depth=prong_depth,
            prong_height=prong_height,
            prong_opening_width=prong_opening_width,
            prong_opening_depth=prong_opening_depth,
            prong_opening_z=prong_opening_z,
            prong_center=prong_center,
            prong_min_z=prong_min[2],
            prong_max_z=prong_max[2],
            fit_ratio_x=fit_ratio_x,
            fit_ratio_y=fit_ratio_y,
            center_offset_x=center_offset_x,
            center_offset_y=center_offset_y,
            girdle_to_opening_z=girdle_to_opening_z,
            depth_ratio=depth_ratio,
            scale_x=current_scale[0],
            scale_y=current_scale[1],
            scale_z=current_scale[2]
        )
    
    def extract_metrics_from_assembly(
        self,
        assembly_model: rhino3dm.File3dm
    ) -> AssemblyMetrics:
        """Extract metrics from an already-assembled model with layers"""
        
        stone_objects = []
        prong_objects = []
        
        for obj in assembly_model.Objects:
            layer_idx = obj.Attributes.LayerIndex
            if layer_idx < len(assembly_model.Layers):
                layer_name = assembly_model.Layers[layer_idx].Name.lower()
                if "stone" in layer_name:
                    stone_objects.append(obj)
                elif "prong" in layer_name or "setting" in layer_name:
                    prong_objects.append(obj)
        
        # Calculate bounding boxes for each group
        def get_bbox(objects):
            min_v = [float('inf')] * 3
            max_v = [float('-inf')] * 3
            for obj in objects:
                if obj.Geometry:
                    bb = obj.Geometry.GetBoundingBox()
                    min_v[0] = min(min_v[0], bb.Min.X)
                    min_v[1] = min(min_v[1], bb.Min.Y)
                    min_v[2] = min(min_v[2], bb.Min.Z)
                    max_v[0] = max(max_v[0], bb.Max.X)
                    max_v[1] = max(max_v[1], bb.Max.Y)
                    max_v[2] = max(max_v[2], bb.Max.Z)
            return min_v, max_v
        
        stone_min, stone_max = get_bbox(stone_objects)
        prong_min, prong_max = get_bbox(prong_objects)
        
        # Calculate all metrics
        stone_width = stone_max[0] - stone_min[0]
        stone_depth = stone_max[1] - stone_min[1]
        stone_height = stone_max[2] - stone_min[2]
        stone_center = tuple((stone_min[i] + stone_max[i]) / 2 for i in range(3))
        stone_girdle_z = stone_min[2] + stone_height * 0.35
        
        prong_width = prong_max[0] - prong_min[0]
        prong_depth = prong_max[1] - prong_min[1]
        prong_height = prong_max[2] - prong_min[2]
        prong_center = tuple((prong_min[i] + prong_max[i]) / 2 for i in range(3))
        prong_opening_width = prong_width * 0.80
        prong_opening_depth = prong_depth * 0.80
        prong_opening_z = prong_max[2] - prong_height * 0.12
        
        fit_ratio_x = stone_width / prong_opening_width if prong_opening_width > 0 else 0
        fit_ratio_y = stone_depth / prong_opening_depth if prong_opening_depth > 0 else 0
        center_offset_x = stone_center[0] - prong_center[0]
        center_offset_y = stone_center[1] - prong_center[1]
        girdle_to_opening_z = prong_opening_z - stone_girdle_z
        depth_ratio = girdle_to_opening_z / prong_height if prong_height > 0 else 0
        
        return AssemblyMetrics(
            stone_width=stone_width,
            stone_depth=stone_depth,
            stone_height=stone_height,
            stone_girdle_width=stone_width,
            stone_girdle_depth=stone_depth,
            stone_girdle_z=stone_girdle_z,
            stone_center=stone_center,
            stone_min_z=stone_min[2],
            stone_max_z=stone_max[2],
            prong_width=prong_width,
            prong_depth=prong_depth,
            prong_height=prong_height,
            prong_opening_width=prong_opening_width,
            prong_opening_depth=prong_opening_depth,
            prong_opening_z=prong_opening_z,
            prong_center=prong_center,
            prong_min_z=prong_min[2],
            prong_max_z=prong_max[2],
            fit_ratio_x=fit_ratio_x,
            fit_ratio_y=fit_ratio_y,
            center_offset_x=center_offset_x,
            center_offset_y=center_offset_y,
            girdle_to_opening_z=girdle_to_opening_z,
            depth_ratio=depth_ratio
        )
    
    def analyze_and_correct(
        self,
        metrics: AssemblyMetrics,
        iteration: int = 1
    ) -> Dict:
        """Use AI to analyze assembly and determine corrections"""
        
        prompt = self.ANALYSIS_PROMPT.format(metrics=metrics.summary())
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            result = json.loads(text)
            return result
            
        except Exception as e:
            Logger.warning(f"AI analysis error: {e}")
            # Fallback to basic heuristics
            return self._fallback_analysis(metrics)
    
    def final_check(
        self,
        metrics: AssemblyMetrics,
        iteration: int
    ) -> Dict:
        """AI final quality check"""
        
        prompt = self.FINAL_CHECK_PROMPT.format(
            metrics=metrics.summary(),
            iteration=iteration
        )
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            result = json.loads(text)
            return result
            
        except Exception as e:
            Logger.warning(f"AI check error: {e}")
            return {
                "approved": metrics.fit_ratio_x > 0.92 and metrics.fit_ratio_x < 1.02,
                "quality_score": 70,
                "remaining_issues": [],
                "final_adjustments": {
                    "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0,
                    "translate_x": 0, "translate_y": 0, "translate_z": 0
                },
                "summary": "Fallback approval based on basic metrics"
            }
    
    def _fallback_analysis(self, metrics: AssemblyMetrics) -> Dict:
        """Fallback heuristic analysis if AI fails"""
        issues = []
        corrections = {
            "scale_x": 1.0, "scale_y": 1.0, "scale_z": 1.0,
            "translate_x": 0.0, "translate_y": 0.0, "translate_z": 0.0
        }
        
        # Check fit
        target_fit = 0.97
        if metrics.fit_ratio_x > 1.0 or metrics.fit_ratio_x < 0.90:
            corrections["scale_x"] = target_fit / metrics.fit_ratio_x
            issues.append({
                "type": "fit_x",
                "severity": "critical" if metrics.fit_ratio_x > 1.0 else "warning",
                "description": f"Fit ratio X is {metrics.fit_ratio_x:.1%}",
                "current_value": metrics.fit_ratio_x,
                "ideal_value": target_fit
            })
        
        if metrics.fit_ratio_y > 1.0 or metrics.fit_ratio_y < 0.90:
            corrections["scale_y"] = target_fit / metrics.fit_ratio_y
            issues.append({
                "type": "fit_y", 
                "severity": "critical" if metrics.fit_ratio_y > 1.0 else "warning",
                "description": f"Fit ratio Y is {metrics.fit_ratio_y:.1%}",
                "current_value": metrics.fit_ratio_y,
                "ideal_value": target_fit
            })
        
        # Check depth
        if metrics.depth_ratio < 0.15:
            move_down = (0.25 - metrics.depth_ratio) * metrics.prong_height
            corrections["translate_z"] = move_down
            issues.append({
                "type": "depth",
                "severity": "warning",
                "description": f"Stone not deep enough ({metrics.depth_ratio:.1%})",
                "current_value": metrics.depth_ratio,
                "ideal_value": 0.25
            })
        
        corrections["scale_z"] = (corrections["scale_x"] + corrections["scale_y"]) / 2
        
        return {
            "is_valid": len([i for i in issues if i["severity"] == "critical"]) == 0,
            "quality_score": max(0, 100 - len(issues) * 15),
            "issues": issues,
            "corrections": corrections,
            "reasoning": "Fallback heuristic analysis"
        }


class AIAssistedAssembler:
    """
    Complete AI-assisted assembly pipeline.
    Uses SmartAssemblyAI for dynamic analysis and correction.
    Returns only the final assembled model path.
    """
    
    def __init__(self):
        self.ai = SmartAssemblyAI()
        self.output_dir = Path(__file__).parent.parent / "outputs" / "assemblies"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def assemble(
        self,
        stone_path: str,
        prong_path: str,
        max_iterations: int = 5,
        ring_size: float = 7.0,
        output_filename: str = None
    ) -> Dict:
        """
        AI-assisted assembly with iterative correction.
        Returns single output file path.
        """
        Logger.header("AI-POWERED ASSEMBLY")
        
        # Load models
        Logger.section("Loading CAD Models")
        stone_model = rhino3dm.File3dm.Read(stone_path)
        prong_model = rhino3dm.File3dm.Read(prong_path)
        
        if not stone_model or not prong_model:
            Logger.error("Failed to load CAD models")
            return {"success": False, "error": "Failed to load models"}
        
        Logger.detail("Stone", f"{Path(stone_path).name} ({len(stone_model.Objects)} objects)")
        Logger.detail("Prong", f"{Path(prong_path).name} ({len(prong_model.Objects)} objects)")
        
        # Center prong at origin
        self._center_model(prong_model)
        
        # Initial metrics
        metrics = self.ai.extract_metrics(stone_model, prong_model)
        
        Logger.section("Initial Analysis")
        Logger.detail("Stone size", f"{metrics.stone_width:.2f} x {metrics.stone_depth:.2f} mm")
        Logger.detail("Prong opening", f"{metrics.prong_opening_width:.2f} x {metrics.prong_opening_depth:.2f} mm")
        Logger.detail("Initial fit", f"X={metrics.fit_ratio_x:.1%}, Y={metrics.fit_ratio_y:.1%}")
        
        # Cumulative transformations
        total_scale = [1.0, 1.0, 1.0]
        total_translate = [0.0, 0.0, 0.0]
        
        # Iterative correction loop
        Logger.section("Iterative Correction")
        for iteration in range(1, max_iterations + 1):
            Logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Get AI analysis
            analysis = self.ai.analyze_and_correct(metrics, iteration)
            
            quality = analysis.get('quality_score', 'N/A')
            is_valid = analysis.get('is_valid', False)
            Logger.detail("Quality", f"{quality}/100")
            
            if analysis.get('issues'):
                for issue in analysis['issues'][:2]:
                    Logger.info(f"  - [{issue['severity']}] {issue['description']}")
            
            # Check if already valid
            if is_valid and quality >= 85:
                Logger.success("Assembly approved by AI")
                break
            
            # Apply corrections
            corrections = analysis.get('corrections', {})
            scale_x = corrections.get('scale_x', 1.0)
            scale_y = corrections.get('scale_y', 1.0)
            scale_z = corrections.get('scale_z', 1.0)
            
            Logger.detail("Correction", f"Scale: {scale_x:.3f} x {scale_y:.3f} x {scale_z:.3f}")
            
            # Update cumulative transforms
            total_scale[0] *= scale_x
            total_scale[1] *= scale_y
            total_scale[2] *= scale_z
            total_translate[0] += corrections.get('translate_x', 0.0)
            total_translate[1] += corrections.get('translate_y', 0.0)
            total_translate[2] += corrections.get('translate_z', 0.0)
            
            # Apply scale to stone
            self._apply_scale(stone_model, scale_x, scale_y, scale_z)
            
            # Apply translation
            translate_x = corrections.get('translate_x', 0.0)
            translate_y = corrections.get('translate_y', 0.0)
            translate_z = corrections.get('translate_z', 0.0)
            
            if abs(translate_x) > 0.001 or abs(translate_y) > 0.001 or abs(translate_z) > 0.001:
                self._translate_model(stone_model, (translate_x, translate_y, translate_z))
            
            # Re-extract metrics
            metrics = self.ai.extract_metrics(stone_model, prong_model, tuple(total_scale))
            Logger.detail("Result", f"Fit: X={metrics.fit_ratio_x:.1%}, Y={metrics.fit_ratio_y:.1%}, Depth: {metrics.depth_ratio:.1%}")
        
        # Final quality check
        Logger.section("Final Quality Check")
        final_metrics = self.ai.extract_metrics(stone_model, prong_model, tuple(total_scale))
        final_check = self.ai.final_check(final_metrics, iteration)
        
        if not final_check.get('approved', False):
            Logger.info("Applying final adjustments...")
            final_adj = final_check.get('final_adjustments', {})
            
            if abs(final_adj.get('scale_x', 1.0) - 1.0) > 0.01 or \
               abs(final_adj.get('scale_y', 1.0) - 1.0) > 0.01:
                self._apply_scale(
                    stone_model,
                    final_adj.get('scale_x', 1.0),
                    final_adj.get('scale_y', 1.0),
                    final_adj.get('scale_z', 1.0)
                )
            
            tx = final_adj.get('translate_x', 0.0)
            ty = final_adj.get('translate_y', 0.0)
            tz = final_adj.get('translate_z', 0.0)
            if abs(tx) > 0.01 or abs(ty) > 0.01 or abs(tz) > 0.01:
                self._translate_model(stone_model, (tx, ty, tz))
            
            final_metrics = self.ai.extract_metrics(stone_model, prong_model, tuple(total_scale))
        
        Logger.detail("Quality Score", f"{final_check.get('quality_score', 0)}/100")
        Logger.detail("Status", "Approved" if final_check.get('approved', False) else "Completed with warnings")
        
        # Create output model
        if output_filename:
            output_path = self.output_dir / output_filename
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"head_assembly_{timestamp}.3dm"
        
        output_model = rhino3dm.File3dm()
        
        # Add layers
        prong_layer = rhino3dm.Layer()
        prong_layer.Name = "Prong_Setting"
        prong_idx = output_model.Layers.Add(prong_layer)
        
        stone_layer = rhino3dm.Layer()
        stone_layer.Name = "Stone"
        stone_idx = output_model.Layers.Add(stone_layer)
        
        # Add objects
        for obj in prong_model.Objects:
            if obj.Geometry:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = prong_idx
                output_model.Objects.Add(obj.Geometry, attr)
        
        for obj in stone_model.Objects:
            if obj.Geometry:
                attr = rhino3dm.ObjectAttributes()
                attr.LayerIndex = stone_idx
                output_model.Objects.Add(obj.Geometry, attr)
        
        output_model.Write(str(output_path), 7)
        
        Logger.section("Assembly Complete")
        Logger.detail("Output", output_path.name)
        Logger.detail("Iterations", str(iteration))
        Logger.detail("Final Scale", f"{total_scale[0]:.3f} x {total_scale[1]:.3f} x {total_scale[2]:.3f}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "iterations": iteration,
            "total_scale": total_scale,
            "quality_score": final_check.get('quality_score', 0),
            "approved": final_check.get('approved', False),
            "final_metrics": final_metrics.to_dict()
        }
    
    def _center_model(self, model: rhino3dm.File3dm):
        """Center model at origin"""
        min_v = [float('inf')] * 3
        max_v = [float('-inf')] * 3
        for obj in model.Objects:
            if obj.Geometry:
                bb = obj.Geometry.GetBoundingBox()
                for i in range(3):
                    min_v[i] = min(min_v[i], [bb.Min.X, bb.Min.Y, bb.Min.Z][i])
                    max_v[i] = max(max_v[i], [bb.Max.X, bb.Max.Y, bb.Max.Z][i])
        
        center = tuple((min_v[i] + max_v[i]) / 2 for i in range(3))
        self._translate_model(model, (-center[0], -center[1], -center[2]))
    
    def _translate_model(self, model: rhino3dm.File3dm, translation: Tuple[float, float, float]):
        """Translate all objects in model"""
        transform = rhino3dm.Transform.Translation(*translation)
        for obj in model.Objects:
            if obj.Geometry:
                obj.Geometry.Transform(transform)
    
    def _apply_scale(
        self,
        model: rhino3dm.File3dm,
        scale_x: float,
        scale_y: float,
        scale_z: float
    ):
        """Apply non-uniform scale to model around its center"""
        # First get center
        min_v = [float('inf')] * 3
        max_v = [float('-inf')] * 3
        for obj in model.Objects:
            if obj.Geometry:
                bb = obj.Geometry.GetBoundingBox()
                for i in range(3):
                    min_v[i] = min(min_v[i], [bb.Min.X, bb.Min.Y, bb.Min.Z][i])
                    max_v[i] = max(max_v[i], [bb.Max.X, bb.Max.Y, bb.Max.Z][i])
        
        center = tuple((min_v[i] + max_v[i]) / 2 for i in range(3))
        
        for obj in model.Objects:
            if obj.Geometry:
                # Move to origin
                t1 = rhino3dm.Transform.Translation(-center[0], -center[1], -center[2])
                obj.Geometry.Transform(t1)
                
                # Scale
                scale_t = rhino3dm.Transform(1.0)
                scale_t.M00 = scale_x
                scale_t.M11 = scale_y
                scale_t.M22 = scale_z
                obj.Geometry.Transform(scale_t)
                
                # Move back
                t2 = rhino3dm.Transform.Translation(center[0], center[1], center[2])
                obj.Geometry.Transform(t2)


# Convenience function
def ai_assemble(stone_path: str, prong_path: str, **kwargs) -> Dict:
    """Quick AI-assisted assembly"""
    assembler = AIAssistedAssembler()
    return assembler.assemble(stone_path, prong_path, **kwargs)


if __name__ == "__main__":
    # Test with sample files
    import sys
    
    base_dir = Path(__file__).parent.parent
    stones_dir = base_dir / "cad_library" / "stones"
    prongs_dir = base_dir / "cad_library" / "prongs"
    
    stones = list(stones_dir.glob("*.3dm"))
    prongs = list(prongs_dir.glob("*.3dm"))
    
    if stones and prongs:
        result = ai_assemble(str(stones[0]), str(prongs[0]))
        Logger.header("TEST COMPLETE")
        Logger.detail("Success", str(result['success']))
        Logger.detail("Quality", f"{result.get('quality_score', 0)}/100")
    else:
        Logger.error("No CAD files found for testing")
