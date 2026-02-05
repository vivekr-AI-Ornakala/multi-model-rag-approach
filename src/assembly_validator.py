"""
Assembly Validator
Validates assembled CAD files and detects issues with stone/prong fit,
positioning, and scaling. Returns detailed diagnostics and correction factors.

Key validations:
1. Stone-Prong Fit: Is the stone properly sized for the prong opening?
2. Stone Position: Is the stone centered and at the correct depth?
3. Component Overlap: Are there problematic intersections?
4. Proportions: Are the relative sizes correct?
"""
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import rhino3dm
except ImportError:
    print("❌ rhino3dm not installed. Install with: pip install rhino3dm")
    raise


class ValidationStatus(Enum):
    """Status of validation check"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    category: str
    severity: ValidationStatus
    message: str
    current_value: float
    expected_value: float
    correction_factor: float = 1.0


@dataclass
class ValidationResult:
    """Complete validation result"""
    is_valid: bool
    overall_status: ValidationStatus
    issues: List[ValidationIssue]
    stone_bbox: Dict
    prong_bbox: Dict
    fit_metrics: Dict
    position_metrics: Dict
    correction_factors: Dict
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "overall_status": self.overall_status.value,
            "issues": [asdict(i) for i in self.issues],
            "stone_bbox": self.stone_bbox,
            "prong_bbox": self.prong_bbox,
            "fit_metrics": self.fit_metrics,
            "position_metrics": self.position_metrics,
            "correction_factors": self.correction_factors
        }


class AssemblyValidator:
    """
    Validates assembled CAD files for correct stone-prong fitting.
    
    Validation criteria:
    1. FIT: Stone girdle should be 94-98% of prong opening (2-6% clearance)
    2. POSITION: Stone center should align with prong center (±0.1mm)
    3. DEPTH: Stone girdle should be inside prong (not floating above)
    4. PROPORTIONS: Stone shouldn't be too small or too large for prong
    """
    
    # Validation thresholds
    TARGET_FIT_RATIO = 0.97  # Stone girdle = 97% of opening (3% clearance)
    FIT_TOLERANCE = 0.05     # ±5% acceptable
    POSITION_TOLERANCE = 0.15  # ±0.15mm center alignment
    MIN_DEPTH_RATIO = 0.15   # Stone must drop at least 15% into prong
    MAX_DEPTH_RATIO = 0.50   # Stone shouldn't drop more than 50% into prong
    
    # Opening ratio estimates for different setting types
    OPENING_RATIOS = {
        "prong": 0.80,
        "bezel": 0.88,
        "cathedral": 0.82,
        "halo": 0.85,
        "default": 0.82
    }
    
    def __init__(self):
        self.model = None
        self.stone_objects = []
        self.prong_objects = []
        self.shank_objects = []
    
    def load_assembly(self, file_path: str) -> bool:
        """Load an assembled .3dm file"""
        try:
            self.model = rhino3dm.File3dm.Read(file_path)
            if not self.model:
                print(f"❌ Failed to read: {file_path}")
                return False
            
            # Categorize objects by layer
            self.stone_objects = []
            self.prong_objects = []
            self.shank_objects = []
            
            for obj in self.model.Objects:
                layer_idx = obj.Attributes.LayerIndex
                if layer_idx < len(self.model.Layers):
                    layer_name = self.model.Layers[layer_idx].Name.lower()
                    
                    if "stone" in layer_name:
                        self.stone_objects.append(obj)
                    elif "prong" in layer_name or "setting" in layer_name:
                        self.prong_objects.append(obj)
                    elif "shank" in layer_name:
                        self.shank_objects.append(obj)
            
            print(f"   Loaded: {len(self.stone_objects)} stone, {len(self.prong_objects)} prong, {len(self.shank_objects)} shank objects")
            return True
            
        except Exception as e:
            print(f"❌ Error loading assembly: {e}")
            return False
    
    def get_layer_bbox(self, objects: List) -> Optional[Dict]:
        """Get combined bounding box for a list of objects"""
        if not objects:
            return None
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for obj in objects:
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
        
        if min_x == float('inf'):
            return None
        
        return {
            "min": [min_x, min_y, min_z],
            "max": [max_x, max_y, max_z],
            "width": max_x - min_x,
            "depth": max_y - min_y,
            "height": max_z - min_z,
            "center": [
                (min_x + max_x) / 2,
                (min_y + max_y) / 2,
                (min_z + max_z) / 2
            ]
        }
    
    def estimate_stone_girdle(self, stone_bbox: Dict) -> Dict:
        """Estimate stone girdle dimensions from bounding box"""
        # Girdle is at approximately 35% height from bottom
        girdle_z = stone_bbox["min"][2] + stone_bbox["height"] * 0.35
        
        return {
            "width": stone_bbox["width"],
            "depth": stone_bbox["depth"],
            "z": girdle_z,
            "center": [stone_bbox["center"][0], stone_bbox["center"][1], girdle_z]
        }
    
    def estimate_prong_opening(self, prong_bbox: Dict, setting_type: str = "default") -> Dict:
        """Estimate prong opening dimensions from bounding box"""
        ratio = self.OPENING_RATIOS.get(setting_type, self.OPENING_RATIOS["default"])
        
        # Opening is near the top of the prong
        opening_z = prong_bbox["max"][2] - prong_bbox["height"] * 0.15
        
        return {
            "width": prong_bbox["width"] * ratio,
            "depth": prong_bbox["depth"] * ratio,
            "z": opening_z,
            "center": [prong_bbox["center"][0], prong_bbox["center"][1], opening_z],
            "ratio_used": ratio
        }
    
    def validate(self, file_path: str = None, setting_type: str = "default") -> ValidationResult:
        """
        Validate an assembly file or the currently loaded model.
        Returns detailed validation result with correction factors.
        """
        if file_path:
            if not self.load_assembly(file_path):
                return ValidationResult(
                    is_valid=False,
                    overall_status=ValidationStatus.FAIL,
                    issues=[ValidationIssue(
                        category="load",
                        severity=ValidationStatus.FAIL,
                        message="Failed to load assembly file",
                        current_value=0, expected_value=1, correction_factor=1.0
                    )],
                    stone_bbox={}, prong_bbox={},
                    fit_metrics={}, position_metrics={},
                    correction_factors={}
                )
        
        if not self.stone_objects or not self.prong_objects:
            return ValidationResult(
                is_valid=False,
                overall_status=ValidationStatus.FAIL,
                issues=[ValidationIssue(
                    category="components",
                    severity=ValidationStatus.FAIL,
                    message="Missing stone or prong objects in assembly",
                    current_value=0, expected_value=1, correction_factor=1.0
                )],
                stone_bbox={}, prong_bbox={},
                fit_metrics={}, position_metrics={},
                correction_factors={}
            )
        
        # Get bounding boxes
        stone_bbox = self.get_layer_bbox(self.stone_objects)
        prong_bbox = self.get_layer_bbox(self.prong_objects)
        
        if not stone_bbox or not prong_bbox:
            return ValidationResult(
                is_valid=False,
                overall_status=ValidationStatus.FAIL,
                issues=[ValidationIssue(
                    category="geometry",
                    severity=ValidationStatus.FAIL,
                    message="Could not compute bounding boxes",
                    current_value=0, expected_value=1, correction_factor=1.0
                )],
                stone_bbox={}, prong_bbox={},
                fit_metrics={}, position_metrics={},
                correction_factors={}
            )
        
        # Estimate key dimensions
        stone_girdle = self.estimate_stone_girdle(stone_bbox)
        prong_opening = self.estimate_prong_opening(prong_bbox, setting_type)
        
        issues = []
        
        # === VALIDATION 1: FIT ===
        fit_x = stone_girdle["width"] / prong_opening["width"] if prong_opening["width"] > 0 else 0
        fit_y = stone_girdle["depth"] / prong_opening["depth"] if prong_opening["depth"] > 0 else 0
        avg_fit = (fit_x + fit_y) / 2
        
        fit_metrics = {
            "fit_x": fit_x,
            "fit_y": fit_y,
            "avg_fit": avg_fit,
            "target_fit": self.TARGET_FIT_RATIO,
            "stone_girdle_width": stone_girdle["width"],
            "stone_girdle_depth": stone_girdle["depth"],
            "prong_opening_width": prong_opening["width"],
            "prong_opening_depth": prong_opening["depth"]
        }
        
        # Check X fit
        if fit_x > 1.0:
            issues.append(ValidationIssue(
                category="fit_x",
                severity=ValidationStatus.FAIL,
                message=f"Stone too WIDE for prong opening ({fit_x:.1%} > 100%)",
                current_value=fit_x,
                expected_value=self.TARGET_FIT_RATIO,
                correction_factor=self.TARGET_FIT_RATIO / fit_x
            ))
        elif fit_x < self.TARGET_FIT_RATIO - self.FIT_TOLERANCE:
            issues.append(ValidationIssue(
                category="fit_x",
                severity=ValidationStatus.WARNING,
                message=f"Stone too NARROW for prong ({fit_x:.1%} < {(self.TARGET_FIT_RATIO - self.FIT_TOLERANCE):.1%})",
                current_value=fit_x,
                expected_value=self.TARGET_FIT_RATIO,
                correction_factor=self.TARGET_FIT_RATIO / fit_x
            ))
        
        # Check Y fit
        if fit_y > 1.0:
            issues.append(ValidationIssue(
                category="fit_y",
                severity=ValidationStatus.FAIL,
                message=f"Stone too DEEP for prong opening ({fit_y:.1%} > 100%)",
                current_value=fit_y,
                expected_value=self.TARGET_FIT_RATIO,
                correction_factor=self.TARGET_FIT_RATIO / fit_y
            ))
        elif fit_y < self.TARGET_FIT_RATIO - self.FIT_TOLERANCE:
            issues.append(ValidationIssue(
                category="fit_y",
                severity=ValidationStatus.WARNING,
                message=f"Stone too SHALLOW for prong ({fit_y:.1%} < {(self.TARGET_FIT_RATIO - self.FIT_TOLERANCE):.1%})",
                current_value=fit_y,
                expected_value=self.TARGET_FIT_RATIO,
                correction_factor=self.TARGET_FIT_RATIO / fit_y
            ))
        
        # === VALIDATION 2: CENTER ALIGNMENT ===
        offset_x = abs(stone_bbox["center"][0] - prong_bbox["center"][0])
        offset_y = abs(stone_bbox["center"][1] - prong_bbox["center"][1])
        
        position_metrics = {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "tolerance": self.POSITION_TOLERANCE,
            "stone_center": stone_bbox["center"],
            "prong_center": prong_bbox["center"]
        }
        
        if offset_x > self.POSITION_TOLERANCE:
            issues.append(ValidationIssue(
                category="position_x",
                severity=ValidationStatus.WARNING,
                message=f"Stone off-center in X by {offset_x:.2f}mm",
                current_value=offset_x,
                expected_value=0,
                correction_factor=1.0  # Position correction is absolute
            ))
        
        if offset_y > self.POSITION_TOLERANCE:
            issues.append(ValidationIssue(
                category="position_y",
                severity=ValidationStatus.WARNING,
                message=f"Stone off-center in Y by {offset_y:.2f}mm",
                current_value=offset_y,
                expected_value=0,
                correction_factor=1.0
            ))
        
        # === VALIDATION 3: DEPTH (Z Position) ===
        # Stone girdle should be BELOW the prong opening
        girdle_to_opening_z = prong_opening["z"] - stone_girdle["z"]
        depth_ratio = girdle_to_opening_z / prong_bbox["height"] if prong_bbox["height"] > 0 else 0
        
        position_metrics["girdle_to_opening_z"] = girdle_to_opening_z
        position_metrics["depth_ratio"] = depth_ratio
        
        if girdle_to_opening_z < 0:
            issues.append(ValidationIssue(
                category="depth",
                severity=ValidationStatus.FAIL,
                message=f"Stone FLOATING above prong opening (girdle {abs(girdle_to_opening_z):.2f}mm above)",
                current_value=depth_ratio,
                expected_value=self.MIN_DEPTH_RATIO,
                correction_factor=1.0
            ))
        elif depth_ratio < self.MIN_DEPTH_RATIO:
            issues.append(ValidationIssue(
                category="depth",
                severity=ValidationStatus.WARNING,
                message=f"Stone not deep enough in prong ({depth_ratio:.1%} < {self.MIN_DEPTH_RATIO:.1%})",
                current_value=depth_ratio,
                expected_value=self.MIN_DEPTH_RATIO,
                correction_factor=1.0
            ))
        elif depth_ratio > self.MAX_DEPTH_RATIO:
            issues.append(ValidationIssue(
                category="depth",
                severity=ValidationStatus.WARNING,
                message=f"Stone too deep in prong ({depth_ratio:.1%} > {self.MAX_DEPTH_RATIO:.1%})",
                current_value=depth_ratio,
                expected_value=self.MAX_DEPTH_RATIO,
                correction_factor=1.0
            ))
        
        # === CALCULATE CORRECTION FACTORS ===
        correction_factors = {
            "scale_x": self.TARGET_FIT_RATIO / fit_x if fit_x > 0 else 1.0,
            "scale_y": self.TARGET_FIT_RATIO / fit_y if fit_y > 0 else 1.0,
            "scale_z": (self.TARGET_FIT_RATIO / avg_fit) if avg_fit > 0 else 1.0,
            "translate_x": prong_bbox["center"][0] - stone_bbox["center"][0],
            "translate_y": prong_bbox["center"][1] - stone_bbox["center"][1],
            "translate_z": 0  # Calculated separately based on depth requirements
        }
        
        # Clamp correction factors to reasonable range
        for key in ["scale_x", "scale_y", "scale_z"]:
            correction_factors[key] = max(0.5, min(2.0, correction_factors[key]))
        
        # === DETERMINE OVERALL STATUS ===
        fail_count = sum(1 for i in issues if i.severity == ValidationStatus.FAIL)
        warning_count = sum(1 for i in issues if i.severity == ValidationStatus.WARNING)
        
        if fail_count > 0:
            overall_status = ValidationStatus.FAIL
            is_valid = False
        elif warning_count > 2:
            overall_status = ValidationStatus.WARNING
            is_valid = False  # Too many warnings = needs fixing
        elif warning_count > 0:
            overall_status = ValidationStatus.WARNING
            is_valid = True  # A few warnings are acceptable
        else:
            overall_status = ValidationStatus.PASS
            is_valid = True
        
        return ValidationResult(
            is_valid=is_valid,
            overall_status=overall_status,
            issues=issues,
            stone_bbox=stone_bbox,
            prong_bbox=prong_bbox,
            fit_metrics=fit_metrics,
            position_metrics=position_metrics,
            correction_factors=correction_factors
        )
    
    def print_report(self, result: ValidationResult):
        """Print a human-readable validation report"""
        print("\n" + "="*60)
        print("🔍 ASSEMBLY VALIDATION REPORT")
        print("="*60)
        
        status_icons = {
            ValidationStatus.PASS: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.FAIL: "❌"
        }
        
        print(f"\n   Overall Status: {status_icons[result.overall_status]} {result.overall_status.value.upper()}")
        print(f"   Valid: {'Yes' if result.is_valid else 'No'}")
        
        # Fit metrics
        fm = result.fit_metrics
        print(f"\n   📏 FIT ANALYSIS:")
        print(f"      Stone girdle: {fm.get('stone_girdle_width', 0):.2f} x {fm.get('stone_girdle_depth', 0):.2f} mm")
        print(f"      Prong opening: {fm.get('prong_opening_width', 0):.2f} x {fm.get('prong_opening_depth', 0):.2f} mm")
        print(f"      Fit ratio X: {fm.get('fit_x', 0):.1%} (target: {fm.get('target_fit', 0):.1%})")
        print(f"      Fit ratio Y: {fm.get('fit_y', 0):.1%} (target: {fm.get('target_fit', 0):.1%})")
        
        # Position metrics
        pm = result.position_metrics
        print(f"\n   📍 POSITION ANALYSIS:")
        print(f"      Center offset X: {pm.get('offset_x', 0):.3f} mm")
        print(f"      Center offset Y: {pm.get('offset_y', 0):.3f} mm")
        print(f"      Depth ratio: {pm.get('depth_ratio', 0):.1%}")
        
        # Issues
        if result.issues:
            print(f"\n   🔧 ISSUES ({len(result.issues)}):")
            for issue in result.issues:
                icon = status_icons[issue.severity]
                print(f"      {icon} [{issue.category}] {issue.message}")
        
        # Correction factors
        if not result.is_valid:
            cf = result.correction_factors
            print(f"\n   📐 CORRECTION FACTORS:")
            print(f"      Scale X: {cf.get('scale_x', 1):.3f}")
            print(f"      Scale Y: {cf.get('scale_y', 1):.3f}")
            print(f"      Scale Z: {cf.get('scale_z', 1):.3f}")
            if abs(cf.get('translate_x', 0)) > 0.01:
                print(f"      Translate X: {cf.get('translate_x', 0):.3f} mm")
            if abs(cf.get('translate_y', 0)) > 0.01:
                print(f"      Translate Y: {cf.get('translate_y', 0):.3f} mm")
        
        print("\n" + "="*60)


def validate_assembly(file_path: str, setting_type: str = "default") -> ValidationResult:
    """Convenience function to validate an assembly file"""
    validator = AssemblyValidator()
    result = validator.validate(file_path, setting_type)
    validator.print_report(result)
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python assembly_validator.py <assembly.3dm> [setting_type]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    setting_type = sys.argv[2] if len(sys.argv) > 2 else "default"
    
    result = validate_assembly(file_path, setting_type)
    sys.exit(0 if result.is_valid else 1)
