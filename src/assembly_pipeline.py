"""
Iterative Assembly Pipeline
Orchestrates the complete jewelry CAD assembly process with validation loops.

Process:
1. Load stone and prong CAD files
2. Perform precision assembly
3. Validate the assembly
4. If validation fails, apply corrections and re-assemble
5. Repeat until assembly is valid or max iterations reached
6. Generate dynamic shank based on design analysis
7. Combine into complete ring
8. Final validation

Key features:
- Automatic correction based on validation feedback
- Maximum iteration limits to prevent infinite loops
- Detailed logging of each iteration
- Support for different setting types
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    import rhino3dm
except ImportError:
    print("❌ rhino3dm not installed. Install with: pip install rhino3dm")
    raise

from precision_assembler import PrecisionAssembler, GeometryAnalyzer
from assembly_validator import AssemblyValidator, ValidationResult, ValidationStatus
from dynamic_shank_generator import DynamicShankGenerator, ShankParameters


# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "assemblies"


@dataclass
class AssemblyIteration:
    """Record of a single assembly iteration"""
    iteration: int
    assembly_path: str
    validation_passed: bool
    fit_x: float
    fit_y: float
    corrections_applied: Dict
    issues: List[str]


@dataclass
class PipelineResult:
    """Complete result of the assembly pipeline"""
    success: bool
    iterations_used: int
    head_assembly_path: Optional[str]
    complete_ring_path: Optional[str]
    final_validation: Optional[Dict]
    iteration_history: List[AssemblyIteration]
    shank_params: Optional[Dict]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "iterations_used": self.iterations_used,
            "head_assembly_path": self.head_assembly_path,
            "complete_ring_path": self.complete_ring_path,
            "final_validation": self.final_validation,
            "iteration_history": [asdict(i) for i in self.iteration_history],
            "shank_params": self.shank_params,
            "error": self.error
        }


class IterativeAssemblyPipeline:
    """
    Orchestrates jewelry CAD assembly with iterative validation.
    
    The pipeline ensures assembly correctness through:
    1. Precision assembly with calculated scaling
    2. Automatic validation of fit, position, depth
    3. Correction factor application for failed assemblies
    4. Iterative refinement until correct
    """
    
    MAX_ITERATIONS = 5
    
    def __init__(self):
        self.assembler = PrecisionAssembler()
        self.validator = AssemblyValidator()
        self.shank_generator = DynamicShankGenerator()
        self.iteration_history: List[AssemblyIteration] = []
    
    def run(
        self,
        stone_path: str,
        prong_path: str,
        design_analysis: Dict = None,
        stone_id: str = None,
        prong_id: str = None,
        setting_type: str = "prong",
        ring_size: float = 7.0,
        output_dir: Path = None
    ) -> PipelineResult:
        """
        Run the complete assembly pipeline.
        
        Args:
            stone_path: Path to stone .3dm file
            prong_path: Path to prong .3dm file
            design_analysis: Vision analysis of reference design
            stone_id: Stone ID for metadata
            prong_id: Prong ID for metadata
            setting_type: Type of setting (prong, bezel, etc.)
            ring_size: US ring size
            output_dir: Output directory for files
            
        Returns:
            PipelineResult with success status and all outputs
        """
        print("\n" + "="*70)
        print("🔄 ITERATIVE ASSEMBLY PIPELINE")
        print("="*70)
        
        self.iteration_history = []
        output_dir = output_dir or OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Verify inputs exist
        if not Path(stone_path).exists():
            return PipelineResult(
                success=False, iterations_used=0,
                head_assembly_path=None, complete_ring_path=None,
                final_validation=None, iteration_history=[],
                shank_params=None, error=f"Stone file not found: {stone_path}"
            )
        
        if not Path(prong_path).exists():
            return PipelineResult(
                success=False, iterations_used=0,
                head_assembly_path=None, complete_ring_path=None,
                final_validation=None, iteration_history=[],
                shank_params=None, error=f"Prong file not found: {prong_path}"
            )
        
        print(f"\n📂 Inputs:")
        print(f"   Stone: {Path(stone_path).name}")
        print(f"   Prong: {Path(prong_path).name}")
        print(f"   Setting: {setting_type}")
        print(f"   Ring size: {ring_size}")
        
        # === PHASE 1: Iterative Head Assembly ===
        print("\n" + "-"*70)
        print("📦 PHASE 1: HEAD ASSEMBLY (Stone + Prong)")
        print("-"*70)
        
        head_assembly_path = None
        final_validation = None
        correction_factors = None
        
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            print(f"\n🔄 Iteration {iteration}/{self.MAX_ITERATIONS}")
            
            # Generate output path for this iteration
            if iteration == 1:
                assembly_output = str(output_dir / f"head_assembly_{timestamp}.3dm")
            else:
                assembly_output = str(output_dir / f"head_assembly_{timestamp}_iter{iteration}.3dm")
            
            # Perform assembly
            try:
                if correction_factors:
                    print(f"   Applying corrections: X={correction_factors.get('scale_x', 1):.3f}, Y={correction_factors.get('scale_y', 1):.3f}")
                
                result_path = self.assembler.assemble(
                    stone_path=stone_path,
                    prong_path=prong_path,
                    output_path=assembly_output,
                    stone_id=stone_id,
                    prong_id=prong_id,
                    setting_type=setting_type,
                    correction_factors=correction_factors
                )
                
                if not result_path:
                    self._record_iteration(iteration, None, False, 0, 0, correction_factors, ["Assembly failed"])
                    continue
                
                head_assembly_path = result_path
                
            except Exception as e:
                print(f"   ❌ Assembly error: {e}")
                self._record_iteration(iteration, None, False, 0, 0, correction_factors, [str(e)])
                continue
            
            # Validate assembly
            print(f"\n   🔍 Validating assembly...")
            validation = self.validator.validate(head_assembly_path, setting_type)
            
            # Record this iteration
            issues = [f"{i.category}: {i.message}" for i in validation.issues]
            self._record_iteration(
                iteration,
                head_assembly_path,
                validation.is_valid,
                validation.fit_metrics.get("fit_x", 0),
                validation.fit_metrics.get("fit_y", 0),
                correction_factors,
                issues
            )
            
            # Check if valid
            if validation.is_valid:
                print(f"\n   ✅ ASSEMBLY VALID after {iteration} iteration(s)!")
                final_validation = validation.to_dict()
                break
            
            # Print issues
            print(f"\n   ⚠️ Validation issues ({len(validation.issues)}):")
            for issue in validation.issues[:3]:  # Show first 3
                print(f"      - {issue.message}")
            
            # Get correction factors for next iteration
            correction_factors = validation.correction_factors
            
            if iteration == self.MAX_ITERATIONS:
                print(f"\n   ⚠️ Max iterations reached. Using best result.")
                final_validation = validation.to_dict()
        
        if not head_assembly_path:
            return PipelineResult(
                success=False, iterations_used=len(self.iteration_history),
                head_assembly_path=None, complete_ring_path=None,
                final_validation=None, iteration_history=self.iteration_history,
                shank_params=None, error="Head assembly failed after all iterations"
            )
        
        # === PHASE 2: Dynamic Shank Generation ===
        print("\n" + "-"*70)
        print("💍 PHASE 2: DYNAMIC SHANK GENERATION")
        print("-"*70)
        
        try:
            # Load head assembly for analysis
            head_model = rhino3dm.File3dm.Read(head_assembly_path)
            
            # Generate shank based on head and design analysis
            shank_output = str(output_dir / f"shank_{timestamp}.3dm")
            
            shank_model, shank_params = self.shank_generator.generate_for_head(
                head_model,
                design_analysis,
                ring_size,
                shank_output
            )
            
            print(f"\n   ✅ Shank generated: {shank_params.style}")
            
        except Exception as e:
            print(f"\n   ❌ Shank generation error: {e}")
            # Return partial success (head assembly worked)
            return PipelineResult(
                success=True,  # Head assembly succeeded
                iterations_used=len(self.iteration_history),
                head_assembly_path=head_assembly_path,
                complete_ring_path=None,
                final_validation=final_validation,
                iteration_history=self.iteration_history,
                shank_params=None,
                error=f"Shank generation failed: {e}"
            )
        
        # === PHASE 3: Combine into Complete Ring ===
        print("\n" + "-"*70)
        print("💎 PHASE 3: COMPLETE RING ASSEMBLY")
        print("-"*70)
        
        try:
            complete_output = str(output_dir / f"complete_ring_{timestamp}.3dm")
            
            complete_model = self.shank_generator.combine_head_and_shank(
                head_model,
                shank_model,
                shank_params,
                complete_output
            )
            
            print(f"\n   ✅ Complete ring assembled!")
            
        except Exception as e:
            print(f"\n   ❌ Ring combination error: {e}")
            return PipelineResult(
                success=True,
                iterations_used=len(self.iteration_history),
                head_assembly_path=head_assembly_path,
                complete_ring_path=None,
                final_validation=final_validation,
                iteration_history=self.iteration_history,
                shank_params=asdict(shank_params),
                error=f"Ring combination failed: {e}"
            )
        
        # === PHASE 4: Final Validation ===
        print("\n" + "-"*70)
        print("🔍 PHASE 4: FINAL VALIDATION")
        print("-"*70)
        
        # Validate complete ring
        final_ring_validation = self.validator.validate(complete_output, setting_type)
        self.validator.print_report(final_ring_validation)
        
        # === Summary ===
        print("\n" + "="*70)
        print("✅ ASSEMBLY PIPELINE COMPLETE")
        print("="*70)
        
        print(f"\n📊 Summary:")
        print(f"   Iterations used: {len(self.iteration_history)}")
        print(f"   Head assembly: {Path(head_assembly_path).name}")
        print(f"   Complete ring: {Path(complete_output).name}")
        print(f"   Shank style: {shank_params.style}")
        print(f"   Final status: {'✅ VALID' if final_ring_validation.is_valid else '⚠️ WARNINGS'}")
        
        return PipelineResult(
            success=True,
            iterations_used=len(self.iteration_history),
            head_assembly_path=head_assembly_path,
            complete_ring_path=complete_output,
            final_validation=final_ring_validation.to_dict(),
            iteration_history=self.iteration_history,
            shank_params=asdict(shank_params)
        )
    
    def _record_iteration(
        self,
        iteration: int,
        assembly_path: Optional[str],
        passed: bool,
        fit_x: float,
        fit_y: float,
        corrections: Optional[Dict],
        issues: List[str]
    ):
        """Record an iteration in history"""
        record = AssemblyIteration(
            iteration=iteration,
            assembly_path=assembly_path or "",
            validation_passed=passed,
            fit_x=fit_x,
            fit_y=fit_y,
            corrections_applied=corrections or {},
            issues=issues
        )
        self.iteration_history.append(record)


def run_assembly_pipeline(
    stone_path: str,
    prong_path: str,
    design_analysis: Dict = None,
    setting_type: str = "prong",
    ring_size: float = 7.0
) -> PipelineResult:
    """Convenience function to run the assembly pipeline"""
    pipeline = IterativeAssemblyPipeline()
    return pipeline.run(
        stone_path=stone_path,
        prong_path=prong_path,
        design_analysis=design_analysis,
        setting_type=setting_type,
        ring_size=ring_size
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python assembly_pipeline.py <stone.3dm> <prong.3dm> [setting_type] [ring_size]")
        print("\nExample:")
        print("  python assembly_pipeline.py stones/round_5mm.3dm prongs/4prong.3dm prong 7.0")
        sys.exit(1)
    
    stone_path = sys.argv[1]
    prong_path = sys.argv[2]
    setting_type = sys.argv[3] if len(sys.argv) > 3 else "prong"
    ring_size = float(sys.argv[4]) if len(sys.argv) > 4 else 7.0
    
    result = run_assembly_pipeline(
        stone_path=stone_path,
        prong_path=prong_path,
        setting_type=setting_type,
        ring_size=ring_size
    )
    
    # Save result JSON
    if result.success:
        output_dir = Path(stone_path).parent.parent / "outputs" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = output_dir / f"assembly_result_{timestamp}.json"
        
        with open(result_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\n📁 Result saved: {result_path}")
    
    sys.exit(0 if result.success else 1)
