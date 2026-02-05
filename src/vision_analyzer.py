"""
Vision LLM Analyzer using Gemini 2.5 Pro / Flash
Handles:
1. Analyzing jewelry images to extract component requirements
2. Verifying if retrieved components match requirements
3. Generating component reference images for retrieval
"""
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import google.generativeai as genai
from PIL import Image

from config import GEMINI_API_KEY, GEMINI_MODEL_ANALYSIS, GEMINI_MODEL_VERIFY
from models import (
    ComponentType, ComponentRequirement, CADComponent,
    VerificationResult, VerificationStatus, RetrievalResult
)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class VisionAnalyzer:
    """Vision LLM for analyzing jewelry designs and verifying components"""
    
    def __init__(
        self, 
        analysis_model: str = GEMINI_MODEL_ANALYSIS,
        verify_model: str = GEMINI_MODEL_VERIFY
    ):
        # Use Pro model for analysis (better quality)
        self.analysis_model = genai.GenerativeModel(analysis_model)
        # Use Flash model for verification (faster, many calls)
        self.verify_model = genai.GenerativeModel(verify_model)
        
        print(f"🧠 Vision LLM initialized:")
        print(f"   Analysis: {analysis_model}")
        print(f"   Verification: {verify_model}")
    
    def _load_image(self, image_path: Path) -> Image.Image:
        """Load an image from path"""
        return Image.open(image_path)
    
    def analyze_design_comprehensive(self, image_path: Path) -> Dict:
        """
        Comprehensive analysis of jewelry design - extracts ALL component info.
        
        Returns:
            {
                "stone": {
                    "shape": "oval",
                    "color": "pink",
                    "size_mm": 10.0,
                    "cut_style": "brilliant",
                    "description": "..."
                },
                "prong": {
                    "style": "bezel",
                    "prong_count": 0,
                    "shape": "oval",
                    "description": "..."
                },
                "shank": {
                    "style": "cathedral",
                    "width_mm": 2.5,
                    "thickness_mm": 1.8,
                    "features": [],
                    "description": "..."
                },
                "ring_size_estimate": 7.0
            }
        """
        image = self._load_image(image_path)
        
        prompt = """You are an expert jewelry CAD designer. Analyze this jewelry ring image and extract COMPLETE specifications for ALL components.

Analyze with PRECISION:

1. STONE (center gemstone):
   - shape: EXACT shape (round, oval, pear, marquise, cushion, emerald, princess, heart, radiant, asscher, trillion)
   - color: observed color (pink, blue, clear, etc.)
   - size_mm: estimated size in mm (typical range: 5-15mm for center stones)
   - cut_style: brilliant, step, mixed, cabochon
   
2. PRONG/SETTING (metal holding stone):
   - style: EXACT type (bezel, 4-prong, 6-prong, 3-prong, tension, channel, pave, halo, cathedral-prong)
   - prong_count: number of prongs (0 for bezel, 4, 6, etc.)
   - shape: opening shape (matches stone usually)
   
3. SHANK (ring band):
   - style: plain, cathedral, split, tapered, knife-edge, twisted
   - width_mm: band width (thin=1.5-2, medium=2.5-3, thick=3.5-4)
   - thickness_mm: band thickness (1.5-2.5 typical)
   - features: any decorations (pave, milgrain, engraving, channel-set)

4. RING SIZE: estimate based on proportions (US sizes 4-13)

Return ONLY valid JSON:
{
    "stone": {
        "shape": "oval",
        "color": "pink",
        "size_mm": 10.0,
        "cut_style": "brilliant",
        "description": "Large oval pink gemstone with brilliant faceting"
    },
    "prong": {
        "style": "bezel",
        "prong_count": 0,
        "shape": "oval",
        "description": "Polished bezel setting wrapping around oval stone"
    },
    "shank": {
        "style": "cathedral",
        "width_mm": 2.5,
        "thickness_mm": 1.8,
        "features": ["smooth", "polished"],
        "description": "Cathedral band with arched shoulders"
    },
    "ring_size_estimate": 7.0
}"""

        try:
            response = self.analysis_model.generate_content([prompt, image])
            response_text = response.text.strip()
            
            # Clean up JSON
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
            
        except Exception as e:
            print(f"❌ Error in comprehensive analysis: {e}")
            return {}

    def analyze_design(self, image_path: Path) -> list[ComponentRequirement]:
        """
        Analyze a jewelry design image and extract component requirements
        """
        image = self._load_image(image_path)
        
        prompt = """You are an expert jewelry CAD designer. Analyze this jewelry design image and identify all components needed to recreate it.

Focus on these component types:
1. PRONGS - Metal settings that hold stones (4-prong, 6-prong, bezel, channel, shared prong, etc.)
2. STONES - Gemstones/diamonds (round, oval, pear, marquise, princess, emerald, cushion, heart, etc.)

For EACH component found, provide detailed specifications in this JSON format:
{
    "components": [
        {
            "component_type": "prongs" or "stones",
            "description": "Detailed description of the component appearance and style",
            "shape": "Shape (round, oval, pear, marquise, heart, cushion, emerald, princess, etc.)",
            "size": "Estimated size (e.g., 5mm, small, medium, large)",
            "style": "Style (e.g., 4-prong, 6-prong, bezel, solitaire, halo, etc.)",
            "quantity": "Number needed",
            "position": "Where in the design (center, side, halo, band, etc.)",
            "additional_notes": "Any distinctive features"
        }
    ]
}

Be VERY specific about:
- Prong count and style (4-prong cathedral, 6-prong tiffany, etc.)
- Stone cuts (brilliant, step cut, modified brilliant, etc.)
- Proportions and relationships between components

Return ONLY valid JSON, no markdown formatting or extra text."""

        try:
            response = self.analysis_model.generate_content([prompt, image])
            response_text = response.text.strip()
            
            # Clean up response
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            data = json.loads(response_text.strip())
            
            requirements = []
            for comp in data.get("components", []):
                comp_type_str = comp.get("component_type", "").lower()
                
                if comp_type_str == "prongs":
                    comp_type = ComponentType.PRONGS
                elif comp_type_str == "stones":
                    comp_type = ComponentType.STONES
                else:
                    continue
                
                requirement = ComponentRequirement(
                    component_type=comp_type,
                    description=comp.get("description", ""),
                    shape=comp.get("shape"),
                    size=comp.get("size"),
                    style=comp.get("style"),
                    additional_details={
                        "quantity": comp.get("quantity"),
                        "position": comp.get("position"),
                        "notes": comp.get("additional_notes")
                    }
                )
                requirements.append(requirement)
            
            return requirements
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing LLM response: {e}")
            return []
        except Exception as e:
            print(f"❌ Error analyzing design: {e}")
            return []
    
    def _get_stone_verification_prompt(self, requirement: ComponentRequirement) -> str:
        """Get verification prompt specifically for stones"""
        # Extract shape from requirement
        req_shape = (requirement.shape or '').lower()
        desc_lower = requirement.description.lower()
        
        # Determine shape family
        shape_hint = ""
        if 'oval' in desc_lower or 'oval' in req_shape:
            shape_hint = "LOOKING FOR: OVAL shape (elongated ellipse)"
        elif 'round' in desc_lower or 'round' in req_shape:
            shape_hint = "LOOKING FOR: ROUND shape (circular)"
        elif 'pear' in desc_lower or 'pear' in req_shape:
            shape_hint = "LOOKING FOR: PEAR shape (teardrop)"
        elif 'marquise' in desc_lower or 'marquise' in req_shape:
            shape_hint = "LOOKING FOR: MARQUISE shape (pointed both ends)"
        
        return f"""You are a jewelry CAD expert. Your job is to APPROVE stones that match the required shape.

ORIGINAL JEWELRY: [First Image]
CANDIDATE CAD STONE: [Second Image - wireframe/technical view]

{shape_hint}

SHAPE FAMILIES (similar shapes are OK):
- OVAL family: oval, ellipse, elongated round
- ROUND family: round, circular, brilliant
- PEAR family: pear, teardrop
- RECTANGULAR family: emerald, radiant, baguette, princess, cushion

RULES:
1. APPROVE if the CAD stone shape is in the SAME FAMILY as required
2. APPROVE if it looks close enough to work
3. Size doesn't matter - we can scale
4. Color doesn't matter - CAD is grayscale

ONLY REJECT if the shape is COMPLETELY WRONG (e.g., heart when oval needed)

BE LENIENT. When in doubt, APPROVE.

Respond with ONLY valid JSON:
{{
    "status": "approved" or "rejected",
    "confidence": 0.8,
    "reasoning": "Brief explanation",
    "shape_match": true or false,
    "style_match": true,
    "usable": true,
    "suggested_modifications": null
}}"""

    def _get_prong_verification_prompt(self, requirement: ComponentRequirement) -> str:
        """Get verification prompt specifically for prongs/settings"""
        desc_lower = requirement.description.lower()
        
        # Detect setting type
        is_bezel = 'bezel' in desc_lower
        is_prong = any(x in desc_lower for x in ['prong', '4-prong', '3-prong', '6-prong'])
        
        # Detect shape requirement
        shape_hint = ""
        if 'oval' in desc_lower:
            shape_hint = "LOOKING FOR: Setting for OVAL stone (oval/elliptical opening)"
        elif 'round' in desc_lower:
            shape_hint = "LOOKING FOR: Setting for ROUND stone (circular opening)"
        elif 'pear' in desc_lower:
            shape_hint = "LOOKING FOR: Setting for PEAR stone"
        
        # Setting type hint
        if is_bezel:
            type_hint = "LOOKING FOR: BEZEL setting (metal rim wraps around stone, NO prongs)"
        elif is_prong:
            type_hint = "LOOKING FOR: PRONG setting (metal claws hold stone)"
        else:
            type_hint = "Any setting type is acceptable"
        
        return f"""You are a jewelry CAD expert. Your job is to APPROVE settings that match the requirement.

ORIGINAL JEWELRY: [First Image]
CANDIDATE CAD SETTING: [Second Image - wireframe/technical view]

{type_hint}
{shape_hint}

SETTING TYPES:
- BEZEL: Metal rim completely surrounds stone edge. NO prongs visible. Smooth rim.
- PRONG: Individual metal claws (2,3,4,6,8) grip the stone. Prongs visible as separate posts.
- TENSION: Stone held by pressure, minimal metal visible.

RULES:
1. If looking for BEZEL: Approve any bezel-style setting (rim around stone)
2. If looking for PRONG: Approve if it has prongs
3. Shape of opening should roughly match stone shape (oval for oval, round for round)
4. Size doesn't matter - we can scale

BE LENIENT. The CAD is wireframe and may look different from photo.
When in doubt, APPROVE.

ONLY REJECT if:
- Looking for bezel but it clearly has prongs
- Looking for prongs but it's clearly a bezel
- Shape is completely wrong (round opening for pear stone)

Respond with ONLY valid JSON:
{{
    "status": "approved" or "rejected",
    "confidence": 0.8,
    "prong_count_observed": 0,
    "reasoning": "Brief explanation",
    "shape_match": true,
    "prong_count_match": true,
    "usable": true,
    "suggested_modifications": null
}}"""

    def verify_component(
        self,
        original_image: Path,
        requirement: ComponentRequirement,
        candidate_component: CADComponent
    ) -> VerificationResult:
        """
        Verify if a candidate component matches the requirement
        """
        original_img = self._load_image(original_image)
        component_img = self._load_image(candidate_component.screenshot_path)
        
        # Use different prompts for stones vs prongs
        if requirement.component_type == ComponentType.STONES:
            prompt = self._get_stone_verification_prompt(requirement)
        else:
            prompt = self._get_prong_verification_prompt(requirement)

        try:
            # Use faster Flash model for verification (many calls)
            response = self.verify_model.generate_content([prompt, original_img, component_img])
            response_text = response.text.strip()
            
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            data = json.loads(response_text.strip())
            
            status_str = data.get("status", "rejected").lower()
            status = VerificationStatus.APPROVED if status_str == "approved" else VerificationStatus.REJECTED
            
            return VerificationResult(
                component=candidate_component,
                status=status,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                suggested_modifications=data.get("suggested_modifications")
            )
            
        except Exception as e:
            print(f"❌ Error verifying component: {e}")
            return VerificationResult(
                component=candidate_component,
                status=VerificationStatus.REJECTED,
                confidence=0.0,
                reasoning=f"Verification error: {e}"
            )
    
    def verify_batch(
        self,
        original_image: Path,
        requirement: ComponentRequirement,
        candidates: list[RetrievalResult]
    ) -> list[VerificationResult]:
        """Verify multiple candidate components"""
        results = []
        for candidate in candidates:
            result = self.verify_component(
                original_image,
                requirement,
                candidate.component
            )
            results.append(result)
        return results
    
    def select_best_component(
        self,
        original_image: Path,
        requirement: ComponentRequirement,
        candidates: list[RetrievalResult]
    ) -> Optional[VerificationResult]:
        """
        Select the best matching component from candidates
        Returns the first approved component with highest confidence
        """
        verification_results = self.verify_batch(original_image, requirement, candidates)
        
        approved = [r for r in verification_results if r.status == VerificationStatus.APPROVED]
        
        if approved:
            return max(approved, key=lambda r: r.confidence)
        
        return None
