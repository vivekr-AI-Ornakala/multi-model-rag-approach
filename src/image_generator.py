"""
Component Image Generator using Gemini 2.0 Flash
Generates CAD-style reference images for each component from a jewelry design.
"""
import json
import base64
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from PIL import Image
import io
from datetime import datetime

from google import genai
from google.genai import types

from config import GEMINI_API_KEY


class ComponentImageGenerator:
    """
    Generate CAD-style reference images for individual jewelry components.
    
    Uses Gemini 2.0 Flash to:
    1. Analyze the reference jewelry image
    2. Generate isolated CAD-style images for each component (stone, prong, shank)
    """
    
    def __init__(self):
        # Initialize the new google-genai client
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model = "gemini-2.0-flash-exp-image-generation"
        
        # Output directory for generated images
        self.output_dir = Path(__file__).parent.parent / "outputs" / "generated_components"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎨 Image Generator initialized")
        print(f"   Model: {self.model}")
    
    def _load_image_as_base64(self, image_path: Path) -> str:
        """Load image and convert to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    def _save_generated_image(self, image_data: bytes, component_type: str) -> Path:
        """Save generated image to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{component_type}_{timestamp}.png"
        output_path = self.output_dir / filename
        
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        return output_path
    
    def analyze_and_generate(self, image_path: Path) -> Dict[str, any]:
        """
        Analyze jewelry image and generate component specifications + images.
        
        Returns:
            {
                "stone": {
                    "description": "...",
                    "shape": "oval",
                    "generated_image": Path or None,
                    "specs": {...}
                },
                "prong": {
                    "description": "...",
                    "style": "bezel",
                    "prong_count": 0,
                    "generated_image": Path or None,
                    "specs": {...}
                },
                "shank": {
                    "description": "...",
                    "style": "cathedral",
                    "generated_image": Path or None,
                    "specs": {...}
                }
            }
        """
        print(f"\n🔍 Analyzing jewelry design...")
        
        # First, analyze the image to understand components
        analysis = self._analyze_components(image_path)
        
        # Generate images for each component
        result = {
            "stone": None,
            "prong": None,
            "shank": None
        }
        
        if analysis.get("stone"):
            print(f"\n💎 Generating stone reference image...")
            stone_result = self._generate_component_image(
                image_path, 
                "stone",
                analysis["stone"]
            )
            result["stone"] = stone_result
        
        if analysis.get("prong"):
            print(f"\n🔧 Generating prong/setting reference image...")
            prong_result = self._generate_component_image(
                image_path,
                "prong", 
                analysis["prong"]
            )
            result["prong"] = prong_result
        
        if analysis.get("shank"):
            print(f"\n💍 Generating shank reference image...")
            shank_result = self._generate_component_image(
                image_path,
                "shank",
                analysis["shank"]
            )
            result["shank"] = shank_result
        
        return result
    
    def _analyze_components(self, image_path: Path) -> Dict:
        """Analyze the jewelry image to extract component details"""
        
        image = Image.open(image_path)
        
        prompt = """Analyze this jewelry ring image and identify the components.

For each component found, provide detailed specifications:

1. STONE (the gemstone):
   - shape: exact shape (round, oval, pear, marquise, cushion, emerald, princess, heart, etc.)
   - color: observed color
   - cut_style: brilliant, step cut, mixed, cabochon, etc.
   - size_estimate: small/medium/large or mm estimate
   - clarity: if visible (clear, included, opaque)

2. PRONG/SETTING (metal that holds the stone):
   - style: bezel, 4-prong, 6-prong, 3-prong, tension, channel, pave, halo, etc.
   - prong_count: number of prongs (0 for bezel)
   - shape: shape of opening (matches stone or different)
   - metal_appearance: polished, textured, matte

3. SHANK (ring band):
   - style: plain, cathedral, split, tapered, knife-edge, comfort-fit, twisted, braided
   - width: thin, medium, thick
   - features: any decorative elements, pave, milgrain, engraving
   - profile: flat, rounded, D-shape

Return ONLY valid JSON:
{
    "stone": {
        "shape": "oval",
        "color": "pink",
        "cut_style": "brilliant",
        "size_estimate": "large",
        "description": "Large oval pink gemstone with brilliant cut faceting"
    },
    "prong": {
        "style": "bezel",
        "prong_count": 0,
        "shape": "oval",
        "metal_appearance": "polished",
        "description": "Polished bezel setting with oval opening"
    },
    "shank": {
        "style": "cathedral",
        "width": "medium",
        "features": "smooth shoulders rising to head",
        "profile": "rounded",
        "description": "Cathedral style band with arched shoulders"
    }
}"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            response_text = response.text.strip()
            
            # Clean up JSON
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            analysis = json.loads(response_text.strip())
            
            print(f"   ✅ Found components:")
            if analysis.get("stone"):
                print(f"      💎 Stone: {analysis['stone'].get('shape', 'unknown')} {analysis['stone'].get('color', '')}")
            if analysis.get("prong"):
                print(f"      🔧 Setting: {analysis['prong'].get('style', 'unknown')}")
            if analysis.get("shank"):
                print(f"      💍 Shank: {analysis['shank'].get('style', 'unknown')}")
            
            return analysis
            
        except Exception as e:
            print(f"   ⚠️ Analysis error: {e}")
            return {}
    
    def _generate_component_image(
        self,
        reference_image: Path,
        component_type: str,
        specs: Dict
    ) -> Dict:
        """Generate a CAD-style reference image for a specific component"""
        
        ref_image = Image.open(reference_image)
        
        # Build prompt based on component type
        if component_type == "stone":
            prompt = self._get_stone_generation_prompt(specs)
        elif component_type == "prong":
            prompt = self._get_prong_generation_prompt(specs)
        elif component_type == "shank":
            prompt = self._get_shank_generation_prompt(specs)
        else:
            return {"specs": specs, "generated_image": None}
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, ref_image],
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
            
            # Extract generated image from response
            generated_image_path = None
            description = specs.get("description", "")
            
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    # Save the generated image
                    image_data = part.inline_data.data
                    generated_image_path = self._save_generated_image(
                        image_data, 
                        component_type
                    )
                    print(f"   ✅ Generated: {generated_image_path.name}")
                elif part.text:
                    # Capture any text description
                    description = part.text
            
            return {
                "specs": specs,
                "generated_image": generated_image_path,
                "description": description
            }
            
        except Exception as e:
            print(f"   ⚠️ Generation error: {e}")
            # Return specs without generated image
            return {
                "specs": specs,
                "generated_image": None,
                "description": specs.get("description", "")
            }
    
    def _get_stone_generation_prompt(self, specs: Dict) -> str:
        """Generate prompt for stone image generation"""
        shape = specs.get("shape", "round")
        color = specs.get("color", "clear")
        cut = specs.get("cut_style", "brilliant")
        
        return f"""Generate a clean, isolated CAD-style rendering of a gemstone with these specifications:

STONE SPECIFICATIONS:
- Shape: {shape}
- Color: {color} (but render as grayscale/silver CAD style)
- Cut style: {cut}

RENDERING STYLE:
- Technical CAD wireframe or solid render
- Isolated on plain white background
- Show facet structure clearly
- Top-down view (looking at table facet)
- Clean, professional jewelry CAD appearance
- Grayscale/metallic coloring (not the actual gemstone color)

Generate ONLY the gemstone, isolated, no setting or ring."""

    def _get_prong_generation_prompt(self, specs: Dict) -> str:
        """Generate prompt for prong/setting image generation"""
        style = specs.get("style", "4-prong")
        shape = specs.get("shape", "round")
        prong_count = specs.get("prong_count", 4)
        
        if "bezel" in style.lower():
            setting_desc = f"bezel setting (metal rim around {shape} opening, no prongs)"
        else:
            setting_desc = f"{prong_count}-prong setting for {shape} stone"
        
        return f"""Generate a clean, isolated CAD-style rendering of a jewelry setting:

SETTING SPECIFICATIONS:
- Type: {setting_desc}
- Opening shape: {shape}
- Style: {style}

RENDERING STYLE:
- Technical CAD wireframe or solid render
- Isolated on plain white background
- Top-down view showing the opening/prongs
- Silver/metallic coloring
- Professional jewelry CAD appearance
- Show prongs or bezel rim clearly

Generate ONLY the setting/prong head, NO stone inside, NO ring band."""

    def _get_shank_generation_prompt(self, specs: Dict) -> str:
        """Generate prompt for shank/band image generation"""
        style = specs.get("style", "plain")
        width = specs.get("width", "medium")
        profile = specs.get("profile", "rounded")
        features = specs.get("features", "none")
        
        return f"""Generate a clean, isolated CAD-style rendering of a ring band/shank:

SHANK SPECIFICATIONS:
- Style: {style}
- Width: {width}
- Profile: {profile}
- Features: {features}

RENDERING STYLE:
- Technical CAD wireframe or solid render
- Isolated on plain white background
- 3/4 view showing band shape and profile
- Silver/metallic coloring
- Professional jewelry CAD appearance

Generate ONLY the ring band/shank, NO stone, NO setting head.
Show the band as a complete circle (ring shape)."""


class SmartShankAnalyzer:
    """
    Analyze reference image to determine optimal shank parameters.
    """
    
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def analyze_shank_style(self, image_path: Path) -> Dict:
        """
        Analyze reference image to determine shank style and parameters.
        
        Returns:
            {
                "style": "cathedral" | "plain" | "split" | "tapered",
                "width": 2.5,  # mm
                "thickness": 1.8,  # mm  
                "features": [...],
                "confidence": 0.85
            }
        """
        image = Image.open(image_path)
        
        prompt = """Analyze the ring band/shank in this jewelry image.

Determine the SHANK STYLE from these options:
1. PLAIN - Simple, uniform band with no decorative shoulders
2. CATHEDRAL - Arched shoulders that rise up toward the center stone
3. SPLIT - Band splits into two paths near the center
4. TAPERED - Band narrows toward the center

Also estimate:
- Band width (thin ~1.5-2mm, medium ~2.5-3mm, thick ~3.5-4mm)
- Any decorative features (pave, milgrain, engraving, twisted)

Return ONLY valid JSON:
{
    "style": "cathedral",
    "width_mm": 2.5,
    "thickness_mm": 1.8,
    "features": ["smooth", "polished"],
    "confidence": 0.85,
    "reasoning": "The band shows arched shoulders rising toward the center stone, typical of cathedral style"
}"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image]
            )
            
            response_text = response.text.strip()
            
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text.strip())
            
            # Normalize style name
            style = result.get("style", "plain").lower()
            if style not in ["plain", "cathedral", "split", "tapered"]:
                style = "plain"
            
            return {
                "style": style,
                "width_mm": result.get("width_mm", 2.5),
                "thickness_mm": result.get("thickness_mm", 1.8),
                "features": result.get("features", []),
                "confidence": result.get("confidence", 0.7),
                "reasoning": result.get("reasoning", "")
            }
            
        except Exception as e:
            print(f"   ⚠️ Shank analysis error: {e}")
            return {
                "style": "plain",
                "width_mm": 2.5,
                "thickness_mm": 1.8,
                "features": [],
                "confidence": 0.5,
                "reasoning": "Default fallback"
            }


def test_generator():
    """Test the image generator"""
    print("\n" + "="*60)
    print("🧪 Testing Component Image Generator")
    print("="*60)
    
    generator = ComponentImageGenerator()
    
    # Test with a sample image if available
    test_images = list(Path(__file__).parent.parent.glob("test_images/*.png"))
    if not test_images:
        test_images = list(Path(__file__).parent.parent.glob("test_images/*.jpg"))
    
    if test_images:
        test_image = test_images[0]
        print(f"\n📸 Testing with: {test_image.name}")
        
        result = generator.analyze_and_generate(test_image)
        
        print(f"\n📋 Results:")
        for comp, data in result.items():
            if data:
                print(f"   {comp}: {data.get('specs', {}).get('shape', 'N/A')}")
                if data.get('generated_image'):
                    print(f"         Image: {data['generated_image']}")
    else:
        print("\n⚠️ No test images found. Add images to test_images/ folder.")
    
    # Test shank analyzer
    print("\n" + "-"*40)
    print("Testing Shank Analyzer...")
    
    analyzer = SmartShankAnalyzer()
    if test_images:
        shank_result = analyzer.analyze_shank_style(test_images[0])
        print(f"   Style: {shank_result['style']}")
        print(f"   Width: {shank_result['width_mm']}mm")
        print(f"   Confidence: {shank_result['confidence']:.0%}")


if __name__ == "__main__":
    test_generator()
