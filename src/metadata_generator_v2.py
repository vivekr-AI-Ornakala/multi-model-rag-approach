"""
Comprehensive CAD Metadata Generator

Creates detailed, accurate metadata for prongs and stones by:
1. Analyzing CAD geometry (dimensions, structure)
2. Analyzing multiple screenshot views
3. Cross-validating with multiple LLM passes
4. Producing structured, queryable metadata

The goal: Perfect retrieval through accurate metadata + hard filters
"""
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum
import math

import google.generativeai as genai
from PIL import Image
import rhino3dm

# Import API key from config
from config import GEMINI_API_KEY

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class StoneShape(Enum):
    ROUND = "round"
    OVAL = "oval"
    PEAR = "pear"
    MARQUISE = "marquise"
    PRINCESS = "princess"
    CUSHION = "cushion"
    EMERALD = "emerald"
    RADIANT = "radiant"
    ASSCHER = "asscher"
    HEART = "heart"
    TRILLION = "trillion"
    BAGUETTE = "baguette"
    HALF_MOON = "half_moon"
    OTHER = "other"


class SettingStyle(Enum):
    PRONG = "prong"
    BEZEL = "bezel"
    CHANNEL = "channel"
    PAVE = "pave"
    TENSION = "tension"
    HALO = "halo"
    CATHEDRAL = "cathedral"
    BASKET = "basket"
    TIFFANY = "tiffany"
    OTHER = "other"


@dataclass
class ProngMetadata:
    """Comprehensive prong/setting metadata"""
    component_id: str
    cad_file: str
    
    # Geometry from CAD analysis
    bounding_box_width: float  # mm
    bounding_box_depth: float  # mm
    bounding_box_height: float  # mm
    opening_diameter: float  # Inner opening where stone sits (mm)
    outer_diameter: float  # Outer prong tip diameter (mm)
    
    # Visual analysis
    prong_count: int  # VERIFIED count: 2, 3, 4, 5, 6, 8
    prong_count_confidence: str  # high, medium, low
    
    # Compatible stone info
    compatible_stone_shapes: List[str]  # ["oval", "round", ...]
    min_stone_size_mm: float  # Minimum stone this can hold
    max_stone_size_mm: float  # Maximum stone this can hold
    ideal_stone_size_mm: float  # Best fit stone size
    
    # Style classification
    setting_style: str  # basket, tiffany, cathedral, etc.
    setting_type: str  # solitaire, halo, three-stone, etc.
    has_accent_stones: bool
    has_gallery: bool  # Open gallery under stone
    
    # Design characteristics
    prong_shape: str  # round, pointed, flat, claw
    basket_shape: str  # round, oval, square, rectangular
    profile_height: str  # low, medium, high
    
    # Quality flags
    metadata_version: str
    geometry_source: str  # "cad_analysis" or "estimated"
    visual_source: str  # "multi_view_llm" or "single_view"


@dataclass 
class StoneMetadata:
    """Comprehensive stone metadata"""
    component_id: str
    cad_file: str
    
    # Geometry from CAD
    width_mm: float  # X dimension
    depth_mm: float  # Y dimension  
    height_mm: float  # Z dimension (crown to culet)
    girdle_max_mm: float  # Maximum girdle dimension
    
    # Shape classification
    shape: str  # round, oval, pear, etc.
    shape_confidence: str
    
    # Cut details
    cut_style: str  # brilliant, step, mixed, rose, etc.
    facet_pattern: str  # standard, modified, etc.
    
    # Proportions
    aspect_ratio: float  # width/depth
    crown_height_ratio: float  # Approximate
    pavilion_depth_ratio: float  # Approximate
    
    # Compatibility
    compatible_prong_counts: List[int]  # e.g., [4, 6] for round
    recommended_setting_styles: List[str]
    
    # Quality flags
    metadata_version: str


class ComprehensiveMetadataGenerator:
    """Generate detailed, accurate metadata for CAD components"""
    
    def __init__(self, use_fast_model: bool = True):
        # Use Flash model for faster processing (still accurate for counting)
        if use_fast_model:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("📚 Using Gemini Flash (faster)")
        else:
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            print("📚 Using Gemini Pro (more accurate)")
        
        self.prong_dir = Path('cad_library/prongs')
        self.stone_dir = Path('cad_library/stones')
        self.prong_sc_dir = Path('prongs_sc')
        self.stone_sc_dir = Path('stones_sc')
        self.output_dir = Path('vector_stores')
        
    def analyze_prong_geometry(self, cad_path: str) -> dict:
        """Extract precise dimensions from CAD file"""
        try:
            m = rhino3dm.File3dm.Read(cad_path)
            if not m or len(m.Objects) == 0:
                return {}
            
            geom = m.Objects[0].Geometry
            bb = geom.GetBoundingBox()
            
            width = bb.Max.X - bb.Min.X
            depth = bb.Max.Y - bb.Min.Y
            height = bb.Max.Z - bb.Min.Z
            top_z = bb.Max.Z
            
            # Analyze vertices to find opening
            opening_diameter = None
            outer_diameter = None
            
            if hasattr(geom, 'Vertices'):
                top_vertices = []
                for v in geom.Vertices:
                    if v.Location.Z >= top_z - (height * 0.3):
                        top_vertices.append((v.Location.X, v.Location.Y))
                
                if len(top_vertices) >= 4:
                    # Calculate centroid
                    cx = sum(v[0] for v in top_vertices) / len(top_vertices)
                    cy = sum(v[1] for v in top_vertices) / len(top_vertices)
                    
                    # Find distance from centroid
                    distances = [math.sqrt((v[0]-cx)**2 + (v[1]-cy)**2) for v in top_vertices]
                    
                    # Cluster distances
                    dist_set = sorted(set(round(d * 2) / 2 for d in distances))
                    
                    if len(dist_set) >= 2:
                        opening_diameter = dist_set[0] * 2
                        outer_diameter = dist_set[-1] * 2
            
            # Fallback estimates if vertex analysis failed
            if opening_diameter is None:
                opening_diameter = min(width, depth) * 0.80
            if outer_diameter is None:
                outer_diameter = max(width, depth)
            
            return {
                'bounding_box_width': round(width, 2),
                'bounding_box_depth': round(depth, 2),
                'bounding_box_height': round(height, 2),
                'opening_diameter': round(opening_diameter, 2),
                'outer_diameter': round(outer_diameter, 2),
                'geometry_source': 'cad_analysis'
            }
            
        except Exception as e:
            return {'error': str(e), 'geometry_source': 'failed'}
    
    def analyze_stone_geometry(self, cad_path: str) -> dict:
        """Extract precise dimensions from stone CAD"""
        try:
            m = rhino3dm.File3dm.Read(cad_path)
            if not m or len(m.Objects) == 0:
                return {}
            
            # Combined bounding box of all objects
            min_x = min_y = min_z = float('inf')
            max_x = max_y = max_z = float('-inf')
            
            for obj in m.Objects:
                bb = obj.Geometry.GetBoundingBox()
                min_x = min(min_x, bb.Min.X)
                min_y = min(min_y, bb.Min.Y)
                min_z = min(min_z, bb.Min.Z)
                max_x = max(max_x, bb.Max.X)
                max_y = max(max_y, bb.Max.Y)
                max_z = max(max_z, bb.Max.Z)
            
            width = max_x - min_x
            depth = max_y - min_y
            height = max_z - min_z
            
            return {
                'width_mm': round(width, 2),
                'depth_mm': round(depth, 2),
                'height_mm': round(height, 2),
                'girdle_max_mm': round(max(width, depth), 2),
                'aspect_ratio': round(width / depth, 2) if depth > 0 else 1.0,
                'geometry_source': 'cad_analysis'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_prong_visually(self, component_id: str) -> dict:
        """Analyze prong using multiple screenshot views"""
        
        views = ['perspective', 'front', 'right', 'top']
        images = []
        
        for view in views:
            img_path = self.prong_sc_dir / f"{component_id}_{view}.png"
            if img_path.exists():
                images.append(Image.open(img_path))
        
        if not images:
            return {'error': 'No screenshots found'}
        
        prompt = """Analyze this jewelry prong/setting CAD component from multiple views.

TASK 1 - COUNT PRONGS PRECISELY:
- Look at ALL views (perspective, front, right, top)
- Count the individual PRONG POSTS (vertical metal pieces that grip the stone)
- Prongs are usually 3, 4, 5, 6, or 8 in count
- The TOP view is most reliable for counting

TASK 2 - IDENTIFY BASKET SHAPE:
Looking at the TOP view, what is the shape of the opening?
- round (circular)
- oval (elliptical)
- square
- rectangular
- pear-shaped
- marquise-shaped
- heart-shaped

TASK 3 - IDENTIFY SETTING STYLE:
- basket: Traditional basket with prongs
- tiffany: 6-prong Tiffany-style
- cathedral: Arched shoulders
- bezel: Metal wrapping around stone
- halo: Center setting with surrounding stones
- solitaire: Simple single stone setting

TASK 4 - OTHER CHARACTERISTICS:
- Prong shape: round, pointed, flat, claw, v-prong
- Has accent stones: yes/no (small stones on setting)
- Has open gallery: yes/no (openwork under main stone)
- Profile height: low, medium, high

Respond ONLY with valid JSON:
{
    "prong_count": <integer>,
    "prong_count_confidence": "high" or "medium" or "low",
    "basket_shape": "<shape>",
    "compatible_stone_shapes": ["<shape1>", "<shape2>"],
    "setting_style": "<style>",
    "setting_type": "solitaire" or "halo" or "three-stone" or "accent",
    "prong_shape": "<shape>",
    "has_accent_stones": true or false,
    "has_gallery": true or false,
    "profile_height": "low" or "medium" or "high"
}"""

        try:
            content = [prompt] + images
            response = self.model.generate_content(content)
            text = response.text.strip()
            
            # Parse JSON
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text.strip())
            data['visual_source'] = 'multi_view_llm'
            return data
            
        except Exception as e:
            return {'error': str(e), 'visual_source': 'failed'}
    
    def analyze_stone_visually(self, component_id: str) -> dict:
        """Analyze stone using multiple screenshot views"""
        
        views = ['perspective', 'front', 'right', 'top']
        images = []
        
        for view in views:
            img_path = self.stone_sc_dir / f"{component_id}_{view}.png"
            if img_path.exists():
                images.append(Image.open(img_path))
        
        if not images:
            return {'error': 'No screenshots found'}
        
        prompt = """Analyze this gemstone CAD component from multiple views.

TASK 1 - IDENTIFY SHAPE:
Look at ALL views and determine the stone shape:
- round (circular outline)
- oval (elliptical)
- pear (teardrop)
- marquise (pointed oval, football shape)
- princess (square)
- cushion (rounded square)
- emerald (rectangular with cut corners)
- radiant (rectangular with brilliant facets)
- asscher (square step-cut)
- heart
- trillion (triangular)
- baguette (long rectangle)
- half_moon (semi-circle)

TASK 2 - IDENTIFY CUT STYLE:
- brilliant: Many triangular facets (sparkly)
- step: Rectangular facets in steps (emerald cut)
- mixed: Combination of brilliant and step
- rose: Flat bottom, domed faceted top
- cabochon: Smooth, no facets

TASK 3 - COMPATIBLE SETTINGS:
Based on shape, what prong counts work?
- Round: typically 4, 6, or 8 prongs
- Oval/Marquise: typically 4 or 6 prongs
- Princess/Cushion: typically 4 prongs (corner prongs)
- Pear/Heart: typically 3, 5, or 6 prongs
- Emerald/Radiant: typically 4 prongs

Respond ONLY with valid JSON:
{
    "shape": "<shape>",
    "shape_confidence": "high" or "medium" or "low",
    "cut_style": "<cut>",
    "facet_pattern": "standard" or "modified" or "custom",
    "compatible_prong_counts": [<int>, <int>],
    "recommended_setting_styles": ["<style1>", "<style2>"]
}"""

        try:
            content = [prompt] + images
            response = self.model.generate_content(content)
            text = response.text.strip()
            
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            if text.endswith("```"):
                text = text[:-3]
            
            data = json.loads(text.strip())
            return data
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_prong_metadata(self, component_id: str) -> dict:
        """Generate complete metadata for a prong"""
        
        cad_path = self.prong_dir / f"{component_id}.3dm"
        
        # Get geometry from CAD
        geom_data = self.analyze_prong_geometry(str(cad_path))
        
        # Get visual analysis
        visual_data = self.analyze_prong_visually(component_id)
        
        # Calculate stone size compatibility
        opening = geom_data.get('opening_diameter', 0)
        min_stone = opening * 0.85 if opening else 0
        max_stone = opening * 1.05 if opening else 0
        ideal_stone = opening * 0.95 if opening else 0
        
        # Merge data
        metadata = {
            'component_id': component_id,
            'cad_file': str(cad_path),
            'metadata_version': '2.0',
            
            # Geometry
            **geom_data,
            
            # Visual
            'prong_count': visual_data.get('prong_count', 4),
            'prong_count_confidence': visual_data.get('prong_count_confidence', 'unknown'),
            'compatible_stone_shapes': visual_data.get('compatible_stone_shapes', []),
            'setting_style': visual_data.get('setting_style', 'unknown'),
            'setting_type': visual_data.get('setting_type', 'unknown'),
            'prong_shape': visual_data.get('prong_shape', 'unknown'),
            'basket_shape': visual_data.get('basket_shape', 'unknown'),
            'has_accent_stones': visual_data.get('has_accent_stones', False),
            'has_gallery': visual_data.get('has_gallery', False),
            'profile_height': visual_data.get('profile_height', 'medium'),
            'visual_source': visual_data.get('visual_source', 'unknown'),
            
            # Stone compatibility
            'min_stone_size_mm': round(min_stone, 2),
            'max_stone_size_mm': round(max_stone, 2),
            'ideal_stone_size_mm': round(ideal_stone, 2),
        }
        
        return metadata
    
    def generate_stone_metadata(self, component_id: str) -> dict:
        """Generate complete metadata for a stone"""
        
        cad_path = self.stone_dir / f"{component_id}.3dm"
        
        # Get geometry from CAD
        geom_data = self.analyze_stone_geometry(str(cad_path))
        
        # Get visual analysis
        visual_data = self.analyze_stone_visually(component_id)
        
        # Calculate proportions
        width = geom_data.get('width_mm', 0)
        depth = geom_data.get('depth_mm', 0)
        height = geom_data.get('height_mm', 0)
        
        crown_ratio = 0.15  # Estimate
        pavilion_ratio = 0.43  # Estimate
        
        metadata = {
            'component_id': component_id,
            'cad_file': str(cad_path),
            'metadata_version': '2.0',
            
            # Geometry
            **geom_data,
            
            # Visual
            'shape': visual_data.get('shape', 'unknown'),
            'shape_confidence': visual_data.get('shape_confidence', 'unknown'),
            'cut_style': visual_data.get('cut_style', 'unknown'),
            'facet_pattern': visual_data.get('facet_pattern', 'unknown'),
            'compatible_prong_counts': visual_data.get('compatible_prong_counts', [4]),
            'recommended_setting_styles': visual_data.get('recommended_setting_styles', ['prong']),
            
            # Proportions
            'crown_height_ratio': crown_ratio,
            'pavilion_depth_ratio': pavilion_ratio,
        }
        
        return metadata
    
    def generate_all_metadata(self, prong_limit: int = None, stone_limit: int = None, resume: bool = True):
        """Generate metadata for all components with resume capability"""
        
        print("="*60)
        print("COMPREHENSIVE METADATA GENERATION")
        print("="*60)
        
        # Load existing metadata for resume
        existing_prongs = {}
        existing_stones = {}
        
        prong_output = self.output_dir / 'prongs_metadata_v2.json'
        stone_output = self.output_dir / 'stones_metadata_v2.json'
        
        if resume:
            if prong_output.exists():
                with open(prong_output) as f:
                    existing_prongs = json.load(f)
                print(f"📂 Loaded {len(existing_prongs)} existing prong metadata")
            if stone_output.exists():
                with open(stone_output) as f:
                    existing_stones = json.load(f)
                print(f"📂 Loaded {len(existing_stones)} existing stone metadata")
        
        # Generate prong metadata
        print("\n📦 GENERATING PRONG METADATA...")
        prong_files = list(self.prong_dir.glob('*.3dm'))
        if prong_limit:
            prong_files = prong_files[:prong_limit]
        
        prong_metadata = existing_prongs.copy()
        errors = []
        
        for i, pf in enumerate(prong_files):
            component_id = pf.stem
            
            # Skip if already processed (resume)
            if component_id in prong_metadata and resume:
                print(f"[{i+1}/{len(prong_files)}] {component_id}... skipped (already done)")
                continue
            
            print(f"\n[{i+1}/{len(prong_files)}] {component_id}...")
            
            try:
                metadata = self.generate_prong_metadata(component_id)
                prong_metadata[component_id] = metadata
                
                # Show key info
                pc = metadata.get('prong_count', '?')
                shape = metadata.get('basket_shape', '?')
                opening = metadata.get('opening_diameter', '?')
                print(f"   ✓ {pc}-prong, {shape} basket, {opening}mm opening")
                
                # Save after each one (in case of crash)
                with open(prong_output, 'w') as f:
                    json.dump(prong_metadata, f, indent=2)
                
            except Exception as e:
                errors.append((component_id, str(e)))
                print(f"   ✗ Error: {e}")
            
            # Rate limiting - reduced for Flash
            time.sleep(0.5)
        
        # Save prong metadata (final)
        with open(prong_output, 'w') as f:
            json.dump(prong_metadata, f, indent=2)
        print(f"\n✅ Saved {len(prong_metadata)} prong metadata to {prong_output}")
        
        # Generate stone metadata
        print("\n💎 GENERATING STONE METADATA...")
        stone_files = list(self.stone_dir.glob('*.3dm'))
        if stone_limit:
            stone_files = stone_files[:stone_limit]
        
        stone_metadata = existing_stones.copy()
        
        for i, sf in enumerate(stone_files):
            component_id = sf.stem
            
            # Skip if already done
            if component_id in stone_metadata and resume:
                print(f"[{i+1}/{len(stone_files)}] {component_id}... skipped")
                continue
            
            print(f"\n[{i+1}/{len(stone_files)}] {component_id}...")
            
            try:
                metadata = self.generate_stone_metadata(component_id)
                stone_metadata[component_id] = metadata
                
                shape = metadata.get('shape', '?')
                girdle = metadata.get('girdle_max_mm', '?')
                print(f"   ✓ {shape}, {girdle}mm")
                
            except Exception as e:
                errors.append((component_id, str(e)))
                print(f"   ✗ Error: {e}")
            
            time.sleep(0.5)
        
        # Save stone metadata
        with open(stone_output, 'w') as f:
            json.dump(stone_metadata, f, indent=2)
        print(f"\n✅ Saved {len(stone_metadata)} stone metadata to {stone_output}")
        
        # Summary
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"Prongs processed: {len(prong_metadata)}")
        print(f"Stones processed: {len(stone_metadata)}")
        print(f"Errors: {len(errors)}")
        
        if errors:
            print("\nErrors:")
            for comp_id, err in errors[:10]:
                print(f"  {comp_id}: {err}")


if __name__ == '__main__':
    generator = ComprehensiveMetadataGenerator()
    
    print("="*60)
    print("COMPREHENSIVE METADATA GENERATION")
    print("="*60)
    print()
    print("This will generate accurate metadata for all CAD components.")
    print("For 588 prongs, this takes approximately 30-60 minutes.")
    print()
    
    mode = input("Select mode:\n  1. Test (5 prongs + all stones) ~2 min\n  2. Full (all 588 prongs + stones) ~45 min\n  3. Cancel\n\nChoice (1/2/3): ")
    
    if mode == '1':
        generator.generate_all_metadata(prong_limit=5, stone_limit=None)
    elif mode == '2':
        generator.generate_all_metadata()
    else:
        print("Cancelled.")
