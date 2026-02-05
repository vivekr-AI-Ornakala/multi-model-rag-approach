"""
CLI Interface for Multi-Modal RAG CAD Component Agent
"""
import argparse
import sys
from pathlib import Path

from config import MAX_ITERATIONS, TOP_K_RESULTS


def cmd_index(args):
    """Build embedding indices"""
    from embedding_indexer import EmbeddingIndexer
    from models import ComponentType
    
    indexer = EmbeddingIndexer()
    
    if args.component == "all":
        indexer.index_components(force_reindex=args.force)
    elif args.component == "prongs":
        indexer.index_components(ComponentType.PRONGS, force_reindex=args.force)
    elif args.component == "stones":
        indexer.index_components(ComponentType.STONES, force_reindex=args.force)


def cmd_search(args):
    """Search for components"""
    from agent import quick_search
    
    results = quick_search(
        component_type=args.type,
        description=args.query,
        top_k=args.top_k
    )
    
    if args.show_images:
        from PIL import Image
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, min(len(results), 5), figsize=(4*min(len(results), 5), 4))
        if len(results) == 1:
            axes = [axes]
        
        for ax, result in zip(axes, results[:5]):
            img = Image.open(result.component.screenshot_path)
            ax.imshow(img)
            ax.set_title(f"{result.component.component_id}\n{result.similarity_score:.3f}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig("search_results.png", dpi=150)
        print(f"\n📊 Results saved to: search_results.png")
        plt.show()


def cmd_process(args):
    """Process a jewelry design"""
    from agent import CADComponentAgent
    
    image_path = Path(args.image)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    agent = CADComponentAgent()
    
    analysis = agent.process_design(
        image_path=image_path,
        max_iterations=args.max_iter,
        top_k=args.top_k,
        verbose=not args.quiet
    )
    
    if args.output:
        agent.export_results(analysis, Path(args.output))
    else:
        agent.export_results(analysis)
    
    if args.show_results:
        from PIL import Image
        import matplotlib.pyplot as plt
        
        selected = [s for s in analysis.component_selections if s.selected_component]
        
        if selected:
            n_images = len(selected) + 1
            fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
            
            original = Image.open(image_path)
            axes[0].imshow(original)
            axes[0].set_title("Original Design")
            axes[0].axis('off')
            
            for ax, selection in zip(axes[1:], selected):
                img = Image.open(selection.selected_component.screenshot_path)
                ax.imshow(img)
                ax.set_title(f"{selection.requirement.component_type.value}\n{selection.selected_component.component_id}")
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig("results_visualization.png", dpi=150)
            print(f"\n📊 Visualization saved to: results_visualization.png")
            plt.show()
    
    return 0 if analysis.success else 1


def cmd_demo(args):
    """Run demo"""
    from embedding_indexer import EmbeddingIndexer
    from rag_retriever import RAGRetriever
    from models import ComponentType, ComponentRequirement
    
    print("="*60)
    print("🎯 Multi-Modal RAG CAD Agent - Demo")
    print("="*60)
    
    print("\n1️⃣ Loading indices...")
    indexer = EmbeddingIndexer()
    retriever = RAGRetriever(indexer)
    
    prongs = retriever.get_available_count(ComponentType.PRONGS)
    stones = retriever.get_available_count(ComponentType.STONES)
    
    print(f"   📊 Loaded {prongs} prongs, {stones} stones")
    
    demo_queries = [
        ("prongs", "4-prong round solitaire diamond setting"),
        ("prongs", "bezel setting for oval stone"),
        ("stones", "round brilliant cut diamond"),
        ("stones", "pear shaped gemstone"),
    ]
    
    print("\n2️⃣ Demo searches:")
    
    for comp_type, query in demo_queries:
        print(f"\n   🔍 '{query}'")
        component_type = ComponentType.PRONGS if comp_type == "prongs" else ComponentType.STONES
        
        requirement = ComponentRequirement(
            component_type=component_type,
            description=query
        )
        
        results = retriever.search_by_text(requirement, top_k=3)
        
        for r in results:
            print(f"      • {r.component.component_id} ({r.similarity_score:.3f})")
    
    print("\n" + "="*60)
    print("✅ Demo complete!")
    print("   Use: python cli.py process <image.jpg>")
    print("="*60)


def cmd_stats(args):
    """Show index statistics"""
    from embedding_indexer import EmbeddingIndexer
    from models import ComponentType
    from pathlib import Path
    from config import VECTOR_STORE_DIR
    
    indexer = EmbeddingIndexer()
    
    # Check for metadata files
    prongs_meta = VECTOR_STORE_DIR / "prongs_metadata.json"
    stones_meta = VECTOR_STORE_DIR / "stones_metadata.json"
    
    prongs_meta_count = 0
    stones_meta_count = 0
    
    if prongs_meta.exists():
        import json
        with open(prongs_meta) as f:
            prongs_meta_count = len(json.load(f))
    
    if stones_meta.exists():
        import json
        with open(stones_meta) as f:
            stones_meta_count = len(json.load(f))
    
    print("\n📊 Index Statistics:")
    print("-" * 40)
    print(f"   Prongs: {indexer.get_component_count(ComponentType.PRONGS)} indexed")
    print(f"           {prongs_meta_count} with metadata (descriptions)")
    print(f"   Stones: {indexer.get_component_count(ComponentType.STONES)} indexed")
    print(f"           {stones_meta_count} with metadata (descriptions)")
    print("-" * 40)
    
    if prongs_meta_count == 0 and stones_meta_count == 0:
        print("\n⚠️  No metadata generated yet!")
        print("   Run: python cli.py metadata --component all")
        print("   This will improve search accuracy significantly.")


def cmd_metadata(args):
    """Generate metadata descriptions for components using Vision LLM"""
    from metadata_generator import MetadataGenerator
    
    print("\n" + "="*60)
    print("📝 Generating Component Metadata with Vision LLM")
    print("="*60)
    print("\nThis uses Gemini to analyze each component and create")
    print("detailed descriptions for improved search accuracy.\n")
    
    generator = MetadataGenerator()
    
    if args.component == "all":
        generator.generate_all(force=args.force, prong_limit=args.limit)
    elif args.component == "prongs":
        generator.generate_prong_metadata(force=args.force, limit=args.limit)
    elif args.component == "stones":
        generator.generate_stone_metadata(force=args.force)
    
    print("\n✅ Metadata generation complete!")
    print("   Now re-run indexing to include metadata:")
    print("   python cli.py index --force")


def cmd_assemble(args):
    """Assemble retrieved components into a .3dm file"""
    from rhino3dm_assembler import Rhino3dmAssembler, get_latest_results
    from pathlib import Path
    
    results_path = args.results
    if not results_path:
        results_path = get_latest_results()
        if results_path:
            print(f"📂 Using latest results: {results_path}")
        else:
            print("❌ No results file found. Run 'python cli.py process <image>' first.")
            return
    
    if not Path(results_path).exists():
        print(f"❌ File not found: {results_path}")
        return
    
    assembler = Rhino3dmAssembler()
    output = assembler.assemble_from_results(results_path, args.output)
    
    if output:
        print(f"\n🎉 Open in Rhino: {output}")


def cmd_complete(args):
    """Complete ring - add shank to head assembly"""
    import sys
    from pathlib import Path
    from config import ASSEMBLIES_DIR
    
    # Add moltbot skill path
    skill_path = Path(__file__).parent.parent / "scripts" / "moltbot_skills" / "jewelry_shank"
    sys.path.insert(0, str(skill_path))
    
    from rhino3dm_assembler import get_latest_assembly
    
    # Get head assembly
    head_path = args.head
    if not head_path:
        head_path = get_latest_assembly()
        if head_path:
            print(f"📂 Using latest assembly: {head_path}")
        else:
            print("❌ No assembly file found. Run 'python cli.py process <image>' first.")
            return
    
    head_path = Path(head_path)
    if not head_path.exists():
        print(f"❌ File not found: {head_path}")
        return
    
    # Reference image (optional - for AI-driven style detection)
    ref_image = Path(args.reference) if args.reference else None
    
    if ref_image and not ref_image.exists():
        print(f"❌ Reference image not found: {ref_image}")
        return
    
    try:
        from main import JewelryShankSkill
        from shank_generator import ShankGenerator, RingParameters
        
        skill = JewelryShankSkill()
        
        if ref_image:
            # Use AI to analyze reference image and generate matching shank
            output = skill.complete_ring(
                head_3dm_path=head_path,
                reference_image=ref_image,
                ring_size=args.size,
                output_path=args.output
            )
        else:
            # Use specified style directly
            print(f"\n💍 Completing ring with {args.style} shank...")
            print(f"   Ring size: {args.size}")
            
            import rhino3dm
            from datetime import datetime
            
            # Load head
            head_model = rhino3dm.File3dm.Read(str(head_path))
            if not head_model:
                raise ValueError(f"Could not load: {head_path}")
            
            # Get head bounding box
            head_min_x = head_min_y = head_min_z = float('inf')
            head_max_x = head_max_y = head_max_z = float('-inf')
            for obj in head_model.Objects:
                geom = obj.Geometry
                if geom:
                    try:
                        bbox = geom.GetBoundingBox()
                        head_min_x = min(head_min_x, bbox.Min.X)
                        head_min_y = min(head_min_y, bbox.Min.Y)
                        head_min_z = min(head_min_z, bbox.Min.Z)
                        head_max_x = max(head_max_x, bbox.Max.X)
                        head_max_y = max(head_max_y, bbox.Max.Y)
                        head_max_z = max(head_max_z, bbox.Max.Z)
                    except:
                        pass
            
            head_width = max(head_max_x - head_min_x, head_max_y - head_min_y)
            head_height = head_max_z - head_min_z
            
            # Generate shank
            params = RingParameters(
                ring_size=args.size,
                band_width=args.width,
                band_thickness=args.thickness,
                style=args.style
            )
            
            generator = ShankGenerator()
            shank_model = generator.generate(params)
            
            # Calculate proper scale for head relative to shank
            # The assembled head already has correct stone-prong proportions
            # We just need to ensure head looks right on the ring
            shank_outer_diameter = (params.inner_radius + params.band_thickness) * 2
            
            # Standard solitaire: head should be ~5-8mm for a typical ring
            # If head is already in reasonable range (4-10mm), don't scale much
            # If head is too big/small, scale to ~6mm target
            target_head_width = 6.0  # Standard solitaire head size in mm
            
            if head_width < 4.0 or head_width > 12.0:
                # Head is outside reasonable range, scale to target
                head_scale = target_head_width / head_width if head_width > 0 else 1.0
            else:
                # Head is in reasonable range, only minor adjustment
                # Scale to make it look proportional (45% of ring diameter max)
                max_head = shank_outer_diameter * 0.45
                if head_width > max_head:
                    head_scale = max_head / head_width
                else:
                    head_scale = 1.0  # Keep original size
            
            # Clamp scale to reasonable range
            head_scale = max(0.30, min(1.5, head_scale))
            
            print(f"   Head original width: {head_width:.1f}mm")
            print(f"   Shank outer diameter: {shank_outer_diameter:.1f}mm")
            print(f"   Scaling head by: {head_scale:.2f}x")
            
            # Shank is centered at Z=0, extends from -radius to +radius
            shank_radius = params.inner_radius + params.band_thickness
            
            # Position shank so top is at Z=0 (will position head above)
            shank_top_z = 0
            shank_translation_z = shank_top_z - shank_radius
            
            # Scale head and position it so bottom sits on shank top
            # After scaling, head height becomes head_height * head_scale
            scaled_head_height = head_height * head_scale
            # Head bottom should be at shank_top_z, so translate head up
            head_center_z = (head_min_z + head_max_z) / 2
            # After scaling around origin, head extends from head_min_z*scale to head_max_z*scale
            scaled_head_min_z = head_min_z * head_scale
            head_translation_z = shank_top_z - scaled_head_min_z
            
            print(f"   Shank top at Z={shank_top_z:.1f}")
            print(f"   Head positioned at Z={head_translation_z:.1f}")
            
            # Combine
            final_model = rhino3dm.File3dm()
            
            # Add layers
            shank_layer = rhino3dm.Layer()
            shank_layer.Name = "Ring_Shank"
            shank_idx = final_model.Layers.Add(shank_layer)
            
            head_layer = rhino3dm.Layer()
            head_layer.Name = "Ring_Head"
            head_idx = final_model.Layers.Add(head_layer)
            
            # Add shank with translation
            for obj in shank_model.Objects:
                geom = obj.Geometry
                if geom:
                    # Translate shank down 
                    transform = rhino3dm.Transform.Translation(0, 0, shank_translation_z)
                    geom.Transform(transform)
                    
                    attr = rhino3dm.ObjectAttributes()
                    attr.LayerIndex = shank_idx
                    final_model.Objects.Add(geom, attr)
            
            # Add head with scaling and translation
            for obj in head_model.Objects:
                geom = obj.Geometry
                if geom:
                    # Scale head uniformly around origin
                    scale_transform = rhino3dm.Transform.Scale(
                        rhino3dm.Point3d(0, 0, 0), 
                        head_scale
                    )
                    geom.Transform(scale_transform)
                    
                    # Then translate to sit on shank
                    translate_transform = rhino3dm.Transform.Translation(0, 0, head_translation_z)
                    geom.Transform(translate_transform)
                    
                    attr = rhino3dm.ObjectAttributes()
                    attr.LayerIndex = head_idx
                    final_model.Objects.Add(geom, attr)
                    attr.LayerIndex = head_idx
                    final_model.Objects.Add(obj.Geometry, attr)
            
            # Save
            if not args.output:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ASSEMBLIES_DIR.mkdir(parents=True, exist_ok=True)
                output = str(ASSEMBLIES_DIR / f"complete_ring_{timestamp}.3dm")
            else:
                output = args.output
            
            final_model.Write(output, 7)
            print(f"\n✅ Complete ring saved: {output}")
            
    except ImportError as e:
        print(f"❌ Could not import shank generator: {e}")
        print("   Make sure you're in the correct directory.")


def main():
    parser = argparse.ArgumentParser(
        description="🎯 Multi-Modal RAG CAD Component Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build indices (first time)
  python cli.py index --component all
  
  # Search for components
  python cli.py search --type prongs --query "4-prong setting" --show
  
  # Process a jewelry design
  python cli.py process ring.jpg --show
  
  # Run demo
  python cli.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Index
    index_p = subparsers.add_parser("index", help="Build embedding indices")
    index_p.add_argument("--component", choices=["prongs", "stones", "all"], default="all")
    index_p.add_argument("--force", action="store_true", help="Force rebuild")
    
    # Search
    search_p = subparsers.add_parser("search", help="Search components")
    search_p.add_argument("--type", "-t", choices=["prongs", "stones"], required=True)
    search_p.add_argument("--query", "-q", required=True)
    search_p.add_argument("--top-k", type=int, default=TOP_K_RESULTS)
    search_p.add_argument("--show", dest="show_images", action="store_true")
    
    # Process
    process_p = subparsers.add_parser("process", help="Process jewelry design")
    process_p.add_argument("image", nargs="?", help="Path to design image")
    process_p.add_argument("--max-iter", type=int, default=MAX_ITERATIONS)
    process_p.add_argument("--top-k", type=int, default=TOP_K_RESULTS)
    process_p.add_argument("--output", "-o", help="Output JSON path")
    process_p.add_argument("--quiet", "-q", action="store_true")
    process_p.add_argument("--show", dest="show_results", action="store_true")
    
    # Demo
    subparsers.add_parser("demo", help="Run demonstration")
    
    # Stats
    subparsers.add_parser("stats", help="Show index statistics")
    
    # Metadata generation
    meta_p = subparsers.add_parser("metadata", help="Generate component descriptions using Vision LLM")
    meta_p.add_argument("--component", choices=["prongs", "stones", "all"], default="all")
    meta_p.add_argument("--force", action="store_true", help="Regenerate existing metadata")
    meta_p.add_argument("--limit", type=int, help="Limit prongs to process (for testing)")
    
    # Assembly
    assemble_p = subparsers.add_parser("assemble", help="Assemble retrieved components into .3dm file")
    assemble_p.add_argument("--results", "-r", help="Path to results JSON file (uses latest if not specified)")
    assemble_p.add_argument("--output", "-o", help="Output .3dm file path")
    
    # Complete ring (add shank)
    complete_p = subparsers.add_parser("complete", help="Complete ring by adding shank to head assembly")
    complete_p.add_argument("--head", "-H", help="Path to head assembly .3dm file (uses latest if not specified)")
    complete_p.add_argument("--reference", "-r", help="Reference image for AI style detection")
    complete_p.add_argument("--style", "-s", choices=["plain", "cathedral", "split", "tapered"], default="plain")
    complete_p.add_argument("--size", type=float, default=7.0, help="Ring size (default: 7)")
    complete_p.add_argument("--width", type=float, default=2.5, help="Band width in mm (default: 2.5)")
    complete_p.add_argument("--thickness", type=float, default=1.8, help="Band thickness in mm (default: 1.8)")
    complete_p.add_argument("--output", "-o", help="Output .3dm file path")
    
    args = parser.parse_args()
    
    # If no command given, run interactive mode
    if args.command is None:
        interactive_mode()
        return
    
    commands = {
        "index": cmd_index,
        "search": cmd_search,
        "process": cmd_process,
        "demo": cmd_demo,
        "stats": cmd_stats,
        "metadata": cmd_metadata,
        "assemble": cmd_assemble,
        "complete": cmd_complete
    }
    
    commands[args.command](args)


def interactive_mode():
    """Interactive mode - prompts user for image input"""
    print("\n" + "="*60)
    print("🎯 Multi-Modal RAG CAD Component Agent")
    print("="*60)
    print("\nThis tool finds matching CAD components for jewelry designs.\n")
    
    # Get image path from user
    while True:
        image_path = input("📸 Enter the path to your jewelry image: ").strip()
        
        # Remove quotes if user wrapped path in quotes
        image_path = image_path.strip('"').strip("'")
        
        if not image_path:
            print("❌ Please enter an image path.")
            continue
            
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"❌ File not found: {image_path}")
            print("   Please check the path and try again.\n")
            continue
        
        if not image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            print(f"❌ Invalid image format: {image_path.suffix}")
            print("   Supported formats: jpg, jpeg, png, webp, bmp\n")
            continue
            
        break
    
    # Ask for options
    print("\n📋 Options:")
    show_results = input("   Show visualization after processing? (y/n) [y]: ").strip().lower()
    show_results = show_results != 'n'
    
    print("\n" + "-"*60)
    print("🚀 Starting processing...")
    print("-"*60 + "\n")
    
    # Process the image
    from agent import CADComponentAgent
    
    agent = CADComponentAgent()
    
    analysis = agent.process_design(
        image_path=image_path,
        max_iterations=MAX_ITERATIONS,
        top_k=TOP_K_RESULTS,
        verbose=True
    )
    
    # Export results
    agent.export_results(analysis)
    
    # Show visualization
    if show_results and any(s.selected_component for s in analysis.component_selections):
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
            
            selected = [s for s in analysis.component_selections if s.selected_component]
            n_images = len(selected) + 1
            
            fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
            if n_images == 1:
                axes = [axes]
            
            original = Image.open(image_path)
            axes[0].imshow(original)
            axes[0].set_title("Original Design")
            axes[0].axis('off')
            
            for ax, selection in zip(axes[1:], selected):
                img = Image.open(selection.selected_component.screenshot_path)
                ax.imshow(img)
                ax.set_title(f"{selection.requirement.component_type.value}\n{selection.selected_component.component_id}")
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save to outputs/visualizations folder
            from config import VISUALIZATIONS_DIR
            VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
            vis_path = VISUALIZATIONS_DIR / "results_visualization.png"
            plt.savefig(str(vis_path), dpi=150)
            print(f"\n📊 Visualization saved to: {vis_path}")
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not show visualization: {e}")
    
    # Auto-assemble into .3dm file using SMART ASSEMBLER
    if analysis.success:
        print("\n" + "-"*60)
        print("🔧 Smart CAD Assembly (with shape adaptation)...")
        print("-"*60)
        
        output_3dm = None  # Track the assembly path
        try:
            from smart_assembler import SmartAssembler
            from rhino3dm_assembler import get_latest_results, get_latest_assembly
            
            results_path = get_latest_results()
            if results_path:
                assembler = SmartAssembler()
                output_3dm = assembler.assemble_from_results(results_path)
                
                if output_3dm:
                    print(f"\n🎉 SMART ASSEMBLY READY!")
                    print(f"   📁 {output_3dm}")
        except Exception as e:
            print(f"⚠️ Could not assemble with smart assembler: {e}")
            # Fall back to basic assembler
            try:
                from rhino3dm_assembler import Rhino3dmAssembler
                print("   Trying basic assembler...")
                assembler = Rhino3dmAssembler()
                output_3dm = assembler.assemble_from_results(results_path)
                if output_3dm:
                    print(f"   📁 {output_3dm}")
            except Exception as e2:
                print(f"   Basic assembler also failed: {e2}")
        
        # Ask about adding shank to complete ring
        print("\n" + "-"*60)
        print("💍 Complete Ring - Add Shank")
        print("-"*60)
        
        add_shank = input("\n   Add ring shank to complete the ring? (y/n) [y]: ").strip().lower()
        
        if add_shank != 'n':
            print("\n   Available shank styles:")
            print("   1. plain     - Simple comfort-fit band")
            print("   2. cathedral - Arched shoulders rising toward head")
            print("   3. split     - Band splits into two paths near head")
            
            style_choice = input("\n   Select style (1/2/3) [1]: ").strip()
            style_map = {'1': 'plain', '2': 'cathedral', '3': 'split', '': 'plain'}
            shank_style = style_map.get(style_choice, 'plain')
            
            size_input = input("   Ring size (4-13) [7]: ").strip()
            try:
                ring_size = float(size_input) if size_input else 7.0
            except:
                ring_size = 7.0
            
            print(f"\n   Generating {shank_style} shank, size {ring_size}...")
            
            try:
                import sys
                from config import ASSEMBLIES_DIR
                from datetime import datetime
                
                # Add skill path
                skill_path = Path(__file__).parent.parent / "scripts" / "moltbot_skills" / "jewelry_shank"
                sys.path.insert(0, str(skill_path))
                
                from shank_generator import ShankGenerator, RingParameters
                import rhino3dm
                
                # Use the just-created head assembly, or find the latest
                head_path = output_3dm if output_3dm else get_latest_assembly()
                if head_path:
                    head_model = rhino3dm.File3dm.Read(str(head_path))
                    
                    # Get head bounding box
                    head_min_x = head_min_y = head_min_z = float('inf')
                    head_max_x = head_max_y = head_max_z = float('-inf')
                    for obj in head_model.Objects:
                        geom = obj.Geometry
                        if geom:
                            try:
                                bbox = geom.GetBoundingBox()
                                head_min_x = min(head_min_x, bbox.Min.X)
                                head_min_y = min(head_min_y, bbox.Min.Y)
                                head_min_z = min(head_min_z, bbox.Min.Z)
                                head_max_x = max(head_max_x, bbox.Max.X)
                                head_max_y = max(head_max_y, bbox.Max.Y)
                                head_max_z = max(head_max_z, bbox.Max.Z)
                            except:
                                pass
                    
                    head_width = max(head_max_x - head_min_x, head_max_y - head_min_y)
                    
                    # Generate shank
                    params = RingParameters(
                        ring_size=ring_size,
                        band_width=2.5,
                        band_thickness=1.8,
                        style=shank_style
                    )
                    
                    generator = ShankGenerator()
                    shank_model = generator.generate(params)
                    
                    # Calculate proper head scale to fit shank
                    # Shank outer diameter = inner_diameter + 2*band_thickness
                    shank_outer_diameter = (params.inner_radius + params.band_thickness) * 2
                    
                    # Head width should be proportional to shank
                    # Typical solitaire: head is 50-70% of shank outer diameter for balance
                    target_head_width = shank_outer_diameter * 0.55
                    head_scale = target_head_width / head_width if head_width > 0 else 1.0
                    
                    print(f"   → Head width: {head_width:.1f} → scaled to {target_head_width:.1f}mm (scale: {head_scale:.3f})")
                    
                    # Position head on top of shank
                    shank_radius = params.inner_radius + params.band_thickness
                    shank_top_z = 0
                    shank_translation_z = shank_top_z - shank_radius
                    
                    # Position scaled head so its bottom sits on top of shank
                    scaled_head_min_z = head_min_z * head_scale
                    head_translation_z = shank_top_z - scaled_head_min_z

                    
                    # Combine into final model
                    final_model = rhino3dm.File3dm()
                    
                    shank_layer = rhino3dm.Layer()
                    shank_layer.Name = "Ring_Shank"
                    shank_idx = final_model.Layers.Add(shank_layer)
                    
                    head_layer = rhino3dm.Layer()
                    head_layer.Name = "Ring_Head"
                    head_idx = final_model.Layers.Add(head_layer)
                    
                    # Add shank
                    for obj in shank_model.Objects:
                        geom = obj.Geometry
                        if geom:
                            transform = rhino3dm.Transform.Translation(0, 0, shank_translation_z)
                            geom.Transform(transform)
                            attr = rhino3dm.ObjectAttributes()
                            attr.LayerIndex = shank_idx
                            final_model.Objects.Add(geom, attr)
                    
                    # Add head (scale to fit shank, then position)
                    for obj in head_model.Objects:
                        geom = obj.Geometry
                        if geom:
                            # Scale from origin
                            scale_transform = rhino3dm.Transform.Scale(rhino3dm.Point3d(0, 0, 0), head_scale)
                            geom.Transform(scale_transform)
                            # Then translate
                            translate_transform = rhino3dm.Transform.Translation(0, 0, head_translation_z)
                            geom.Transform(translate_transform)
                            attr = rhino3dm.ObjectAttributes()
                            attr.LayerIndex = head_idx
                            final_model.Objects.Add(geom, attr)
                    
                    # Save
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ASSEMBLIES_DIR.mkdir(parents=True, exist_ok=True)
                    complete_path = str(ASSEMBLIES_DIR / f"complete_ring_{timestamp}.3dm")
                    final_model.Write(complete_path, 7)
                    
                    print(f"\n   ✅ COMPLETE RING SAVED!")
                    print(f"   📁 {complete_path}")
                else:
                    print("   ⚠️ No head assembly found")
            except Exception as e:
                print(f"   ⚠️ Could not add shank: {e}")
                print("   Run 'python run.py complete --style plain --size 7' manually.")
    
    print("\n" + "="*60)
    if analysis.success:
        print("✅ All components found and assembled!")
    else:
        print("⚠️ Some components could not be found.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
