#!/usr/bin/env python
"""
Jewelry CAD RAG Agent - Main Entry Point
=========================================
Run this script to start the jewelry CAD component retrieval and assembly system.

Usage:
    python run.py                    # Interactive mode (SMART - no prompts)
    python run.py image.jpg          # Process specific image (fully automatic)
    python run.py --legacy           # Use legacy interactive mode with prompts
    python run.py --help             # Show all commands
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def smart_mode(image_path: str = None):
    """
    Smart automated mode - no user prompts required.
    Analyzes image, generates component images, retrieves matches, assembles, and adds shank.
    """
    from smart_pipeline import SmartRAGPipeline
    
    # Get image path
    if not image_path:
        # Check for command line argument
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
            image_path = sys.argv[1]
        else:
            # Simple prompt for image path only
            print("\n" + "="*60)
            print("🤖 SMART JEWELRY CAD RAG AGENT")
            print("="*60)
            print("\nFully automated pipeline - no additional prompts required!")
            print("Just provide the image and the system handles everything.\n")
            
            image_path = input("📸 Enter jewelry image path: ").strip().strip('"').strip("'")
    
    path = Path(image_path)
    
    if not path.exists():
        print(f"❌ Image not found: {path}")
        sys.exit(1)
    
    # Run smart pipeline
    pipeline = SmartRAGPipeline()
    results = pipeline.process(path)
    
    return results.get("success", False)


def legacy_mode():
    """Legacy interactive mode with prompts"""
    from cli import main
    main()


if __name__ == "__main__":
    # Parse arguments
    if "--legacy" in sys.argv:
        legacy_mode()
    elif "--help" in sys.argv or "-h" in sys.argv:
        # Show help and run CLI for full command list
        from cli import main
        main()
    else:
        # Default: Smart mode
        image_arg = None
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                image_arg = arg
                break
        
        success = smart_mode(image_arg)
        sys.exit(0 if success else 1)

