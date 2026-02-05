"""
Configuration settings for the Multi-Modal RAG CAD Component Agent
Using state-of-the-art models for best performance
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths - Go up one level from src/
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent

# CAD Library paths
CAD_LIBRARY_DIR = BASE_DIR / "cad_library"
PRONGS_CAD_DIR = CAD_LIBRARY_DIR / "prongs"
STONES_CAD_DIR = CAD_LIBRARY_DIR / "stones"

# Screenshot directories (original wireframe)
PRONGS_SCREENSHOTS_DIR = BASE_DIR / "prongs_sc"
STONES_SCREENSHOTS_DIR = BASE_DIR / "stones_sc"

# Multi-view screenshot directories (shaded renders)
PRONGS_MULTIVIEW_DIR = BASE_DIR / "prongs_multiview"
STONES_MULTIVIEW_DIR = BASE_DIR / "stones_multiview"

# Vector store paths
VECTOR_STORE_DIR = BASE_DIR / "vector_stores"
CHROMA_DB_DIR = VECTOR_STORE_DIR / "chroma_db"  # Original
CHROMA_MULTIVIEW_DIR = VECTOR_STORE_DIR / "chroma_multiview"  # Multi-view (improved)

# Which vector store to use: "original" or "multiview"
ACTIVE_VECTOR_STORE = "multiview"  # Using multi-view shaded screenshots

# Output paths
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSEMBLIES_DIR = OUTPUTS_DIR / "assemblies"
RESULTS_DIR = OUTPUTS_DIR / "results"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
LOGS_DIR = BASE_DIR / "logs"

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional: for GPT-4V

# =============================================================================
# MODEL SETTINGS - Using best-in-class models
# =============================================================================

# Embedding Model Options (ranked by performance):
# 1. "google/siglip-so400m-patch14-384" - Google's SigLIP (excellent for image-text)
# 2. "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" - OpenCLIP bigG (best open CLIP)
# 3. "openai/clip-vit-large-patch14-336" - OpenAI CLIP Large
# 4. "microsoft/Florence-2-large" - Microsoft Florence (multimodal)

EMBEDDING_MODEL = "google/siglip-so400m-patch14-384"  # Best balance of speed/quality
EMBEDDING_DIMENSION = 1152  # SigLIP dimension (adjust if changing model)

# Vision LLM - Using your best available models:
# Available: gemini-2.5-pro, gemini-3-pro-preview, gemini-2.5-flash, gemini-2.0-flash
# 
# For ANALYSIS (design breakdown): Use Pro for best quality
# For VERIFICATION (component matching): Use Flash for speed

GEMINI_MODEL_ANALYSIS = "gemini-2.5-pro"  # Best for analyzing designs
GEMINI_MODEL_VERIFY = "gemini-2.5-flash"  # Fast for verification loops
GEMINI_MODEL = GEMINI_MODEL_ANALYSIS  # Default

# Vector Store Settings
VECTOR_STORE_TYPE = "chroma"  # Options: "chroma", "faiss"
USE_GPU_FOR_SEARCH = True  # Use GPU acceleration for FAISS if available

# =============================================================================
# RAG SETTINGS
# =============================================================================
TOP_K_RESULTS = 5  # Number of top matches to return
MAX_ITERATIONS = 5  # Maximum verification iterations per component
SIMILARITY_THRESHOLD = 0.0  # Minimum similarity (0 = no threshold, let LLM decide)

# Batch processing
BATCH_SIZE = 32  # Batch size for embedding generation
NUM_WORKERS = 4  # Parallel workers for data loading

# Caching
ENABLE_CACHE = True
CACHE_DIR = BASE_DIR / ".cache"

# Component types
COMPONENT_TYPES = ["prongs", "stones"]

# =============================================================================
# ENSURE DIRECTORIES EXIST
# =============================================================================
VECTOR_STORE_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
CHROMA_MULTIVIEW_DIR.mkdir(exist_ok=True)
PRONGS_MULTIVIEW_DIR.mkdir(exist_ok=True)
STONES_MULTIVIEW_DIR.mkdir(exist_ok=True)
(VECTOR_STORE_DIR / "multiview").mkdir(exist_ok=True)
if ENABLE_CACHE:
    CACHE_DIR.mkdir(exist_ok=True)
