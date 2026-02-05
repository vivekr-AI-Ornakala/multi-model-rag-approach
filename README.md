# 💎 Jewelry CAD RAG Agent

> **Multi-modal RAG (Retrieval-Augmented Generation) system for jewelry CAD component retrieval and intelligent assembly.**

An AI-powered pipeline that takes a jewelry design image and automatically produces a complete, manufacturable CAD file (`.3dm`) by:
1. Analyzing the design with Vision AI (Gemini)
2. Retrieving matching CAD components using SigLIP embeddings
3. Assembling components with Physics + AI hybrid system
4. Generating a parametric shank based on design analysis

---

## 📑 Table of Contents

- [System Overview](#-system-overview)
- [Mermaid Pipeline Diagrams](#-mermaid-pipeline-diagrams)
- [Complete Pipeline Flowchart](#-complete-pipeline-flowchart)
- [File Structure & Module Descriptions](#-file-structure--module-descriptions)
- [Core Modules Deep Dive](#-core-modules-deep-dive)
- [Assembly System Architecture](#-assembly-system-architecture)
- [Quick Start](#-quick-start)
- [Commands Reference](#-commands-reference)
- [Technology Stack](#-technology-stack)
- [Configuration](#-configuration)

---

## 🔄 System Overview

### High-Level Architecture

```mermaid
flowchart LR
    subgraph INPUT["📸 INPUT"]
        A[Jewelry Design Image]
    end
    
    subgraph ANALYSIS["🧠 ANALYSIS"]
        B[Gemini Vision AI]
    end
    
    subgraph RETRIEVAL["🔍 RETRIEVAL"]
        C[SigLIP + ChromaDB]
    end
    
    subgraph ASSEMBLY["⚙️ ASSEMBLY"]
        D[Physics + AI Engine]
    end
    
    subgraph OUTPUT["💍 OUTPUT"]
        E[Complete Ring .3dm]
    end
    
    A --> B --> C --> D --> E
```

### Two Operational Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Offline Preprocessing** | Index CAD files, generate embeddings, create metadata | Run once per library update |
| **Online Processing** | Analyze image → Retrieve → Assemble | Run per design request |

---

## 📊 Mermaid Pipeline Diagrams

### 🔷 Master Pipeline Overview

```mermaid
flowchart TB
    subgraph MASTER["🎯 JEWELRY CAD RAG PIPELINE"]
        direction TB
        
        START([📸 Input Image]) --> STEP1
        
        subgraph STEP1["STEP 1: Vision Analysis"]
            V1[Load Image] --> V2[Gemini 2.5 Pro]
            V2 --> V3[Extract Requirements]
        end
        
        STEP1 --> STEP2
        
        subgraph STEP2["STEP 2: Component Retrieval"]
            R1[Hard Filtering] --> R2[Vector Search]
            R2 --> R3[LLM Verification]
        end
        
        STEP2 --> STEP3
        
        subgraph STEP3["STEP 3: Physics Assembly"]
            P1[OBB Analysis] --> P2[Drop Test]
            P2 --> P3[Collision Check]
        end
        
        STEP3 --> STEP4
        
        subgraph STEP4["STEP 4: Shank Generation"]
            S1[Analyze Style] --> S2[Generate Mesh]
        end
        
        STEP4 --> STEP5
        
        subgraph STEP5["STEP 5: Final Assembly"]
            F1[Combine Layers] --> F2[Z-Alignment]
        end
        
        STEP5 --> FINISH([💍 Complete Ring .3dm])
    end
```

---

### 📍 STEP 1: Vision Analysis (Detailed)

```mermaid
flowchart TB
    subgraph VISION["STEP 1: VISION ANALYSIS"]
        direction TB
        
        INPUT[/"📸 Input Image"/] --> LOAD[Load with PIL]
        LOAD --> GEMINI["🤖 Gemini 2.5 Pro<br/>Vision LLM"]
        
        GEMINI --> PROMPT["Structured Prompt:<br/>Analyze jewelry components"]
        
        PROMPT --> EXTRACT["Extract JSON Response"]
        
        EXTRACT --> STONE_INFO
        EXTRACT --> PRONG_INFO
        EXTRACT --> SHANK_INFO
        EXTRACT --> SIZE_INFO
        
        subgraph STONE_INFO["💎 Stone Info"]
            ST1["shape: oval"]
            ST2["size_mm: 10.0"]
            ST3["color: pink"]
            ST4["cut_style: brilliant"]
        end
        
        subgraph PRONG_INFO["🔧 Prong Info"]
            PR1["style: 4-prong"]
            PR2["prong_count: 4"]
            PR3["shape: oval"]
        end
        
        subgraph SHANK_INFO["💫 Shank Info"]
            SH1["style: cathedral"]
            SH2["width_mm: 2.5"]
            SH3["thickness_mm: 1.8"]
        end
        
        subgraph SIZE_INFO["📏 Size Info"]
            SZ1["ring_size_estimate: 7.0"]
        end
    end
    
    STONE_INFO --> OUTPUT[/"Design Analysis Object"/]
    PRONG_INFO --> OUTPUT
    SHANK_INFO --> OUTPUT
    SIZE_INFO --> OUTPUT
```

**Key File:** `vision_analyzer.py` → `VisionAnalyzer.analyze_design_comprehensive()`

---

### 📍 STEP 2: Component Retrieval (Detailed)

```mermaid
flowchart TB
    subgraph RETRIEVAL["STEP 2: COMPONENT RETRIEVAL"]
        direction TB
        
        INPUT[/"Design Requirements"/] --> PARSE["Parse Requirements"]
        
        PARSE --> PRONG_COUNT["Extract prong_count: 4"]
        PARSE --> STONE_SIZE["Extract size_mm: 10.0"]
        PARSE --> STONE_SHAPE["Extract shape: oval"]
        
        subgraph HARD_FILTER["🚫 HARD FILTERING (Must Match)"]
            HF1["Prong count == 4<br/>❌ Reject 3-prong, 6-prong"]
            HF2["Size within 70-130%<br/>7mm - 13mm range"]
            HF3["Shape compatible<br/>oval → oval prong"]
        end
        
        PRONG_COUNT --> HF1
        STONE_SIZE --> HF2
        STONE_SHAPE --> HF3
        
        HF1 --> CANDIDATES["Filtered Candidates<br/>(~50 from 588)"]
        HF2 --> CANDIDATES
        HF3 --> CANDIDATES
        
        subgraph VECTOR_SEARCH["🔍 SEMANTIC SEARCH"]
            VS1["SigLIP Text Encoder"]
            VS2["Query: 'oval 4-prong basket'"]
            VS3["ChromaDB Cosine Search"]
            VS4["Top-K Results (K=5)"]
        end
        
        CANDIDATES --> VS1 --> VS2 --> VS3 --> VS4
        
        subgraph LLM_VERIFY["✅ LLM VERIFICATION"]
            LV1["Gemini 2.5 Flash"]
            LV2["Compare: Original vs Candidate"]
            LV3["Score: 0-100%"]
            LV4["Accept if > 70%"]
        end
        
        VS4 --> LV1 --> LV2 --> LV3 --> LV4
        
        LV4 --> BEST_STONE["📄 Best Stone Match<br/>118_118_526_S.3dm"]
        LV4 --> BEST_PRONG["📄 Best Prong Match<br/>114_114_574_S.3dm"]
    end
    
    BEST_STONE --> OUTPUT[/"Matched CAD Files"/]
    BEST_PRONG --> OUTPUT
```

**Key Files:** 
- `v2_retriever.py` → Hard filtering
- `embedding_indexer.py` → SigLIP + ChromaDB
- `vision_analyzer.py` → LLM verification

---

### 📍 STEP 3: Physics + AI Assembly (Detailed)

```mermaid
flowchart TB
    subgraph PHYSICS["STEP 3: PHYSICS + AI ASSEMBLY"]
        direction TB
        
        INPUT1[/"Stone.3dm"/] --> LOAD1["Load with rhino3dm"]
        INPUT2[/"Prong.3dm"/] --> LOAD2["Load with rhino3dm"]
        
        LOAD1 --> TRIMESH1["Convert to Trimesh"]
        LOAD2 --> TRIMESH2["Convert to Trimesh"]
        
        subgraph OBB["1️⃣ OBB ANALYSIS (Oriented Bounding Box)"]
            direction LR
            O1["Extract Vertices"]
            O2["Apply PCA<br/>(Principal Component Analysis)"]
            O3["Compute True Dimensions<br/>Width × Depth × Height"]
            O1 --> O2 --> O3
        end
        
        TRIMESH1 --> OBB
        TRIMESH2 --> OBB
        
        OBB --> STONE_DIM["Stone: 10.12 × 8.51 mm"]
        OBB --> PRONG_DIM["Prong Opening: 12.5 × 10.2 mm"]
        
        subgraph SCALING["📐 UNIFORM SCALING"]
            SC1["target_fit = 0.95 (95%)"]
            SC2["scale = prong_opening × 0.95 / stone_dim"]
            SC3["Apply uniform scale to stone"]
        end
        
        STONE_DIM --> SCALING
        PRONG_DIM --> SCALING
        
        subgraph DROP_TEST["2️⃣ RAY CASTING DROP TEST"]
            direction LR
            D1["Sample 100 points<br/>from stone girdle"]
            D2["Cast rays downward<br/>direction = (0, 0, -1)"]
            D3["Find intersection<br/>with prong mesh"]
            D4["Min distance = seat Z<br/>0.28mm"]
            D1 --> D2 --> D3 --> D4
        end
        
        SCALING --> DROP_TEST
        
        subgraph COLLISION["3️⃣ MESH COLLISION DETECTION"]
            direction LR
            C1["Trimesh CollisionManager"]
            C2["Check mesh intersection"]
            C3{"Collision?"}
            C4["✅ NO - Valid Fit"]
            C5["❌ YES - Adjust"]
            C1 --> C2 --> C3
            C3 -->|No| C4
            C3 -->|Yes| C5
        end
        
        DROP_TEST --> COLLISION
        
        subgraph AESTHETIC["4️⃣ AESTHETIC JUDGE (AI)"]
            direction LR
            A1["Gemini 2.5 Flash"]
            A2["'Does this look proportional?'"]
            A3["Score: 85/100"]
            A1 --> A2 --> A3
        end
        
        COLLISION --> AESTHETIC
    end
    
    AESTHETIC --> OUTPUT[/"Assembled Head<br/>(Stone + Prong)"/]
```

**Key File:** `smart_assembly_physics.py`
- `GeometryEngine.compute_obb()` - OBB via PCA
- `GeometryEngine.drop_test()` - Ray casting
- `GeometryEngine.check_collision()` - Mesh intersection
- `AestheticJudge` - AI style check

---

### 📍 STEP 4: Shank Generation (Detailed)

```mermaid
flowchart TB
    subgraph SHANK["STEP 4: SHANK GENERATION"]
        direction TB
        
        INPUT[/"Design Analysis"/] --> EXTRACT["Extract Shank Params"]
        
        EXTRACT --> RING_SIZE["ring_size: 7"]
        EXTRACT --> STYLE["style: cathedral"]
        EXTRACT --> WIDTH["width_mm: 2.5"]
        EXTRACT --> THICKNESS["thickness_mm: 1.8"]
        
        subgraph FORMULA["📐 RING SIZE FORMULA"]
            F1["diameter_mm = (US_size × 0.825) + 12.5"]
            F2["diameter = (7 × 0.825) + 12.5 = 18.275mm"]
            F3["radius = 9.14mm"]
        end
        
        RING_SIZE --> FORMULA
        
        subgraph STYLES["🎨 SHANK STYLES"]
            direction LR
            PLAIN["Plain<br/>Simple torus"]
            CATHEDRAL["Cathedral<br/>Arched sides"]
            SPLIT["Split<br/>Divided band"]
            TAPERED["Tapered<br/>Narrowing"]
        end
        
        STYLE --> STYLES
        
        subgraph MESH_GEN["🔧 MESH GENERATION"]
            M1["Create base torus mesh"]
            M2["Apply style modifications"]
            M3["Boolean operations"]
            M4["Export to rhino3dm"]
        end
        
        FORMULA --> MESH_GEN
        STYLES --> MESH_GEN
        WIDTH --> MESH_GEN
        THICKNESS --> MESH_GEN
    end
    
    MESH_GEN --> OUTPUT[/"Shank Mesh .3dm"/]
```

**Key File:** `dynamic_shank_generator.py` → `DynamicShankGenerator.generate()`

---

### 📍 STEP 5: Final Assembly (Detailed)

```mermaid
flowchart TB
    subgraph FINAL["STEP 5: FINAL ASSEMBLY"]
        direction TB
        
        INPUT1[/"Assembled Head<br/>(Stone + Prong)"/] --> LOAD1["Load Head Model"]
        INPUT2[/"Shank Mesh"/] --> LOAD2["Load Shank Model"]
        
        subgraph Z_CALC["📏 Z-POSITION CALCULATION"]
            Z1["Ring diameter = 18.275mm"]
            Z2["Band thickness = 1.8mm"]
            Z3["center_radius = inner_radius + thickness/2"]
            Z4["shank_top_z = center_radius + thickness/2"]
            Z5["shank_top_z ≈ 10.73mm"]
        end
        
        LOAD2 --> Z_CALC
        
        subgraph POSITION["🎯 COMPONENT POSITIONING"]
            P1["Shank: Z = -10.73 to +10.73<br/>(centered at finger hole)"]
            P2["Head: Translate so bottom = shank_top_z"]
            P3["Prong: Z = +10.73 to +21.47"]
            P4["Stone: Z = +11.14 to +18.67<br/>(inside prong via drop test)"]
        end
        
        Z_CALC --> POSITION
        LOAD1 --> POSITION
        
        subgraph LAYERS["📁 LAYER CREATION"]
            L1["Ring_Shank<br/>Color: Gold (200,180,100)"]
            L2["Prong_Setting<br/>Color: Silver (192,192,192)"]
            L3["Stone<br/>Color: Pink (255,0,100)"]
        end
        
        POSITION --> LAYERS
        
        subgraph COMBINE["🔗 COMBINE & SAVE"]
            CB1["Create new File3dm"]
            CB2["Add layers"]
            CB3["Add objects with attributes"]
            CB4["Write to .3dm file"]
        end
        
        LAYERS --> COMBINE
    end
    
    COMBINE --> OUTPUT[/"complete_ring_YYYYMMDD.3dm"/]
```

**Key File:** `smart_pipeline.py` → `SmartRAGPipeline._generate_complete_ring()`

---

### 🔷 Data Flow Diagram

```mermaid
flowchart LR
    subgraph INPUTS["📥 INPUTS"]
        IMG["📸 Design Image"]
        CAD_LIB["📁 CAD Library<br/>588 prongs + 15 stones"]
        VECTORS["🗄️ ChromaDB<br/>Vector Embeddings"]
    end
    
    subgraph PROCESSING["⚙️ PROCESSING"]
        direction TB
        VISION["Gemini Vision"]
        RETRIEVAL["SigLIP Search"]
        PHYSICS["Trimesh Physics"]
        SHANK_GEN["Shank Generator"]
    end
    
    subgraph OUTPUTS["📤 OUTPUTS"]
        RING["💍 complete_ring.3dm"]
        RESULTS["📄 results.json"]
        VIZ["🖼️ visualization.png"]
    end
    
    IMG --> VISION
    VISION --> RETRIEVAL
    CAD_LIB --> RETRIEVAL
    VECTORS --> RETRIEVAL
    RETRIEVAL --> PHYSICS
    PHYSICS --> SHANK_GEN
    SHANK_GEN --> RING
    SHANK_GEN --> RESULTS
    SHANK_GEN --> VIZ
```

---

### 🔷 Technology Stack Diagram

```mermaid
flowchart TB
    subgraph STACK["🔧 TECHNOLOGY STACK"]
        direction TB
        
        subgraph AI_LAYER["🤖 AI Layer"]
            GEMINI_PRO["Gemini 2.5 Pro<br/>Vision Analysis"]
            GEMINI_FLASH["Gemini 2.5 Flash<br/>Verification"]
        end
        
        subgraph EMBED_LAYER["🔍 Embedding Layer"]
            SIGLIP["SigLIP<br/>siglip-so400m-patch14-384"]
            CHROMA["ChromaDB<br/>HNSW Index"]
        end
        
        subgraph PHYSICS_LAYER["⚙️ Physics Layer"]
            TRIMESH["Trimesh<br/>3D Mesh Operations"]
            SCIPY["SciPy<br/>PCA, ConvexHull"]
            RTREE["RTree<br/>Spatial Indexing"]
        end
        
        subgraph CAD_LAYER["📐 CAD Layer"]
            RHINO3DM["rhino3dm<br/>.3dm File I/O"]
            NUMPY["NumPy<br/>Matrix Operations"]
        end
        
        AI_LAYER --> EMBED_LAYER --> PHYSICS_LAYER --> CAD_LAYER
    end
```

---

### 🔷 Class Relationship Diagram

```mermaid
classDiagram
    class SmartRAGPipeline {
        +process(image_path) Dict
        -_analyze_design()
        -_retrieve_stone()
        -_retrieve_prong()
        -_assemble_complete_ring()
        -_generate_complete_ring()
    }
    
    class VisionAnalyzer {
        +analyze_design_comprehensive() Dict
        +verify_component_match() bool
        -analysis_model: GenerativeModel
        -verify_model: GenerativeModel
    }
    
    class EmbeddingIndexer {
        +index_components()
        +search_similar() List
        -model: SiglipModel
        -chroma_client: PersistentClient
    }
    
    class V2Retriever {
        +filter_prongs_by_requirements() List
        +retrieve_with_hard_filter() Dict
        -prong_metadata: Dict
        -stone_metadata: Dict
    }
    
    class PhysicsAIAssembler {
        +assemble() Dict
        -geometry_engine: GeometryEngine
        -aesthetic_judge: AestheticJudge
    }
    
    class GeometryEngine {
        +rhino_to_trimesh() Trimesh
        +compute_obb() Dict
        +drop_test() float
        +check_collision() Dict
    }
    
    class DynamicShankGenerator {
        +generate(params) str
        -_create_plain_shank()
        -_create_cathedral_shank()
        -_create_split_shank()
    }
    
    SmartRAGPipeline --> VisionAnalyzer
    SmartRAGPipeline --> EmbeddingIndexer
    SmartRAGPipeline --> V2Retriever
    SmartRAGPipeline --> PhysicsAIAssembler
    SmartRAGPipeline --> DynamicShankGenerator
    PhysicsAIAssembler --> GeometryEngine
    V2Retriever --> EmbeddingIndexer
```

---

## 📊 Complete Pipeline Flowchart

### Stage 1: Offline Preprocessing (Run Once)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE PREPROCESSING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────────────┐                                                              │
│   │  CAD Files (.3dm)│                                                              │
│   │  ────────────────│                                                              │
│   │  cad_library/    │                                                              │
│   │  ├── prongs/     │  588 prong components                                        │
│   │  └── stones/     │  15 stone components                                         │
│   └────────┬─────────┘                                                              │
│            │                                                                        │
│            ▼                                                                        │
│   ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐   │
│   │  Rhino Screenshots│         │  Multi-View      │         │  LLM Metadata    │   │
│   │  (rhino_plugins/) │  ───►  │  Renders         │  ───►  │  (Gemini)        │   │
│   │  Generate .jpg    │         │  prongs_multiview│         │  Analyze each    │   │
│   │  from CAD files   │         │  stones_multiview│         │  component       │   │
│   └──────────────────┘         └────────┬─────────┘         └────────┬─────────┘   │
│                                         │                            │              │
│                                         ▼                            ▼              │
│                               ┌──────────────────┐         ┌──────────────────┐    │
│                               │  SigLIP Embeddings│         │  Metadata JSON   │    │
│                               │  (1152-dim vectors)│         │  prongs_metadata │    │
│                               │  embedding_indexer │         │  _v2.json        │    │
│                               └────────┬─────────┘         └────────┬─────────┘    │
│                                         │                            │              │
│                                         ▼                            ▼              │
│                               ┌─────────────────────────────────────────────────┐   │
│                               │              ChromaDB Vector Database           │   │
│                               │  ─────────────────────────────────────────────  │   │
│                               │  • Image embeddings (SigLIP 1152-dim)           │   │
│                               │  • Text metadata (prong count, shape, etc.)     │   │
│                               │  • Persistent storage in vector_stores/         │   │
│                               └─────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Stage 2: Online Processing (Per Query)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              STEP 1: DESIGN ANALYSIS                                │
│                              (vision_analyzer.py)                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────────────┐                          ┌──────────────────────────────┐   │
│   │                  │                          │  Extracted Requirements      │   │
│   │  📸 Reference    │      Gemini 2.5 Pro      │  ──────────────────────────  │   │
│   │     Image        │  ─────────────────────►  │  STONE:                      │   │
│   │                  │       (Vision LLM)       │  • shape: "oval"             │   │
│   │  [User's ring    │                          │  • size_mm: 10.0             │   │
│   │   design photo]  │                          │  • color: "pink"             │   │
│   │                  │                          │                              │   │
│   │                  │                          │  PRONG:                      │   │
│   │                  │                          │  • style: "4-prong"          │   │
│   │                  │                          │  • prong_count: 4            │   │
│   │                  │                          │                              │   │
│   │                  │                          │  SHANK:                      │   │
│   │                  │                          │  • style: "cathedral"        │   │
│   │                  │                          │  • width_mm: 2.5             │   │
│   └──────────────────┘                          └──────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              STEP 2: COMPONENT RETRIEVAL                            │
│                              (v2_retriever.py + embedding_indexer.py)               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────────────┐                                                              │
│   │  Requirements    │                                                              │
│   │  from Step 1     │                                                              │
│   └────────┬─────────┘                                                              │
│            │                                                                        │
│            ▼                                                                        │
│   ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐   │
│   │  HARD FILTERING  │         │  SEMANTIC SEARCH │         │  LLM VERIFICATION│   │
│   │  ────────────────│   ───►  │  ────────────────│   ───►  │  ────────────────│   │
│   │  • Prong count   │         │  • SigLIP encode │         │  Gemini Flash    │   │
│   │    MUST match    │         │    query text    │         │  compares:       │   │
│   │  • Size range    │         │  • ChromaDB      │         │  • Original img  │   │
│   │    70-130%       │         │    cosine search │         │  • Candidate img │   │
│   │  • Shape compat  │         │  • Top-K results │         │  → Yes/No match  │   │
│   └──────────────────┘         └──────────────────┘         └──────────────────┘   │
│                                                                      │              │
│                                                                      ▼              │
│                                                        ┌──────────────────────┐    │
│                                                        │  MATCHED COMPONENTS  │    │
│                                                        │  ────────────────────│    │
│                                                        │  📄 Stone: 118_xxx.3dm│   │
│                                                        │  📄 Prong: 114_xxx.3dm│   │
│                                                        │  🎯 Confidence: 95%   │   │
│                                                        └──────────────────────┘    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              STEP 3: PHYSICS + AI ASSEMBLY                          │
│                              (smart_assembly_physics.py)                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────────────┐ │
│   │                        GEOMETRY ENGINE (Trimesh)                             │ │
│   │  ────────────────────────────────────────────────────────────────────────── │ │
│   │                                                                              │ │
│   │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │ │
│   │  │ 1. OBB Analysis │    │ 2. DROP TEST    │    │ 3. COLLISION    │          │ │
│   │  │ ───────────────│    │ ───────────────│    │ ───────────────│          │ │
│   │  │ Oriented       │    │ Ray casting    │    │ Mesh-to-mesh   │          │ │
│   │  │ Bounding Box   │────►│ simulates      │────►│ intersection   │          │ │
│   │  │ via PCA        │    │ gravity drop   │    │ detection      │          │ │
│   │  │                │    │                │    │                │          │ │
│   │  │ Result:        │    │ Result:        │    │ Result:        │          │ │
│   │  │ TRUE dimensions│    │ EXACT Z-height │    │ YES/NO fit     │          │ │
│   │  │ (rotation-     │    │ for seating    │    │                │          │ │
│   │  │  invariant)    │    │                │    │                │          │ │
│   │  └─────────────────┘    └─────────────────┘    └─────────────────┘          │ │
│   │                                                                              │ │
│   └──────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                          │
│                                         ▼                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────┐ │
│   │                        AESTHETIC JUDGE (Gemini AI)                           │ │
│   │  ────────────────────────────────────────────────────────────────────────── │ │
│   │  AI ONLY handles style judgment:                                             │ │
│   │  • "Does this look proportional?"                                            │ │
│   │  • "Is the setting style appropriate?"                                       │ │
│   │  • NO math calculations (that's Trimesh's job)                               │ │
│   └──────────────────────────────────────────────────────────────────────────────┘ │
│                                         │                                          │
│                                         ▼                                          │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐                              │
│   │   Stone    │    │   Prong    │    │ ASSEMBLED  │                              │
│   │   (scaled) │  + │ (centered) │  = │   HEAD     │                              │
│   │            │    │            │    │            │                              │
│   └────────────┘    └────────────┘    └────────────┘                              │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              STEP 4: SHANK GENERATION                               │
│                              (dynamic_shank_generator.py)                           │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐   │
│   │  Design Analysis │         │  Parametric      │         │  Shank Styles    │   │
│   │  ────────────────│   ───►  │  Generator       │   ───►  │  ────────────────│   │
│   │  • ring_size: 7  │         │  ────────────────│         │  • Plain         │   │
│   │  • style: split  │         │  Creates mesh    │         │  • Cathedral     │   │
│   │  • width: 2.5mm  │         │  using torus +   │         │  • Split         │   │
│   │                  │         │  boolean ops     │         │  • Tapered       │   │
│   └──────────────────┘         └──────────────────┘         └──────────────────┘   │
│                                                                                     │
│   Ring Size Formula: diameter_mm = (US_size × 0.825) + 12.5                         │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              STEP 5: FINAL ASSEMBLY                                 │
│                              (smart_pipeline.py)                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────────────┐     │
│   │   Stone    │    │   Prong    │    │   Shank    │    │   COMPLETE RING    │     │
│   │   Layer    │  + │   Layer    │  + │   Layer    │  = │   complete_ring_   │     │
│   │   (.3dm)   │    │   (.3dm)   │    │   (.3dm)   │    │   YYYYMMDD.3dm     │     │
│   └────────────┘    └────────────┘    └────────────┘    └────────────────────┘     │
│                                                                                     │
│   Z-Alignment:                                                                      │
│   ─────────────                                                                     │
│   Shank:  Z = -10.7 to +10.7  (centered at finger)                                 │
│   Prong:  Z = +10.7 to +21.5  (sits on top of shank)                               │
│   Stone:  Z = +11.1 to +18.7  (inside prong, drop-tested)                          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                               ┌──────────────────────┐
                               │  📁 OUTPUT FILES     │
                               │  ────────────────────│
                               │  outputs/assemblies/ │
                               │   └── complete_ring_ │
                               │       20260204.3dm   │
                               │                      │
                               │  outputs/results/    │
                               │   └── results.json   │
                               │                      │
                               │  outputs/visualizations/│
                               │   └── result.png     │
                               └──────────────────────┘
```

---

## 📁 File Structure & Module Descriptions

### Root Level Files

| File | Purpose | Usage |
|------|---------|-------|
| `run.py` | **Main Entry Point** - Handles CLI routing and smart mode execution | `python run.py [image.jpg]` |
| `requirements.txt` | Python dependencies list | `pip install -r requirements.txt` |
| `.env` | Environment variables (API keys) | Create with `GEMINI_API_KEY=your_key` |

### Source Code (`src/`)

#### 🧠 Core Pipeline Orchestration

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `smart_pipeline.py` | **Master Orchestrator** - Coordinates the entire automated pipeline from image to CAD | `SmartRAGPipeline.process()` |
| `config.py` | **Configuration Hub** - All paths, model settings, API keys, thresholds | Constants: `GEMINI_API_KEY`, `EMBEDDING_MODEL`, etc. |
| `models.py` | **Data Models** - Type definitions for components, results, requirements | `ComponentType`, `CADComponent`, `RetrievalResult` |
| `cli.py` | **Command Line Interface** - Legacy interactive mode with prompts | `main()` |

#### 🔍 Retrieval System

| File | Purpose | Key Features |
|------|---------|--------------|
| `embedding_indexer.py` | **Embedding Engine** - SigLIP model for image/text embeddings + ChromaDB storage | `EmbeddingIndexer.index_components()`, GPU acceleration |
| `v2_retriever.py` | **Smart Retriever** - V2 retrieval with HARD filtering (prong count must match) | `V2Retriever.filter_prongs_by_requirements()` |
| `rag_retriever.py` | **Legacy Retriever** - Original retrieval without hard filtering | `RAGRetriever` |
| `vision_analyzer.py` | **Vision AI** - Gemini for image analysis and component verification | `VisionAnalyzer.analyze_design_comprehensive()` |

#### ⚙️ Assembly System (v3.0 - Physics + AI Hybrid)

| File | Purpose | Key Algorithms |
|------|---------|----------------|
| `smart_assembly_physics.py` | **Physics Assembly Engine (v3.0)** - Trimesh-based computational geometry with AI aesthetics | `GeometryEngine.compute_obb()`, `drop_test()`, `check_collision()`, `AestheticJudge` |
| `smart_assembly_ai.py` | **AI Assembly Engine (v2.0)** - Pure AI-based assembly with iterative correction | `SmartAssemblyAI`, `AIAssistedAssembler` |
| `smart_assembler.py` | **Smart Assembler** - Shape-aware assembly logic | Shape detection, scaling |
| `precision_assembler.py` | **Precision Assembler** - Vertex-level geometry analysis for fitting | 97% fit ratio targeting |
| `assembly_validator.py` | **Validation System** - Validates fit ratio, alignment, depth | Correction factor generation |
| `assembly_pipeline.py` | **Pipeline Orchestration** - Iterative assembly with validation loop | Max 5 iterations |
| `dynamic_shank_generator.py` | **Shank Generator** - Parametric ring band generation | Plain, Cathedral, Split, Tapered styles |

#### 📊 Metadata & Utilities

| File | Purpose | Output |
|------|---------|--------|
| `metadata_generator_v2.py` | **V2 Metadata** - Comprehensive accurate metadata with prong counts | `prongs_metadata_v2.json` |
| `image_generator.py` | **Visualization** - Result image generation | Comparison images |

### Data Directories

| Directory | Contents | Usage |
|-----------|----------|-------|
| `cad_library/prongs/` | 588 prong CAD files (.3dm) | Source components |
| `cad_library/stones/` | 15 stone CAD files (.3dm) | Source components |
| `prongs_sc/` | Prong screenshots (wireframe) | Original indexing |
| `stones_sc/` | Stone screenshots (wireframe) | Original indexing |
| `prongs_multiview/` | Prong multi-view renders (shaded) | Improved retrieval |
| `stones_multiview/` | Stone multi-view renders (shaded) | Improved retrieval |
| `vector_stores/` | ChromaDB + metadata JSONs | Persistent embeddings |
| `outputs/assemblies/` | Generated complete ring .3dm files | Final output |
| `outputs/results/` | Search results JSON files | Retrieval logs |
| `outputs/visualizations/` | Result comparison images | Visual verification |

---

## 🔧 Core Modules Deep Dive

### 1. Vision Analyzer (`vision_analyzer.py`)

**Purpose**: Extract complete jewelry specifications from a reference image using Gemini Vision.

```python
# Key method signature
def analyze_design_comprehensive(self, image_path: Path) -> Dict:
    """
    Returns:
    {
        "stone": {"shape": "oval", "size_mm": 10.0, "color": "pink"},
        "prong": {"style": "4-prong", "prong_count": 4, "shape": "oval"},
        "shank": {"style": "cathedral", "width_mm": 2.5},
        "ring_size_estimate": 7.0
    }
    """
```

**Models Used**:
- Analysis: `gemini-2.5-pro` (higher quality)
- Verification: `gemini-2.5-flash` (faster)

### 2. Embedding Indexer (`embedding_indexer.py`)

**Purpose**: Create and manage vector embeddings for CAD component images.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Component     │     │     SigLIP      │     │    ChromaDB     │
│   Screenshot    │────►│   Encoder       │────►│   Collection    │
│   (.jpg)        │     │   (1152-dim)    │     │   (persistent)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Key Features**:
- GPU acceleration (CUDA) when available
- Batch processing (default: 32)
- Cosine similarity for retrieval
- Dual collections: prongs + stones

### 3. V2 Retriever (`v2_retriever.py`)

**Purpose**: Enhanced retrieval with HARD filtering (guarantees correct prong count).

**Filtering Pipeline**:
```
Query: "4-prong basket setting for 10mm stone"
         │
         ▼
┌─────────────────┐
│  1. HARD FILTER │  Prong count MUST = 4 (not 3, not 6)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. SIZE FILTER │  Opening must fit 10mm ± 30%
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. VECTOR SEARCH│  SigLIP cosine similarity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. LLM VERIFY  │  Gemini confirms visual match
└─────────────────┘
```

### 4. Physics Assembly Engine (`smart_assembly_physics.py`)

**Purpose**: v3.0 assembly using computational geometry (not AI guessing).

#### Problem → Solution Mapping:

| Problem | Old Approach (v1-v2) | v3.0 Solution |
|---------|---------------------|---------------|
| **Rotation** | AABB (45° rotated square = rectangle) | OBB via PCA (true dimensions) |
| **Seating Height** | Guess: `prong_height × 0.15` | Ray Casting Drop Test |
| **Collision** | Box-in-box overlap check | Trimesh mesh intersection |

#### Architecture:
```
┌─────────────────────────────────────────────────────────────────┐
│                    PhysicsAIAssembler                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────┐      ┌─────────────────────┐          │
│  │   GeometryEngine    │      │   AestheticJudge    │          │
│  │   (Trimesh)         │      │   (Gemini AI)       │          │
│  ├─────────────────────┤      ├─────────────────────┤          │
│  │ • rhino_to_trimesh()│      │ • "Does this look   │          │
│  │ • compute_obb()     │      │    proportional?"   │          │
│  │ • drop_test()       │      │ • Style judgment    │          │
│  │ • check_collision() │      │ • NO math here      │          │
│  │ • compute_fit_metrics│      │                     │          │
│  └─────────────────────┘      └─────────────────────┘          │
│           │                            │                        │
│           │ Physics handles all math   │ AI handles aesthetics  │
│           └────────────┬───────────────┘                        │
│                        ▼                                        │
│              ┌─────────────────┐                                │
│              │ Assembled Head  │                                │
│              │ (stone + prong) │                                │
│              └─────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Dynamic Shank Generator (`dynamic_shank_generator.py`)

**Purpose**: Generate parametric ring bands matching the design style.

**Supported Styles**:
| Style | Description |
|-------|-------------|
| `plain` | Simple circular band |
| `cathedral` | Arched sides rising to meet the setting |
| `split` | Band splits into two before meeting setting |
| `tapered` | Band narrows toward the setting |

**Parameters**:
```python
@dataclass
class ShankParameters:
    ring_size: float = 7.0      # US ring size
    style: str = "plain"         # plain/cathedral/split/tapered
    band_width: float = 2.5      # mm
    band_thickness: float = 1.8  # mm
```

---

## 🏗️ Assembly System Architecture

### v3.0 Physics + AI Hybrid

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ASSEMBLY SYSTEM v3.0                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   PHASE 1: GEOMETRY ANALYSIS (100% Computational - No AI)                          │
│   ─────────────────────────────────────────────────────────                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐      │
│   │                                                                         │      │
│   │   Stone.3dm ──► Trimesh ──► OBB Analysis ──► TRUE dimensions            │      │
│   │                              (PCA-based)      10.12 x 8.51mm            │      │
│   │                                                                         │      │
│   │   Prong.3dm ──► Trimesh ──► OBB Analysis ──► Opening size               │      │
│   │                              (PCA-based)      12.5 x 10.2mm             │      │
│   │                                                                         │      │
│   └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                     │
│   PHASE 2: SCALING (Uniform Scale Based on OBB)                                    │
│   ───────────────────────────────────────────────                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐      │
│   │                                                                         │      │
│   │   Scale Factor = (prong_opening × 0.95) / stone_dimension               │      │
│   │   Applied uniformly to preserve stone proportions                       │      │
│   │                                                                         │      │
│   └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                     │
│   PHASE 3: POSITIONING (Ray Casting Drop Test)                                     │
│   ──────────────────────────────────────────────                                    │
│   ┌─────────────────────────────────────────────────────────────────────────┐      │
│   │                                                                         │      │
│   │   100 rays cast downward from stone girdle ──► Find first contact      │      │
│   │   Minimum drop distance = exact seating Z ──► 0.28mm (example)         │      │
│   │                                                                         │      │
│   └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                     │
│   PHASE 4: VALIDATION (Mesh Collision Detection)                                   │
│   ───────────────────────────────────────────────                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐      │
│   │                                                                         │      │
│   │   Trimesh CollisionManager ──► is_collision: False ──► VALID FIT       │      │
│   │   (actual mesh intersection, not box overlap)                          │      │
│   │                                                                         │      │
│   └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                     │
│   PHASE 5: AESTHETIC CHECK (AI - Style Only)                                       │
│   ────────────────────────────────────────────                                      │
│   ┌─────────────────────────────────────────────────────────────────────────┐      │
│   │                                                                         │      │
│   │   Gemini Flash ──► "Are proportions pleasing?" ──► Score: 85/100       │      │
│   │   (NO math, just visual judgment)                                       │      │
│   │                                                                         │      │
│   └─────────────────────────────────────────────────────────────────────────┘      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Assembly Metrics & Thresholds

| Metric | Target | Description |
|--------|--------|-------------|
| **Fit Ratio** | 95% | Stone = 95% of prong opening (5% clearance) |
| **Drop Distance** | Exact via ray cast | No guessing, physics simulation |
| **Collision** | None | Must pass mesh intersection test |
| **Aesthetic Score** | ≥70/100 | AI subjective quality check |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and enter directory
cd "c:\Users\vivek\Desktop\code space\RAG"

# Activate virtual environment
.\rag\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install physics engine (for v3.0 assembly)
pip install trimesh scipy rtree
```

### 2. Configuration

Create `.env` file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run

```bash
# Smart mode (fully automated)
python run.py jewelry_image.jpg

# Interactive prompt
python run.py

# Legacy mode with prompts
python run.py --legacy
```

---

## 📋 Commands Reference

| Command | Description |
|---------|-------------|
| `python run.py` | Interactive smart mode |
| `python run.py image.jpg` | Process specific image |
| `python run.py --legacy` | Legacy interactive mode |
| `python run.py index --component all` | Index all CAD components |
| `python run.py metadata --component all` | Generate metadata |
| `python run.py stats` | Show library statistics |

---

## 🔧 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Embeddings** | Google SigLIP | siglip-so400m-patch14-384 |
| **Vector DB** | ChromaDB | With HNSW |
| **Analysis LLM** | Gemini 2.5 Pro | Vision capable |
| **Verification LLM** | Gemini 2.5 Flash | Fast inference |
| **CAD Manipulation** | rhino3dm | Python bindings |
| **Physics Engine** | trimesh | 4.11.1 |
| **Spatial Indexing** | rtree | 1.4.1 (for ray casting) |
| **Scientific Computing** | scipy, numpy | ConvexHull, PCA |

---

## ⚙️ Configuration

### Key Settings (`src/config.py`)

```python
# Model Selection
EMBEDDING_MODEL = "google/siglip-so400m-patch14-384"
GEMINI_MODEL_ANALYSIS = "gemini-2.5-pro"
GEMINI_MODEL_VERIFY = "gemini-2.5-flash"

# Retrieval Settings
TOP_K_RESULTS = 5
MAX_ITERATIONS = 5
ACTIVE_VECTOR_STORE = "multiview"  # or "original"

# Batch Processing
BATCH_SIZE = 32
NUM_WORKERS = 4
```

### Assembly Parameters

```python
# In smart_assembly_physics.py
target_fit = 0.95      # 95% fit ratio (5% clearance)
clearance = 0.02       # 0.02mm clearance for drop test
max_iterations = 5     # Max correction iterations
```

---

## 📊 Current Library Statistics

| Component Type | Count | Source |
|----------------|-------|--------|
| Prongs | 588 | `cad_library/prongs/` |
| Stones | 15 | `cad_library/stones/` |
| Total CAD Files | 603 | - |

---

## 📄 License

Internal project - Jewelry CAD RAG Agent

---

*Last Updated: February 4, 2026 - Physics v3.0 Assembly Engine*
