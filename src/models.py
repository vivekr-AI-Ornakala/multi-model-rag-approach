"""
Data models for the Multi-Modal RAG CAD Component Agent
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum


class ComponentType(Enum):
    PRONGS = "prongs"
    STONES = "stones"


class VerificationStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ComponentRequirement:
    """Requirements extracted by Vision LLM for a specific component"""
    component_type: ComponentType
    description: str
    shape: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None
    material: Optional[str] = None
    additional_details: dict = field(default_factory=dict)


@dataclass
class CADComponent:
    """Represents a CAD component from the library"""
    component_id: str
    component_type: ComponentType
    cad_file_path: Path
    screenshot_path: Path
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "cad_file_path": str(self.cad_file_path),
            "screenshot_path": str(self.screenshot_path),
            "metadata": self.metadata
        }


@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    component: CADComponent
    similarity_score: float
    rank: int


@dataclass 
class VerificationResult:
    """Result from Vision LLM verification"""
    component: CADComponent
    status: VerificationStatus
    confidence: float
    reasoning: str
    suggested_modifications: Optional[str] = None


@dataclass
class ComponentSelection:
    """Final selected component after verification loop"""
    requirement: ComponentRequirement
    selected_component: Optional[CADComponent]
    verification_result: Optional[VerificationResult]
    iterations_taken: int
    rejected_components: list = field(default_factory=list)


@dataclass
class DesignAnalysis:
    """Complete analysis of a jewelry design image"""
    original_image_path: str
    component_requirements: list  # List[ComponentRequirement]
    component_selections: list  # List[ComponentSelection]
    success: bool
    summary: str
