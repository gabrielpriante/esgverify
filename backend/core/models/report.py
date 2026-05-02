"""
Core data models for ESGVerify.

These Pydantic models define the shape of data flowing through the
pipeline. Using explicit models (rather than raw dicts) gives us:
  - Automatic validation at every boundary
  - Self-documenting API schemas in Swagger UI
  - Type safety throughout the codebase

Model hierarchy:
  DocumentInput          Raw document submitted by the user
  ExtractedClaim         A single ESG assertion found in the document
  SupportingEvidence     Evidence found that relates to a claim
  ClaimAnalysis          One claim + its evidence + gap score
  AnalysisReport         The full report returned to the user
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ESGCategory(str, Enum):
    """Top-level ESG pillar a claim belongs to."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    UNKNOWN = "unknown"


class SubstantiationLevel(str, Enum):
    """How well a claim is supported by evidence in the document."""
    STRONG   = "strong"    # Quantified data, verified certification, methodology
    MODERATE = "moderate"  # Partial data or vague reference to methodology
    WEAK     = "weak"      # Assertion with no supporting evidence
    NONE     = "none"      # No attempt at substantiation


class RiskLevel(str, Enum):
    """Overall greenwashing / ESG risk flag for a claim."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ---------------------------------------------------------------------------
# Pipeline models
# ---------------------------------------------------------------------------

class DocumentInput(BaseModel):
    """Represents a document submitted for analysis."""
    filename: str
    content: str = Field(..., description="Full extracted text of the document")
    source_type: str = Field(..., description="pdf | docx | txt")


class ExtractedClaim(BaseModel):
    """A single ESG claim identified by the LLM in the document text."""
    claim_id: str = Field(..., description="Unique identifier, e.g. 'claim_001'")
    text: str = Field(..., description="The verbatim claim text from the document")
    context: str = Field(..., description="Surrounding sentences for context")
    esg_category: ESGCategory
    framework_tags: list[str] = Field(
        default_factory=list,
        description="Relevant framework codes, e.g. ['GRI 305', 'TCFD Strategy']"
    )
    page_reference: Optional[str] = Field(
        None,
        description="Page or section where the claim appears, if extractable"
    )


class SupportingEvidence(BaseModel):
    """A piece of evidence found in the document that relates to a claim."""
    evidence_id: str
    text: str = Field(..., description="The evidence text")
    evidence_type: str = Field(
        ...,
        description="data | certification | methodology | third_party_reference | none"
    )
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Semantic similarity score between claim and evidence"
    )


class ClaimAnalysis(BaseModel):
    """Full analysis of a single claim: the claim, its evidence, and our verdict."""
    claim: ExtractedClaim
    evidence: list[SupportingEvidence] = Field(default_factory=list)
    substantiation_level: SubstantiationLevel
    risk_level: RiskLevel
    gap_explanation: str = Field(
        ...,
        description="LLM-generated explanation of the gap between claim and evidence"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Model confidence in this analysis"
    )


class AnalysisSummary(BaseModel):
    """Aggregate statistics across all claims in a document."""
    total_claims: int
    by_esg_category: dict[str, int]
    by_substantiation_level: dict[str, int]
    by_risk_level: dict[str, int]
    overall_risk_level: RiskLevel
    key_findings: list[str] = Field(
        ...,
        description="Top 3–5 human-readable findings from the analysis"
    )


class AnalysisReport(BaseModel):
    """
    The complete report returned to the user after a full pipeline run.
    This is the top-level response model for the /analyze endpoint.
    """
    report_id: str
    filename: str
    analysis_version: str
    claims: list[ClaimAnalysis]
    summary: AnalysisSummary
