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

Commitment extraction (Paper #4 benchmark) — see the COMMITMENT EXTRACTION
section at the bottom of this module:
  StructuredField        One structural attribute + its stated/not-stated status
  ExtractedCommitment    A sentence judged for commitment status + 7 fields

The commitment models are deliberately separate from ExtractedClaim. The claim
path answers "is this ESG assertion substantiated?" (greenwashing framing); the
commitment path answers "is this a pledge to a future action, and what is its
structure?" (benchmark framing). They share SubstantiationLevel and nothing else.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


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
    """Verifiability level: how well a claim is supported by evidence.

    Levels and boundary rule are defined in the annotation guideline
    (paper/annotation_guideline_v1.md) and must stay in sync with it.

    WEAK vs NONE: WEAK if any supporting context for the commitment exists
    elsewhere in the document, however thin. NONE only if the commitment
    appears with zero corroborating evidence anywhere in the document.
    The test is presence of evidence, not specificity of the commitment.
    """
    STRONG   = "strong"    # Specific data, open/checkable datasets, third-party verification
    MODERATE = "moderate"  # Partial supporting evidence, but gaps remain
    WEAK     = "weak"      # Thin supporting evidence exists somewhere in the document
    NONE     = "none"      # Zero corroborating evidence anywhere in the document
    UNSURE   = "unsure"    # Unable to verify either way


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


# ---------------------------------------------------------------------------
# COMMITMENT EXTRACTION (Paper #4 benchmark)
#
# Definitions are locked in paper/task_definition.md and notes/decisions.md
# (entry dated 07/25/2026). Any change to the rulings below must be made in
# those documents first, then mirrored here.
#
#   Commitment:  a stated intention in writing to take a future environmental
#                action or reach a future environmental outcome.
#   NOT a commitment: past action, values statement with no action, factual
#                disclosure with no promise, product/process/org description.
#   Annotation unit: SENTENCE-LEVEL for v1 (span-level is future work).
# ---------------------------------------------------------------------------


class CommitmentDecision(str, Enum):
    """The yes/no commitment judgement for one annotation unit.

    UNSURE is a first-class outcome, not a failure mode. The guideline states
    that an unsure option must be available for every field and that no choice
    is ever forced.
    """
    YES    = "yes"
    NO     = "no"
    UNSURE = "unsure"


class FieldStatus(str, Enum):
    """Why a structural field holds (or does not hold) a value.

    These are kept distinct on purpose — collapsing them loses signal that the
    benchmark needs:

    STATED          the document gives a value; ``StructuredField.value`` is set
    NOT_STATED      the document genuinely does not say
    NOT_APPLICABLE  the field cannot apply to this commitment type
    UNSURE          the document might say, but the annotator cannot tell
    """
    STATED         = "stated"
    NOT_STATED     = "not_stated"
    NOT_APPLICABLE = "not_applicable"
    UNSURE         = "unsure"


class FlagValue(str, Enum):
    """A yes/no attribute that also admits not-stated, N/A, and unsure."""
    YES            = "yes"
    NO             = "no"
    NOT_STATED     = "not_stated"
    NOT_APPLICABLE = "not_applicable"
    UNSURE         = "unsure"


class StructuredField(BaseModel):
    """One structural attribute of a commitment.

    Separating ``status`` from ``value`` means "not stated", "not applicable"
    and "unsure" are recorded as data rather than smuggled in as magic strings
    or nulls that later collapse into each other.
    """
    status: FieldStatus
    value: Optional[str] = Field(
        None,
        description="The extracted value; set only when status is 'stated'",
    )

    @model_validator(mode="after")
    def _check_value_matches_status(self) -> "StructuredField":
        if self.status is FieldStatus.STATED:
            if self.value is None or not str(self.value).strip():
                raise ValueError("status 'stated' requires a non-empty value")
        elif self.value is not None:
            raise ValueError(
                f"status '{self.status.value}' must not carry a value "
                f"(got {self.value!r})"
            )
        return self

    @classmethod
    def stated(cls, value: str) -> "StructuredField":
        return cls(status=FieldStatus.STATED, value=value)

    @classmethod
    def not_stated(cls) -> "StructuredField":
        return cls(status=FieldStatus.NOT_STATED)

    @classmethod
    def not_applicable(cls) -> "StructuredField":
        return cls(status=FieldStatus.NOT_APPLICABLE)

    @classmethod
    def unsure(cls) -> "StructuredField":
        return cls(status=FieldStatus.UNSURE)


class ExtractedCommitment(BaseModel):
    """One annotation unit (a sentence) judged against the commitment definition.

    Units judged NOT a commitment are still recorded. The benchmark needs
    negatives: precision cannot be measured against positives alone.

    The seven structural fields mirror paper/task_definition.md exactly.
    Note that ``depends_on_outside_factors`` IS the conditional flag — the
    task definition lists it as the seventh structural field, so it is stored
    once here rather than duplicated as a separate attribute.
    """

    commitment_id: str = Field(..., description="Unique identifier")
    text: str = Field(..., description="Verbatim sentence from the document")
    context: str = Field(..., description="Surrounding sentences for context")
    page_reference: Optional[str] = Field(
        None, description="Page or chunk where the sentence appears"
    )

    # --- Decision -----------------------------------------------------------
    is_commitment: CommitmentDecision = Field(
        ...,
        description="Does this sentence pledge a future environmental action/outcome?",
    )
    rejection_reason: Optional[str] = Field(
        None,
        description=(
            "When is_commitment is 'no', which not-a-commitment category applied: "
            "past_action | values_statement | factual_disclosure | description"
        ),
    )

    # --- The seven structural fields ---------------------------------------
    target: StructuredField = Field(..., description="What is being promised")
    quantity: StructuredField = Field(..., description="How much (number/percentage)")
    deadline: StructuredField = Field(..., description="By when (year or date)")
    baseline: StructuredField = Field(..., description="Starting point (baseline year)")
    business_unit: StructuredField = Field(..., description="What part of the business")
    emissions_scope: StructuredField = Field(..., description="What part of emissions")
    depends_on_outside_factors: FlagValue = Field(
        ...,
        description=(
            "Conditional flag. Ruling: conditional commitments COUNT — the "
            "caveat is recorded here, never used to exclude the commitment."
        ),
    )

    # --- Locked rulings -----------------------------------------------------
    restated: FlagValue = Field(
        default=FlagValue.NOT_STATED,
        description=(
            "Ruling: restatements COUNT, flagged so repeat disclosure can be "
            "filtered out and does not inflate commitment counts."
        ),
    )
    is_evidence: FlagValue = Field(
        default=FlagValue.NO,
        description=(
            "Ruling: third-party validation statements are EVIDENCE, not "
            "commitments. Set yes and is_commitment=no; evidence retrieval "
            "links them back to the commitment they support."
        ),
    )
    supports_commitment_id: Optional[str] = Field(
        None,
        description="When is_evidence is yes, the commitment this evidence backs",
    )

    # --- Verifiability ------------------------------------------------------
    verifiability: SubstantiationLevel = Field(
        default=SubstantiationLevel.UNSURE,
        description=(
            "Five-level scale shared with the claim path. Evidence scope for "
            "v1 is within-document only; external verification is out of scope "
            "and recorded as a limitation."
        ),
    )
    annotator_notes: Optional[str] = Field(
        None,
        description="Free-text ambiguity notes; never affects the level assigned",
    )

    # --- Provenance ---------------------------------------------------------
    # Recorded per record, not per run: a single output file may mix results
    # from a resumed or re-prompted run, and the paper has to be able to say
    # exactly which model and which prompt wording produced each row.
    model: Optional[str] = Field(
        None, description="Ollama model tag that produced this record"
    )
    detect_prompt_id: Optional[str] = Field(
        None, description="Stage 1 prompt identifier, '<file>@<sha256[:12]>'"
    )
    enrich_prompt_id: Optional[str] = Field(
        None,
        description="Stage 2 prompt identifier; None if enrichment did not run",
    )
