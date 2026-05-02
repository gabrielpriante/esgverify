"""
Analysis endpoints.

POST /analyze   — Upload a document and receive a full ESG analysis report.

The endpoint is intentionally thin. It:
  1. Validates and reads the uploaded file
  2. Hands off to the analysis pipeline
  3. Returns the structured report

All business logic lives in backend/core/pipeline/.
"""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.core.config import settings
from backend.core.models.report import AnalysisReport
from backend.utils.document_parser import parse_uploaded_file

router = APIRouter()
logger = structlog.get_logger(__name__)

# Maximum upload size enforced before touching the pipeline
MAX_BYTES = settings.max_document_size_mb * 1024 * 1024


@router.post(
    "/analyze",
    response_model=AnalysisReport,
    status_code=status.HTTP_200_OK,
    summary="Analyze an ESG document",
    description=(
        "Upload a PDF, DOCX, or TXT file. "
        "Returns a structured report with extracted claims, "
        "supporting evidence, and substantiation scores."
    ),
)
async def analyze_document(
    file: Annotated[UploadFile, File(description="ESG document to analyze")]
) -> AnalysisReport:
    # --- Validate file type ---
    extension = _get_extension(file.filename)
    if extension not in settings.supported_file_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{extension}'. "
                   f"Accepted: {settings.supported_file_types}",
        )

    # --- Validate file size ---
    content_bytes = await file.read()
    if len(content_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_document_size_mb} MB.",
        )

    logger.info("Analysis request received", filename=file.filename, bytes=len(content_bytes))

    # --- Parse document to plain text ---
    document = await parse_uploaded_file(
        filename=file.filename,
        content_bytes=content_bytes,
        extension=extension,
    )

    # --- Run analysis pipeline ---
    # TODO (Phase 2): Replace stub with real pipeline call
    # from backend.core.pipeline.orchestrator import run_pipeline
    # report = await run_pipeline(document)
    report = _stub_report(file.filename)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_extension(filename: str | None) -> str:
    """Extract the lowercase file extension, without the leading dot."""
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _stub_report(filename: str) -> AnalysisReport:
    """
    Temporary placeholder returned while the pipeline is being built.
    Remove this once backend/core/pipeline/orchestrator.py is complete.
    """
    from backend.core.models.report import (
        AnalysisSummary, ClaimAnalysis, ESGCategory,
        ExtractedClaim, RiskLevel, SubstantiationLevel,
    )

    stub_claim = ExtractedClaim(
        claim_id="claim_001",
        text="Our operations are carbon neutral.",
        context="We are proud to announce that our operations are carbon neutral as of 2023.",
        esg_category=ESGCategory.ENVIRONMENTAL,
        framework_tags=["GRI 305", "TCFD Strategy"],
        page_reference="p. 4",
    )

    stub_analysis = ClaimAnalysis(
        claim=stub_claim,
        evidence=[],
        substantiation_level=SubstantiationLevel.WEAK,
        risk_level=RiskLevel.HIGH,
        gap_explanation=(
            "The claim asserts carbon neutrality but no emissions data, "
            "third-party verification, or offset methodology is referenced."
        ),
        confidence=0.82,
    )

    return AnalysisReport(
        report_id=str(uuid.uuid4()),
        filename=filename,
        analysis_version="stub-0.1",
        claims=[stub_analysis],
        summary=AnalysisSummary(
            total_claims=1,
            by_esg_category={"environmental": 1},
            by_substantiation_level={"weak": 1},
            by_risk_level={"high": 1},
            overall_risk_level=RiskLevel.HIGH,
            key_findings=["Pipeline not yet implemented — this is a stub response."],
        ),
    )
