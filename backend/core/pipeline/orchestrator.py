"""
Analysis pipeline orchestrator.

This module coordinates the full ESG analysis pipeline. Each stage is a
separate module so that stages can be tested, swapped, and improved
independently.

Current status: STUB — returns empty results.
Phase 2 will implement each stage below.

Pipeline stages
---------------
1. Chunker         Split the document into overlapping text chunks
2. ClaimExtractor  Use Ollama (LLaMA 3.1) to identify ESG claims per chunk
3. EvidenceRetriever  Use ChromaDB to find evidence related to each claim
4. ClaimClassifier    Use ClimateBERT to classify claims by ESG category
5. GapScorer          Use Ollama to reason about substantiation quality
6. ReportBuilder      Aggregate results into an AnalysisReport
"""

import uuid
import structlog

from backend.core.models.report import AnalysisReport, AnalysisSummary, RiskLevel
from backend.core.models.report import DocumentInput

logger = structlog.get_logger(__name__)


async def run_pipeline(document: DocumentInput) -> AnalysisReport:
    """
    Run the full ESG analysis pipeline on a parsed document.

    Args:
        document: Parsed document text wrapped in a DocumentInput model.

    Returns:
        AnalysisReport with all claims analyzed and summarized.
    """
    logger.info("Pipeline started", filename=document.filename)

    # Stage 1 — Chunking
    # chunks = chunk_document(document.content)

    # Stage 2 — Claim extraction
    # claims = await extract_claims(chunks)

    # Stage 3 — Evidence retrieval
    # claims_with_evidence = await retrieve_evidence(claims, document.content)

    # Stage 4 — ClimateBERT classification
    # classified_claims = classify_claims(claims_with_evidence)

    # Stage 5 — Gap scoring
    # analyzed_claims = await score_gaps(classified_claims)

    # Stage 6 — Report assembly
    # report = build_report(document.filename, analyzed_claims)

    # Placeholder until stages are implemented
    logger.warning("Pipeline is a stub — returning empty report")
    return _empty_report(document.filename)


def _empty_report(filename: str) -> AnalysisReport:
    return AnalysisReport(
        report_id=str(uuid.uuid4()),
        filename=filename,
        analysis_version="0.1.0-stub",
        claims=[],
        summary=AnalysisSummary(
            total_claims=0,
            by_esg_category={},
            by_substantiation_level={},
            by_risk_level={},
            overall_risk_level=RiskLevel.LOW,
            key_findings=["Pipeline not yet implemented."],
        ),
    )
