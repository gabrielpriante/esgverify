"""
Analysis pipeline orchestrator.

Coordinates the full ESG analysis pipeline. Each stage is a separate module
so that stages can be tested, swapped, and improved independently.

Pipeline stages
---------------
1. Chunker            Split document text into overlapping chunks
2. ClaimExtractor     Identify ESG claims per chunk via Ollama (LLaMA 3.1)
3. ClaimAnalyzer      Score each claim for substantiation and greenwashing risk
4. EvidenceRetriever  Find supporting evidence passages via ChromaDB + embeddings
5. ReportBuilder      Aggregate results into an AnalysisReport  ← stub
"""

from __future__ import annotations

import uuid
from collections import Counter

import structlog

from backend.core.models.report import (
    AnalysisReport,
    AnalysisSummary,
    ClaimAnalysis,
    DocumentInput,
    RiskLevel,
    SubstantiationLevel,
)
from backend.core.pipeline.chunker import chunk_text
from backend.core.pipeline.claim_extractor import extract_claims
from backend.core.pipeline.claim_analyzer import analyze_claims
from backend.core.pipeline.evidence_retriever import build_document_index, retrieve_evidence

logger = structlog.get_logger(__name__)


async def run_pipeline(document: DocumentInput) -> AnalysisReport:
    """Run the full ESG analysis pipeline on a parsed document.

    Executes Stages 1–4 in sequence and assembles the results into an
    :class:`AnalysisReport`. Stage 5 (report generation / PDF export) is
    not yet implemented and is represented by the inline report builder
    at the bottom of this function.

    Args:
        document: Parsed document text wrapped in a :class:`DocumentInput` model.

    Returns:
        :class:`AnalysisReport` with all claims analyzed and summarized.
    """
    log = logger.bind(filename=document.filename)
    log.info("pipeline_started")

    # ------------------------------------------------------------------
    # Stage 1 — Chunking
    # ------------------------------------------------------------------
    chunks = chunk_text(document.content)
    log.info("stage_1_complete", chunks=len(chunks))

    if not chunks:
        log.warning("document_produced_no_chunks")
        return _empty_report(document.filename, reason="Document contained no extractable text.")

    # ------------------------------------------------------------------
    # Stage 2 — Claim extraction
    # ------------------------------------------------------------------
    extracted_claims = await extract_claims(chunks)
    log.info("stage_2_complete", claims_extracted=len(extracted_claims))

    if not extracted_claims:
        log.warning("no_claims_extracted")
        return _empty_report(document.filename, reason="No ESG claims found in document.")

    # ------------------------------------------------------------------
    # Stage 3 — Claim analysis (substantiation + risk scoring)
    # ------------------------------------------------------------------
    analyses = await analyze_claims(extracted_claims)
    log.info("stage_3_complete", claims_analyzed=len(analyses))

    # ------------------------------------------------------------------
    # Stage 4 — Evidence retrieval
    # Build the vector index from the document chunks, then query it for
    # each claim. Attach the retrieved evidence to each ClaimAnalysis.
    # ------------------------------------------------------------------
    build_document_index(chunks, document.filename)
    evidence_map = retrieve_evidence(extracted_claims, document.filename)

    # Attach retrieved evidence to each analysis, replacing the LLM-only
    # evidence list with the richer ChromaDB-retrieved passages.
    enriched_analyses: list[ClaimAnalysis] = []
    for analysis in analyses:
        retrieved = evidence_map.get(analysis.claim.claim_id, [])
        enriched = ClaimAnalysis(
            claim=analysis.claim,
            evidence=retrieved if retrieved else analysis.evidence,
            substantiation_level=analysis.substantiation_level,
            risk_level=analysis.risk_level,
            gap_explanation=analysis.gap_explanation,
            confidence=analysis.confidence,
        )
        enriched_analyses.append(enriched)

    log.info("stage_4_complete", total_evidence_passages=sum(
        len(e.evidence) for e in enriched_analyses
    ))

    # ------------------------------------------------------------------
    # Stage 5 — Report assembly (inline until Stage 5 module is built)
    # ------------------------------------------------------------------
    report = _build_report(document.filename, enriched_analyses)
    log.info("pipeline_complete", report_id=report.report_id, total_claims=len(enriched_analyses))
    return report


# ---------------------------------------------------------------------------
# Report builder (inline Stage 5 placeholder)
# ---------------------------------------------------------------------------


def _build_report(filename: str, analyses: list[ClaimAnalysis]) -> AnalysisReport:
    """Assemble a full :class:`AnalysisReport` from a list of analyzed claims.

    Args:
        filename: Source document filename.
        analyses: List of enriched :class:`ClaimAnalysis` objects.

    Returns:
        Complete :class:`AnalysisReport`.
    """
    by_category: Counter[str] = Counter()
    by_substantiation: Counter[str] = Counter()
    by_risk: Counter[str] = Counter()

    for a in analyses:
        by_category[a.claim.esg_category.value] += 1
        by_substantiation[a.substantiation_level.value] += 1
        by_risk[a.risk_level.value] += 1

    overall_risk = _compute_overall_risk(by_risk)
    key_findings = _generate_key_findings(analyses, by_category, by_risk)

    return AnalysisReport(
        report_id=str(uuid.uuid4()),
        filename=filename,
        analysis_version="0.2.0",
        claims=analyses,
        summary=AnalysisSummary(
            total_claims=len(analyses),
            by_esg_category=dict(by_category),
            by_substantiation_level=dict(by_substantiation),
            by_risk_level=dict(by_risk),
            overall_risk_level=overall_risk,
            key_findings=key_findings,
        ),
    )


def _compute_overall_risk(by_risk: Counter[str]) -> RiskLevel:
    """Derive a single document-level risk from per-claim risk counts.

    Args:
        by_risk: Counter mapping risk level strings to claim counts.

    Returns:
        The highest risk level present, weighted by prevalence.
    """
    if by_risk.get(RiskLevel.HIGH.value, 0) > 0:
        return RiskLevel.HIGH
    if by_risk.get(RiskLevel.MEDIUM.value, 0) > by_risk.get(RiskLevel.LOW.value, 0):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _generate_key_findings(
    analyses: list[ClaimAnalysis],
    by_category: Counter[str],
    by_risk: Counter[str],
) -> list[str]:
    """Generate 3–5 human-readable findings from the analysis results.

    Args:
        analyses: All analyzed claims.
        by_category: Claim counts by ESG category.
        by_risk: Claim counts by risk level.

    Returns:
        List of finding strings for the report summary.
    """
    findings: list[str] = []
    total = len(analyses)

    findings.append(f"{total} ESG claim{'s' if total != 1 else ''} identified across the document.")

    high_risk_count = by_risk.get(RiskLevel.HIGH.value, 0)
    if high_risk_count:
        findings.append(
            f"{high_risk_count} claim{'s' if high_risk_count != 1 else ''} flagged as high greenwashing risk."
        )

    # Most common category
    if by_category:
        top_cat, top_count = by_category.most_common(1)[0]
        findings.append(f"Most claims relate to {top_cat.capitalize()} ({top_count} claim{'s' if top_count != 1 else ''}).")

    # Unsubstantiated claims
    weak_none = sum(
        1 for a in analyses
        if a.substantiation_level in (SubstantiationLevel.WEAK, SubstantiationLevel.NONE)
    )
    if weak_none:
        findings.append(
            f"{weak_none} claim{'s' if weak_none != 1 else ''} {'are' if weak_none != 1 else 'is'} "
            f"weakly substantiated or unsubstantiated."
        )

    # Strong claims
    strong = sum(1 for a in analyses if a.substantiation_level == SubstantiationLevel.STRONG)
    if strong:
        findings.append(f"{strong} claim{'s' if strong != 1 else ''} {'are' if strong != 1 else 'is'} well-substantiated.")

    # Unsure claims — reported separately, never folded into weak/none.
    #
    # UNSURE was added to SubstantiationLevel on 07/25/2026 to match
    # annotation_guideline_v1.md. Until now nothing counted it, so an
    # undetermined claim was invisible in every summary. Counting it as
    # unsubstantiated would be worse than silence: "we could not tell" is not
    # a finding about the company, it is a finding about our own coverage.
    unsure = sum(
        1 for a in analyses if a.substantiation_level == SubstantiationLevel.UNSURE
    )
    if unsure:
        findings.append(
            f"{unsure} claim{'s' if unsure != 1 else ''} could not be assessed "
            f"either way and {'require' if unsure != 1 else 'requires'} review."
        )

    return findings[:5]


def _empty_report(filename: str, reason: str = "No claims found.") -> AnalysisReport:
    """Return a minimal report when the pipeline produces no results.

    Args:
        filename: Source document filename.
        reason: Human-readable explanation included in key_findings.

    Returns:
        Empty :class:`AnalysisReport`.
    """
    return AnalysisReport(
        report_id=str(uuid.uuid4()),
        filename=filename,
        analysis_version="0.2.0",
        claims=[],
        summary=AnalysisSummary(
            total_claims=0,
            by_esg_category={},
            by_substantiation_level={},
            by_risk_level={},
            overall_risk_level=RiskLevel.LOW,
            key_findings=[reason],
        ),
    )
