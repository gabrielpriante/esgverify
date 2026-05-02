"""Unit tests for backend.core.pipeline.claim_analyzer.

Tests cover:
- Successful analysis from a well-formed Ollama response
- Graceful handling of malformed JSON (no exception raised)
- Unknown SubstantiationLevel / RiskLevel coercion to safe defaults
- Confidence clamping (out-of-range and invalid values)
- Retry / skip behavior on transient Ollama failures
- Empty claims list handling
- SupportingEvidence parsing (valid items, missing text, invalid evidence_type,
  out-of-range relevance_score)
- Multiple claims aggregated into flat output list
- High-risk claim filtering by caller
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from backend.core.models.report import (
    ESGCategory,
    ExtractedClaim,
    RiskLevel,
    SubstantiationLevel,
)
from backend.core.pipeline.claim_analyzer import (
    _dict_to_claim_analysis,
    _parse_analysis_json,
    analyze_claims,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_claim(
    text: str = "We achieved carbon neutrality in 2023.",
    context: str = "As part of our net-zero strategy, we offset all Scope 1 and 2 emissions.",
    esg_category: ESGCategory = ESGCategory.ENVIRONMENTAL,
    framework_tags: list[str] | None = None,
) -> ExtractedClaim:
    """Helper to create a minimal ExtractedClaim for testing."""
    return ExtractedClaim(
        claim_id=str(uuid.uuid4()),
        text=text,
        context=context,
        esg_category=esg_category,
        framework_tags=framework_tags or ["GHG Protocol"],
        page_reference="p.5",
    )


def valid_ollama_response(
    substantiation_level: str = "moderate",
    risk_level: str = "medium",
    gap_explanation: str = "The claim references offsets but lacks third-party verification.",
    confidence: float = 0.75,
    supporting_evidence: list[dict] | None = None,
) -> str:
    """Build a valid JSON string as the LLM would return for claim analysis."""
    return json.dumps(
        {
            "substantiation_level": substantiation_level,
            "risk_level": risk_level,
            "gap_explanation": gap_explanation,
            "confidence": confidence,
            "supporting_evidence": supporting_evidence
            or [
                {
                    "text": "we offset all Scope 1 and 2 emissions",
                    "evidence_type": "data",
                    "relevance_score": 0.85,
                }
            ],
        }
    )


MOCK_VALID_RESPONSE = valid_ollama_response()


# ---------------------------------------------------------------------------
# _parse_analysis_json
# ---------------------------------------------------------------------------


class TestParseAnalysisJson:
    def test_valid_json_returns_dict(self) -> None:
        result = _parse_analysis_json(MOCK_VALID_RESPONSE, "claim_abc")
        assert isinstance(result, dict)
        assert result["substantiation_level"] == "moderate"

    def test_malformed_json_returns_none(self) -> None:
        result = _parse_analysis_json("this is not json }{", "claim_abc")
        assert result is None

    def test_markdown_fences_are_stripped(self) -> None:
        fenced = "```json\n" + MOCK_VALID_RESPONSE + "\n```"
        result = _parse_analysis_json(fenced, "claim_abc")
        assert result is not None
        assert "substantiation_level" in result

    def test_non_dict_json_returns_none(self) -> None:
        result = _parse_analysis_json(json.dumps([1, 2, 3]), "claim_abc")
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        result = _parse_analysis_json("", "claim_abc")
        assert result is None


# ---------------------------------------------------------------------------
# _dict_to_claim_analysis
# ---------------------------------------------------------------------------


class TestDictToClaimAnalysis:
    def test_valid_dict_returns_analysis(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.claim is claim
        assert analysis.substantiation_level == SubstantiationLevel.MODERATE
        assert analysis.risk_level == RiskLevel.MEDIUM
        assert analysis.confidence == 0.75
        assert len(analysis.evidence) == 1

    def test_unknown_substantiation_level_coerces_to_none(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["substantiation_level"] = "super_strong_made_up"
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.substantiation_level == SubstantiationLevel.NONE

    def test_unknown_risk_level_coerces_to_medium(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["risk_level"] = "catastrophic"
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.risk_level == RiskLevel.MEDIUM

    def test_missing_gap_explanation_uses_fallback(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw.pop("gap_explanation", None)
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.gap_explanation == "No explanation provided."

    def test_confidence_clamped_above_one(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["confidence"] = 1.5
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.confidence == 1.0

    def test_confidence_clamped_below_zero(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["confidence"] = -0.3
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.confidence == 0.0

    def test_invalid_confidence_defaults_to_half(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["confidence"] = "not_a_number"
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.confidence == 0.5

    def test_empty_supporting_evidence_list(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["supporting_evidence"] = []
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.evidence == []

    def test_evidence_missing_text_is_skipped(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["supporting_evidence"] = [
            {"text": "", "evidence_type": "data", "relevance_score": 0.9},
            {"text": "valid evidence text", "evidence_type": "certification", "relevance_score": 0.7},
        ]
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert len(analysis.evidence) == 1
        assert analysis.evidence[0].evidence_type == "certification"

    def test_invalid_evidence_type_coerces_to_none(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["supporting_evidence"] = [
            {"text": "some evidence", "evidence_type": "made_up_type", "relevance_score": 0.6}
        ]
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.evidence[0].evidence_type == "none"

    def test_relevance_score_clamped(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["supporting_evidence"] = [
            {"text": "some evidence", "evidence_type": "data", "relevance_score": 2.5}
        ]
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.evidence[0].relevance_score == 1.0

    def test_evidence_ids_are_unique(self) -> None:
        claim = make_claim()
        raw = json.loads(MOCK_VALID_RESPONSE)
        raw["supporting_evidence"] = [
            {"text": "evidence A", "evidence_type": "data", "relevance_score": 0.8},
            {"text": "evidence B", "evidence_type": "methodology", "relevance_score": 0.6},
        ]
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        ids = [e.evidence_id for e in analysis.evidence]
        assert len(ids) == len(set(ids))

    def test_esg_category_preserved_from_claim(self) -> None:
        claim = make_claim(esg_category=ESGCategory.GOVERNANCE)
        raw = json.loads(MOCK_VALID_RESPONSE)
        analysis = _dict_to_claim_analysis(raw, claim)
        assert analysis is not None
        assert analysis.claim.esg_category == ESGCategory.GOVERNANCE

    def test_all_substantiation_levels_accepted(self) -> None:
        claim = make_claim()
        for level in ("strong", "moderate", "weak", "none"):
            raw = json.loads(valid_ollama_response(substantiation_level=level))
            analysis = _dict_to_claim_analysis(raw, claim)
            assert analysis is not None
            assert analysis.substantiation_level == SubstantiationLevel(level)

    def test_all_risk_levels_accepted(self) -> None:
        claim = make_claim()
        for level in ("low", "medium", "high"):
            raw = json.loads(valid_ollama_response(risk_level=level))
            analysis = _dict_to_claim_analysis(raw, claim)
            assert analysis is not None
            assert analysis.risk_level == RiskLevel(level)


# ---------------------------------------------------------------------------
# analyze_claims (integration-level with mocked _call_ollama)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_claims_success() -> None:
    """analyze_claims returns one analysis per claim on valid Ollama responses."""
    claims = [make_claim(), make_claim(text="Our supply chain is fully traceable.")]

    with patch(
        "backend.core.pipeline.claim_analyzer._call_ollama",
        new=AsyncMock(return_value=MOCK_VALID_RESPONSE),
    ):
        analyses = await analyze_claims(claims)

    assert len(analyses) == 2
    assert all(a.substantiation_level == SubstantiationLevel.MODERATE for a in analyses)


@pytest.mark.asyncio
async def test_analyze_claims_empty_list_returns_empty() -> None:
    """analyze_claims on an empty list returns [] without calling Ollama."""
    with patch(
        "backend.core.pipeline.claim_analyzer._call_ollama",
        new=AsyncMock(),
    ) as mock_ollama:
        analyses = await analyze_claims([])

    assert analyses == []
    mock_ollama.assert_not_called()


@pytest.mark.asyncio
async def test_analyze_claims_malformed_json_skips_claim() -> None:
    """analyze_claims skips claims with malformed JSON — never raises."""
    claims = [make_claim(), make_claim(text="Second claim.")]
    responses = ["not valid json {{", MOCK_VALID_RESPONSE]

    with patch(
        "backend.core.pipeline.claim_analyzer._call_ollama",
        new=AsyncMock(side_effect=responses),
    ):
        analyses = await analyze_claims(claims)

    assert len(analyses) == 1


@pytest.mark.asyncio
async def test_analyze_claims_http_error_skips_claim() -> None:
    """analyze_claims skips a claim when all Ollama retries are exhausted."""
    claims = [make_claim()]

    with patch(
        "backend.core.pipeline.claim_analyzer._call_ollama",
        new=AsyncMock(side_effect=httpx.HTTPError("connection refused")),
    ):
        analyses = await analyze_claims(claims)

    assert analyses == []


@pytest.mark.asyncio
async def test_analyze_claims_retry_then_succeed() -> None:
    """analyze_claims continues to the next claim after a failure."""
    claims = [make_claim(), make_claim(text="Second claim.")]
    responses = [httpx.HTTPError("connection refused"), MOCK_VALID_RESPONSE]
    call_count = 0

    async def side_effect(prompt: str, claim_id: str) -> str:
        nonlocal call_count
        result = responses[call_count]
        call_count += 1
        if isinstance(result, Exception):
            raise result
        return result

    with patch(
        "backend.core.pipeline.claim_analyzer._call_ollama",
        new=side_effect,
    ):
        analyses = await analyze_claims(claims)

    assert call_count == 2
    assert len(analyses) == 1


@pytest.mark.asyncio
async def test_analyze_claims_multiple_claims_aggregated() -> None:
    """Analyses from multiple claims are combined into a single flat list."""
    claims = [make_claim(text=f"Claim {i}") for i in range(4)]

    with patch(
        "backend.core.pipeline.claim_analyzer._call_ollama",
        new=AsyncMock(return_value=MOCK_VALID_RESPONSE),
    ):
        analyses = await analyze_claims(claims)

    assert len(analyses) == 4


@pytest.mark.asyncio
async def test_analyze_claims_high_risk_filtering() -> None:
    """Caller can filter returned analyses by risk level."""
    high_risk_response = valid_ollama_response(
        substantiation_level="none",
        risk_level="high",
        gap_explanation="No data provided to support this claim.",
        confidence=0.9,
        supporting_evidence=[],
    )
    claims = [make_claim(), make_claim(text="We are 100% sustainable.")]
    responses = [MOCK_VALID_RESPONSE, high_risk_response]

    with patch(
        "backend.core.pipeline.claim_analyzer._call_ollama",
        new=AsyncMock(side_effect=responses),
    ):
        analyses = await analyze_claims(claims)

    high_risk = [a for a in analyses if a.risk_level == RiskLevel.HIGH]
    assert len(high_risk) == 1
    assert high_risk[0].claim.text == "We are 100% sustainable."
