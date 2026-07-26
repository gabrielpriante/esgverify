"""Unit tests for backend.core.pipeline.orchestrator summary generation.

Focused on the UNSURE gap: SubstantiationLevel.UNSURE was added on 07/25/2026
to match annotation_guideline_v1.md, but nothing in the summary counted it, so
an undetermined claim was invisible in every report.
"""

from __future__ import annotations

from collections import Counter

import pytest

from backend.core.models.report import (
    ClaimAnalysis,
    ESGCategory,
    ExtractedClaim,
    RiskLevel,
    SubstantiationLevel,
)
from backend.core.pipeline.orchestrator import _generate_key_findings


def make_analysis(
    level: SubstantiationLevel,
    risk: RiskLevel = RiskLevel.MEDIUM,
) -> ClaimAnalysis:
    return ClaimAnalysis(
        claim=ExtractedClaim(
            claim_id="claim_001",
            text="We will reduce emissions by 50% by 2030.",
            context="surrounding context",
            esg_category=ESGCategory.ENVIRONMENTAL,
        ),
        evidence=[],
        substantiation_level=level,
        risk_level=risk,
        gap_explanation="explanation",
        confidence=0.5,
    )


def findings_for(*levels: SubstantiationLevel) -> list[str]:
    analyses = [make_analysis(level) for level in levels]
    return _generate_key_findings(
        analyses,
        by_category=Counter({"environmental": len(analyses)}),
        by_risk=Counter({"medium": len(analyses)}),
    )


class TestUnsureIsCounted:

    def test_unsure_claims_appear_in_findings(self):
        text = " ".join(findings_for(SubstantiationLevel.UNSURE, SubstantiationLevel.STRONG))
        assert "could not be assessed" in text

    def test_unsure_is_not_folded_into_unsubstantiated(self):
        """'We could not tell' is a statement about our coverage, not about the
        company. Counting it as weak/none would misreport both."""
        text = " ".join(findings_for(SubstantiationLevel.UNSURE))
        assert "could not be assessed" in text
        assert "unsubstantiated" not in text

    def test_weak_and_none_still_counted_together(self):
        text = " ".join(
            findings_for(SubstantiationLevel.WEAK, SubstantiationLevel.NONE)
        )
        assert "2 claims" in text
        assert "unsubstantiated" in text

    def test_unsure_excluded_from_the_weak_none_count(self):
        text = " ".join(
            findings_for(
                SubstantiationLevel.WEAK,
                SubstantiationLevel.NONE,
                SubstantiationLevel.UNSURE,
            )
        )
        # two weak/none, one unsure — not three unsubstantiated
        assert "2 claims" in text and "unsubstantiated" in text
        assert "1 claim could not be assessed" in text

    def test_no_unsure_line_when_none_are_unsure(self):
        text = " ".join(findings_for(SubstantiationLevel.STRONG))
        assert "could not be assessed" not in text

    @pytest.mark.parametrize(
        "levels,expected",
        [
            ((SubstantiationLevel.UNSURE,), "1 claim could not be assessed"),
            ((SubstantiationLevel.UNSURE, SubstantiationLevel.UNSURE),
             "2 claims could not be assessed"),
        ],
    )
    def test_singular_and_plural_agreement(self, levels, expected):
        assert expected in " ".join(findings_for(*levels))


class TestFindingsShape:

    def test_total_is_always_reported_first(self):
        findings = findings_for(SubstantiationLevel.STRONG, SubstantiationLevel.WEAK)
        assert findings[0].startswith("2 ESG claims")

    def test_findings_are_capped_at_five(self):
        findings = findings_for(*([SubstantiationLevel.UNSURE] * 3))
        assert len(findings) <= 5
