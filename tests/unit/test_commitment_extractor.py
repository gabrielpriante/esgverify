"""Unit tests for backend.core.pipeline.commitment_extractor.

These tests are OFFLINE. Every Ollama call is mocked. Nothing here proves the
model behaves correctly on a real document — that requires the live pass on the
Microsoft development PDF, run separately.

What these tests do prove:
- The five rulings locked in notes/decisions.md (07/25/2026) survive the round
  trip from LLM JSON to ExtractedCommitment, using worked examples 6-10 from
  paper/task_definition.md as fixtures.
- "not stated", "not applicable" and "unsure" stay distinct rather than
  collapsing into each other or into null.
- Malformed model output degrades to "unsure", never to a confident value.
- The five-level verifiability scale (including UNSURE) round-trips.
- The two-stage flow: negatives skip enrichment, positives trigger it, and a
  failed enrichment degrades to 'unsure' rather than inventing values.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from backend.core.models.report import (
    CommitmentDecision,
    ExtractedCommitment,
    FieldStatus,
    FlagValue,
    StructuredField,
    SubstantiationLevel,
)
from backend.core.pipeline.chunker import TextChunk
from backend.core.pipeline.commitment_extractor import (
    _coerce_field,
    _coerce_flag,
    _dict_to_commitment,
    _parse_commitments_json,
    extract_commitments,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(text: str = "irrelevant", index: int = 0) -> TextChunk:
    return TextChunk(index=index, text=text, char_start=0, char_end=len(text))


def field(status: str, value: str | None = None) -> dict:
    d: dict = {"status": status}
    if value is not None:
        d["value"] = value
    return d


def base_record(**overrides) -> dict:
    """A minimally valid raw record; override per fixture."""
    record = {
        "text": "placeholder",
        "context": "placeholder context",
        "is_commitment": "no",
        "rejection_reason": None,
        "target": field("not_stated"),
        "quantity": field("not_stated"),
        "deadline": field("not_stated"),
        "baseline": field("not_stated"),
        "business_unit": field("not_stated"),
        "emissions_scope": field("not_stated"),
        "depends_on_outside_factors": "no",
        "restated": "no",
        "is_evidence": "no",
        "verifiability": "unsure",
        "annotator_notes": None,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# The five locked rulings — fixtures from task_definition.md examples 6-10
# ---------------------------------------------------------------------------


class TestLockedRulings:
    """One test per ruling in notes/decisions.md, dated 07/25/2026."""

    def test_ruling_conditional_commitments_count(self):
        """Example 6: conditional commitments COUNT, caveat recorded not excluded."""
        raw = base_record(
            text=(
                "Subject to supportive government policy and the availability of "
                "renewable energy infrastructure, we intend to transition our fleet "
                "to electric vehicles by 2035."
            ),
            is_commitment="yes",
            target=field("stated", "transition fleet to electric vehicles"),
            deadline=field("stated", "2035"),
            business_unit=field("stated", "vehicle fleet"),
            emissions_scope=field("not_stated"),
            depends_on_outside_factors="yes",
            verifiability="weak",
        )

        record = _dict_to_commitment(raw, make_chunk())

        assert record is not None
        # The ruling: it counts.
        assert record.is_commitment is CommitmentDecision.YES
        # The caveat is captured as data, not used to reject.
        assert record.depends_on_outside_factors is FlagValue.YES
        assert record.rejection_reason is None
        assert record.deadline.value == "2035"

    def test_ruling_aspirations_are_not_commitments(self):
        """Example 7: vague aspiration is rejected as a values statement."""
        raw = base_record(
            text=(
                "We aspire to be a leader in circular economy practices and aim to "
                "significantly increase our use of recycled inputs over the coming years."
            ),
            is_commitment="no",
            rejection_reason="values_statement",
            verifiability="none",
        )

        record = _dict_to_commitment(raw, make_chunk())

        assert record is not None
        assert record.is_commitment is CommitmentDecision.NO
        assert record.rejection_reason == "values_statement"

    def test_ruling_restatements_count_with_flag(self):
        """Example 8: restatements COUNT and carry the restated flag."""
        raw = base_record(
            text=(
                "As announced in our 2021 report, we remain committed to sourcing "
                "100% of our electricity from renewable sources by 2030."
            ),
            is_commitment="yes",
            target=field("stated", "source electricity from renewable sources"),
            quantity=field("stated", "100%"),
            deadline=field("stated", "2030"),
            emissions_scope=field("not_applicable"),
            restated="yes",
            verifiability="moderate",
        )

        record = _dict_to_commitment(raw, make_chunk())

        assert record is not None
        assert record.is_commitment is CommitmentDecision.YES
        # The flag is what makes deduplication possible in the Section 11 analysis.
        assert record.restated is FlagValue.YES
        assert record.quantity.value == "100%"

    def test_ruling_annotation_unit_is_sentence_level(self):
        """Example 9: a mixed past/future sentence is judged whole, and rejected.

        Sentence-level annotation means the vague forward clause does not rescue
        the sentence. Under span-level annotation this case could split — that is
        recorded as future work, not v1 behaviour.
        """
        raw = base_record(
            text=(
                "We have installed solar capacity at twelve facilities and will "
                "continue this program."
            ),
            is_commitment="no",
            rejection_reason="past_action",
            verifiability="weak",
        )

        record = _dict_to_commitment(raw, make_chunk())

        assert record is not None
        assert record.is_commitment is CommitmentDecision.NO
        assert record.rejection_reason == "past_action"
        # Judged as one unit: the whole sentence is retained verbatim.
        assert record.text.startswith("We have installed")
        assert record.text.endswith("continue this program.")

    def test_ruling_third_party_validation_is_evidence(self):
        """Example 10: SBTi validation is evidence, not a commitment."""
        raw = base_record(
            text=(
                "Through our participation in the Science Based Targets initiative, "
                "our emissions reduction targets have been validated as consistent "
                "with a 1.5 degree pathway."
            ),
            is_commitment="no",
            rejection_reason="factual_disclosure",
            is_evidence="yes",
            verifiability="strong",
        )

        record = _dict_to_commitment(raw, make_chunk())

        assert record is not None
        assert record.is_commitment is CommitmentDecision.NO
        assert record.is_evidence is FlagValue.YES
        # Evidence must be linkable back to the commitment it supports.
        assert "supports_commitment_id" in ExtractedCommitment.model_fields


# ---------------------------------------------------------------------------
# Distinctness of not_stated / not_applicable / unsure
# ---------------------------------------------------------------------------


class TestFieldStatusDistinctness:
    """These three must never collapse into one another or into null."""

    @pytest.mark.parametrize(
        "status,expected",
        [
            ("not_stated", FieldStatus.NOT_STATED),
            ("not_applicable", FieldStatus.NOT_APPLICABLE),
            ("unsure", FieldStatus.UNSURE),
        ],
    )
    def test_each_non_stated_status_round_trips(self, status, expected):
        result = _coerce_field(field(status), "target", 0)
        assert result.status is expected
        assert result.value is None

    def test_three_statuses_are_mutually_distinct(self):
        a = _coerce_field(field("not_stated"), "target", 0)
        b = _coerce_field(field("not_applicable"), "target", 0)
        c = _coerce_field(field("unsure"), "target", 0)
        assert len({a.status, b.status, c.status}) == 3

    def test_stated_carries_its_value(self):
        result = _coerce_field(field("stated", "2030"), "deadline", 0)
        assert result.status is FieldStatus.STATED
        assert result.value == "2030"

    def test_stated_without_value_degrades_to_unsure_not_not_stated(self):
        """A model claiming 'stated' with nothing behind it has not earned
        'not_stated' — that would assert the document is silent."""
        result = _coerce_field({"status": "stated"}, "deadline", 0)
        assert result.status is FieldStatus.UNSURE

    def test_stated_with_blank_value_degrades_to_unsure(self):
        result = _coerce_field(field("stated", "   "), "deadline", 0)
        assert result.status is FieldStatus.UNSURE

    def test_model_rejects_stated_without_value(self):
        with pytest.raises(ValidationError):
            StructuredField(status=FieldStatus.STATED, value=None)

    def test_model_rejects_value_on_non_stated_status(self):
        with pytest.raises(ValidationError):
            StructuredField(status=FieldStatus.NOT_STATED, value="2030")


# ---------------------------------------------------------------------------
# Degradation on malformed input
# ---------------------------------------------------------------------------


class TestMalformedInputDegradesToUnsure:

    @pytest.mark.parametrize("bad", [None, "2030", 42, [], {"status": "banana"}])
    def test_bad_field_becomes_unsure(self, bad):
        assert _coerce_field(bad, "target", 0).status is FieldStatus.UNSURE

    @pytest.mark.parametrize("bad", [None, "maybe", 42, ""])
    def test_bad_flag_becomes_unsure_not_no(self, bad):
        """Defaulting to NO would manufacture negative annotations."""
        assert _coerce_flag(bad, "restated", 0) is FlagValue.UNSURE

    def test_unknown_decision_becomes_unsure(self):
        record = _dict_to_commitment(base_record(is_commitment="probably"), make_chunk())
        assert record is not None
        assert record.is_commitment is CommitmentDecision.UNSURE

    def test_unknown_verifiability_becomes_unsure(self):
        record = _dict_to_commitment(
            base_record(verifiability="highly verifiable"), make_chunk()
        )
        assert record is not None
        assert record.verifiability is SubstantiationLevel.UNSURE

    def test_missing_text_returns_none(self):
        assert _dict_to_commitment(base_record(text="  "), make_chunk()) is None

    def test_rejection_reason_dropped_on_positive_judgement(self):
        record = _dict_to_commitment(
            base_record(is_commitment="yes", rejection_reason="past_action"),
            make_chunk(),
        )
        assert record is not None
        assert record.rejection_reason is None


# ---------------------------------------------------------------------------
# Verifiability scale
# ---------------------------------------------------------------------------


class TestVerifiabilityScale:

    @pytest.mark.parametrize(
        "level", ["strong", "moderate", "weak", "none", "unsure"]
    )
    def test_all_five_levels_round_trip(self, level):
        record = _dict_to_commitment(base_record(verifiability=level), make_chunk())
        assert record is not None
        assert record.verifiability == SubstantiationLevel(level)

    def test_scale_has_exactly_five_levels(self):
        assert len(list(SubstantiationLevel)) == 5

    def test_unsure_level_exists(self):
        """Added 07/25/2026 so the code matches annotation_guideline_v1.md."""
        assert SubstantiationLevel.UNSURE.value == "unsure"


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestParseCommitmentsJson:

    def test_parses_well_formed_payload(self):
        payload = json.dumps({"commitments": [base_record()]})
        assert len(_parse_commitments_json(payload, make_chunk())) == 1

    def test_strips_markdown_fences(self):
        payload = "```json\n" + json.dumps({"commitments": []}) + "\n```"
        assert _parse_commitments_json(payload, make_chunk()) == []

    @pytest.mark.parametrize(
        "payload",
        [
            "not json at all",
            "{",
            json.dumps({"claims": []}),          # wrong key — old claim schema
            json.dumps({"commitments": "nope"}),  # wrong type
            json.dumps([1, 2, 3]),                # not an object
        ],
    )
    def test_malformed_payloads_return_empty_without_raising(self, payload):
        assert _parse_commitments_json(payload, make_chunk()) == []


# ---------------------------------------------------------------------------
# extract_commitments orchestration (Ollama fully mocked)
# ---------------------------------------------------------------------------


class TestExtractCommitments:

    @pytest.mark.asyncio
    async def test_empty_chunk_list_returns_empty(self):
        assert await extract_commitments([]) == []

    @pytest.mark.asyncio
    async def test_negatives_are_retained(self):
        """The benchmark needs negatives; precision cannot be measured without them."""
        payload = json.dumps({
            "commitments": [
                base_record(text="We will cut emissions 50% by 2030.", is_commitment="yes"),
                base_record(text="Sustainability is at our heart.",
                            is_commitment="no", rejection_reason="values_statement"),
            ]
        })
        with patch(
            "backend.core.pipeline.commitment_extractor._call_ollama",
            new=AsyncMock(return_value=payload),
        ):
            records = await extract_commitments([make_chunk()])

        assert len(records) == 2
        assert sum(1 for r in records if r.is_commitment is CommitmentDecision.YES) == 1
        assert sum(1 for r in records if r.is_commitment is CommitmentDecision.NO) == 1

    @pytest.mark.asyncio
    async def test_failed_chunk_is_skipped_not_fatal(self):
        payload = json.dumps({"commitments": [base_record()]})
        responses = [httpx.TimeoutException("boom"), payload]

        async def side_effect(prompt, chunk_index, system_prompt):
            result = responses[chunk_index]
            if isinstance(result, Exception):
                raise result
            return result

        with patch(
            "backend.core.pipeline.commitment_extractor._call_ollama",
            new=AsyncMock(side_effect=side_effect),
        ):
            records = await extract_commitments([make_chunk(index=0), make_chunk(index=1)])

        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_page_reference_falls_back_to_chunk_index(self):
        payload = json.dumps({"commitments": [base_record()]})
        with patch(
            "backend.core.pipeline.commitment_extractor._call_ollama",
            new=AsyncMock(return_value=payload),
        ):
            records = await extract_commitments([make_chunk(index=7)])

        assert records[0].page_reference == "chunk_7"

    @pytest.mark.asyncio
    async def test_commitment_ids_are_unique(self):
        payload = json.dumps({"commitments": [base_record(), base_record()]})
        with patch(
            "backend.core.pipeline.commitment_extractor._call_ollama",
            new=AsyncMock(return_value=payload),
        ):
            records = await extract_commitments([make_chunk()])

        assert len({r.commitment_id for r in records}) == 2


# ---------------------------------------------------------------------------
# Prompt content guards
#
# The prompt is the experimental instrument. These assertions fail loudly if a
# future edit reintroduces the claim-extraction framing.
# ---------------------------------------------------------------------------


class TestPromptsEncodeTheDefinition:
    """The prompts are the experimental instrument.

    These assertions fail loudly if a future edit reintroduces the claim
    framing, drops a ruling, or lets a prompt grow past the size that made the
    model degenerate on real hardware.
    """

    def _detect(self) -> str:
        from backend.core.pipeline.commitment_extractor import _DETECT_SYSTEM_PROMPT
        return _DETECT_SYSTEM_PROMPT

    def _enrich(self) -> str:
        from backend.core.pipeline.commitment_extractor import _ENRICH_SYSTEM_PROMPT
        return _ENRICH_SYSTEM_PROMPT

    # --- size guard -------------------------------------------------------

    def test_both_prompts_stay_within_token_budget(self):
        """A combined ~1,190-token prompt plus a ~400-token chunk produced
        word-salad from llama3.1:8b on a mid-range GPU. Two-stage exists to
        keep each call well under that ceiling; this test keeps it that way."""
        from backend.core.pipeline.commitment_extractor import PROMPT_TOKEN_BUDGET
        for name, prompt in (("detect", self._detect()), ("enrich", self._enrich())):
            approx_tokens = len(prompt) // 4
            assert approx_tokens < PROMPT_TOKEN_BUDGET, (
                f"{name} prompt is ~{approx_tokens} tokens, "
                f"budget is {PROMPT_TOKEN_BUDGET}"
            )

    # --- detection prompt -------------------------------------------------

    def test_past_action_example_is_not_present_as_positive(self):
        """The old claim prompt used 'reduced Scope 1 emissions by 30%' as a
        positive example. The guideline excludes past actions outright."""
        assert "reduced Scope 1 emissions by 30%" not in self._detect()

    def test_future_intention_is_the_core_criterion(self):
        prompt = self._detect().lower()
        assert "future" in prompt
        assert "future intention is the core test" in prompt

    def test_all_four_rejection_categories_present(self):
        prompt = self._detect()
        for category in (
            "past_action",
            "values_statement",
            "factual_disclosure",
            "description",
        ):
            assert category in prompt

    def test_three_locked_rulings_present(self):
        prompt = self._detect().lower()
        assert "conditional commitments count" in prompt
        assert "restated commitments count" in prompt
        assert "evidence, not a commitment" in prompt

    def test_sentence_level_unit_declared(self):
        assert "ONE SENTENCE AT A TIME" in self._detect()

    def test_negatives_are_requested(self):
        assert "including the ones you reject" in self._detect()

    # --- enrichment prompt ------------------------------------------------

    def test_all_seven_structural_fields_present(self):
        prompt = self._enrich()
        for field_name in (
            "target", "quantity", "deadline", "baseline",
            "business_unit", "emissions_scope", "depends_on_outside_factors",
        ):
            assert field_name in prompt

    def test_all_four_field_statuses_present(self):
        prompt = self._enrich()
        for status in ("stated", "not_stated", "not_applicable", "unsure"):
            assert status in prompt

    def test_all_five_verifiability_levels_present(self):
        prompt = self._enrich()
        for level in ("strong", "moderate", "weak", "none", "unsure"):
            assert level in prompt

    def test_weak_none_boundary_rule_present(self):
        assert "PRESENCE of evidence" in self._enrich()

    # --- both -------------------------------------------------------------

    def test_neither_prompt_solicits_governance_claims(self):
        """These extractors are environmental-commitment only."""
        assert "governance" not in self._detect().lower()
        assert "governance" not in self._enrich().lower()


# ---------------------------------------------------------------------------
# Two-stage flow
#
# Stage 1 (detect) judges a whole chunk. Stage 2 (enrich) runs per positive
# sentence. The split exists because one combined prompt exceeded the effective
# context window on real hardware and produced word-salad.
# ---------------------------------------------------------------------------


def detect_payload(*records) -> str:
    return json.dumps({"commitments": list(records)})


def detection(text: str, decision: str, **extra) -> dict:
    d = {"text": text, "context": f"context for {text}",
         "is_commitment": decision, "restated": "no", "is_evidence": "no"}
    d.update(extra)
    return d


ENRICHMENT = json.dumps({
    "target": {"status": "stated", "value": "become carbon negative"},
    "quantity": {"status": "not_stated"},
    "deadline": {"status": "stated", "value": "2030"},
    "baseline": {"status": "not_stated"},
    "business_unit": {"status": "not_stated"},
    "emissions_scope": {"status": "not_stated"},
    "depends_on_outside_factors": "no",
    "verifiability": "weak",
    "annotator_notes": None,
})


class TestTwoStageFlow:

    @pytest.mark.asyncio
    async def test_negative_skips_enrichment(self):
        """A rejected sentence must not cost a second model call."""
        calls = []

        async def fake(prompt, chunk_index, system_prompt):
            calls.append(system_prompt)
            return detect_payload(
                detection("In 2023 we reduced water use by 12%.", "no",
                          rejection_reason="past_action")
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert len(calls) == 1          # detection only
        assert len(records) == 1
        assert records[0].is_commitment is CommitmentDecision.NO

    @pytest.mark.asyncio
    async def test_negative_fields_are_not_applicable_not_not_stated(self):
        """A non-commitment has no deadline — that is not_applicable, which is
        a different claim from 'the document did not say'."""
        async def fake(prompt, chunk_index, system_prompt):
            return detect_payload(
                detection("Sustainability is at our heart.", "no",
                          rejection_reason="values_statement")
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        r = records[0]
        assert r.deadline.status is FieldStatus.NOT_APPLICABLE
        assert r.target.status is FieldStatus.NOT_APPLICABLE
        assert r.depends_on_outside_factors is FlagValue.NOT_APPLICABLE

    @pytest.mark.asyncio
    async def test_positive_triggers_enrichment_and_merges_fields(self):
        async def fake(prompt, chunk_index, system_prompt):
            from backend.core.pipeline.commitment_extractor import (
                _DETECT_SYSTEM_PROMPT,
            )
            if system_prompt is _DETECT_SYSTEM_PROMPT:
                return detect_payload(
                    detection("We will be carbon negative by 2030.", "yes")
                )
            return ENRICHMENT

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        r = records[0]
        assert r.is_commitment is CommitmentDecision.YES
        assert r.deadline.value == "2030"
        assert r.target.value == "become carbon negative"
        assert r.verifiability is SubstantiationLevel.WEAK

    @pytest.mark.asyncio
    async def test_enrichment_failure_degrades_to_unsure(self):
        """If stage 2 fails, the record survives with unsure fields rather than
        being dropped or filled with invented values."""
        async def fake(prompt, chunk_index, system_prompt):
            from backend.core.pipeline.commitment_extractor import (
                _DETECT_SYSTEM_PROMPT,
            )
            if system_prompt is _DETECT_SYSTEM_PROMPT:
                return detect_payload(
                    detection("We will be carbon negative by 2030.", "yes")
                )
            raise httpx.TimeoutException("enrich timed out")

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        r = records[0]
        assert r.is_commitment is CommitmentDecision.YES   # detection survives
        assert r.deadline.status is FieldStatus.UNSURE
        assert r.verifiability is SubstantiationLevel.UNSURE

    @pytest.mark.asyncio
    async def test_enrichment_garbage_json_degrades_to_unsure(self):
        async def fake(prompt, chunk_index, system_prompt):
            from backend.core.pipeline.commitment_extractor import (
                _DETECT_SYSTEM_PROMPT,
            )
            if system_prompt is _DETECT_SYSTEM_PROMPT:
                return detect_payload(
                    detection("We will be carbon negative by 2030.", "yes")
                )
            return '{ "I\'m the ": "word salad'

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert records[0].target.status is FieldStatus.UNSURE

    @pytest.mark.asyncio
    async def test_enrich_call_count_scales_with_positives(self):
        enrich_calls = []

        async def fake(prompt, chunk_index, system_prompt):
            from backend.core.pipeline.commitment_extractor import (
                _DETECT_SYSTEM_PROMPT,
            )
            if system_prompt is _DETECT_SYSTEM_PROMPT:
                return detect_payload(
                    detection("We will cut emissions 50% by 2030.", "yes"),
                    detection("We will be water positive by 2030.", "yes"),
                    detection("In 2023 we cut water use.", "no",
                              rejection_reason="past_action"),
                )
            enrich_calls.append(prompt)
            return ENRICHMENT

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert len(records) == 3
        assert len(enrich_calls) == 2   # only the two positives
