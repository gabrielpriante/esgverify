"""Unit tests for backend.core.pipeline.commitment_extractor.

These tests are OFFLINE. Every Ollama call is mocked. Nothing here proves the
model behaves correctly on a real document — that requires the live pass on the
Microsoft development PDF, run separately.

What these tests do prove:
- The five rulings locked in notes/decisions.md (07/25/2026) survive the round
  trip from model JSON to ExtractedCommitment, using worked examples 6-10 from
  paper/task_definition.md as fixtures.
- "not stated", "not applicable" and "unsure" stay distinct rather than
  collapsing into each other or into null.
- Malformed model output degrades to "unsure", never to a confident value.
- RECALL: every sentence gets a record. A sentence the model omits becomes
  "unsure", not a silent disappearance.
- Record text always comes from the source document, never the model's echo.
- Every record carries the model tag and both prompt identifiers.
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
    DETECT_PROMPT_FILE,
    ENRICH_PROMPT_FILE,
    _coerce_field,
    _coerce_flag,
    _dict_to_commitment,
    _parse_verdicts_json,
    _split_sentences,
    extract_commitments,
)
from backend.core.pipeline.prompts import PROMPT_TOKEN_BUDGET, load_prompt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TWO_SENTENCE_CHUNK = (
    "We are driving robust programs in efforts to be carbon negative by 2030. "
    "More than 95% of our Scope 2 emissions were reduced by renewable energy."
)


def make_chunk(text: str = TWO_SENTENCE_CHUNK, index: int = 0) -> TextChunk:
    return TextChunk(index=index, text=text, char_start=0, char_end=len(text))


def field(status: str, value: str | None = None) -> dict:
    d: dict = {"status": status}
    if value is not None:
        d["value"] = value
    return d


def base_record(**overrides) -> dict:
    """A minimally valid merged record, as _dict_to_commitment receives it."""
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


def verdicts_payload(*verdicts) -> str:
    return json.dumps({"verdicts": list(verdicts)})


def verdict(vid: int, decision: str, **extra) -> dict:
    v = {"id": vid, "is_commitment": decision, "restated": "no", "is_evidence": "no"}
    v.update(extra)
    return v


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


def detect_text() -> str:
    return load_prompt(DETECT_PROMPT_FILE)[0]


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
            depends_on_outside_factors="yes",
            verifiability="weak",
        )

        record = _dict_to_commitment(raw, make_chunk())

        assert record is not None
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
        """Example 9: a mixed past/future sentence is judged whole, and rejected."""
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
        assert _coerce_field(field("stated", "   "), "deadline", 0).status is FieldStatus.UNSURE

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

    @pytest.mark.parametrize("level", ["strong", "moderate", "weak", "none", "unsure"])
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
# Sentence splitting
# ---------------------------------------------------------------------------


class TestSplitSentences:

    def test_splits_on_sentence_boundaries(self):
        assert len(_split_sentences(TWO_SENTENCE_CHUNK)) == 2

    def test_collapses_pdf_hard_line_breaks(self):
        """PDF extraction breaks lines mid-sentence; that must not create
        spurious sentence boundaries."""
        pdf_style = "We will be carbon\nnegative by 2030. We also plan much more."
        out = _split_sentences(pdf_style)
        assert out[0] == "We will be carbon negative by 2030."

    def test_numeric_continuation_does_not_behead_a_sentence(self):
        """Real chunk 39 layout: 'our Scope' ends a line and '2 emissions...'
        begins the next. Treating the numeric line as a new unit produced the
        fragment '2 emissions were reduced by renewable energy' — losing the
        subject of the exact past-action sentence we need to reject."""
        pdf_style = "More than 95% of our Scope\n2 emissions were reduced by renewable energy."
        out = _split_sentences(pdf_style)
        assert len(out) == 1
        assert out[0].startswith("More than 95%")

    def test_drops_navigation_furniture(self):
        """'Overview', 'Earn trust', 'Learn more' are layout, not claims."""
        assert _split_sentences("Overview\nEarn trust\nLearn more") == []

    def test_preserves_document_order(self):
        out = _split_sentences(TWO_SENTENCE_CHUNK)
        assert out[0].startswith("We are driving")
        assert out[1].startswith("More than 95%")


# ---------------------------------------------------------------------------
# Id-keyed verdict parsing
# ---------------------------------------------------------------------------


class TestParseVerdictsJson:

    def test_parses_well_formed_payload(self):
        raw = verdicts_payload(verdict(1, "yes"), verdict(2, "no"))
        assert set(_parse_verdicts_json(raw, make_chunk(), 2)) == {1, 2}

    def test_strips_markdown_fences(self):
        raw = "```json\n" + verdicts_payload(verdict(1, "yes")) + "\n```"
        assert set(_parse_verdicts_json(raw, make_chunk(), 1)) == {1}

    def test_discards_out_of_range_ids(self):
        """An invented id means the model lost track of the list; mapping it
        onto a real sentence would attach a verdict to text it never saw."""
        raw = verdicts_payload(verdict(1, "yes"), verdict(99, "yes"))
        assert set(_parse_verdicts_json(raw, make_chunk(), 2)) == {1}

    def test_discards_verdicts_without_id(self):
        raw = json.dumps({"verdicts": [{"is_commitment": "yes"}]})
        assert _parse_verdicts_json(raw, make_chunk(), 2) == {}

    @pytest.mark.parametrize("payload", [
        "not json at all",
        "{",
        json.dumps({"commitments": []}),      # the old stage-1 shape
        json.dumps({"verdicts": "nope"}),
        json.dumps([1, 2, 3]),
    ])
    def test_malformed_payloads_return_empty_without_raising(self, payload):
        assert _parse_verdicts_json(payload, make_chunk(), 2) == {}


# ---------------------------------------------------------------------------
# Recall — the failure this design exists to prevent
#
# An earlier version asked the model to find and quote commitment sentences
# itself. On chunk 39 of the Microsoft Impact Summary it returned four
# positives, zero rejections, and silently omitted a Scope 2 past-action
# sentence rather than rejecting it. Invisible in the output, and it would have
# inflated apparent precision.
# ---------------------------------------------------------------------------


class TestRecall:

    @pytest.mark.asyncio
    async def test_skipped_sentence_becomes_unsure_not_dropped(self):
        async def fake(prompt, chunk_index, system_prompt):
            # verdict for sentence 1 only; sentence 2 omitted entirely
            return verdicts_payload(
                verdict(1, "no", rejection_reason="values_statement")
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert len(records) == 2, "an omitted sentence must still produce a record"
        skipped = records[1]
        assert skipped.is_commitment is CommitmentDecision.UNSURE
        assert "no verdict" in (skipped.annotator_notes or "")

    @pytest.mark.asyncio
    async def test_past_action_returns_as_explicit_rejection(self):
        async def fake(prompt, chunk_index, system_prompt):
            if system_prompt == detect_text():
                return verdicts_payload(
                    verdict(1, "yes"),
                    verdict(2, "no", rejection_reason="past_action"),
                )
            return ENRICHMENT

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        rejected = [r for r in records if r.is_commitment is CommitmentDecision.NO]
        assert len(rejected) == 1
        assert rejected[0].rejection_reason == "past_action"
        assert "Scope 2 emissions were reduced" in rejected[0].text

    @pytest.mark.asyncio
    async def test_record_text_comes_from_source_not_model_echo(self):
        """Guards against transcription drift and invented quotes."""
        async def fake(prompt, chunk_index, system_prompt):
            return verdicts_payload(
                verdict(1, "no", rejection_reason="values_statement",
                        text="A SENTENCE THE MODEL MADE UP"),
                verdict(2, "no", rejection_reason="past_action"),
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert all("MADE UP" not in r.text for r in records)
        assert records[0].text.startswith("We are driving")

    @pytest.mark.asyncio
    async def test_every_sentence_gets_a_record_even_when_model_returns_nothing(self):
        async def fake(prompt, chunk_index, system_prompt):
            return verdicts_payload()

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert len(records) == len(_split_sentences(TWO_SENTENCE_CHUNK))
        assert all(r.is_commitment is CommitmentDecision.UNSURE for r in records)


# ---------------------------------------------------------------------------
# Two-stage flow
# ---------------------------------------------------------------------------


class TestTwoStageFlow:

    @pytest.mark.asyncio
    async def test_empty_chunk_list_returns_empty(self):
        assert await extract_commitments([]) == []

    @pytest.mark.asyncio
    async def test_chunk_with_no_judgeable_sentences_is_skipped(self):
        called = []

        async def fake(prompt, chunk_index, system_prompt):
            called.append(1)
            return verdicts_payload()

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk("Overview\nLearn more")])

        assert records == []
        assert called == [], "must not spend a model call on layout furniture"

    @pytest.mark.asyncio
    async def test_negatives_skip_enrichment(self):
        systems = []

        async def fake(prompt, chunk_index, system_prompt):
            systems.append(system_prompt)
            return verdicts_payload(
                verdict(1, "no", rejection_reason="values_statement"),
                verdict(2, "no", rejection_reason="past_action"),
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert len(systems) == 1, "detection only; negatives cost no second call"
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_negative_fields_are_not_applicable(self):
        async def fake(prompt, chunk_index, system_prompt):
            return verdicts_payload(
                verdict(1, "no", rejection_reason="values_statement"),
                verdict(2, "no", rejection_reason="past_action"),
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
            if system_prompt == detect_text():
                return verdicts_payload(
                    verdict(1, "yes"),
                    verdict(2, "no", rejection_reason="past_action"),
                )
            return ENRICHMENT

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        r = records[0]
        assert r.is_commitment is CommitmentDecision.YES
        assert r.deadline.value == "2030"
        assert r.verifiability is SubstantiationLevel.WEAK

    @pytest.mark.asyncio
    async def test_enrichment_failure_degrades_to_unsure(self):
        """Keep the judgement we earned; don't invent the rest."""
        async def fake(prompt, chunk_index, system_prompt):
            if system_prompt == detect_text():
                return verdicts_payload(
                    verdict(1, "yes"),
                    verdict(2, "no", rejection_reason="past_action"),
                )
            raise httpx.TimeoutException("enrich timed out")

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        r = records[0]
        assert r.is_commitment is CommitmentDecision.YES
        assert r.deadline.status is FieldStatus.UNSURE
        assert r.verifiability is SubstantiationLevel.UNSURE

    @pytest.mark.asyncio
    async def test_enrichment_garbage_degrades_to_unsure(self):
        async def fake(prompt, chunk_index, system_prompt):
            if system_prompt == detect_text():
                return verdicts_payload(
                    verdict(1, "yes"),
                    verdict(2, "no", rejection_reason="past_action"),
                )
            return '{ "word": "salad'

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert records[0].target.status is FieldStatus.UNSURE

    @pytest.mark.asyncio
    async def test_failed_detection_skips_chunk_without_crashing(self):
        async def fake(prompt, chunk_index, system_prompt):
            raise httpx.TimeoutException("detect timed out")

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert records == []

    @pytest.mark.asyncio
    async def test_enrich_calls_scale_with_positives_only(self):
        enrich_calls = []

        async def fake(prompt, chunk_index, system_prompt):
            if system_prompt == detect_text():
                return verdicts_payload(
                    verdict(1, "yes"),
                    verdict(2, "no", rejection_reason="past_action"),
                )
            enrich_calls.append(prompt)
            return ENRICHMENT

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert len(records) == 2
        assert len(enrich_calls) == 1

    @pytest.mark.asyncio
    async def test_commitment_ids_are_unique(self):
        async def fake(prompt, chunk_index, system_prompt):
            return verdicts_payload(
                verdict(1, "no", rejection_reason="past_action"),
                verdict(2, "no", rejection_reason="past_action"),
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        assert len({r.commitment_id for r in records}) == 2

    @pytest.mark.asyncio
    async def test_page_reference_falls_back_to_chunk_index(self):
        async def fake(prompt, chunk_index, system_prompt):
            return verdicts_payload(
                verdict(1, "no", rejection_reason="past_action"),
                verdict(2, "no", rejection_reason="past_action"),
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk(index=7)])

        assert records[0].page_reference == "chunk_7"


# ---------------------------------------------------------------------------
# Provenance — every record traceable to a model and an exact prompt version
# ---------------------------------------------------------------------------


class TestProvenance:

    @pytest.mark.asyncio
    async def test_every_record_carries_model_and_detect_prompt_id(self):
        async def fake(prompt, chunk_index, system_prompt):
            return verdicts_payload(
                verdict(1, "no", rejection_reason="values_statement"),
                verdict(2, "no", rejection_reason="past_action"),
            )

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        for r in records:
            assert r.model, "model tag must be recorded per record, not per run"
            assert r.detect_prompt_id.startswith(DETECT_PROMPT_FILE + "@")

    @pytest.mark.asyncio
    async def test_enrich_prompt_id_only_on_enriched_records(self):
        async def fake(prompt, chunk_index, system_prompt):
            if system_prompt == detect_text():
                return verdicts_payload(
                    verdict(1, "yes"),
                    verdict(2, "no", rejection_reason="past_action"),
                )
            return ENRICHMENT

        with patch("backend.core.pipeline.commitment_extractor._call_ollama",
                   new=AsyncMock(side_effect=fake)):
            records = await extract_commitments([make_chunk()])

        positive = next(r for r in records if r.is_commitment is CommitmentDecision.YES)
        negative = next(r for r in records if r.is_commitment is CommitmentDecision.NO)
        assert positive.enrich_prompt_id.startswith(ENRICH_PROMPT_FILE + "@")
        assert negative.enrich_prompt_id is None

    def test_prompt_id_is_stable_and_content_addressed(self):
        a = load_prompt(DETECT_PROMPT_FILE)[1]
        b = load_prompt(DETECT_PROMPT_FILE)[1]
        assert a == b
        assert len(a.split("@")[1]) == 12


# ---------------------------------------------------------------------------
# Prompt content guards — the prompts are the experimental instrument
# ---------------------------------------------------------------------------


class TestPromptsEncodeTheDefinition:

    def _detect(self) -> str:
        return load_prompt(DETECT_PROMPT_FILE)[0]

    def _enrich(self) -> str:
        return load_prompt(ENRICH_PROMPT_FILE)[0]

    def test_both_prompts_stay_within_token_budget(self):
        """A ~1,590-token prompt produced word-salad from llama3.1:8b on a
        mid-range GPU; ~1,240 was fine. Two-stage exists to stay clear of that
        ceiling, and this keeps it that way."""
        for name, prompt in (("detect", self._detect()), ("enrich", self._enrich())):
            approx = len(prompt) // 4
            assert approx < PROMPT_TOKEN_BUDGET, f"{name} is ~{approx} tokens"

    def test_prompts_live_on_disk_for_publication(self):
        from backend.core.pipeline.prompts import PROMPT_DIR
        assert (PROMPT_DIR / DETECT_PROMPT_FILE).is_file()
        assert (PROMPT_DIR / ENRICH_PROMPT_FILE).is_file()

    # --- detection prompt -------------------------------------------------

    def test_no_past_action_used_as_a_positive_example(self):
        assert "reduced Scope 1 emissions by 30%" not in self._detect()

    def test_future_intention_is_the_core_criterion(self):
        assert "Future intention is the core test" in self._detect()

    def test_requires_a_specific_outcome(self):
        """The precision failure on chunk 39: 'committed to meeting our own
        goals' and 'taking responsibility for our operational footprint' were
        both accepted as commitments."""
        prompt = self._detect()
        assert "SPECIFIC environmental action or outcome" in prompt
        assert "our own goals" in prompt
        assert "operational" in prompt

    def test_all_four_rejection_categories_present(self):
        for category in ("past_action", "values_statement",
                         "factual_disclosure", "description"):
            assert category in self._detect()

    def test_scope_2_past_action_named_as_a_rejection_example(self):
        """The exact sentence the previous version silently dropped."""
        assert "Scope 2 emissions were reduced" in self._detect()

    def test_three_locked_rulings_present(self):
        prompt = self._detect().lower()
        assert "conditional commitments count" in prompt
        assert "restated commitments count" in prompt
        assert "is evidence" in prompt

    def test_demands_a_verdict_for_every_id(self):
        prompt = self._detect()
        assert "every id" in prompt.lower()
        assert "Do not skip ids" in prompt

    # --- enrichment prompt ------------------------------------------------

    def test_all_seven_structural_fields_present(self):
        for f in ("target", "quantity", "deadline", "baseline",
                  "business_unit", "emissions_scope", "depends_on_outside_factors"):
            assert f in self._enrich()

    def test_all_four_field_statuses_present(self):
        for status in ("stated", "not_stated", "not_applicable", "unsure"):
            assert status in self._enrich()

    def test_all_five_verifiability_levels_present(self):
        for level in ("strong", "moderate", "weak", "none", "unsure"):
            assert level in self._enrich()

    def test_weak_none_boundary_rule_present(self):
        assert "PRESENCE of evidence" in self._enrich()

    def test_forbids_inferring_absent_values(self):
        """Chunk 39 record 2 hallucinated emissions_scope='Scope 1'."""
        assert "Do not infer a value that is not in the" in self._enrich()

    # --- both -------------------------------------------------------------

    def test_neither_prompt_solicits_governance_claims(self):
        assert "governance" not in self._detect().lower()
        assert "governance" not in self._enrich().lower()
