"""Environmental commitment extraction via local Ollama LLM.

Stage 2 of the Paper #4 benchmark pipeline. Sends document text chunks to a
locally running Ollama instance and parses the responses into structured
:class:`ExtractedCommitment` objects.

This module is deliberately separate from :mod:`claim_extractor`. That module
extracts ESG *claims* across all three pillars for the greenwashing framing.
This module extracts environmental *commitments* — pledges to a future action
or outcome — which is a narrower and differently-shaped task.

Definitions are locked in paper/task_definition.md and notes/decisions.md
(entry dated 07/25/2026). The prompt below encodes those rulings verbatim.
If a ruling changes, change the documents first, then this prompt, then the
fixtures in tests/unit/test_commitment_extractor.py.

Recommended model: ``llama3.1:8b-instruct-q4_K_M``
  - Quantization: Q4_K_M (~4.7 GB VRAM)
  - Pull with: ``ollama pull llama3.1:8b-instruct-q4_K_M``
  - Set in config: ``OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M``
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.config import settings
from backend.core.models.report import (
    CommitmentDecision,
    ExtractedCommitment,
    FieldStatus,
    FlagValue,
    StructuredField,
    SubstantiationLevel,
)
from backend.core.pipeline.chunker import TextChunk
from backend.core.pipeline.prompts import (
    PROMPT_TOKEN_BUDGET,  # noqa: F401  (re-exported for tests)
    load_prompt,
)

logger = structlog.get_logger(__name__)

# Max output tokens per chunk. Commitment records are large (7 nested field
# objects each, negatives included), so this is higher than the claim
# extractor's 1024. It is the dominant cost driver for wall-clock time —
# lower it if runs are timing out. Overridable via the runner's --num-predict.
NUM_PREDICT = 2048

# Context window, set EXPLICITLY. Ollama's default num_ctx is small (2048 on
# many setups). This prompt needs roughly:
#     ~1190 tokens  system prompt
#     ~ 405 tokens  user prompt with a 1500-char chunk
#      2048 tokens  reserved for output (NUM_PREDICT)
#     ------------
#     ~3640 tokens  minimum
# Under a 2048-token window the system prompt is silently truncated and the
# model emits degenerate word-salad rather than JSON — which looks like a
# broken model but is really a context overflow. Do not lower this below
# NUM_PREDICT + 1600 without also shortening the prompt.
NUM_CTX = 8192

# When set to a directory path, every raw model response is written there as
# chunk_<index>.txt before parsing. Truncated log previews are not enough to
# diagnose a model that is emitting degenerate text. Set via --dump-raw.
RAW_DUMP_DIR: str | None = None

# ---------------------------------------------------------------------------
# Prompts — TWO STAGE, loaded from scripts/prompts/
#
# Empirically, an 8B q4 model degenerates into word-salad once the combined
# system+user prompt exceeds roughly 1,300 tokens, even with num_ctx set to
# 8192 (the effective window is clamped below what is requested). So the work
# is split, and each prompt is budget-guarded on load:
#
#   STAGE 1  detect   numbered sentences -> one verdict per sentence id
#   STAGE 2  enrich   one sentence       -> seven fields + verifiability
#
# Stage 1 is given sentences WE have already split, numbered, and it must
# return a verdict for every id. Earlier the model was asked to find and quote
# sentences itself; it silently omitted inconvenient ones — notably a Scope 2
# past-action sentence that should have been an explicit rejection. Numbering
# makes omissions detectable and fillable, and means record text comes from the
# source document rather than the model's transcription of it.
#
# Definitions are locked in paper/task_definition.md and notes/decisions.md
# (07/25/2026).
# ---------------------------------------------------------------------------

DETECT_PROMPT_FILE = "detect_v2.txt"
ENRICH_PROMPT_FILE = "enrich_v1.txt"

# Sentences shorter than this are navigation furniture in PDF-extracted text
# ("Overview", "Earn trust", "Learn more") and are not judged. They are
# excluded before numbering, so they never consume an id or output tokens.
MIN_SENTENCE_CHARS = 25

# Cap on sentences per detection call, to keep the user turn bounded.
MAX_SENTENCES_PER_CALL = 25

_DETECT_USER_TEMPLATE = """\
Judge each numbered sentence. Return exactly one verdict per id.

{numbered}
"""

_ENRICH_USER_TEMPLATE = """\
Sentence:
{text}

Surrounding context:
{context}
"""


def _split_sentences(text: str) -> list[str]:
    """Split chunk text into candidate sentences for numbering.

    Two competing problems in PDF-extracted text:

    - Sentences are hard-wrapped mid-clause, so raw newlines are not sentence
      boundaries: "We will be carbon\\nnegative by 2030."
    - Navigation furniture arrives as its own short lines ("Overview", "Earn
      trust", "Learn more"). Collapsing all whitespace glues these into one
      long pseudo-sentence that then passes a naive length filter — this is
      what polluted chunk 39 of the Microsoft Impact Summary.

    So a line break is treated as a continuation only when the current buffer
    has no terminal punctuation *and* the next line begins lowercase. Anything
    else starts a new unit. Units are then split on sentence punctuation and
    kept only if they are long enough and actually end a sentence.

    Requiring terminal punctuation drops sentences truncated by a chunk
    boundary. That is intended: a half sentence should not be judged, and the
    200-character chunk overlap means it usually appears whole in a neighbour.

    Args:
        text: Raw chunk text.

    Returns:
        Sentences worth a verdict, in document order.
    """
    units: list[str] = []
    buffer = ""

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if buffer:
                units.append(buffer)
                buffer = ""
            continue

        if not buffer:
            buffer = line
            continue

        # Continuation of a hard-wrapped sentence? A digit counts: PDF layout
        # splits "our Scope\n2 emissions were reduced..." and treating the
        # numeric line as a new unit beheads the sentence.
        if not buffer.endswith((".", "!", "?", ":", ";")) and (
            line[:1].islower() or line[:1].isdigit()
        ):
            buffer = f"{buffer} {line}"
        else:
            units.append(buffer)
            buffer = line

    if buffer:
        units.append(buffer)

    sentences: list[str] = []
    for unit in units:
        for part in re.split(r"(?<=[.!?])\s+", unit):
            candidate = part.strip()
            if len(candidate) >= MIN_SENTENCE_CHARS and candidate.endswith(
                (".", "!", "?")
            ):
                sentences.append(candidate)

    return sentences


def _number_sentences(sentences: list[str]) -> str:
    """Render sentences as a numbered list for the detection prompt."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))


# ---------------------------------------------------------------------------
# Retry-decorated Ollama call
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _call_ollama(prompt: str, chunk_index: int, system_prompt: str) -> str:
    """Send a prompt to Ollama and return the raw response text.

    Retries up to 3 times with exponential backoff on network or timeout
    errors. Uses the non-streaming ``/api/chat`` endpoint.

    Args:
        prompt: The user-turn message to send to the model.
        chunk_index: Used only for logging context.

    Returns:
        Raw text content from the model's response.

    Raises:
        httpx.HTTPError: If all 3 retry attempts fail due to HTTP errors.
        httpx.TimeoutException: If all 3 retry attempts time out.
    """
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        # Constrained decoding: Ollama forces syntactically valid JSON. Belt and
        # braces alongside the "return ONLY JSON" instruction — the instruction
        # alone is not reliable at 8B scale.
        "format": "json",
        "options": {
            # Temperature 0 — deterministic structured output for a benchmark
            "temperature": 0,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }

    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
    ) as client:
        logger.debug(
            "ollama_request_sent",
            chunk_index=chunk_index,
            model=settings.ollama_model,
        )
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()

    data = response.json()
    content: str = data["message"]["content"]
    logger.debug(
        "ollama_response_received",
        chunk_index=chunk_index,
        response_length=len(content),
    )
    return content


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_commitments_json(raw: str, chunk: TextChunk) -> list[dict[str, Any]]:
    """Parse the LLM's raw output into a list of commitment dicts.

    Strips accidental markdown fences before parsing. Returns an empty list
    (and logs a warning) on any parse failure — never raises.

    Args:
        raw: Raw string returned by the LLM.
        chunk: The source chunk, used for logging context.

    Returns:
        List of raw commitment dicts, or an empty list on failure.
    """
    if RAW_DUMP_DIR:
        try:
            from pathlib import Path
            d = Path(RAW_DUMP_DIR)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"chunk_{chunk.index}.txt").write_text(raw, encoding="utf-8")
        except OSError as exc:  # never let debugging output break a run
            logger.warning("raw_dump_failed", chunk_index=chunk.index, error=str(exc))

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.splitlines()
            if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "json_parse_failed",
            chunk_index=chunk.index,
            error=str(exc),
            raw_preview=cleaned[:200],
        )
        return []

    if not isinstance(parsed, dict) or "commitments" not in parsed:
        logger.warning(
            "unexpected_json_shape",
            chunk_index=chunk.index,
            keys=list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
        )
        return []

    commitments = parsed["commitments"]
    if not isinstance(commitments, list):
        logger.warning("commitments_field_not_list", chunk_index=chunk.index)
        return []

    return commitments


def _coerce_field(raw: Any, field_name: str, chunk_index: int) -> StructuredField:
    """Coerce one raw structural field into a :class:`StructuredField`.

    Falls back to ``unsure`` rather than ``not_stated`` on malformed input.
    That distinction matters: ``not_stated`` is an annotation claiming the
    document is silent, which is a substantive assertion the model has not
    earned when its own output was unparseable.

    Args:
        raw: The raw field value from the LLM (expected: a dict).
        field_name: Field name, for logging.
        chunk_index: Chunk index, for logging.

    Returns:
        A valid :class:`StructuredField`; never raises.
    """
    if not isinstance(raw, dict):
        logger.warning(
            "field_not_object",
            field=field_name,
            chunk_index=chunk_index,
            raw=str(raw)[:100],
        )
        return StructuredField.unsure()

    raw_status = str(raw.get("status", "")).lower().strip()
    try:
        status = FieldStatus(raw_status)
    except ValueError:
        logger.warning(
            "unknown_field_status",
            field=field_name,
            chunk_index=chunk_index,
            raw_status=raw_status,
        )
        return StructuredField.unsure()

    if status is not FieldStatus.STATED:
        return StructuredField(status=status)

    value = raw.get("value")
    if value is None or not str(value).strip():
        # Model said "stated" but gave nothing — that is an unsure, not a value.
        logger.warning(
            "stated_field_missing_value",
            field=field_name,
            chunk_index=chunk_index,
        )
        return StructuredField.unsure()

    return StructuredField.stated(str(value).strip())


def _coerce_flag(raw: Any, field_name: str, chunk_index: int) -> FlagValue:
    """Coerce a raw yes/no-style attribute into a :class:`FlagValue`.

    Unrecognised input becomes ``UNSURE`` rather than ``NO``. Defaulting to
    ``NO`` would silently manufacture negative annotations.
    """
    try:
        return FlagValue(str(raw).lower().strip())
    except ValueError:
        logger.warning(
            "unknown_flag_value",
            field=field_name,
            chunk_index=chunk_index,
            raw=str(raw)[:100],
        )
        return FlagValue.UNSURE


def _dict_to_commitment(
    raw: dict[str, Any],
    chunk: TextChunk,
) -> ExtractedCommitment | None:
    """Convert a raw commitment dict from the LLM into an :class:`ExtractedCommitment`.

    Every enum coercion degrades to ``unsure`` on invalid input, so a confused
    model produces a flagged-for-review record rather than a confident wrong one.
    Returns ``None`` (and logs a warning) only when the record has no text.

    Args:
        raw: A single commitment dict as returned by the LLM.
        chunk: The source chunk, for page fallback and logging.

    Returns:
        A validated :class:`ExtractedCommitment`, or ``None`` on failure.
    """
    try:
        text = str(raw.get("text", "")).strip()
        if not text:
            logger.warning("commitment_missing_text", chunk_index=chunk.index)
            return None

        raw_decision = str(raw.get("is_commitment", "")).lower().strip()
        try:
            decision = CommitmentDecision(raw_decision)
        except ValueError:
            logger.warning(
                "unknown_commitment_decision",
                chunk_index=chunk.index,
                raw_decision=raw_decision,
            )
            decision = CommitmentDecision.UNSURE

        raw_verifiability = str(raw.get("verifiability", "")).lower().strip()
        try:
            verifiability = SubstantiationLevel(raw_verifiability)
        except ValueError:
            logger.warning(
                "unknown_verifiability_level",
                chunk_index=chunk.index,
                raw_level=raw_verifiability,
            )
            verifiability = SubstantiationLevel.UNSURE

        rejection_reason = raw.get("rejection_reason")
        if rejection_reason is not None:
            rejection_reason = str(rejection_reason).strip() or None
        # A rejection reason on a positive judgement is contradictory; drop it.
        if decision is not CommitmentDecision.NO:
            rejection_reason = None

        notes = raw.get("annotator_notes")
        if notes is not None:
            notes = str(notes).strip() or None

        return ExtractedCommitment(
            commitment_id=str(uuid.uuid4()),
            text=text,
            context=str(raw.get("context", "")).strip() or text,
            page_reference=raw.get("page_reference") or f"chunk_{chunk.index}",
            is_commitment=decision,
            rejection_reason=rejection_reason,
            target=_coerce_field(raw.get("target"), "target", chunk.index),
            quantity=_coerce_field(raw.get("quantity"), "quantity", chunk.index),
            deadline=_coerce_field(raw.get("deadline"), "deadline", chunk.index),
            baseline=_coerce_field(raw.get("baseline"), "baseline", chunk.index),
            business_unit=_coerce_field(
                raw.get("business_unit"), "business_unit", chunk.index
            ),
            emissions_scope=_coerce_field(
                raw.get("emissions_scope"), "emissions_scope", chunk.index
            ),
            depends_on_outside_factors=_coerce_flag(
                raw.get("depends_on_outside_factors"),
                "depends_on_outside_factors",
                chunk.index,
            ),
            restated=_coerce_flag(raw.get("restated"), "restated", chunk.index),
            is_evidence=_coerce_flag(raw.get("is_evidence"), "is_evidence", chunk.index),
            verifiability=verifiability,
            annotator_notes=notes,
            model=raw.get("model"),
            detect_prompt_id=raw.get("detect_prompt_id"),
            enrich_prompt_id=raw.get("enrich_prompt_id"),
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "commitment_conversion_failed",
            chunk_index=chunk.index,
            error=str(exc),
            raw_commitment=str(raw)[:200],
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def _enrich(
    detection: dict[str, Any],
    chunk: TextChunk,
) -> dict[str, Any]:
    """Stage 2: ask the model for the structural fields of one commitment.

    Returns the detection dict merged with whatever enrichment came back. On
    any failure the detection dict is returned unchanged — the caller's
    coercion layer then fills every field with ``unsure``, which is the honest
    outcome when stage 2 produced nothing.
    """
    enrich_prompt, _ = load_prompt(ENRICH_PROMPT_FILE)
    text = str(detection.get("text", "")).strip()
    prompt = _ENRICH_USER_TEMPLATE.format(
        text=text,
        context=str(detection.get("context", "")).strip() or text,
    )

    try:
        raw = await _call_ollama(prompt, chunk.index, enrich_prompt)
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.warning(
            "enrich_call_failed",
            chunk_index=chunk.index,
            error=repr(exc),
            error_type=type(exc).__name__,
        )
        return detection

    if RAW_DUMP_DIR:
        try:
            from pathlib import Path as _P
            d = _P(RAW_DUMP_DIR)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"chunk_{chunk.index}_enrich.txt").write_text(raw, encoding="utf-8")
        except OSError:
            pass

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")
        ).strip()

    try:
        fields = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "enrich_json_parse_failed",
            chunk_index=chunk.index,
            error=str(exc),
            raw_preview=cleaned[:200],
        )
        return detection

    if not isinstance(fields, dict):
        logger.warning("enrich_not_object", chunk_index=chunk.index)
        return detection

    merged = dict(detection)
    merged.update(fields)
    return merged


def _mark_fields_not_applicable(detection: dict[str, Any]) -> dict[str, Any]:
    """Fill structural fields for a NON-commitment.

    A sentence that is not a pledge has no target, deadline or baseline — that
    is ``not_applicable``, which is deliberately distinct from ``not_stated``
    (the document was silent) and ``unsure`` (we could not tell).
    """
    out = dict(detection)
    for name in (
        "target", "quantity", "deadline", "baseline",
        "business_unit", "emissions_scope",
    ):
        out.setdefault(name, {"status": "not_applicable"})
    out.setdefault("depends_on_outside_factors", "not_applicable")
    return out


def _parse_verdicts_json(
    raw: str,
    chunk: TextChunk,
    sentence_count: int,
) -> dict[int, dict[str, Any]]:
    """Parse stage 1 output into ``{sentence_id: verdict}``.

    Ids outside ``1..sentence_count`` are discarded rather than trusted — a
    model that invents an id has lost track of the list, and mapping that
    verdict onto a real sentence would attach a judgement to text the model
    was not looking at.

    Returns an empty dict on any parse failure; never raises.
    """
    if RAW_DUMP_DIR:
        try:
            from pathlib import Path as _P
            d = _P(RAW_DUMP_DIR)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"chunk_{chunk.index}_detect.txt").write_text(raw, encoding="utf-8")
        except OSError as exc:
            logger.warning("raw_dump_failed", chunk_index=chunk.index, error=str(exc))

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "json_parse_failed",
            chunk_index=chunk.index,
            error=str(exc),
            raw_preview=cleaned[:200],
        )
        return {}

    if not isinstance(parsed, dict) or "verdicts" not in parsed:
        logger.warning(
            "unexpected_json_shape",
            chunk_index=chunk.index,
            keys=list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
        )
        return {}

    verdicts = parsed["verdicts"]
    if not isinstance(verdicts, list):
        logger.warning("verdicts_field_not_list", chunk_index=chunk.index)
        return {}

    out: dict[int, dict[str, Any]] = {}
    for v in verdicts:
        if not isinstance(v, dict):
            continue
        try:
            vid = int(v.get("id"))
        except (TypeError, ValueError):
            logger.warning("verdict_missing_id", chunk_index=chunk.index)
            continue
        if not 1 <= vid <= sentence_count:
            logger.warning(
                "verdict_id_out_of_range",
                chunk_index=chunk.index,
                verdict_id=vid,
                sentence_count=sentence_count,
            )
            continue
        out[vid] = v

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_commitments(chunks: list[TextChunk]) -> list[ExtractedCommitment]:
    """Extract environmental commitments from document chunks, in two stages.

    Stage 1 splits a chunk into numbered sentences and asks the model for one
    verdict per id. Stage 2 sends each positive sentence back on its own and
    asks for its seven structural fields and verifiability level.

    Two properties follow from numbering the sentences ourselves:

    - **No silent omissions.** Any id the model fails to return is recorded as
      ``unsure`` with a note, rather than vanishing. An earlier version let the
      model choose which sentences to mention and it quietly dropped a
      past-action sentence instead of rejecting it.
    - **No transcription drift.** Record text is taken from the source
      document by id, so a record can never contain a sentence the model
      paraphrased or invented.

    Records judged NOT a commitment are retained — the benchmark needs
    negatives to measure precision. They skip stage 2 and their structural
    fields are marked ``not_applicable``.

    Every record carries the model tag and the identifiers of both prompts.

    Args:
        chunks: Ordered list of :class:`TextChunk` objects from
            :func:`backend.core.pipeline.chunker.chunk_text`.

    Returns:
        Flat list of all :class:`ExtractedCommitment` objects across all chunks.
    """
    if not chunks:
        logger.warning("extract_commitments called with empty chunk list")
        return []

    detect_prompt, detect_id = load_prompt(DETECT_PROMPT_FILE)
    _, enrich_id = load_prompt(ENRICH_PROMPT_FILE)
    model = settings.ollama_model

    logger.info(
        "commitment_extraction_start",
        model=model,
        detect_prompt=detect_id,
        enrich_prompt=enrich_id,
        chunks=len(chunks),
    )

    all_records: list[ExtractedCommitment] = []

    for chunk in chunks:
        log = logger.bind(chunk_index=chunk.index, total_chunks=len(chunks))

        sentences = _split_sentences(chunk.text)[:MAX_SENTENCES_PER_CALL]
        if not sentences:
            log.info("chunk_has_no_judgeable_sentences")
            continue

        # --- Stage 1: detection over numbered sentences ----------------------
        detect_user = _DETECT_USER_TEMPLATE.format(
            numbered=_number_sentences(sentences)
        )

        try:
            raw_response = await _call_ollama(
                detect_user, chunk.index, detect_prompt
            )
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            # str() on httpx timeout exceptions is often empty — use repr so the
            # log says which failure it was instead of "error=".
            log.error(
                "ollama_call_failed_after_retries",
                error=repr(exc),
                error_type=type(exc).__name__,
                timeout_seconds=settings.ollama_timeout_seconds,
                chunk_skipped=True,
            )
            continue

        verdicts = _parse_verdicts_json(raw_response, chunk, len(sentences))
        missing = [i for i in range(1, len(sentences) + 1) if i not in verdicts]
        if missing:
            log.warning(
                "detection_incomplete",
                missing_ids=missing,
                returned=len(verdicts),
                expected=len(sentences),
            )

        # --- Stage 2: enrichment, positives only -----------------------------
        chunk_records: list[ExtractedCommitment] = []
        enriched_count = 0

        for idx, sentence in enumerate(sentences, start=1):
            verdict = verdicts.get(idx)

            if verdict is None:
                # Model skipped this id. Record it as unsure rather than
                # dropping it — a silently missing sentence is invisible in
                # the results and would inflate apparent precision.
                detection: dict[str, Any] = {
                    "text": sentence,
                    "is_commitment": "unsure",
                    "annotator_notes": "stage 1 returned no verdict for this sentence",
                }
            else:
                detection = dict(verdict)
                # Always use OUR sentence text, never the model's echo.
                detection["text"] = sentence

            detection.setdefault("context", chunk.text[:400])

            decision = str(detection.get("is_commitment", "")).lower().strip()
            if decision == "yes":
                merged = await _enrich(detection, chunk)
                enriched_count += 1
            else:
                merged = _mark_fields_not_applicable(detection)

            merged["model"] = model
            merged["detect_prompt_id"] = detect_id
            merged["enrich_prompt_id"] = enrich_id if decision == "yes" else None

            record = _dict_to_commitment(merged, chunk)
            if record is not None:
                chunk_records.append(record)

        log.info(
            "chunk_processed",
            sentences=len(sentences),
            records_found=len(chunk_records),
            positives=sum(
                1 for r in chunk_records
                if r.is_commitment is CommitmentDecision.YES
            ),
            rejections=sum(
                1 for r in chunk_records
                if r.is_commitment is CommitmentDecision.NO
            ),
            unsure=sum(
                1 for r in chunk_records
                if r.is_commitment is CommitmentDecision.UNSURE
            ),
            enrich_calls=enriched_count,
        )
        all_records.extend(chunk_records)

    logger.info(
        "commitment_extraction_complete",
        total_chunks=len(chunks),
        total_records=len(all_records),
        positives=sum(
            1 for r in all_records if r.is_commitment is CommitmentDecision.YES
        ),
        rejections=sum(
            1 for r in all_records if r.is_commitment is CommitmentDecision.NO
        ),
    )
    return all_records
