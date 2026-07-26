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
# Prompts — TWO STAGE
#
# Empirically, an 8B q4 model on a mid-range GPU degenerates into word-salad
# once the combined system+user prompt exceeds roughly 1,300 tokens, even with
# num_ctx set to 8192 (the effective window is clamped below what is requested,
# most likely by available VRAM). A single ~1,190-token prompt plus a
# ~400-token chunk crossed that line and produced unusable output.
#
# So the work is split. Each call keeps its prompt well under the ceiling:
#
#   STAGE 1  detect   chunk -> which sentences are commitments (yes/no/unsure)
#   STAGE 2  enrich   one sentence -> seven structural fields + verifiability
#
# Stage 2 runs only for sentences stage 1 judged "yes", so cost scales with
# commitment density rather than document length.
#
# Definitions are locked in paper/task_definition.md and notes/decisions.md
# (07/25/2026). Keep both prompts under PROMPT_TOKEN_BUDGET.
# ---------------------------------------------------------------------------

# Rough ceiling per prompt (chars/4). Guarded by a unit test.
PROMPT_TOKEN_BUDGET = 800

_DETECT_SYSTEM_PROMPT = """\
You find ENVIRONMENTAL COMMITMENTS in corporate sustainability reports.

A commitment is a stated intention to take a FUTURE environmental action or
reach a FUTURE environmental outcome. Future intention is the core test: if a
sentence does not point forward in time, it is not a commitment, however
environmental it sounds.

Reject, with the reason given:
  past_action        already done. "In 2023 we reduced water use by 12%."
  values_statement   values or vague aspiration, no measurable action.
                     "Sustainability is at the heart of what we do."
                     "We aim to significantly increase recycling."
  factual_disclosure a fact carrying no promise.
  description        a product, process, or organization as it exists.

Rulings:
- Conditional commitments COUNT. "Subject to government policy, we intend to
  electrify our fleet by 2035" is yes.
- Restated commitments COUNT. "As announced in 2021, we remain committed to
  100% renewable electricity by 2030" is yes, restated yes.
- Third-party validation (e.g. "our targets were validated by SBTi") is
  EVIDENCE, not a commitment: is_commitment no, is_evidence yes.
- A past action with a vague forward clause is past_action. "We installed
  solar at twelve sites and will continue this program" is no.

Judge ONE SENTENCE AT A TIME. Do not split or merge sentences.
Judge every environmental sentence, including the ones you reject.

Return ONLY JSON:
{"commitments": [
  {"text": "<exact sentence>",
   "is_commitment": "yes|no|unsure",
   "rejection_reason": "past_action|values_statement|factual_disclosure|description|null",
   "restated": "yes|no|unsure",
   "is_evidence": "yes|no|unsure"}
]}
If the excerpt has no environmental sentences, return {"commitments": []}
"""

_ENRICH_SYSTEM_PROMPT = """\
You record the structure of ONE environmental commitment sentence.

Fill seven fields. Each takes a status, and a value ONLY when "stated":
  stated          the text gives it        (supply "value")
  not_stated      the text does not say    (omit "value")
  not_applicable  cannot apply here        (omit "value")
  unsure          cannot tell              (omit "value")
Never guess. "unsure" is always allowed.

  target           what is promised
  quantity         how much (number or percentage)
  deadline         by when (year or date)
  baseline         starting point (baseline year)
  business_unit    what part of the business
  emissions_scope  what part of emissions (Scope 1/2/3)

depends_on_outside_factors: yes if the pledge is conditional on policy,
infrastructure, or third parties; otherwise no.

verifiability, judged from the given text and context ONLY:
  strong    specific data, checkable datasets, third-party verification
  moderate  partial supporting evidence, gaps remain
  weak      thin supporting evidence exists
  none      no corroborating evidence at all
  unsure    cannot determine
weak vs none is about PRESENCE of evidence, not how specific the pledge is.

Return ONLY JSON:
{"target": {"status": "stated", "value": "..."},
 "quantity": {"status": "not_stated"},
 "deadline": {"status": "stated", "value": "2030"},
 "baseline": {"status": "not_stated"},
 "business_unit": {"status": "not_stated"},
 "emissions_scope": {"status": "not_stated"},
 "depends_on_outside_factors": "yes|no|unsure",
 "verifiability": "strong|moderate|weak|none|unsure",
 "annotator_notes": "<note or null>"}
"""

_DETECT_USER_TEMPLATE = """\
Judge the environmental sentences in this excerpt:

---
{chunk_text}
---
"""

_ENRICH_USER_TEMPLATE = """\
Sentence:
{text}

Surrounding context:
{context}
"""

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
    text = str(detection.get("text", "")).strip()
    prompt = _ENRICH_USER_TEMPLATE.format(
        text=text,
        context=str(detection.get("context", "")).strip() or text,
    )

    try:
        raw = await _call_ollama(prompt, chunk.index, _ENRICH_SYSTEM_PROMPT)
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


async def extract_commitments(chunks: list[TextChunk]) -> list[ExtractedCommitment]:
    """Extract environmental commitments from document chunks, in two stages.

    Stage 1 sends each chunk to the model and asks only which sentences are
    commitments. Stage 2 sends each positive sentence back on its own and asks
    for its seven structural fields and verifiability level.

    The split exists because a single combined prompt exceeded the effective
    context window of an 8B model on mid-range hardware and produced degenerate
    output. Each stage's prompt stays under ``PROMPT_TOKEN_BUDGET``.

    Records judged NOT a commitment are retained — the benchmark needs
    negatives to measure precision. They skip stage 2 and their structural
    fields are marked ``not_applicable``.

    Malformed model output is logged and skipped; this function never raises
    because of it.

    Args:
        chunks: Ordered list of :class:`TextChunk` objects from
            :func:`backend.core.pipeline.chunker.chunk_text`.

    Returns:
        Flat list of all :class:`ExtractedCommitment` objects across all chunks.

    Example:
        >>> chunks = chunk_text(document_text)
        >>> records = await extract_commitments(chunks)
        >>> positives = [r for r in records if r.is_commitment.value == "yes"]
    """
    if not chunks:
        logger.warning("extract_commitments called with empty chunk list")
        return []

    all_records: list[ExtractedCommitment] = []

    for chunk in chunks:
        log = logger.bind(chunk_index=chunk.index, total_chunks=len(chunks))

        # --- Stage 1: detection ---------------------------------------------
        detect_prompt = _DETECT_USER_TEMPLATE.format(chunk_text=chunk.text)

        try:
            raw_response = await _call_ollama(
                detect_prompt, chunk.index, _DETECT_SYSTEM_PROMPT
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

        detections = _parse_commitments_json(raw_response, chunk)

        # --- Stage 2: enrichment, positives only -----------------------------
        chunk_records: list[ExtractedCommitment] = []
        enriched_count = 0

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            decision = str(detection.get("is_commitment", "")).lower().strip()
            if decision == "yes":
                merged = await _enrich(detection, chunk)
                enriched_count += 1
            else:
                merged = _mark_fields_not_applicable(detection)

            record = _dict_to_commitment(merged, chunk)
            if record is not None:
                chunk_records.append(record)

        log.info(
            "chunk_processed",
            records_found=len(chunk_records),
            positives=sum(
                1 for r in chunk_records
                if r.is_commitment is CommitmentDecision.YES
            ),
            enrich_calls=enriched_count,
            detections_received=len(detections),
        )
        all_records.extend(chunk_records)

    logger.info(
        "commitment_extraction_complete",
        total_chunks=len(chunks),
        total_records=len(all_records),
        positives=sum(
            1 for r in all_records if r.is_commitment is CommitmentDecision.YES
        ),
    )
    return all_records
