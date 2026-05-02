"""ESG claim extraction via local Ollama LLM.

Sends document text chunks to a locally running Ollama instance and parses
the responses into structured :class:`ExtractedClaim` objects.

Recommended model: ``llama3.1:8b-instruct-q4_K_M``
  - Quantization: Q4_K_M (~4.7 GB VRAM)
  - Fits comfortably within 6 GB VRAM budget on 8 GB cards
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
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.core.config import settings
from backend.core.models.report import ESGCategory, ExtractedClaim
from backend.core.pipeline.chunker import TextChunk

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an ESG (Environmental, Social, Governance) claim extraction specialist.
Your task is to identify explicit ESG claims in corporate documents and return
them as structured JSON.

Rules:
- Extract only claims that make a specific environmental, social, or governance
  assertion (e.g. "We reduced Scope 1 emissions by 30%", "Our supply chain is
  fully traceable", "We achieved gender pay parity in 2023").
- Ignore generic mission statements, product descriptions, or financial data
  unless they contain a specific ESG claim.
- For each claim assign a category: "environmental", "social", "governance",
  or "unknown".
- Identify any framework tags that apply (e.g. "GHG Protocol", "GRI", "TCFD",
  "UN SDGs", "CDP", "SASB"). Use an empty list if none apply.
- Return ONLY a valid JSON object — no prose, no markdown, no code fences.

Response format (strict):
{
  "claims": [
    {
      "text": "<exact or near-exact quote of the claim>",
      "context": "<one or two surrounding sentences that give context>",
      "esg_category": "<environmental|social|governance|unknown>",
      "framework_tags": ["<tag1>", "<tag2>"],
      "page_reference": null
    }
  ]
}

If no ESG claims are found, return: {"claims": []}
"""

_USER_PROMPT_TEMPLATE = """\
Extract all ESG claims from the following document excerpt (chunk {chunk_index}):

---
{chunk_text}
---
"""

# ---------------------------------------------------------------------------
# Retry-decorated Ollama call
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, "warning"),  # type: ignore[arg-type]
    reraise=True,
)
async def _call_ollama(prompt: str, chunk_index: int) -> str:
    """Send a prompt to Ollama and return the raw response text.

    Retries up to 3 times with exponential backoff on network or timeout
    errors. Uses the non-streaming ``/api/chat`` endpoint so the full
    response arrives as a single JSON object.

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
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            # Keep temperature at 0 — we want deterministic structured output
            "temperature": 0,
            "num_predict": 1024,
        },
    }

    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
    ) as client:
        logger.debug("ollama_request_sent", chunk_index=chunk_index, model=settings.ollama_model)
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()

    data = response.json()
    content: str = data["message"]["content"]
    logger.debug("ollama_response_received", chunk_index=chunk_index, response_length=len(content))
    return content


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_claims_json(raw: str, chunk: TextChunk) -> list[dict[str, Any]]:
    """Attempt to parse the LLM's raw output as a JSON claims list.

    Strips any accidental markdown fences before parsing. Returns an empty
    list (and logs a warning) on any parse failure — never raises.

    Args:
        raw: Raw string returned by the LLM.
        chunk: The source chunk, used for logging context.

    Returns:
        List of raw claim dicts, or an empty list on failure.
    """
    # Strip accidental markdown fences the model may produce despite instructions
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

    if not isinstance(parsed, dict) or "claims" not in parsed:
        logger.warning(
            "unexpected_json_shape",
            chunk_index=chunk.index,
            keys=list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__,
        )
        return []

    claims = parsed["claims"]
    if not isinstance(claims, list):
        logger.warning("claims_field_not_list", chunk_index=chunk.index)
        return []

    return claims


def _dict_to_extracted_claim(raw_claim: dict[str, Any], chunk: TextChunk) -> ExtractedClaim | None:
    """Convert a raw claim dict from the LLM into an :class:`ExtractedClaim`.

    Validates required fields and coerces ``esg_category`` to the
    :class:`ESGCategory` enum, defaulting to ``unknown`` on invalid values.
    Returns ``None`` (and logs a warning) if the dict is malformed.

    Args:
        raw_claim: A single claim dict as returned by the LLM.
        chunk: The source chunk, used to populate ``page_reference`` fallback
            and for logging context.

    Returns:
        A validated :class:`ExtractedClaim`, or ``None`` on failure.
    """
    try:
        text = str(raw_claim.get("text", "")).strip()
        if not text:
            logger.warning("claim_missing_text", chunk_index=chunk.index)
            return None

        # Coerce category — fall back to unknown rather than crashing
        raw_cat = str(raw_claim.get("esg_category", "unknown")).lower().strip()
        try:
            category = ESGCategory(raw_cat)
        except ValueError:
            logger.warning(
                "unknown_esg_category",
                chunk_index=chunk.index,
                raw_category=raw_cat,
            )
            category = ESGCategory.unknown

        framework_tags: list[str] = []
        raw_tags = raw_claim.get("framework_tags", [])
        if isinstance(raw_tags, list):
            framework_tags = [str(t) for t in raw_tags if t]

        page_ref: str | None = raw_claim.get("page_reference") or f"chunk_{chunk.index}"

        return ExtractedClaim(
            claim_id=str(uuid.uuid4()),
            text=text,
            context=str(raw_claim.get("context", "")).strip() or text,
            esg_category=category,
            framework_tags=framework_tags,
            page_reference=page_ref,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "claim_conversion_failed",
            chunk_index=chunk.index,
            error=str(exc),
            raw_claim=str(raw_claim)[:200],
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_claims(chunks: list[TextChunk]) -> list[ExtractedClaim]:
    """Extract ESG claims from a list of document text chunks.

    For each chunk, sends the text to the local Ollama LLM, parses the
    JSON response, and converts raw dicts into :class:`ExtractedClaim`
    objects. Malformed responses are logged and skipped — this function
    never raises due to bad LLM output.

    Args:
        chunks: Ordered list of :class:`TextChunk` objects produced by
            :func:`backend.core.pipeline.chunker.chunk_text`.

    Returns:
        Flat list of all :class:`ExtractedClaim` objects found across all
        chunks. May be empty if no claims are detected or all chunks fail.

    Example:
        >>> from backend.core.pipeline.chunker import chunk_text
        >>> from backend.core.pipeline.claim_extractor import extract_claims
        >>> chunks = chunk_text(document_text)
        >>> claims = await extract_claims(chunks)
        >>> print(f"Found {len(claims)} claims")
    """
    if not chunks:
        logger.warning("extract_claims called with empty chunk list")
        return []

    all_claims: list[ExtractedClaim] = []

    for chunk in chunks:
        log = logger.bind(chunk_index=chunk.index, total_chunks=len(chunks))

        prompt = _USER_PROMPT_TEMPLATE.format(
            chunk_index=chunk.index,
            chunk_text=chunk.text,
        )

        try:
            raw_response = await _call_ollama(prompt, chunk.index)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.error(
                "ollama_call_failed_after_retries",
                error=str(exc),
                chunk_skipped=True,
            )
            continue

        raw_claim_dicts = _parse_claims_json(raw_response, chunk)

        chunk_claims: list[ExtractedClaim] = []
        for raw_claim in raw_claim_dicts:
            claim = _dict_to_extracted_claim(raw_claim, chunk)
            if claim is not None:
                chunk_claims.append(claim)

        log.info(
            "chunk_processed",
            claims_found=len(chunk_claims),
            raw_dicts_received=len(raw_claim_dicts),
        )
        all_claims.extend(chunk_claims)

    logger.info(
        "extraction_complete",
        total_chunks=len(chunks),
        total_claims=len(all_claims),
    )
    return all_claims
