"""ESG claim analysis via local Ollama LLM.

For each :class:`ExtractedClaim` produced by Stage 2, sends the claim text
and its surrounding context back to Ollama. The model determines:

- **SubstantiationLevel** — how well the claim is backed by evidence in the
  document (``strong``, ``moderate``, ``weak``, ``none``).
- **RiskLevel** — greenwashing risk for the claim (``low``, ``medium``, ``high``).
- **SupportingEvidence** — specific data points found in context that relate to
  the claim, each with an evidence type and a relevance score.
- **ClaimAnalysis** — the full structured output combining the above.

Recommended model: ``llama3.1:8b-instruct-q4_K_M``
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
    ClaimAnalysis,
    ExtractedClaim,
    RiskLevel,
    SubstantiationLevel,
    SupportingEvidence,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an ESG (Environmental, Social, Governance) greenwashing analysis specialist.
Your task is to evaluate a single ESG claim extracted from a corporate document and
determine whether it is substantiated by evidence.

Rules:
- Base your analysis ONLY on the claim text and context provided — do not invent
  supporting evidence.
- substantiation_level values:
    "strong"   — claim backed by specific, quantified data, verified certification,
                 or a named methodology.
    "moderate" — claim backed by partial data or a vague reference to methodology.
    "weak"     — claim is an assertion with no supporting evidence in the context.
    "none"     — no attempt at substantiation; purely marketing language.
- risk_level values:
    "low"    — claim is credible and well-supported.
    "medium" — claim is plausible but lacks full substantiation.
    "high"   — claim is likely unsubstantiated, vague, or uses known greenwashing
               tactics.
- For each piece of evidence found in the context, classify its type as one of:
    "data" | "certification" | "methodology" | "third_party_reference" | "none"
- relevance_score is a float from 0.0 to 1.0 reflecting how directly the evidence
  supports or contradicts the claim.
- confidence is a float from 0.0 to 1.0 reflecting your certainty in the overall
  assessment.
- Return ONLY a valid JSON object — no prose, no markdown, no code fences.

Response format (strict):
{
  "substantiation_level": "<strong|moderate|weak|none>",
  "risk_level": "<low|medium|high>",
  "gap_explanation": "<2-4 sentence explanation of the gap between claim and evidence>",
  "confidence": <0.0-1.0>,
  "supporting_evidence": [
    {
      "text": "<verbatim or near-verbatim excerpt from context>",
      "evidence_type": "<data|certification|methodology|third_party_reference|none>",
      "relevance_score": <0.0-1.0>
    }
  ]
}

If no supporting evidence is found in the context, return an empty list for
"supporting_evidence".
"""

_USER_PROMPT_TEMPLATE = """\
Evaluate the following ESG claim:

CLAIM: {claim_text}

CONTEXT (surrounding text from the document):
---
{claim_context}
---

ESG Category: {esg_category}
Framework Tags: {framework_tags}
"""

# ---------------------------------------------------------------------------
# Retry-decorated Ollama call
# ---------------------------------------------------------------------------


import logging as stdlib_logging

@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
async def _call_ollama(prompt: str, claim_id: str) -> str:
    """Send a claim analysis prompt to Ollama and return the raw response text.

    Retries up to 3 times with exponential backoff on network or timeout
    errors. Uses the non-streaming ``/api/chat`` endpoint.

    Args:
        prompt: The user-turn message to send to the model.
        claim_id: Used only for logging context.

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
            "temperature": 0,
            "num_predict": 1024,
        },
    }

    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
    ) as client:
        logger.debug("ollama_request_sent", claim_id=claim_id, model=settings.ollama_model)
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()

    data = response.json()
    content: str = data["message"]["content"]
    logger.debug("ollama_response_received", claim_id=claim_id, response_length=len(content))
    return content


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_analysis_json(raw: str, claim_id: str) -> dict[str, Any] | None:
    """Attempt to parse the LLM's raw output as a claim analysis dict.

    Strips any accidental markdown fences before parsing. Returns ``None``
    (and logs a warning) on any parse failure — never raises.

    Args:
        raw: Raw string returned by the LLM.
        claim_id: The source claim ID, used for logging context.

    Returns:
        Parsed analysis dict, or ``None`` on failure.
    """
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
            claim_id=claim_id,
            error=str(exc),
            raw_preview=cleaned[:200],
        )
        return None

    if not isinstance(parsed, dict):
        logger.warning("unexpected_json_shape", claim_id=claim_id, type=type(parsed).__name__)
        return None

    return parsed


def _dict_to_claim_analysis(
    raw: dict[str, Any],
    claim: ExtractedClaim,
) -> ClaimAnalysis | None:
    """Convert a raw analysis dict from the LLM into a :class:`ClaimAnalysis`.

    Validates required fields and coerces enum values, defaulting to safe
    fallbacks on invalid values. Returns ``None`` (and logs a warning) if
    the dict cannot be converted.

    Args:
        raw: Analysis dict as returned by the LLM.
        claim: The source :class:`ExtractedClaim` being analyzed.

    Returns:
        A validated :class:`ClaimAnalysis`, or ``None`` on failure.
    """
    try:
        # --- SubstantiationLevel ---
        raw_sub = str(raw.get("substantiation_level", "none")).lower().strip()
        try:
            substantiation_level = SubstantiationLevel(raw_sub)
        except ValueError:
            logger.warning(
                "unknown_substantiation_level",
                claim_id=claim.claim_id,
                raw_value=raw_sub,
            )
            substantiation_level = SubstantiationLevel.NONE

        # --- RiskLevel ---
        raw_risk = str(raw.get("risk_level", "medium")).lower().strip()
        try:
            risk_level = RiskLevel(raw_risk)
        except ValueError:
            logger.warning(
                "unknown_risk_level",
                claim_id=claim.claim_id,
                raw_value=raw_risk,
            )
            risk_level = RiskLevel.MEDIUM

        # --- Gap explanation ---
        gap_explanation = str(raw.get("gap_explanation", "")).strip() or "No explanation provided."

        # --- Confidence ---
        try:
            confidence = float(raw.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # clamp to [0.0, 1.0]
        except (TypeError, ValueError):
            logger.warning("invalid_confidence_value", claim_id=claim.claim_id)
            confidence = 0.5

        # --- SupportingEvidence ---
        valid_evidence_types = {"data", "certification", "methodology", "third_party_reference", "none"}
        evidence_list: list[SupportingEvidence] = []
        raw_evidence = raw.get("supporting_evidence", [])
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                evidence_type = str(item.get("evidence_type", "none")).strip().lower()
                if evidence_type not in valid_evidence_types:
                    evidence_type = "none"

                try:
                    relevance_score = float(item.get("relevance_score", 0.5))
                    relevance_score = max(0.0, min(1.0, relevance_score))
                except (TypeError, ValueError):
                    relevance_score = 0.5

                evidence_list.append(
                    SupportingEvidence(
                        evidence_id=str(uuid.uuid4()),
                        text=text,
                        evidence_type=evidence_type,
                        relevance_score=relevance_score,
                    )
                )

        return ClaimAnalysis(
            claim=claim,
            evidence=evidence_list,
            substantiation_level=substantiation_level,
            risk_level=risk_level,
            gap_explanation=gap_explanation,
            confidence=confidence,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "analysis_conversion_failed",
            claim_id=claim.claim_id,
            error=str(exc),
        )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def analyze_claims(claims: list[ExtractedClaim]) -> list[ClaimAnalysis]:
    """Analyze a list of extracted ESG claims for substantiation and greenwashing risk.

    For each claim, sends the claim text and context to the local Ollama LLM,
    parses the JSON response, and converts it into a :class:`ClaimAnalysis`
    object. Claims that fail after retries are skipped and logged — this
    function never raises due to bad LLM output or network errors.

    Args:
        claims: List of :class:`ExtractedClaim` objects produced by
            :func:`backend.core.pipeline.claim_extractor.extract_claims`.

    Returns:
        Flat list of :class:`ClaimAnalysis` objects, one per successfully
        analyzed claim. May be shorter than the input list if some claims
        fail.

    Example:
        >>> from backend.core.pipeline.claim_analyzer import analyze_claims
        >>> analyses = await analyze_claims(extracted_claims)
        >>> high_risk = [a for a in analyses if a.risk_level == RiskLevel.HIGH]
    """
    if not claims:
        logger.warning("analyze_claims called with empty claims list")
        return []

    all_analyses: list[ClaimAnalysis] = []

    for claim in claims:
        log = logger.bind(claim_id=claim.claim_id, total_claims=len(claims))

        prompt = _USER_PROMPT_TEMPLATE.format(
            claim_text=claim.text,
            claim_context=claim.context,
            esg_category=claim.esg_category.value,
            framework_tags=", ".join(claim.framework_tags) if claim.framework_tags else "none",
        )

        try:
            raw_response = await _call_ollama(prompt, claim.claim_id)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            log.error(
                "ollama_call_failed_after_retries",
                error=str(exc),
                claim_skipped=True,
            )
            continue

        parsed = _parse_analysis_json(raw_response, claim.claim_id)
        if parsed is None:
            log.warning("claim_analysis_skipped_bad_json")
            continue

        analysis = _dict_to_claim_analysis(parsed, claim)
        if analysis is not None:
            all_analyses.append(analysis)
            log.info(
                "claim_analyzed",
                substantiation_level=analysis.substantiation_level.value,
                risk_level=analysis.risk_level.value,
                confidence=analysis.confidence,
            )
        else:
            log.warning("claim_analysis_skipped_conversion_failed")

    logger.info(
        "analysis_complete",
        total_claims=len(claims),
        total_analyses=len(all_analyses),
        skipped=len(claims) - len(all_analyses),
    )
    return all_analyses
