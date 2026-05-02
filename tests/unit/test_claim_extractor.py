"""Unit tests for backend.core.pipeline.claim_extractor.

Tests cover:
- Successful claim extraction from a well-formed Ollama response
- Graceful handling of malformed JSON (no exception raised)
- Retry behavior on transient Ollama failures
- Empty chunk list handling
- Unknown ESG category coercion to ESGCategory.unknown
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.core.models.report import ESGCategory
from backend.core.pipeline.chunker import TextChunk
from backend.core.pipeline.claim_extractor import (
    _dict_to_extracted_claim,
    _parse_claims_json,
    extract_claims,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chunk(index: int = 0, text: str = "Sample ESG text.") -> TextChunk:
    """Helper to create a minimal TextChunk for testing."""
    return TextChunk(index=index, text=text, char_start=0, char_end=len(text))


def valid_ollama_response(claims: list[dict]) -> str:
    """Build a valid JSON string as the LLM would return it."""
    return json.dumps({"claims": claims})


# ---------------------------------------------------------------------------
# _parse_claims_json
# ---------------------------------------------------------------------------


class TestParseClaimsJson:
    def test_valid_json_returns_claims(self) -> None:
        chunk = make_chunk()
        raw = valid_ollama_response(
            [{"text": "We cut emissions by 30%.", "esg_category": "environmental"}]
        )
        result = _parse_claims_json(raw, chunk)
        assert len(result) == 1
        assert result[0]["text"] == "We cut emissions by 30%."

    def test_empty_claims_list(self) -> None:
        chunk = make_chunk()
        raw = valid_ollama_response([])
        result = _parse_claims_json(raw, chunk)
        assert result == []

    def test_malformed_json_returns_empty_list(self) -> None:
        chunk = make_chunk()
        result = _parse_claims_json("this is not json at all }{", chunk)
        assert result == []

    def test_markdown_fences_are_stripped(self) -> None:
        chunk = make_chunk()
        raw = "```json\n" + valid_ollama_response([{"text": "Net zero by 2040."}]) + "\n```"
        result = _parse_claims_json(raw, chunk)
        assert len(result) == 1

    def test_missing_claims_key_returns_empty(self) -> None:
        chunk = make_chunk()
        result = _parse_claims_json(json.dumps({"data": []}), chunk)
        assert result == []

    def test_claims_not_list_returns_empty(self) -> None:
        chunk = make_chunk()
        result = _parse_claims_json(json.dumps({"claims": "oops"}), chunk)
        assert result == []


# ---------------------------------------------------------------------------
# _dict_to_extracted_claim
# ---------------------------------------------------------------------------


class TestDictToExtractedClaim:
    def test_valid_dict_returns_claim(self) -> None:
        chunk = make_chunk()
        raw = {
            "text": "We achieved carbon neutrality in 2023.",
            "context": "As part of our sustainability strategy...",
            "esg_category": "environmental",
            "framework_tags": ["GHG Protocol", "TCFD"],
            "page_reference": "p.12",
        }
        claim = _dict_to_extracted_claim(raw, chunk)
        assert claim is not None
        assert claim.text == "We achieved carbon neutrality in 2023."
        assert claim.esg_category == ESGCategory.environmental
        assert "TCFD" in claim.framework_tags
        assert claim.page_reference == "p.12"

    def test_missing_text_returns_none(self) -> None:
        chunk = make_chunk()
        claim = _dict_to_extracted_claim({"esg_category": "social"}, chunk)
        assert claim is None

    def test_unknown_category_coerces_to_unknown(self) -> None:
        chunk = make_chunk()
        raw = {
            "text": "Some claim.",
            "esg_category": "made_up_category",
            "framework_tags": [],
        }
        claim = _dict_to_extracted_claim(raw, chunk)
        assert claim is not None
        assert claim.esg_category == ESGCategory.unknown

    def test_missing_framework_tags_defaults_to_empty_list(self) -> None:
        chunk = make_chunk()
        raw = {"text": "Some governance claim.", "esg_category": "governance"}
        claim = _dict_to_extracted_claim(raw, chunk)
        assert claim is not None
        assert claim.framework_tags == []

    def test_claim_id_is_unique(self) -> None:
        chunk = make_chunk()
        raw = {"text": "A.", "esg_category": "social"}
        c1 = _dict_to_extracted_claim(raw, chunk)
        c2 = _dict_to_extracted_claim(raw, chunk)
        assert c1 is not None and c2 is not None
        assert c1.claim_id != c2.claim_id

    def test_page_reference_falls_back_to_chunk_index(self) -> None:
        chunk = make_chunk(index=3)
        raw = {"text": "A claim.", "esg_category": "social", "page_reference": None}
        claim = _dict_to_extracted_claim(raw, chunk)
        assert claim is not None
        assert claim.page_reference == "chunk_3"


# ---------------------------------------------------------------------------
# extract_claims (integration-level with mocked _call_ollama)
# ---------------------------------------------------------------------------


MOCK_VALID_RESPONSE = valid_ollama_response(
    [
        {
            "text": "We reduced Scope 1 emissions by 40%.",
            "context": "As stated in our 2023 Climate Report...",
            "esg_category": "environmental",
            "framework_tags": ["GHG Protocol"],
            "page_reference": "p.5",
        },
        {
            "text": "Board diversity reached 45% women in 2023.",
            "context": "Our governance improvements include...",
            "esg_category": "governance",
            "framework_tags": ["GRI"],
            "page_reference": "p.22",
        },
    ]
)


@pytest.mark.asyncio
async def test_extract_claims_success() -> None:
    """extract_claims returns parsed claims on a valid Ollama response."""
    chunks = [make_chunk(index=0)]

    with patch(
        "backend.core.pipeline.claim_extractor._call_ollama",
        new=AsyncMock(return_value=MOCK_VALID_RESPONSE),
    ):
        claims = await extract_claims(chunks)

    assert len(claims) == 2
    assert claims[0].esg_category == ESGCategory.environmental
    assert claims[1].esg_category == ESGCategory.governance


@pytest.mark.asyncio
async def test_extract_claims_malformed_json_does_not_raise() -> None:
    """extract_claims skips chunks with malformed JSON — never raises."""
    chunks = [make_chunk(index=0), make_chunk(index=1, text="Second chunk.")]

    responses = ["not valid json {{", MOCK_VALID_RESPONSE]

    with patch(
        "backend.core.pipeline.claim_extractor._call_ollama",
        new=AsyncMock(side_effect=responses),
    ):
        claims = await extract_claims(chunks)

    # First chunk failed silently; second chunk succeeded
    assert len(claims) == 2


@pytest.mark.asyncio
async def test_extract_claims_empty_chunks_returns_empty() -> None:
    """extract_claims on an empty list returns [] without calling Ollama."""
    with patch(
        "backend.core.pipeline.claim_extractor._call_ollama",
        new=AsyncMock(),
    ) as mock_ollama:
        claims = await extract_claims([])

    assert claims == []
    mock_ollama.assert_not_called()


@pytest.mark.asyncio
async def test_extract_claims_retry_then_succeed() -> None:
    """extract_claims succeeds after a transient failure on retry."""
    import httpx

    chunks = [make_chunk(index=0)]

    # First call raises a timeout; second call succeeds
    call_count = 0

    async def flaky_ollama(prompt: str, chunk_index: int) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.TimeoutException("timeout")
        return MOCK_VALID_RESPONSE

    with patch(
        "backend.core.pipeline.claim_extractor._call_ollama",
        new=flaky_ollama,
    ):
        # _call_ollama itself has the tenacity decorator, so we test the
        # outer extract_claims behavior when _call_ollama exhausts retries.
        # Here we simulate it by making the mock raise then succeed — in
        # production tenacity wraps _call_ollama, not our mock, so we patch
        # at the extract_claims level to verify the skip-on-failure path.
        claims = await extract_claims(chunks)

    # call_count == 2 means the retry logic triggered internally
    assert call_count == 2
    assert len(claims) == 2


@pytest.mark.asyncio
async def test_extract_claims_all_retries_exhausted_skips_chunk() -> None:
    """extract_claims skips a chunk when all Ollama retries are exhausted."""
    import httpx

    chunks = [make_chunk(index=0)]

    with patch(
        "backend.core.pipeline.claim_extractor._call_ollama",
        new=AsyncMock(side_effect=httpx.HTTPError("connection refused")),
    ):
        claims = await extract_claims(chunks)

    # Chunk skipped, no crash
    assert claims == []


@pytest.mark.asyncio
async def test_extract_claims_multiple_chunks_aggregated() -> None:
    """Claims from multiple chunks are combined into a single flat list."""
    chunks = [make_chunk(index=i) for i in range(3)]

    with patch(
        "backend.core.pipeline.claim_extractor._call_ollama",
        new=AsyncMock(return_value=MOCK_VALID_RESPONSE),
    ):
        claims = await extract_claims(chunks)

    # 2 claims per chunk × 3 chunks = 6
    assert len(claims) == 6
