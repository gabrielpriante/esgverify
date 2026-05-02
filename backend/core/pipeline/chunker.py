"""Text chunking utilities for ESGVerify document pipeline.

Splits raw document text into overlapping chunks suitable for LLM processing,
respecting sentence and paragraph boundaries where possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class TextChunk:
    """A single chunk of document text with positional metadata.

    Attributes:
        index: Zero-based position of this chunk in the full sequence.
        text: The raw text content of this chunk.
        char_start: Starting character offset in the original document.
        char_end: Ending character offset in the original document.
    """

    index: int
    text: str
    char_start: int
    char_end: int


def _split_into_units(text: str) -> list[str]:
    """Split text into sentence/paragraph units for boundary-aware chunking.

    Splits on paragraph breaks first, then on sentence-ending punctuation.
    This ensures chunks never break mid-sentence when possible.

    Args:
        text: Raw document text to split into units.

    Returns:
        List of text units (paragraphs or sentences), each stripped of
        leading/trailing whitespace. Empty units are excluded.
    """
    # Split on paragraph boundaries first (two or more newlines)
    paragraphs = re.split(r"\n{2,}", text)

    units: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Further split long paragraphs on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", para)
        units.extend(s.strip() for s in sentences if s.strip())

    return units


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[TextChunk]:
    """Split document text into overlapping chunks.

    Chunks are assembled by accumulating sentence/paragraph units until the
    target ``chunk_size`` (in characters) is reached. The next chunk rewinds
    by ``chunk_overlap`` characters so that context is not lost at boundaries.

    Args:
        text: Full document text to be chunked.
        chunk_size: Maximum characters per chunk. Defaults to
            ``settings.claim_extraction_chunk_size``.
        chunk_overlap: Number of characters to overlap between consecutive
            chunks. Defaults to ``settings.claim_extraction_chunk_overlap``.

    Returns:
        Ordered list of :class:`TextChunk` objects. Returns an empty list if
        ``text`` is blank.

    Example:
        >>> chunks = chunk_text(document_text)
        >>> for chunk in chunks:
        ...     print(chunk.index, len(chunk.text))
    """
    chunk_size = chunk_size or settings.claim_extraction_chunk_size
    chunk_overlap = chunk_overlap or settings.claim_extraction_chunk_overlap

    text = text.strip()
    if not text:
        logger.warning("chunk_text received empty text; returning no chunks")
        return []

    units = _split_into_units(text)
    if not units:
        logger.warning("No text units found after splitting; returning no chunks")
        return []

    chunks: list[TextChunk] = []
    current_units: list[str] = []
    current_len: int = 0
    # Track absolute character position by scanning original text
    scan_pos: int = 0
    chunk_char_start: int = 0

    for unit in units:
        unit_len = len(unit) + 1  # +1 for the space/newline between units

        if current_len + unit_len > chunk_size and current_units:
            # Emit the current chunk
            chunk_text_str = " ".join(current_units)
            chunk_char_end = chunk_char_start + len(chunk_text_str)
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=chunk_text_str,
                    char_start=chunk_char_start,
                    char_end=chunk_char_end,
                )
            )
            logger.debug(
                "chunk_emitted",
                chunk_index=len(chunks) - 1,
                char_start=chunk_char_start,
                char_end=chunk_char_end,
                length=len(chunk_text_str),
            )

            # Build overlap: keep trailing units that fit within chunk_overlap
            overlap_units: list[str] = []
            overlap_len: int = 0
            for prev_unit in reversed(current_units):
                unit_with_sep = len(prev_unit) + 1
                if overlap_len + unit_with_sep > chunk_overlap:
                    break
                overlap_units.insert(0, prev_unit)
                overlap_len += unit_with_sep

            current_units = overlap_units
            current_len = overlap_len
            # Advance char_start to just after the overlap region
            overlap_text = " ".join(overlap_units)
            chunk_char_start = chunk_char_end - len(overlap_text)

        current_units.append(unit)
        current_len += unit_len

    # Emit the final remaining chunk
    if current_units:
        chunk_text_str = " ".join(current_units)
        chunk_char_end = chunk_char_start + len(chunk_text_str)
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=chunk_text_str,
                char_start=chunk_char_start,
                char_end=chunk_char_end,
            )
        )
        logger.debug(
            "chunk_emitted",
            chunk_index=len(chunks) - 1,
            char_start=chunk_char_start,
            char_end=chunk_char_end,
            length=len(chunk_text_str),
        )

    logger.info(
        "chunking_complete",
        total_chunks=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        original_length=len(text),
    )
    return chunks
