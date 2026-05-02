"""Evidence retrieval for ESG claims via sentence-transformers and ChromaDB.

Stage 4 of the ESGVerify pipeline. For each :class:`ExtractedClaim`, finds the
most semantically similar passages from the source document using dense vector
search. Results are returned as :class:`SupportingEvidence` objects with
relevance scores derived from cosine similarity.

Embedding model: ``all-MiniLM-L6-v2`` (sentence-transformers)
  - ~80 MB download on first use
  - 384-dimensional embeddings
  - Strong performance on semantic similarity tasks

Vector store: ChromaDB (local, persistent)
  - Collection is keyed per document filename to avoid cross-document bleed
  - Directory is created automatically on first run
"""

from __future__ import annotations

import uuid
from pathlib import Path

import chromadb
import structlog
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from backend.core.config import settings
from backend.core.models.report import ExtractedClaim, SupportingEvidence
from backend.core.pipeline.chunker import TextChunk

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Embedding model — loaded once at module level to avoid repeated I/O
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_TOP_K = 5                   # Number of passages to retrieve per claim
_MIN_RELEVANCE_SCORE = 0.25  # Discard passages below this cosine similarity

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    """Return the shared SentenceTransformer instance, loading it on first call.

    Returns:
        Loaded :class:`SentenceTransformer` model.
    """
    global _embedder
    if _embedder is None:
        logger.info("loading_embedding_model", model=_EMBEDDING_MODEL_NAME)
        _embedder = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        logger.info("embedding_model_loaded", model=_EMBEDDING_MODEL_NAME)
    return _embedder


# ---------------------------------------------------------------------------
# ChromaDB client — created once at module level
# ---------------------------------------------------------------------------

_chroma_client: chromadb.ClientAPI | None = None


def _get_chroma_client() -> chromadb.ClientAPI:
    """Return the shared ChromaDB persistent client, creating it on first call.

    The persistence directory is taken from ``settings.chroma_persist_directory``
    and created automatically if it does not exist.

    Returns:
        Persistent :class:`chromadb.ClientAPI` instance.
    """
    global _chroma_client
    if _chroma_client is None:
        persist_path = Path(settings.chroma_persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        logger.info("initializing_chromadb", path=str(persist_path))
        _chroma_client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("chromadb_initialized", path=str(persist_path))
    return _chroma_client


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _collection_name(filename: str) -> str:
    """Derive a safe ChromaDB collection name from a document filename.

    ChromaDB collection names must be 3–63 characters, start and end with
    an alphanumeric character, and contain only alphanumerics, hyphens, or
    underscores.

    Args:
        filename: Original document filename (e.g. ``"annual_report_2023.pdf"``).

    Returns:
        Sanitised collection name string.
    """
    import re
    stem = Path(filename).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    # Ensure length constraints (3–63 chars)
    safe = safe[:63].strip("_-")
    if len(safe) < 3:
        safe = safe.ljust(3, "0")
    return safe


def build_document_index(chunks: list[TextChunk], filename: str) -> None:
    """Embed document chunks and store them in ChromaDB.

    Creates (or resets) a collection keyed to ``filename``, embeds all chunk
    texts using ``all-MiniLM-L6-v2``, and upserts them into ChromaDB. Safe to
    call multiple times — existing collections for the same filename are deleted
    and rebuilt to ensure freshness.

    Args:
        chunks: Ordered list of :class:`TextChunk` objects from the document.
        filename: Source document filename, used to name the collection.
    """
    if not chunks:
        logger.warning("build_document_index called with empty chunks", filename=filename)
        return

    client = _get_chroma_client()
    embedder = _get_embedder()
    collection_name = _collection_name(filename)

    # Delete existing collection for this document to ensure fresh index
    try:
        client.delete_collection(collection_name)
        logger.debug("existing_collection_deleted", collection=collection_name)
    except Exception:  # noqa: BLE001
        pass  # Collection didn't exist yet — that's fine

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )

    texts = [chunk.text for chunk in chunks]
    ids = [f"chunk_{chunk.index}" for chunk in chunks]
    metadatas = [
        {
            "chunk_index": chunk.index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }
        for chunk in chunks
    ]

    logger.info("embedding_chunks", count=len(texts), filename=filename)
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    logger.info(
        "document_index_built",
        filename=filename,
        collection=collection_name,
        chunk_count=len(chunks),
    )


def retrieve_evidence(
    claims: list[ExtractedClaim],
    filename: str,
    top_k: int = _TOP_K,
    min_relevance: float = _MIN_RELEVANCE_SCORE,
) -> dict[str, list[SupportingEvidence]]:
    """Retrieve supporting evidence passages for each claim via vector search.

    Queries the ChromaDB collection for ``filename`` using each claim's text
    as the query vector. Returns a mapping from ``claim_id`` to a list of
    :class:`SupportingEvidence` objects sorted by descending relevance.

    Passages below ``min_relevance`` are discarded. If the collection for
    ``filename`` does not exist, logs a warning and returns empty lists for
    all claims.

    Args:
        claims: List of :class:`ExtractedClaim` objects to retrieve evidence for.
        filename: Source document filename — must match the filename used in
            :func:`build_document_index`.
        top_k: Maximum number of evidence passages to return per claim.
        min_relevance: Minimum cosine similarity threshold (0.0–1.0).

    Returns:
        Dict mapping ``claim_id`` → ``list[SupportingEvidence]``, one entry
        per claim. Claims with no evidence above the threshold map to ``[]``.

    Example:
        >>> evidence_map = retrieve_evidence(claims, "annual_report_2023.pdf")
        >>> for claim_id, evidence in evidence_map.items():
        ...     print(claim_id, len(evidence))
    """
    if not claims:
        logger.warning("retrieve_evidence called with empty claims list")
        return {}

    client = _get_chroma_client()
    embedder = _get_embedder()
    collection_name = _collection_name(filename)

    try:
        collection = client.get_collection(collection_name)
    except Exception:  # noqa: BLE001
        logger.warning(
            "collection_not_found",
            collection=collection_name,
            filename=filename,
        )
        return {claim.claim_id: [] for claim in claims}

    evidence_map: dict[str, list[SupportingEvidence]] = {}

    for claim in claims:
        log = logger.bind(claim_id=claim.claim_id)

        query_embedding = embedder.encode(claim.text, show_progress_bar=False).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "distances"],
        )

        passages: list[SupportingEvidence] = []

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_text, distance in zip(documents, distances):
            # ChromaDB cosine distance = 1 - cosine_similarity
            # Convert back to similarity score in [0, 1]
            relevance_score = round(max(0.0, 1.0 - distance), 4)

            if relevance_score < min_relevance:
                continue

            # Classify evidence type from content heuristics
            evidence_type = _classify_evidence_type(doc_text)

            passages.append(
                SupportingEvidence(
                    evidence_id=str(uuid.uuid4()),
                    text=doc_text,
                    evidence_type=evidence_type,
                    relevance_score=relevance_score,
                )
            )

        # Sort descending by relevance
        passages.sort(key=lambda e: e.relevance_score, reverse=True)
        evidence_map[claim.claim_id] = passages

        log.info(
            "evidence_retrieved",
            passages_found=len(passages),
            top_score=passages[0].relevance_score if passages else None,
        )

    logger.info(
        "retrieval_complete",
        total_claims=len(claims),
        filename=filename,
    )
    return evidence_map


# ---------------------------------------------------------------------------
# Evidence type heuristics
# ---------------------------------------------------------------------------

_DATA_KEYWORDS = {
    "%", "tonnes", "kwh", "mwh", "gwh", "metric tons", "gigawatts", "megawatts",
    "reduced by", "increased by", "decreased by", "scope 1", "scope 2", "scope 3",
    "million", "billion", "thousand", "cubic meters", "liters", "gallons",
}
_CERTIFICATION_KEYWORDS = {
    "certified", "certification", "iso ", "b corp", "leed", "verified",
    "third-party", "audited", "accredited", "validated",
}
_METHODOLOGY_KEYWORDS = {
    "methodology", "protocol", "framework", "standard", "ghg protocol",
    "science-based", "sbti", "tcfd", "gri", "sasb", "cdp",
}
_THIRD_PARTY_KEYWORDS = {
    "according to", "reported by", "as assessed by", "independent",
    "external auditor", "deloitte", "pwc", "kpmg", "ey ", "ernst",
}


def _classify_evidence_type(text: str) -> str:
    """Heuristically classify a passage into an evidence type.

    Checks for keyword patterns in order of specificity: certification >
    third-party reference > methodology > data > none.

    Args:
        text: The evidence passage text.

    Returns:
        One of ``"data"``, ``"certification"``, ``"methodology"``,
        ``"third_party_reference"``, or ``"none"``.
    """
    lower = text.lower()

    if any(kw in lower for kw in _CERTIFICATION_KEYWORDS):
        return "certification"
    if any(kw in lower for kw in _THIRD_PARTY_KEYWORDS):
        return "third_party_reference"
    if any(kw in lower for kw in _METHODOLOGY_KEYWORDS):
        return "methodology"
    if any(kw in lower for kw in _DATA_KEYWORDS):
        return "data"
    return "none"
