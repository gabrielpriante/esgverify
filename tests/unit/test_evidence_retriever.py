"""Unit tests for backend.core.pipeline.evidence_retriever.

Tests cover:
- Collection name sanitisation (_collection_name)
- Evidence type classification (_classify_evidence_type)
- build_document_index: empty chunks, single chunk, multiple chunks,
  collection reset on re-index
- retrieve_evidence: empty claims, missing collection, results below
  threshold filtered out, results sorted by relevance, distance-to-score
  conversion, top_k respected
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from backend.core.models.report import ESGCategory, ExtractedClaim
from backend.core.pipeline.chunker import TextChunk
from backend.core.pipeline.evidence_retriever import (
    _classify_evidence_type,
    _collection_name,
    build_document_index,
    retrieve_evidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chunk(index: int = 0, text: str = "Sample document text.") -> TextChunk:
    return TextChunk(index=index, text=text, char_start=0, char_end=len(text))


def make_claim(text: str = "We achieved carbon neutrality in 2023.") -> ExtractedClaim:
    return ExtractedClaim(
        claim_id=str(uuid.uuid4()),
        text=text,
        context="As part of our sustainability strategy...",
        esg_category=ESGCategory.ENVIRONMENTAL,
        framework_tags=["GHG Protocol"],
        page_reference="p.5",
    )


# ---------------------------------------------------------------------------
# _collection_name
# ---------------------------------------------------------------------------


class TestCollectionName:
    def test_strips_extension(self) -> None:
        assert _collection_name("annual_report.pdf") == "annual_report"

    def test_replaces_spaces_with_underscores(self) -> None:
        name = _collection_name("my report 2023.pdf")
        assert " " not in name

    def test_replaces_dots_in_stem(self) -> None:
        name = _collection_name("report.v2.final.pdf")
        assert name.count(".") == 0

    def test_max_length_63(self) -> None:
        long_name = "a" * 100 + ".pdf"
        assert len(_collection_name(long_name)) <= 63

    def test_min_length_3(self) -> None:
        assert len(_collection_name("ab.pdf")) >= 3

    def test_same_filename_returns_same_name(self) -> None:
        assert _collection_name("report.pdf") == _collection_name("report.pdf")


# ---------------------------------------------------------------------------
# _classify_evidence_type
# ---------------------------------------------------------------------------


class TestClassifyEvidenceType:
    def test_certification_keyword(self) -> None:
        assert _classify_evidence_type("The facility is ISO 14001 certified.") == "certification"

    def test_third_party_keyword(self) -> None:
        assert _classify_evidence_type("According to our external auditor, emissions fell.") == "third_party_reference"

    def test_methodology_keyword(self) -> None:
        assert _classify_evidence_type("We follow the GHG Protocol methodology.") == "methodology"

    def test_data_keyword_percentage(self) -> None:
        assert _classify_evidence_type("Emissions reduced by 30% since 2019.") == "data"

    def test_data_keyword_tonnes(self) -> None:
        assert _classify_evidence_type("We emitted 1,200 tonnes of CO2 equivalent.") == "data"

    def test_no_keywords_returns_none(self) -> None:
        assert _classify_evidence_type("We are committed to a sustainable future.") == "none"

    def test_certification_takes_priority_over_data(self) -> None:
        # Both "certified" and "%" present — certification wins
        text = "Our certified process reduced emissions by 40%."
        assert _classify_evidence_type(text) == "certification"

    def test_case_insensitive(self) -> None:
        assert _classify_evidence_type("CERTIFIED by an independent body.") == "certification"


# ---------------------------------------------------------------------------
# build_document_index
# ---------------------------------------------------------------------------


def make_mock_collection():
    col = MagicMock()
    col.count.return_value = 3
    return col


def make_mock_chroma_client(collection=None):
    client = MagicMock()
    if collection is None:
        collection = make_mock_collection()
    client.create_collection.return_value = collection
    return client


def make_mock_embedder(dim: int = 384):
    import numpy as np
    embedder = MagicMock()
    embedder.encode = MagicMock(
        side_effect=lambda texts, **kwargs: (
            np.random.rand(len(texts), dim)
            if isinstance(texts, list)
            else np.random.rand(dim)
        )
    )
    return embedder


class TestBuildDocumentIndex:
    def test_empty_chunks_does_not_call_chroma(self) -> None:
        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client") as mock_client,
            patch("backend.core.pipeline.evidence_retriever._get_embedder"),
        ):
            build_document_index([], "report.pdf")
            mock_client.assert_not_called()

    def test_creates_collection_with_cosine_space(self) -> None:
        mock_col = make_mock_collection()
        mock_client = make_mock_chroma_client(mock_col)
        mock_embedder = make_mock_embedder()

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            build_document_index([make_chunk()], "report.pdf")

        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args
        assert call_kwargs.kwargs["metadata"]["hnsw:space"] == "cosine"

    def test_upserts_correct_number_of_chunks(self) -> None:
        mock_col = make_mock_collection()
        mock_client = make_mock_chroma_client(mock_col)
        mock_embedder = make_mock_embedder()
        chunks = [make_chunk(i, f"Text {i}") for i in range(5)]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            build_document_index(chunks, "report.pdf")

        mock_col.upsert.assert_called_once()
        upsert_kwargs = mock_col.upsert.call_args.kwargs
        assert len(upsert_kwargs["ids"]) == 5
        assert len(upsert_kwargs["documents"]) == 5

    def test_deletes_existing_collection_before_rebuild(self) -> None:
        mock_col = make_mock_collection()
        mock_client = make_mock_chroma_client(mock_col)
        mock_embedder = make_mock_embedder()

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            build_document_index([make_chunk()], "report.pdf")

        mock_client.delete_collection.assert_called_once()

    def test_chunk_ids_follow_naming_convention(self) -> None:
        mock_col = make_mock_collection()
        mock_client = make_mock_chroma_client(mock_col)
        mock_embedder = make_mock_embedder()
        chunks = [make_chunk(i) for i in range(3)]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            build_document_index(chunks, "report.pdf")

        ids = mock_col.upsert.call_args.kwargs["ids"]
        assert ids == ["chunk_0", "chunk_1", "chunk_2"]


# ---------------------------------------------------------------------------
# retrieve_evidence
# ---------------------------------------------------------------------------


class TestRetrieveEvidence:
    def test_empty_claims_returns_empty_dict(self) -> None:
        with patch("backend.core.pipeline.evidence_retriever._get_chroma_client"):
            result = retrieve_evidence([], "report.pdf")
        assert result == {}

    def test_missing_collection_returns_empty_lists(self) -> None:
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("does not exist")
        mock_embedder = make_mock_embedder()
        claims = [make_claim()]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            result = retrieve_evidence(claims, "report.pdf")

        assert result[claims[0].claim_id] == []

    def test_results_sorted_by_relevance_descending(self) -> None:
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        # distances: lower = more similar; 0.1 → 0.9 score, 0.4 → 0.6 score
        mock_col.query.return_value = {
            "documents": [["passage A", "passage B"]],
            "distances": [[0.4, 0.1]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_col
        mock_embedder = make_mock_embedder()
        claims = [make_claim()]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            result = retrieve_evidence(claims, "report.pdf")

        passages = result[claims[0].claim_id]
        assert passages[0].relevance_score >= passages[1].relevance_score

    def test_passages_below_min_relevance_filtered_out(self) -> None:
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        # distance 0.9 → score 0.1, which is below default threshold 0.25
        mock_col.query.return_value = {
            "documents": [["irrelevant passage"]],
            "distances": [[0.9]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_col
        mock_embedder = make_mock_embedder()
        claims = [make_claim()]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            result = retrieve_evidence(claims, "report.pdf")

        assert result[claims[0].claim_id] == []

    def test_distance_converted_to_relevance_score(self) -> None:
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [["some passage with data and %"]],
            "distances": [[0.3]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_col
        mock_embedder = make_mock_embedder()
        claims = [make_claim()]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            result = retrieve_evidence(claims, "report.pdf")

        passages = result[claims[0].claim_id]
        assert len(passages) == 1
        assert passages[0].relevance_score == pytest.approx(0.7, abs=0.001)

    def test_evidence_ids_are_unique(self) -> None:
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.query.return_value = {
            "documents": [["passage A", "passage B", "passage C"]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_col
        mock_embedder = make_mock_embedder()
        claims = [make_claim()]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            result = retrieve_evidence(claims, "report.pdf")

        passages = result[claims[0].claim_id]
        ids = [p.evidence_id for p in passages]
        assert len(ids) == len(set(ids))

    def test_returns_one_entry_per_claim(self) -> None:
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.query.return_value = {
            "documents": [["relevant passage with % data"]],
            "distances": [[0.2]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_col
        mock_embedder = make_mock_embedder()
        claims = [make_claim(f"Claim {i}") for i in range(3)]

        with (
            patch("backend.core.pipeline.evidence_retriever._get_chroma_client", return_value=mock_client),
            patch("backend.core.pipeline.evidence_retriever._get_embedder", return_value=mock_embedder),
        ):
            result = retrieve_evidence(claims, "report.pdf")

        assert len(result) == 3
        for claim in claims:
            assert claim.claim_id in result
