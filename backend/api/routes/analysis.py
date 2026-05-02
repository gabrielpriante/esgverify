"""
Analysis endpoints.

POST /analyze   — Upload a document and receive a full ESG analysis report.

The endpoint is intentionally thin. It:
  1. Validates and reads the uploaded file
  2. Hands off to the analysis pipeline
  3. Returns the structured report

All business logic lives in backend/core/pipeline/.
"""

from typing import Annotated

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.core.config import settings
from backend.core.models.report import AnalysisReport
from backend.core.pipeline.orchestrator import run_pipeline
from backend.utils.document_parser import parse_uploaded_file

router = APIRouter()
logger = structlog.get_logger(__name__)

MAX_BYTES = settings.max_document_size_mb * 1024 * 1024


@router.post(
    "/analyze",
    response_model=AnalysisReport,
    status_code=status.HTTP_200_OK,
    summary="Analyze an ESG document",
    description=(
        "Upload a PDF, DOCX, or TXT file. "
        "Returns a structured report with extracted claims, "
        "supporting evidence, and substantiation scores."
    ),
)
async def analyze_document(
    file: Annotated[UploadFile, File(description="ESG document to analyze")]
) -> AnalysisReport:
    """Validate, parse, and analyze an uploaded ESG document.

    Args:
        file: Uploaded file from the multipart form request.

    Returns:
        :class:`AnalysisReport` with all claims, evidence, and summary.

    Raises:
        HTTPException 415: Unsupported file type.
        HTTPException 413: File exceeds size limit.
        HTTPException 422: Document could not be parsed.
        HTTPException 500: Unexpected pipeline failure.
    """
    # --- Validate file type ---
    extension = _get_extension(file.filename)
    if extension not in settings.supported_file_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{extension}'. "
                f"Accepted: {settings.supported_file_types}"
            ),
        )

    # --- Validate file size ---
    content_bytes = await file.read()
    if len(content_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_document_size_mb} MB.",
        )

    logger.info("analysis_request_received", filename=file.filename, bytes=len(content_bytes))

    # --- Parse document to plain text ---
    try:
        document = await parse_uploaded_file(
            filename=file.filename,
            content_bytes=content_bytes,
            extension=extension,
        )
    except Exception as exc:
        logger.error("document_parse_failed", filename=file.filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse document: {exc}",
        ) from exc

    # --- Run analysis pipeline ---
    try:
        report = await run_pipeline(document)
    except Exception as exc:
        logger.error("pipeline_failed", filename=file.filename, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis pipeline failed. Check server logs for details.",
        ) from exc

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_extension(filename: str | None) -> str:
    """Extract the lowercase file extension without the leading dot.

    Args:
        filename: Original filename string, may be None.

    Returns:
        Lowercase extension string, or empty string if not determinable.
    """
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()
