"""
Document parsing utilities.

Converts uploaded files (PDF, DOCX, TXT) into a DocumentInput model
containing clean, extracted text. Each parser is isolated so that
adding a new format means adding one function here, nothing else.

Dependencies:
  - pymupdf  (imported as fitz) — PDF extraction
  - python-docx                 — DOCX extraction
"""

import structlog

from backend.core.models.report import DocumentInput

logger = structlog.get_logger(__name__)


async def parse_uploaded_file(
    filename: str,
    content_bytes: bytes,
    extension: str,
) -> DocumentInput:
    """
    Dispatch to the correct parser based on file extension.

    Args:
        filename:      Original filename (used for logging and the model).
        content_bytes: Raw file bytes from the upload.
        extension:     Lowercase extension without dot — 'pdf', 'docx', or 'txt'.

    Returns:
        DocumentInput with the extracted text ready for the pipeline.

    Raises:
        ValueError: If the extension is not supported (should be caught upstream).
    """
    parsers = {
        "pdf":  _parse_pdf,
        "docx": _parse_docx,
        "txt":  _parse_txt,
    }

    parser = parsers.get(extension)
    if parser is None:
        raise ValueError(f"No parser registered for extension '{extension}'")

    text = parser(content_bytes)
    logger.info("Document parsed", filename=filename, characters=len(text))

    return DocumentInput(filename=filename, content=text, source_type=extension)


# ---------------------------------------------------------------------------
# Format-specific parsers
# Each takes raw bytes and returns a plain string.
# ---------------------------------------------------------------------------

def _parse_pdf(content_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    import fitz  # pymupdf

    text_parts: list[str] = []

    with fitz.open(stream=content_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n\n".join(text_parts)


def _parse_docx(content_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx."""
    import io
    from docx import Document

    doc = Document(io.BytesIO(content_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def _parse_txt(content_bytes: bytes) -> str:
    """Decode a plain text file, falling back to latin-1 if UTF-8 fails."""
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed, falling back to latin-1")
        return content_bytes.decode("latin-1")
