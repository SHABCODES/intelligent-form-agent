"""PDF text extraction utilities."""

from __future__ import annotations
from pathlib import Path
from typing import Tuple

import fitz  # PyMuPDF

from src.core.logger import get_logger
from src.core.exceptions import DocumentError

log = get_logger(__name__)


def extract_text_pymupdf(file_path: str | Path) -> Tuple[str, int]:
    """
    Extract text from a PDF using PyMuPDF.

    Returns
    -------
    (text, page_count)
    """
    path = Path(file_path)
    if not path.exists():
        raise DocumentError(f"File not found: {path}")

    text_parts: list[str] = []
    try:
        doc = fitz.open(str(path))
        page_count = len(doc)
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
    except Exception as exc:
        raise DocumentError(f"PyMuPDF extraction failed: {exc}", detail=str(exc))

    return "\n".join(text_parts), page_count


def extract_text_ocr(file_path: str | Path) -> str:
    """
    OCR fallback using pdf2image + pytesseract.
    Only imported/called when needed to avoid heavy startup.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        log.warning("pdf2image / pytesseract not installed — OCR unavailable")
        return ""

    path = Path(file_path)
    ocr_parts: list[str] = []
    try:
        pages = convert_from_path(str(path), dpi=200)
        for page_img in pages:
            ocr_parts.append(pytesseract.image_to_string(page_img))
    except Exception as exc:
        log.warning("OCR extraction error: %s", exc)

    return "\n".join(ocr_parts)


def extract_document_text(file_path: str | Path) -> Tuple[str, int]:
    """
    Smart extraction: tries PyMuPDF first, falls back to OCR if
    the extracted text is suspiciously short (< 100 chars).

    Returns
    -------
    (text, page_count)
    """
    text, page_count = extract_text_pymupdf(file_path)
    if len(text.strip()) < 100:
        log.info("Short text from PyMuPDF (%d chars) — trying OCR", len(text.strip()))
        ocr_text = extract_text_ocr(file_path)
        if len(ocr_text) > len(text):
            text = ocr_text

    return text, page_count
