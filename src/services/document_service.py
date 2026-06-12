"""
Document service — the central orchestrator for the full processing pipeline.

Flow:
  PDF file → text extraction → field extraction → AI summary
            → vector store indexing → cache → persistent registry
"""

from __future__ import annotations
import hashlib
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import get_logger
from src.core.exceptions import DocumentError, UnsupportedFileError, FileTooLargeError
from src.models.schemas import (
    DocumentInfo,
    DocumentListItem,
    ExtractedFields,
)
from src.services import extraction_service, ai_service
from src.services.cache_service import get_cache
from src.services.vector_store import get_vector_store
from src.services.extraction_service import field_completion_pct, parse_amount_value
from src.utils.pdf_utils import extract_document_text
from src.utils.text_utils import clean_text

log = get_logger(__name__)

# ── In-process document registry (replaced by DB in production) ───────────────
# doc_id → DocumentInfo (serialised as dict)
_document_registry: Dict[str, Dict[str, Any]] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """Quick MD5 fingerprint of the file for dedup."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise DocumentError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise UnsupportedFileError(f"Unsupported file type: {suffix}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise FileTooLargeError(
            f"File size {size_mb:.1f} MB exceeds limit {settings.MAX_UPLOAD_SIZE_MB} MB"
        )


def _doc_info_to_dict(doc: DocumentInfo) -> Dict[str, Any]:
    return json.loads(doc.model_dump_json())


def _dict_to_doc_info(d: Dict[str, Any]) -> DocumentInfo:
    return DocumentInfo.model_validate(d)


# ── Public interface ──────────────────────────────────────────────────────────

def process_document(file_path: str | Path) -> DocumentInfo:
    """
    Full processing pipeline for a single PDF document.
    Returns a rich DocumentInfo object with all extracted data.
    """
    path = Path(file_path)
    _validate_file(path)

    t_start = time.perf_counter()

    # Check cache by file hash
    cache = get_cache()
    file_hash = _file_hash(path)
    cached = cache.get(f"doc_hash:{file_hash}")
    if cached:
        log.info("Cache hit for %s", path.name)
        return _dict_to_doc_info(cached)

    log.info("Processing document: %s", path.name)

    # 1. Text extraction
    raw_text, page_count = extract_document_text(path)
    if not raw_text.strip():
        raise DocumentError(f"No text could be extracted from {path.name}")
    text = clean_text(raw_text)

    # 2. Field extraction
    fields: ExtractedFields = extraction_service.extract_fields(text)
    completion_pct = field_completion_pct(fields)

    # 3. AI summary
    summary = ai_service.summarize_document(text)

    # 4. Build document info
    doc_id = str(uuid.uuid4())
    processing_ms = round((time.perf_counter() - t_start) * 1000, 1)

    doc_info = DocumentInfo(
        id=doc_id,
        filename=path.name,
        file_size_bytes=path.stat().st_size,
        page_count=page_count,
        extracted_fields=fields,
        summary=summary,
        text_preview=text[:500] + ("..." if len(text) > 500 else ""),
        field_completion_pct=completion_pct,
        processed_at=datetime.utcnow(),
        processing_time_ms=processing_ms,
    )

    # 5. Store in registry + cache
    serialised = _doc_info_to_dict(doc_info)
    _document_registry[doc_id] = {**serialised, "_text": text}  # keep full text for Q&A
    cache.set(f"doc:{doc_id}", serialised)
    cache.set(f"doc_hash:{file_hash}", serialised)

    # 6. Index in vector store
    vs = get_vector_store()
    vs.add_document(
        doc_id=doc_id,
        text=text,
        metadata={
            "filename": path.name,
            "invoice_number": fields.invoice_number or "",
            "amount": fields.amount or "",
            "date": fields.date or "",
        },
    )

    log.info(
        "Document %s processed in %.0f ms | fields: %.0f%% | pages: %d",
        path.name, processing_ms, completion_pct, page_count,
    )
    return doc_info


def get_document(doc_id: str) -> Optional[DocumentInfo]:
    """Retrieve a processed document by ID."""
    entry = _document_registry.get(doc_id)
    if not entry:
        return None
    # Exclude internal _text key
    public = {k: v for k, v in entry.items() if not k.startswith("_")}
    return _dict_to_doc_info(public)


def get_document_text(doc_id: str) -> Optional[str]:
    """Return full extracted text for a document (used by AI service)."""
    entry = _document_registry.get(doc_id)
    return entry.get("_text") if entry else None


def list_documents() -> List[DocumentListItem]:
    items: List[DocumentListItem] = []
    for doc_id, entry in _document_registry.items():
        info = {k: v for k, v in entry.items() if not k.startswith("_")}
        doc = _dict_to_doc_info(info)
        items.append(
            DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                processed_at=doc.processed_at,
                field_completion_pct=doc.field_completion_pct,
                amount=doc.extracted_fields.amount,
                invoice_number=doc.extracted_fields.invoice_number,
            )
        )
    return sorted(items, key=lambda x: x.processed_at, reverse=True)


def delete_document(doc_id: str) -> bool:
    if doc_id not in _document_registry:
        return False
    del _document_registry[doc_id]
    cache = get_cache()
    cache.delete(f"doc:{doc_id}")
    vs = get_vector_store()
    vs.delete_document(doc_id)
    log.info("Document %s deleted", doc_id)
    return True


def get_collection_analytics() -> Dict[str, Any]:
    """Aggregate analytics across all processed documents."""
    docs = list_documents()
    if not docs:
        return {
            "total_documents": 0,
            "documents_with_amounts": 0,
            "total_amount": 0.0,
            "average_amount": 0.0,
            "currency_breakdown": {},
            "field_stats": [],
            "recent_uploads": [],
        }

    total_amount = 0.0
    docs_with_amounts = 0
    currency_count: Dict[str, int] = {}
    field_hit: Dict[str, int] = {
        f: 0 for f in ["invoice_number", "date", "name", "seller", "email", "phone", "amount", "gst"]
    }

    for item in docs:
        entry = _document_registry.get(item.id, {})
        public = {k: v for k, v in entry.items() if not k.startswith("_")}
        if not public:
            continue
        doc = _dict_to_doc_info(public)
        fields = doc.extracted_fields

        amount_val = parse_amount_value(fields)
        if amount_val:
            total_amount += amount_val
            docs_with_amounts += 1

        if fields.currency:
            currency_count[fields.currency] = currency_count.get(fields.currency, 0) + 1

        for field_name in field_hit:
            if getattr(fields, field_name, None):
                field_hit[field_name] += 1

    n = len(docs)
    field_stats = [
        {
            "field": f,
            "completion_pct": round((count / n) * 100, 1),
            "count": count,
        }
        for f, count in field_hit.items()
    ]

    return {
        "total_documents": n,
        "documents_with_amounts": docs_with_amounts,
        "total_amount": round(total_amount, 2),
        "average_amount": round(total_amount / docs_with_amounts, 2) if docs_with_amounts else 0.0,
        "currency_breakdown": currency_count,
        "field_stats": field_stats,
        "recent_uploads": [d.model_dump() for d in docs[:5]],
    }


def get_all_documents_for_qa() -> List[Dict[str, Any]]:
    """Return all documents as list of {doc_id, filename, text} for cross-doc Q&A."""
    result = []
    for doc_id, entry in _document_registry.items():
        result.append({
            "doc_id": doc_id,
            "filename": entry.get("filename", ""),
            "text": entry.get("_text", ""),
        })
    return result
