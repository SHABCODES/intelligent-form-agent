"""
Document service — central orchestrator for the full processing pipeline.

Full flow for every uploaded PDF:
  1. Validate file (type, size)
  2. Compute MD5 hash → dedup check against DB
  3. Extract text (PyMuPDF → OCR fallback)
  4. Regex field extraction
  5. AI summary (via agent_service)
  6. Persist to SQLAlchemy DB
  7. Index chunks in ChromaDB for RAG
  8. Cache doc metadata for fast reads

All public functions are async. CPU-bound extraction runs via asyncio.to_thread
so it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import DocumentError, FileTooLargeError, UnsupportedFileError
from src.core.logger import get_logger
from src.db.repository import DocumentRepository
from src.models.schemas import DocumentInfo, DocumentListItem, ExtractedFields
from src.services import extraction_service
from src.services.cache_service import get_cache
from src.services.vector_store import get_vector_store
from src.utils.pdf_utils import extract_document_text
from src.utils.text_utils import clean_text

log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    """MD5 fingerprint for deduplication."""
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


def _db_doc_to_schema(db_doc, *, include_text: bool = False) -> DocumentInfo:
    """Convert a SQLAlchemy Document ORM object → Pydantic DocumentInfo schema."""
    fdict = db_doc.fields_as_dict()

    # line_items stored as JSON string in DB
    line_items = []
    raw_li = fdict.pop("line_items", None)
    if raw_li:
        try:
            line_items = json.loads(raw_li)
        except Exception:
            pass

    fields = ExtractedFields(
        invoice_number=fdict.get("invoice_number"),
        date=fdict.get("date"),
        due_date=fdict.get("due_date"),
        name=fdict.get("name"),
        seller=fdict.get("seller"),
        email=fdict.get("email"),
        phone=fdict.get("phone"),
        amount=fdict.get("amount"),
        gst=fdict.get("gst"),
        currency=fdict.get("currency"),
        line_items=line_items,
    )

    return DocumentInfo(
        id=db_doc.id,
        filename=db_doc.filename,
        file_size_bytes=db_doc.file_size_bytes,
        page_count=db_doc.page_count,
        extracted_fields=fields,
        summary=db_doc.summary,
        text_preview=db_doc.text_preview,
        field_completion_pct=db_doc.field_completion_pct,
        processed_at=db_doc.created_at,
        processing_time_ms=db_doc.processing_time_ms,
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def process_document(file_path: str | Path, db: AsyncSession) -> DocumentInfo:
    """
    Full processing pipeline for a single uploaded PDF.

    Parameters
    ----------
    file_path : path to the saved PDF file
    db        : active async SQLAlchemy session (injected via FastAPI Depends)

    Returns
    -------
    DocumentInfo Pydantic schema with all extracted data
    """
    path = Path(file_path)
    _validate_file(path)

    t_start = time.perf_counter()
    cache = get_cache()

    # ── Deduplication via file hash ────────────────────────────────────────
    file_hash = await asyncio.to_thread(_file_hash, path)
    cached = cache.get(f"doc_hash:{file_hash}")
    if cached:
        log.info("Cache hit (dedup) for %s", path.name)
        repo = DocumentRepository(db)
        db_doc = await repo.get_by_id(cached["id"])
        if db_doc:
            return _db_doc_to_schema(db_doc)

    log.info("Processing document: %s", path.name)

    # ── 1. Text extraction (CPU-bound → off event loop) ────────────────────
    raw_text, page_count = await asyncio.to_thread(extract_document_text, path)
    if not raw_text.strip():
        raise DocumentError(f"No text could be extracted from {path.name}")
    text = await asyncio.to_thread(clean_text, raw_text)

    # ── 2. Field extraction (CPU-bound) ────────────────────────────────────
    fields: ExtractedFields = await asyncio.to_thread(
        extraction_service.extract_fields, text
    )
    completion_pct = extraction_service.field_completion_pct(fields)

    # ── 3. AI summary (network I/O → can run in event loop) ───────────────
    summary = await _generate_summary_async(text)

    # ── 4. Persist to DB ───────────────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    processing_ms = round((time.perf_counter() - t_start) * 1000, 1)
    text_preview = text[:500] + ("..." if len(text) > 500 else "")

    repo = DocumentRepository(db)
    db_doc = await repo.create(
        doc_id=doc_id,
        filename=path.name,
        file_hash=file_hash,
        file_size_bytes=path.stat().st_size,
        page_count=page_count,
        raw_text=text,
        summary=summary,
        text_preview=text_preview,
        field_completion_pct=completion_pct,
        processing_time_ms=processing_ms,
        fields=fields.model_dump(),
    )

    doc_info = _db_doc_to_schema(db_doc)

    # ── 5. Cache for fast subsequent reads ────────────────────────────────
    cache.set(f"doc:{doc_id}", {"id": doc_id})
    cache.set(f"doc_hash:{file_hash}", {"id": doc_id})

    # ── 6. Index in ChromaDB for RAG ──────────────────────────────────────
    await asyncio.to_thread(
        _index_in_vector_store,
        doc_id, text,
        {"filename": path.name, "invoice_number": fields.invoice_number or "",
         "amount": fields.amount or "", "date": fields.date or ""},
    )

    log.info(
        "Document %s processed in %.0f ms | completion=%.0f%% | pages=%d",
        path.name, processing_ms, completion_pct, page_count,
    )
    return doc_info


def _index_in_vector_store(doc_id: str, text: str, metadata: dict) -> None:
    """Sync helper called via asyncio.to_thread."""
    vs = get_vector_store()
    vs.add_document(doc_id=doc_id, text=text, metadata=metadata)


async def _generate_summary_async(text: str) -> str:
    """
    Generate an AI summary using the agent's summarize tool.
    Runs the sync LangChain call off the event loop.
    """
    def _summarize():
        try:
            from src.services.agent_service import _get_agent
            import json as _json
            agent = _get_agent()
            if agent is None:
                return _basic_summary(text)
            # Direct tool call — no full ReAct loop needed for summaries
            from langchain_openai import ChatOpenAI
            from langchain_core.output_parsers import JsonOutputParser
            from src.services.agent_service import DocumentSummary
            llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0,
                             api_key=settings.OPENAI_API_KEY)
            parser = JsonOutputParser(pydantic_object=DocumentSummary)
            prompt = (
                f"Summarize this document concisely.\n"
                f"Return ONLY valid JSON: {parser.get_format_instructions()}\n\n"
                f"Document:\n{text[:3000]}"
            )
            resp = llm.invoke(prompt)
            try:
                data = _json.loads(resp.content)
                return data.get("summary", resp.content)
            except Exception:
                return resp.content
        except Exception as exc:
            log.warning("AI summary failed, using basic summary: %s", exc)
            return _basic_summary(text)

    return await asyncio.to_thread(_summarize)


def _basic_summary(text: str) -> str:
    """Fallback: first 300 chars when AI is unavailable."""
    clean = " ".join(text.split())
    return clean[:300] + ("..." if len(clean) > 300 else "")


async def get_document(doc_id: str, db: AsyncSession) -> Optional[DocumentInfo]:
    """Retrieve a processed document by ID."""
    repo = DocumentRepository(db)
    db_doc = await repo.get_by_id(doc_id)
    if not db_doc:
        return None
    return _db_doc_to_schema(db_doc)


async def get_document_text(doc_id: str, db: AsyncSession) -> Optional[str]:
    """Return the full extracted text for RAG / agent context."""
    repo = DocumentRepository(db)
    db_doc = await repo.get_by_id(doc_id)
    return db_doc.raw_text if db_doc else None


async def list_documents(db: AsyncSession) -> List[DocumentListItem]:
    """List all processed documents, newest first."""
    repo = DocumentRepository(db)
    docs = await repo.list_all()
    items = []
    for doc in docs:
        fdict = doc.fields_as_dict()
        items.append(DocumentListItem(
            id=doc.id,
            filename=doc.filename,
            processed_at=doc.created_at,
            field_completion_pct=doc.field_completion_pct,
            amount=fdict.get("amount"),
            invoice_number=fdict.get("invoice_number"),
        ))
    return items


async def delete_document(doc_id: str, db: AsyncSession) -> bool:
    """Delete a document from DB, cache, and vector store."""
    repo = DocumentRepository(db)
    deleted = await repo.delete(doc_id)
    if deleted:
        cache = get_cache()
        cache.delete(f"doc:{doc_id}")
        await asyncio.to_thread(get_vector_store().delete_document, doc_id)
    return deleted


async def get_collection_analytics(db: AsyncSession) -> Dict[str, Any]:
    """Aggregate analytics computed from DB rows."""
    repo = DocumentRepository(db)
    return await repo.get_analytics()


async def get_all_documents_for_qa(db: AsyncSession) -> List[Dict[str, Any]]:
    """Return all documents for cross-document Q&A context."""
    repo = DocumentRepository(db)
    docs = await repo.list_all()
    return [
        {"doc_id": doc.id, "filename": doc.filename, "text": doc.raw_text}
        for doc in docs
    ]
