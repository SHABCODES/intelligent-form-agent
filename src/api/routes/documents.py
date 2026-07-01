"""Document management endpoints — fully async with DB persistence."""

from __future__ import annotations

import csv
import io
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.exceptions import DocumentError, FileTooLargeError, UnsupportedFileError
from src.core.logger import get_logger
from src.db.database import get_db
from src.models.schemas import (
    BatchUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    ExportData,
    UploadResponse,
)
from src.services import document_service
from src.services.vector_store import get_vector_store

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload and process a single PDF document.

    Processing pipeline:
    1. Validate file type and size
    2. Check deduplication via MD5 hash
    3. Extract text (PyMuPDF → OCR fallback)
    4. Run regex field extraction
    5. Generate AI summary (LangChain + GPT-4o-mini)
    6. Persist to SQLite via SQLAlchemy
    7. Index chunks in ChromaDB for RAG
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    tmp_path = settings.UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    try:
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.MAX_UPLOAD_SIZE_MB:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({size_mb:.1f} MB). Limit: {settings.MAX_UPLOAD_SIZE_MB} MB",
            )

        tmp_path.write_bytes(content)
        final_path = settings.UPLOAD_DIR / file.filename
        shutil.copy(tmp_path, final_path)

        doc_info = await document_service.process_document(final_path, db)
        return UploadResponse(success=True, document=doc_info)

    except (DocumentError, UnsupportedFileError, FileTooLargeError) as exc:
        return UploadResponse(success=False, error=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Unexpected error processing %s", file.filename)
        return UploadResponse(success=False, error=f"Processing failed: {exc}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@router.post("/upload-batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload and process multiple PDF documents in one request."""
    results: List[UploadResponse] = []
    for f in files:
        resp = await upload_document(f, db)
        results.append(resp)

    successful = sum(1 for r in results if r.success)
    return BatchUploadResponse(
        total_uploaded=len(files),
        successful=successful,
        failed=len(files) - successful,
        documents=results,
    )


# ── List & Retrieve ───────────────────────────────────────────────────────────

@router.get("", response_model=DocumentListResponse)
async def list_documents(db: AsyncSession = Depends(get_db)):
    """List all processed documents from the database."""
    docs = await document_service.list_documents(db)
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Get full details for a specific document."""
    doc = await document_service.get_document(doc_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc


@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Remove a document from DB, cache, and vector store."""
    deleted = await document_service.delete_document(doc_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"success": True, "message": f"Document {doc_id} deleted"}


# ── Semantic Search ───────────────────────────────────────────────────────────

@router.get("/search/semantic")
async def semantic_search(q: str, top_k: int = 5):
    """
    Semantic similarity search across all indexed documents.

    Uses ChromaDB + sentence-transformer embeddings to find the most
    relevant document passages for a natural language query.
    Unlike keyword search, this finds conceptually related content
    even when the exact words don't match.

    Parameters
    ----------
    q     : natural language search query
    top_k : number of results to return (default 5)
    """
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    vs = get_vector_store()
    if not vs.is_available:
        raise HTTPException(
            status_code=503,
            detail="Semantic search unavailable. ChromaDB not initialized.",
        )

    results = vs.search(q.strip(), n_results=min(top_k, 20))
    return {
        "query": q,
        "total_results": len(results),
        "results": [
            {
                "similarity": r["similarity"],
                "doc_id": r["metadata"].get("doc_id"),
                "filename": r["metadata"].get("filename"),
                "chunk_index": r["metadata"].get("chunk_index"),
                "excerpt": r["text"][:400],
            }
            for r in results
        ],
    }


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/analytics/collection")
async def collection_analytics(db: AsyncSession = Depends(get_db)):
    """Aggregated analytics across all documents in the database."""
    return await document_service.get_collection_analytics(db)


# ── Export ────────────────────────────────────────────────────────────────────

@router.get("/{doc_id}/export/json")
async def export_json(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Export all extracted data from a document as JSON."""
    doc = await document_service.get_document(doc_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    export = ExportData(
        document_id=doc_id,
        filename=doc.filename,
        extracted_fields=doc.extracted_fields.model_dump(),
        summary=doc.summary,
        exported_at=datetime.now(timezone.utc),
    )
    return JSONResponse(content=json.loads(export.model_dump_json()))


@router.get("/{doc_id}/export/csv")
async def export_csv(doc_id: str, db: AsyncSession = Depends(get_db)):
    """Export extracted fields from a document as CSV."""
    doc = await document_service.get_document(doc_id, db)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["field", "value"])
    for field, value in doc.extracted_fields.model_dump().items():
        if field != "line_items":
            writer.writerow([field, value or ""])

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{doc.filename}.csv"'},
    )
