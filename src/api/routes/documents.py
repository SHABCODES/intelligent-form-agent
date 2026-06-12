"""Document management endpoints."""

from __future__ import annotations
import csv
import io
import json
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.config import settings
from src.core.exceptions import DocumentError, FileTooLargeError, UnsupportedFileError
from src.core.logger import get_logger
from src.models.schemas import (
    BatchUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    ExportData,
    UploadResponse,
)
from src.services import document_service

log = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a single PDF document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    # Save to temp location
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
        # Preserve original filename for display
        final_path = settings.UPLOAD_DIR / file.filename
        shutil.copy(tmp_path, final_path)

        doc_info = document_service.process_document(final_path)
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
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload and process multiple PDF documents."""
    results: List[UploadResponse] = []
    for f in files:
        resp = await upload_document(f)
        results.append(resp)

    successful = sum(1 for r in results if r.success)
    return BatchUploadResponse(
        total_uploaded=len(files),
        successful=successful,
        failed=len(files) - successful,
        documents=results,
    )


# ── List & Retrieve ────────────────────────────────────────────────────────────

@router.get("", response_model=DocumentListResponse)
def list_documents():
    """List all processed documents."""
    docs = document_service.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str):
    """Get detailed info for a specific document."""
    doc = document_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc


@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(doc_id: str):
    """Remove a document from the platform."""
    deleted = document_service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"success": True, "message": f"Document {doc_id} deleted"}


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/analytics/collection")
def collection_analytics():
    """Return aggregated analytics across all documents."""
    return document_service.get_collection_analytics()


# ── Export ─────────────────────────────────────────────────────────────────────

@router.get("/{doc_id}/export/json")
def export_json(doc_id: str):
    """Export extracted data from a document as JSON."""
    doc = document_service.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    export = ExportData(
        document_id=doc_id,
        filename=doc.filename,
        extracted_fields=doc.extracted_fields.model_dump(),
        summary=doc.summary,
        exported_at=datetime.utcnow(),
    )
    return JSONResponse(content=json.loads(export.model_dump_json()))


@router.get("/{doc_id}/export/csv")
def export_csv(doc_id: str):
    """Export extracted fields from a document as CSV."""
    doc = document_service.get_document(doc_id)
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
