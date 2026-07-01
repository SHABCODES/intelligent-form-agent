"""Health and metrics endpoints."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.logger import get_logger
from src.db.database import get_db
from src.models.schemas import HealthStatus
from src.services.cache_service import get_cache
from src.services.vector_store import get_vector_store

log = get_logger(__name__)
router = APIRouter()
_start_time = time.monotonic()


@router.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Platform health check.
    Verifies: DB connectivity, cache, vector store availability.
    """
    # Cache health
    cache = get_cache()
    cache_ok = True
    try:
        cache.set("__health_ping__", 1, ttl=5)
        cache_ok = cache.get("__health_ping__") == 1
    except Exception:
        cache_ok = False

    # DB health
    db_ok = True
    total_docs = 0
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM documents"))
        total_docs = result.scalar() or 0
    except Exception as exc:
        log.warning("DB health check failed: %s", exc)
        db_ok = False

    # Vector store health
    vs = get_vector_store()
    vector_store_ok = vs.is_available

    status = "healthy" if (db_ok and cache_ok) else "degraded"

    return HealthStatus(
        status=status,
        version=settings.APP_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        models_loaded=True,
        vector_store_ok=vector_store_ok,
        cache_ok=cache_ok,
        total_documents=total_docs,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/metrics", tags=["System"])
async def metrics(db: AsyncSession = Depends(get_db)):
    """Runtime metrics: uptime, DB counts, cache stats, vector store."""
    cache = get_cache()
    vs = get_vector_store()

    doc_count = 0
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM documents"))
        doc_count = result.scalar() or 0
    except Exception:
        pass

    msg_count = 0
    try:
        result = await db.execute(text("SELECT COUNT(*) FROM conversation_msgs"))
        msg_count = result.scalar() or 0
    except Exception:
        pass

    return {
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "database": {
            "total_documents": doc_count,
            "total_conversation_messages": msg_count,
            "url": settings.DATABASE_URL.split("///")[-1],  # path only, no credentials
        },
        "cache": cache.stats(),
        "vector_store": {
            "available": vs.is_available,
            "chunk_count": vs.document_count() if vs.is_available else 0,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
    }
