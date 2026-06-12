"""Health and metrics endpoints."""

from __future__ import annotations
import time
from datetime import datetime

from fastapi import APIRouter

from src.core.config import settings
from src.models.schemas import HealthStatus
from src.services import document_service
from src.services.cache_service import get_cache
from src.services.vector_store import get_vector_store

router = APIRouter()
_start_time = time.monotonic()


@router.get("/health", response_model=HealthStatus, tags=["System"])
def health_check():
    """Return platform health status."""
    cache = get_cache()
    vs = get_vector_store()

    cache_ok = True
    try:
        cache.set("__health_ping__", 1, ttl=5)
        cache_ok = cache.get("__health_ping__") == 1
    except Exception:
        cache_ok = False

    return HealthStatus(
        status="healthy" if cache_ok else "degraded",
        version=settings.APP_VERSION,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        models_loaded=True,
        vector_store_ok=vs.is_available,
        cache_ok=cache_ok,
        total_documents=len(document_service._document_registry),
        timestamp=datetime.utcnow(),
    )


@router.get("/metrics", tags=["System"])
def metrics():
    """Return runtime metrics."""
    cache = get_cache()
    vs = get_vector_store()
    analytics = document_service.get_collection_analytics()

    return {
        "uptime_seconds": round(time.monotonic() - _start_time, 1),
        "documents": analytics,
        "cache": cache.stats(),
        "vector_store": {
            "available": vs.is_available,
            "chunk_count": vs.document_count(),
        },
    }
