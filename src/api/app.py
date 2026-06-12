"""
FastAPI application factory.
"""

from __future__ import annotations
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.core.config import settings
from src.core.logger import get_logger
from src.api.routes import health, documents, chat
from src.services.ai_service import preload_models
from src.services.cache_service import get_cache
from src.services.vector_store import get_vector_store

log = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=settings.APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        return response

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")

    # ── Static files & frontend SPA ──────────────────────────────────────
    frontend_dir = settings.FRONTEND_DIR
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_spa():
            return FileResponse(str(frontend_dir / "index.html"))

    # ── Startup ────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup_event():
        log.info("=" * 60)
        log.info("  %s v%s", settings.APP_NAME, settings.APP_VERSION)
        log.info("=" * 60)

        # Ensure upload directory exists
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Warm up cache
        cache = get_cache()
        log.info("Cache backend: %s", cache.stats().get("backend", "unknown"))

        # Warm up vector store
        vs = get_vector_store()
        if vs.is_available:
            log.info("ChromaDB ready with %d chunks indexed", vs.document_count())
        else:
            log.warning("ChromaDB not available — semantic search disabled")

        # Pre-load AI models
        log.info("Warming up AI models...")
        preload_models()
        log.info("Platform ready — visit http://%s:%d", settings.HOST, settings.PORT)

    @app.on_event("shutdown")
    async def shutdown_event():
        log.info("Shutting down %s", settings.APP_NAME)

    return app
