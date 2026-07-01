"""
FastAPI application factory.

Startup sequence:
  1. Create DB tables (SQLAlchemy + SQLite/Postgres)
  2. Warm up ChromaDB vector store
  3. Log agent status
  4. Mount frontend SPA
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

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ──────────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        return response

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(health.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")

    # ── Static files & frontend SPA ───────────────────────────────────────────
    frontend_dir = settings.FRONTEND_DIR
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_spa():
            return FileResponse(str(frontend_dir / "index.html"))

    # ── Startup ────────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup_event():
        log.info("=" * 60)
        log.info("  %s v%s", settings.APP_NAME, settings.APP_VERSION)
        log.info("=" * 60)

        # Ensure directories exist
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize DB tables
        from src.db.database import create_tables
        await create_tables()
        log.info("Database ready at %s", settings.DATABASE_URL)

        # Warm up AI services (logs key/model status)
        preload_models()

        log.info("Platform ready — http://%s:%d", settings.HOST, settings.PORT)
        log.info("API docs     — http://%s:%d/docs", settings.HOST, settings.PORT)

    @app.on_event("shutdown")
    async def shutdown_event():
        log.info("Shutting down %s", settings.APP_NAME)

    return app
