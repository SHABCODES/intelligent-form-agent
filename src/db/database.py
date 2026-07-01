"""
Async SQLAlchemy engine and session factory.

Usage in FastAPI:
    async def route(db: AsyncSession = Depends(get_db)):
        repo = DocumentRepository(db)
        ...

Upgrading to Postgres:
    Set DATABASE_URL=postgresql+asyncpg://user:pass@host/db in .env
    pip install asyncpg
    That's it — zero application code changes needed.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,        # SQL logging only in debug mode
    future=True,
    # SQLite-specific: allow shared connections across threads
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)

# ── Session factory ───────────────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ── Declarative base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """All ORM models inherit from this."""
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncSession:  # type: ignore[misc]
    """
    Yield an async DB session per request.
    Automatically rolls back on exception, always closes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Table creation helper ─────────────────────────────────────────────────────

async def create_tables() -> None:
    """Create all tables if they don't exist. Called at app startup."""
    from src.db import models as _  # noqa: F401 — ensures models are registered
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database tables created/verified at %s", settings.DATABASE_URL)
