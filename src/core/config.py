"""Application configuration using Pydantic Settings."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "Intelligent Document AI Platform"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = (
        "A production-grade agentic AI platform for intelligent document "
        "processing, RAG-backed Q&A, and structured extraction."
    )
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── Paths ─────────────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    CHROMA_DIR: Path = BASE_DIR / "chroma_db"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/documents.db"

    # ── AI / LLM ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"

    # ── Vector Store / RAG ────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_COLLECTION: str = "documents"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 5

    # ── Cache ─────────────────────────────────────────────────────────────────
    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_ITEMS: int = 1000

    # ── Upload Limits ─────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".pdf"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
