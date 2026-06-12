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

    # ── App ──────────────────────────────────────────────────────────────────
    APP_NAME: str = "Intelligent Document AI Platform"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = (
        "A production-grade AI platform for intelligent document processing, "
        "extraction, and natural-language querying."
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

    # ── AI Models ─────────────────────────────────────────────────────────────
    # Models
    LLM_MODEL: str = "google/flan-t5-small"
    QA_MODEL: str = "distilbert-base-cased-distilled-squad"
    # Summarization
    SUMMARIZER_MODEL: str = "sshleifer/distilbart-cnn-12-6"
    # Embeddings for ChromaDB
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Model behaviour
    MAX_INPUT_TOKENS: int = 3000
    SUMMARY_MAX_LENGTH: int = 200
    SUMMARY_MIN_LENGTH: int = 60
    QA_CONFIDENCE_THRESHOLD: float = 0.15

    # ── Vector Store ──────────────────────────────────────────────────────────
    CHROMA_COLLECTION: str = "documents"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # ── Cache (in-memory with Redis-compatible interface) ─────────────────────
    USE_REDIS: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600          # 1 hour default
    CACHE_MAX_ITEMS: int = 1000

    # ── Upload limits ─────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = [".pdf"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
