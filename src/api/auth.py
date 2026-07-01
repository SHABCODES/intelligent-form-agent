"""API key authentication dependency for production endpoints."""

from __future__ import annotations
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.core.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """
    Dependency to verify the X-API-Key request header.
    If API_KEY is not configured in settings, authentication is disabled.
    """
    if not settings.API_KEY:
        return None

    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing X-API-Key header",
        )
    return api_key
