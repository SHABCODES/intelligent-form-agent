"""Custom exception hierarchy for the platform."""

from __future__ import annotations


class DocAIError(Exception):
    """Base exception for all platform errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail or message


class DocumentError(DocAIError):
    """Raised when a document cannot be read or parsed."""


class ExtractionError(DocAIError):
    """Raised when field extraction fails."""


class AIError(DocAIError):
    """Raised when an AI model call fails."""


class VectorStoreError(DocAIError):
    """Raised on ChromaDB issues."""


class CacheError(DocAIError):
    """Raised on cache read/write failures."""


class UnsupportedFileError(DocAIError):
    """Raised for unsupported file types."""


class FileTooLargeError(DocAIError):
    """Raised when an upload exceeds the size limit."""
