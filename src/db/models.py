"""
SQLAlchemy ORM models.

Tables:
    documents           — core document record with full text
    extracted_fields    — key-value field rows per document (flexible schema)
    conversation_msgs   — chat history per session
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.database import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


# ── Documents ─────────────────────────────────────────────────────────────────

class Document(Base):
    """Persists every successfully processed PDF document."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_new_uuid
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(32), unique=True, index=True, nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=1)

    # Full extracted text — stored for RAG / agent re-use without re-extracting
    raw_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # AI-generated summary
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Text preview for display (first 500 chars)
    text_preview: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Extraction quality metrics
    field_completion_pct: Mapped[float] = mapped_column(Float, default=0.0)
    processing_time_ms: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    fields: Mapped[List["ExtractedField"]] = relationship(
        "ExtractedField",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r}>"

    def fields_as_dict(self) -> dict:
        """Return extracted fields as a plain dict."""
        return {f.field_name: f.field_value for f in self.fields}


# ── Extracted Fields ──────────────────────────────────────────────────────────

class ExtractedField(Base):
    """
    Flat key-value rows for extracted document fields.

    Storing as rows (not JSON column) lets us:
    - Query by field value: SELECT * WHERE field_name='invoice_number' AND field_value='INV-001'
    - Add new field types without schema migrations
    - Index specific fields easily
    """

    __tablename__ = "extracted_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    field_value: Mapped[str] = mapped_column(Text, nullable=True)

    document: Mapped["Document"] = relationship("Document", back_populates="fields")

    def __repr__(self) -> str:
        return f"<ExtractedField {self.field_name}={self.field_value!r}>"


# ── Conversation Messages ─────────────────────────────────────────────────────

class ConversationMessage(Base):
    """
    Persists conversation turns (user + assistant messages) per session.
    Complements the in-memory checkpointer with durable storage.
    """

    __tablename__ = "conversation_msgs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    doc_id: Mapped[str] = mapped_column(String(36), nullable=True)  # optional context doc
    model_used: Mapped[str] = mapped_column(String(100), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ConversationMessage session={self.session_id!r} role={self.role!r}>"
