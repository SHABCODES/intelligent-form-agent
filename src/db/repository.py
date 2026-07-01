"""
Repository layer — async CRUD operations on top of SQLAlchemy ORM.

Why repositories?
- Single source of truth for all DB interactions
- Routes and services never import sqlalchemy directly
- Easy to mock in tests
- Trivially swappable to a different ORM / DB

Usage:
    async with AsyncSessionLocal() as db:
        repo = DocumentRepository(db)
        doc = await repo.get_by_id("some-uuid")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.db.models import ConversationMessage, Document, ExtractedField

log = get_logger(__name__)


# ── Document Repository ───────────────────────────────────────────────────────

class DocumentRepository:
    """Async CRUD for Document + ExtractedField rows."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(
        self,
        *,
        doc_id: str,
        filename: str,
        file_hash: str,
        file_size_bytes: int,
        page_count: int,
        raw_text: str,
        summary: str,
        text_preview: str,
        field_completion_pct: float,
        processing_time_ms: float,
        fields: Dict[str, Optional[str]],
    ) -> Document:
        """
        Persist a newly processed document and all its extracted fields.
        Runs in a single transaction — either everything commits or nothing does.
        """
        doc = Document(
            id=doc_id,
            filename=filename,
            file_hash=file_hash,
            file_size_bytes=file_size_bytes,
            page_count=page_count,
            raw_text=raw_text,
            summary=summary,
            text_preview=text_preview,
            field_completion_pct=field_completion_pct,
            processing_time_ms=processing_time_ms,
        )
        self._db.add(doc)

        # Insert field rows
        for name, value in fields.items():
            if name == "line_items":
                # Serialize list fields as JSON string
                import json
                value = json.dumps(value) if isinstance(value, (list, dict)) else value
            if value is not None:
                self._db.add(ExtractedField(
                    doc_id=doc_id,
                    field_name=name,
                    field_value=str(value),
                ))

        await self._db.flush()  # get DB-generated values (created_at etc.)
        await self._db.refresh(doc)
        log.info("Document %s persisted to DB", doc_id)
        return doc

    async def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Fetch document by primary key. Returns None if not found."""
        result = await self._db.execute(
            select(Document).where(Document.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, file_hash: str) -> Optional[Document]:
        """Deduplication check — find existing doc with the same file hash."""
        result = await self._db.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 200) -> List[Document]:
        """Return all documents, newest first."""
        result = await self._db.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete(self, doc_id: str) -> bool:
        """Delete document and all child rows (cascade). Returns True if found."""
        result = await self._db.execute(
            delete(Document).where(Document.id == doc_id)
        )
        deleted = result.rowcount > 0
        if deleted:
            log.info("Document %s deleted from DB", doc_id)
        return deleted

    async def get_analytics(self) -> Dict[str, Any]:
        """Return aggregate statistics across all documents."""
        docs = await self.list_all(limit=10_000)
        if not docs:
            return {
                "total_documents": 0,
                "documents_with_amounts": 0,
                "total_amount": 0.0,
                "average_amount": 0.0,
                "currency_breakdown": {},
                "field_stats": [],
                "recent_uploads": [],
            }

        import json, re

        total_amount = 0.0
        docs_with_amounts = 0
        currency_count: Dict[str, int] = {}
        key_fields = ["invoice_number", "date", "name", "seller", "email", "phone", "amount", "gst"]
        field_hit = {f: 0 for f in key_fields}

        for doc in docs:
            fdict = doc.fields_as_dict()
            amt_str = fdict.get("amount")
            if amt_str:
                try:
                    amt = float(re.sub(r"[^\d.]", "", str(amt_str)))
                    total_amount += amt
                    docs_with_amounts += 1
                except ValueError:
                    pass
            currency = fdict.get("currency")
            if currency:
                currency_count[currency] = currency_count.get(currency, 0) + 1
            for f in key_fields:
                if fdict.get(f):
                    field_hit[f] += 1

        n = len(docs)
        field_stats = [
            {"field": f, "completion_pct": round((c / n) * 100, 1), "count": c}
            for f, c in field_hit.items()
        ]

        recent = []
        for doc in docs[:5]:
            fdict = doc.fields_as_dict()
            recent.append({
                "id": doc.id,
                "filename": doc.filename,
                "processed_at": doc.created_at.isoformat(),
                "field_completion_pct": doc.field_completion_pct,
                "amount": fdict.get("amount"),
                "invoice_number": fdict.get("invoice_number"),
            })

        return {
            "total_documents": n,
            "documents_with_amounts": docs_with_amounts,
            "total_amount": round(total_amount, 2),
            "average_amount": round(total_amount / docs_with_amounts, 2) if docs_with_amounts else 0.0,
            "currency_breakdown": currency_count,
            "field_stats": field_stats,
            "recent_uploads": recent,
        }


# ── Conversation Repository ───────────────────────────────────────────────────

class ConversationRepository:
    """Async CRUD for conversation history (persistent complement to in-memory checkpointer)."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def save_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        doc_id: Optional[str] = None,
        model_used: Optional[str] = None,
        latency_ms: Optional[float] = None,
    ) -> ConversationMessage:
        msg = ConversationMessage(
            session_id=session_id,
            role=role,
            content=content,
            doc_id=doc_id,
            model_used=model_used,
            latency_ms=latency_ms,
        )
        self._db.add(msg)
        await self._db.flush()
        return msg

    async def get_session_history(
        self, session_id: str, limit: int = 50
    ) -> List[ConversationMessage]:
        """Return conversation messages for a session, oldest first."""
        result = await self._db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_sessions(self) -> List[str]:
        """Return distinct session IDs that have conversation history."""
        from sqlalchemy import distinct
        result = await self._db.execute(
            select(distinct(ConversationMessage.session_id))
            .order_by(ConversationMessage.session_id)
        )
        return list(result.scalars().all())

    async def delete_session(self, session_id: str) -> int:
        """Delete all messages for a session. Returns count deleted."""
        result = await self._db.execute(
            delete(ConversationMessage).where(
                ConversationMessage.session_id == session_id
            )
        )
        return result.rowcount
