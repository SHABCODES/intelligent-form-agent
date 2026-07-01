"""
DB layer tests — repository CRUD against in-memory SQLite.
Runs fully async; no external DB required.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from src.db.repository import DocumentRepository, ConversationRepository


# ══════════════════════════════════════════════════════════════════════════════
# DocumentRepository Tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestDocumentRepository:

    async def test_create_and_get_by_id(self, db_session):
        repo = DocumentRepository(db_session)
        doc = await repo.create(
            doc_id="test-uuid-001",
            filename="invoice.pdf",
            file_hash="abc123def456",
            file_size_bytes=102400,
            page_count=2,
            raw_text="Invoice Number: INV/001\nTotal: ₹50,000.00",
            summary="A test invoice document.",
            text_preview="Invoice Number: INV/001...",
            field_completion_pct=71.4,
            processing_time_ms=850.0,
            fields={
                "invoice_number": "INV/001",
                "amount": "50,000.00",
                "email": "test@example.com",
                "currency": "INR",
            },
        )
        assert doc.id == "test-uuid-001"
        assert doc.filename == "invoice.pdf"
        assert doc.page_count == 2

        # Retrieve by ID
        fetched = await repo.get_by_id("test-uuid-001")
        assert fetched is not None
        assert fetched.filename == "invoice.pdf"
        assert fetched.summary == "A test invoice document."

    async def test_get_by_id_not_found(self, db_session):
        repo = DocumentRepository(db_session)
        result = await repo.get_by_id("does-not-exist")
        assert result is None

    async def test_get_by_hash(self, db_session):
        repo = DocumentRepository(db_session)
        await repo.create(
            doc_id="test-uuid-002",
            filename="contract.pdf",
            file_hash="unique-hash-xyz",
            file_size_bytes=204800,
            page_count=5,
            raw_text="Contract text here",
            summary="Contract summary",
            text_preview="Contract...",
            field_completion_pct=42.9,
            processing_time_ms=1200.0,
            fields={"seller": "Acme Corp"},
        )

        found = await repo.get_by_hash("unique-hash-xyz")
        assert found is not None
        assert found.filename == "contract.pdf"

        missing = await repo.get_by_hash("nonexistent-hash")
        assert missing is None

    async def test_list_all(self, db_session):
        repo = DocumentRepository(db_session)
        for i in range(3):
            await repo.create(
                doc_id=f"list-test-{i}",
                filename=f"file{i}.pdf",
                file_hash=f"hash-list-{i}",
                file_size_bytes=1024,
                page_count=1,
                raw_text=f"Content {i}",
                summary=f"Summary {i}",
                text_preview=f"Preview {i}",
                field_completion_pct=50.0,
                processing_time_ms=200.0,
                fields={"invoice_number": f"INV-{i}"},
            )

        docs = await repo.list_all()
        ids = [d.id for d in docs]
        for i in range(3):
            assert f"list-test-{i}" in ids

    async def test_delete(self, db_session):
        repo = DocumentRepository(db_session)
        await repo.create(
            doc_id="delete-me",
            filename="delete.pdf",
            file_hash="delete-hash",
            file_size_bytes=512,
            page_count=1,
            raw_text="To be deleted",
            summary="Delete test",
            text_preview="Delete...",
            field_completion_pct=0.0,
            processing_time_ms=100.0,
            fields={},
        )

        deleted = await repo.delete("delete-me")
        assert deleted is True

        gone = await repo.get_by_id("delete-me")
        assert gone is None

    async def test_delete_nonexistent(self, db_session):
        repo = DocumentRepository(db_session)
        deleted = await repo.delete("ghost-id")
        assert deleted is False

    async def test_fields_as_dict(self, db_session):
        repo = DocumentRepository(db_session)
        doc = await repo.create(
            doc_id="fields-test",
            filename="fields.pdf",
            file_hash="fields-hash",
            file_size_bytes=1024,
            page_count=1,
            raw_text="Test",
            summary="Test",
            text_preview="Test",
            field_completion_pct=100.0,
            processing_time_ms=50.0,
            fields={
                "invoice_number": "INV-999",
                "amount": "99,999.00",
                "email": "hello@world.com",
            },
        )
        fdict = doc.fields_as_dict()
        assert fdict["invoice_number"] == "INV-999"
        assert fdict["amount"] == "99,999.00"
        assert fdict["email"] == "hello@world.com"

    async def test_analytics_empty(self, db_session):
        repo = DocumentRepository(db_session)
        analytics = await repo.get_analytics()
        assert analytics["total_documents"] == 0
        assert analytics["total_amount"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# ConversationRepository Tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestConversationRepository:

    async def test_save_and_get_history(self, db_session):
        repo = ConversationRepository(db_session)

        await repo.save_message(
            session_id="session-abc",
            role="user",
            content="What is the invoice total?",
            doc_id="some-doc",
        )
        await repo.save_message(
            session_id="session-abc",
            role="assistant",
            content="The invoice total is ₹88,500.",
            doc_id="some-doc",
            model_used="gpt-4o-mini/langchain-react",
            latency_ms=1230.5,
        )

        history = await repo.get_session_history("session-abc")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[1].model_used == "gpt-4o-mini/langchain-react"
        assert abs(history[1].latency_ms - 1230.5) < 0.01

    async def test_empty_session_history(self, db_session):
        repo = ConversationRepository(db_session)
        history = await repo.get_session_history("no-such-session")
        assert history == []

    async def test_list_sessions(self, db_session):
        repo = ConversationRepository(db_session)
        for sid in ["sess-1", "sess-2", "sess-3"]:
            await repo.save_message(
                session_id=sid,
                role="user",
                content=f"Hello from {sid}",
            )

        sessions = await repo.list_sessions()
        for sid in ["sess-1", "sess-2", "sess-3"]:
            assert sid in sessions

    async def test_delete_session(self, db_session):
        repo = ConversationRepository(db_session)
        await repo.save_message(session_id="to-delete", role="user", content="bye")
        await repo.save_message(session_id="to-delete", role="assistant", content="ok")

        deleted = await repo.delete_session("to-delete")
        assert deleted == 2

        remaining = await repo.get_session_history("to-delete")
        assert remaining == []
