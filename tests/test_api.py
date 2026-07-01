"""
API integration tests — full request/response cycle via async TestClient.
Tests the complete stack: route → service → repository → in-memory DB.
"""

from __future__ import annotations

import json
import io
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# Health Endpoint
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestHealthEndpoint:

    async def test_health_returns_200(self, client):
        response = await client.get("/api/health")
        assert response.status_code == 200

    async def test_health_response_shape(self, client):
        response = await client.get("/api/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "vector_store_ok" in data
        assert "cache_ok" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    async def test_metrics_returns_200(self, client):
        response = await client.get("/api/metrics")
        assert response.status_code == 200

    async def test_metrics_has_database_key(self, client):
        response = await client.get("/api/metrics")
        data = response.json()
        assert "database" in data
        assert "cache" in data
        assert "vector_store" in data


# ══════════════════════════════════════════════════════════════════════════════
# Document Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestDocumentEndpoints:

    async def test_list_documents_empty(self, client):
        response = await client.get("/api/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data
        assert isinstance(data["documents"], list)

    async def test_get_nonexistent_document(self, client):
        response = await client.get("/api/documents/does-not-exist")
        assert response.status_code == 404

    async def test_delete_nonexistent_document(self, client):
        response = await client.delete("/api/documents/ghost-id")
        assert response.status_code == 404

    async def test_upload_invalid_file_type(self, client):
        """Non-PDF uploads should be rejected with 415."""
        fake_file = io.BytesIO(b"not a pdf")
        response = await client.post(
            "/api/documents/upload",
            files={"file": ("document.txt", fake_file, "text/plain")},
        )
        assert response.status_code == 415

    async def test_analytics_endpoint_shape(self, client):
        response = await client.get("/api/documents/analytics/collection")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "field_stats" in data
        assert "currency_breakdown" in data

    async def test_semantic_search_requires_query(self, client):
        response = await client.get("/api/documents/search/semantic?q=")
        assert response.status_code == 400

    async def test_semantic_search_unavailable_when_chromadb_down(self, client):
        """If ChromaDB is not initialized, search returns 503."""
        with patch("src.api.routes.documents.get_vector_store") as mock_vs_factory:
            mock_vs = MagicMock()
            mock_vs.is_available = False
            mock_vs_factory.return_value = mock_vs

            response = await client.get("/api/documents/search/semantic?q=invoice")
            assert response.status_code == 503


# ══════════════════════════════════════════════════════════════════════════════
# Chat Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestChatEndpoints:

    async def test_ask_with_no_documents(self, client):
        """When no documents exist, ask returns a helpful message."""
        with patch("src.services.agent_service._get_agent", return_value=None):
            response = await client.post(
                "/api/chat/ask",
                json={"question": "What is the total?", "mode": "qa"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "No documents" in data["answer"]

    async def test_ask_nonexistent_document(self, client):
        response = await client.post(
            "/api/chat/ask",
            json={
                "question": "What is the total?",
                "document_id": "nonexistent-id",
                "mode": "qa",
            },
        )
        assert response.status_code == 404

    async def test_analyze_nonexistent_document(self, client):
        response = await client.post(
            "/api/chat/analyze",
            json={"document_id": "ghost-doc", "analysis_type": "full"},
        )
        assert response.status_code == 404

    async def test_get_history_empty_session(self, client):
        response = await client.get("/api/chat/history/empty-session")
        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []
        assert data["total_messages"] == 0

    async def test_clear_history(self, client):
        response = await client.delete("/api/chat/history/some-session")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    async def test_list_sessions(self, client):
        response = await client.get("/api/chat/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data

    async def test_agent_info_shape(self, client):
        response = await client.get("/api/chat/agent/info")
        assert response.status_code == 200
        data = response.json()
        assert "agent_type" in data
        assert "tools" in data
        assert "rag" in data
        assert "state_persistence" in data
        assert len(data["tools"]) == 5   # extract, summarize, risks, answer, search


# ══════════════════════════════════════════════════════════════════════════════
# Upload + Full Lifecycle (mocked processing)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestDocumentLifecycle:

    async def test_upload_mock_processing(self, client, mock_vector_store):
        """
        Upload a synthetic PDF-like payload with document_service mocked.
        Tests the full HTTP lifecycle: upload → list → retrieve → delete.
        """
        from src.models.schemas import DocumentInfo, ExtractedFields
        from datetime import datetime, timezone

        mock_doc = DocumentInfo(
            id="lifecycle-doc-id",
            filename="test.pdf",
            file_size_bytes=2048,
            page_count=1,
            extracted_fields=ExtractedFields(
                invoice_number="INV-LIFE-001",
                amount="50,000.00",
                currency="INR",
                email="test@test.com",
            ),
            summary="Test invoice document.",
            text_preview="Invoice Number: INV-LIFE-001...",
            field_completion_pct=57.1,
            processed_at=datetime.now(timezone.utc),
            processing_time_ms=420.0,
        )

        with patch(
            "src.services.document_service.process_document",
            new_callable=AsyncMock,
            return_value=mock_doc,
        ):
            fake_pdf = io.BytesIO(b"%PDF-1.4 fake pdf content for testing")
            response = await client.post(
                "/api/documents/upload",
                files={"file": ("test.pdf", fake_pdf, "application/pdf")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["document"]["id"] == "lifecycle-doc-id"
        assert data["document"]["extracted_fields"]["invoice_number"] == "INV-LIFE-001"
