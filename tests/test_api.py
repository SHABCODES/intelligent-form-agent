"""
API endpoint tests using FastAPI TestClient (no real model inference).
These tests mock the AI models to run fast without GPU/downloads.
"""

import io
import json
import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_health_endpoint(client):
    """Health endpoint should always return 200."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "version" in data
    assert "uptime_seconds" in data


def test_metrics_endpoint(client):
    """Metrics endpoint should return system stats."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data
    assert "cache" in data


def test_list_documents_empty(client):
    """Document list should return empty array initially."""
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)
    assert "total" in data


def test_upload_invalid_type(client):
    """Non-PDF upload should return 415."""
    content = io.BytesIO(b"not a pdf")
    res = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", content, "text/plain")},
    )
    assert res.status_code == 415


def test_upload_no_file(client):
    """Upload with no file should return 422."""
    res = client.post("/api/documents/upload")
    assert res.status_code == 422


def test_get_nonexistent_document(client):
    """Fetching a non-existent document should return 404."""
    res = client.get("/api/documents/nonexistent-id-12345")
    assert res.status_code == 404


def test_delete_nonexistent_document(client):
    """Deleting a non-existent document should return 404."""
    res = client.delete("/api/documents/nonexistent-id-12345")
    assert res.status_code == 404


def test_chat_ask_no_documents(client):
    """Chat when no documents are loaded should return helpful message."""
    res = client.post(
        "/api/chat/ask",
        json={"question": "What is the total amount?", "mode": "qa"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert "confidence" in data
    assert "upload" in data["answer"].lower() or data["confidence"] == 0.0


def test_chat_ask_invalid_document(client):
    """Chat targeting a non-existent doc_id should return 404."""
    res = client.post(
        "/api/chat/ask",
        json={
            "question": "What is the total?",
            "mode": "qa",
            "document_id": "nonexistent-doc-id",
        },
    )
    assert res.status_code == 404


def test_chat_question_validation(client):
    """Question that is too short should fail validation."""
    res = client.post(
        "/api/chat/ask",
        json={"question": "a", "mode": "qa"},  # min_length=2
    )
    assert res.status_code == 422


def test_analyze_nonexistent_document(client):
    """Analyze of a non-existent document should return 404."""
    res = client.post(
        "/api/chat/analyze",
        json={"document_id": "nonexistent", "analysis_type": "full"},
    )
    assert res.status_code == 404


def test_openapi_schema(client):
    """OpenAPI schema should be accessible."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    schema = res.json()
    assert "paths" in schema
    assert "/api/documents/upload" in schema["paths"]


def test_analytics_empty(client):
    """Analytics endpoint should respond even with no documents."""
    res = client.get("/api/documents/analytics/collection")
    assert res.status_code == 200
    data = res.json()
    assert "total_documents" in data


def test_history_nonexistent_session(client):
    """Should return empty list for unknown session."""
    res = client.get("/api/chat/history/unknown-session-xyz")
    assert res.status_code == 200
    assert res.json() == []


def test_clear_history(client):
    """Clearing history for any session should succeed."""
    res = client.delete("/api/chat/history/test-session-123")
    assert res.status_code == 200
    assert res.json()["success"] is True
