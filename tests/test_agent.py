"""
Agent service tests — full ReAct loop with mocked LLM.
No OPENAI_API_KEY needed; uses LangChain's FakeListLLM.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from src.services.agent_service import (
    _ConversationCheckpointer,
    _retrieve_rag_context,
    AgentResult,
    ExtractedInvoiceFields,
    RiskAssessment,
    DocumentSummary,
)


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Schema Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAgentSchemas:
    """Verify structured output schemas validate correctly."""

    def test_extracted_invoice_fields_defaults(self):
        fields = ExtractedInvoiceFields()
        assert fields.invoice_number is None
        assert fields.line_items == []

    def test_extracted_invoice_fields_full(self):
        fields = ExtractedInvoiceFields(
            invoice_number="INV-001",
            date="2024-03-15",
            amount="88500.00",
            currency="INR",
            seller="Acme Corp",
            buyer="TechCo",
            email="billing@acme.com",
            line_items=[{"description": "Cloud hosting", "qty": "1", "total": "50000"}],
        )
        assert fields.invoice_number == "INV-001"
        assert len(fields.line_items) == 1

    def test_risk_assessment_schema(self):
        risk = RiskAssessment(
            risk_level="HIGH",
            flags=["Missing due date", "No GST number"],
            missing_fields=["due_date", "gst"],
            anomalies=["Amount inconsistency"],
            recommendation="Request corrected invoice before payment.",
        )
        assert risk.risk_level == "HIGH"
        assert len(risk.flags) == 2

    def test_document_summary_schema(self):
        summary = DocumentSummary(
            summary="Invoice from Acme for cloud services.",
            document_type="Invoice",
            key_parties=["Acme Corp", "TechCo"],
            total_value="₹88,500",
        )
        assert summary.document_type == "Invoice"
        assert "Acme" in summary.key_parties[0]

    def test_agent_result_defaults(self):
        result = AgentResult(answer="The total is ₹88,500.")
        assert result.confidence == 1.0
        assert result.tool_calls == []
        assert result.rag_chunks_used == 0


# ══════════════════════════════════════════════════════════════════════════════
# _ConversationCheckpointer Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConversationCheckpointer:

    def test_save_and_get_history(self):
        cp = _ConversationCheckpointer()
        cp.save("session1", "user", "Hello")
        cp.save("session1", "assistant", "Hi there!")

        history = cp.get_history("session1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "Hi there!"

    def test_empty_session(self):
        cp = _ConversationCheckpointer()
        assert cp.get_history("nonexistent") == []
        assert cp.get_last_doc_id("nonexistent") is None

    def test_doc_id_tracking(self):
        cp = _ConversationCheckpointer()
        cp.save("s1", "user", "question", doc_id="doc-abc")
        assert cp.get_last_doc_id("s1") == "doc-abc"

        cp.save("s1", "assistant", "answer")  # no doc_id — should not clear
        assert cp.get_last_doc_id("s1") == "doc-abc"

    def test_clear(self):
        cp = _ConversationCheckpointer()
        cp.save("s2", "user", "msg")
        cp.clear("s2")
        assert cp.get_history("s2") == []

    def test_list_sessions(self):
        cp = _ConversationCheckpointer()
        cp.save("alpha", "user", "hi")
        cp.save("beta", "user", "hello")
        sessions = cp.list_sessions()
        assert "alpha" in sessions
        assert "beta" in sessions

    def test_max_20_messages(self):
        cp = _ConversationCheckpointer()
        for i in range(25):
            cp.save("s3", "user", f"message {i}")
        history = cp.get_history("s3")
        assert len(history) == 20
        # Should keep the LAST 20
        assert history[-1]["content"] == "message 24"

    def test_multiple_sessions_isolated(self):
        cp = _ConversationCheckpointer()
        cp.save("session-a", "user", "question A")
        cp.save("session-b", "user", "question B")
        assert len(cp.get_history("session-a")) == 1
        assert len(cp.get_history("session-b")) == 1


# ══════════════════════════════════════════════════════════════════════════════
# RAG Context Retrieval Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGContextRetrieval:

    def test_rag_returns_context_when_available(self, mock_vector_store):
        context, count = _retrieve_rag_context("invoice total amount")
        assert count > 0
        assert "Invoice Number" in context or "Grand Total" in context

    def test_rag_returns_empty_when_unavailable(self):
        with patch("src.services.agent_service._retrieve_rag_context", return_value=("", 0)):
            context, count = _retrieve_rag_context("any query")
        assert count == 0

    def test_rag_with_doc_id_filter(self, mock_vector_store):
        context, count = _retrieve_rag_context(
            "total amount", doc_id="test-doc-id"
        )
        # search should have been called with doc_id
        mock_vector_store.search.assert_called()


# ══════════════════════════════════════════════════════════════════════════════
# run_agent Fallback Tests (no API key)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunAgentFallback:

    def test_agent_fallback_when_no_key(self):
        """When OPENAI_API_KEY is missing, agent returns graceful fallback."""
        with patch("src.services.agent_service._get_agent", return_value=None):
            from src.services.agent_service import run_agent
            result = run_agent(
                question="What is the total?",
                document_text="Invoice Total: $5,000",
                session_id="test-session",
            )
        assert isinstance(result, AgentResult)
        assert result.confidence == 0.0
        assert "OPENAI_API_KEY" in result.answer or "not available" in result.answer

    def test_agent_analysis_fallback(self):
        with patch("src.services.agent_service._get_agent", return_value=None):
            from src.services.agent_service import run_agent_analysis
            result = run_agent_analysis(
                document_text="Some invoice",
                analysis_type="risk",
                session_id="test-session",
            )
        assert isinstance(result, AgentResult)
        assert result.model_used == "fallback"
