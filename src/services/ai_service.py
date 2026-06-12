"""
AI Service — now delegates to LangChain ReAct agent.

Preserves the original function signatures for backward compatibility
while routing all inference through the new agent_service.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger
from src.services.agent_service import run_agent, run_agent_analysis

log = get_logger(__name__)


def preload_models() -> None:
    """Called at FastAPI startup. Agent initializes lazily on first call."""
    log.info("LangChain ReAct agent will initialize on first request.")
    log.info("Ensure OPENAI_API_KEY is set in your .env file.")


def answer_question(question: str, document_text: str, session_id: str = "default") -> Dict[str, Any]:
    """Answer a question about a document using the ReAct agent."""
    result = run_agent(question, document_text, session_id=session_id)
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "model_used": result.model_used,
    }


def answer_across_documents(question: str, docs: List[Dict[str, Any]], session_id: str = "default") -> Dict[str, Any]:
    """Answer a question across multiple documents."""
    combined_text = "\n\n---\n\n".join(
        f"[Document: {d['filename']}]\n{d['text']}" for d in docs
    )
    result = run_agent(question, combined_text, session_id=session_id)
    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "model_used": result.model_used,
        "source_document": docs[0]["filename"] if docs else None,
    }


def analyze_document(document_text: str, analysis_type: str = "full", session_id: str = "default") -> Dict[str, Any]:
    """Run agentic analysis on a document."""
    result = run_agent_analysis(document_text, analysis_type, session_id=session_id)
    key_points: list = []
    if result.structured_data:
        sd = result.structured_data
        key_points = sd.get("flags", []) or sd.get("key_parties", []) or []
    return {
        "findings": result.answer,
        "key_points": key_points[:5],
        "confidence": result.confidence,
        "model_used": result.model_used,
        "latency_ms": result.latency_ms,
    }
