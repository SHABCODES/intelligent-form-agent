"""Chat and Q&A endpoints — powered by LangChain ReAct agent."""

from __future__ import annotations
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.logger import get_logger
from src.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
    ConversationMessage,
)
from src.services import document_service
from src.services.agent_service import (
    checkpointer,
    run_agent,
    run_agent_analysis,
)

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Agent chat endpoint ────────────────────────────────────────────────────────

@router.post("/ask", response_model=ChatResponse)
def ask_question(req: ChatRequest, session_id: str = "default"):
    """
    Ask the LangChain ReAct agent a question about one or all documents.
    The agent autonomously selects the right tool(s) to answer.

    - Set `document_id` to query a specific document.
    - Leave empty to search across ALL documents.
    - Pass `session_id` query param to maintain conversation state.
    """
    t0 = time.perf_counter()

    if req.document_id:
        text = document_service.get_document_text(req.document_id)
        if text is None:
            raise HTTPException(status_code=404, detail=f"Document {req.document_id} not found")
        doc_info = document_service.get_document(req.document_id)
        source = doc_info.filename if doc_info else req.document_id
    else:
        all_docs = document_service.get_all_documents_for_qa()
        if not all_docs:
            return ChatResponse(
                answer="No documents uploaded yet. Please upload a PDF first.",
                confidence=0.0,
                source_document=None,
                mode=req.mode,
                model_used="none",
                latency_ms=0.0,
            )
        # Use combined text from all docs
        text = "\n\n---\n\n".join(
            f"[Document: {d['filename']}]\n{d['text']}" for d in all_docs
        )
        source = "all documents"

    result = run_agent(
        question=req.question,
        document_text=text,
        session_id=session_id,
        doc_id=req.document_id,
    )

    return ChatResponse(
        answer=result.answer,
        confidence=result.confidence,
        source_document=source,
        mode=req.mode,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
    )


# ── Agent analysis endpoint ────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse)
def analyze_document(req: AnalysisRequest, session_id: str = "default"):
    """
    Deep agentic analysis of a specific document.

    `analysis_type` options:
    - `full`     — extract fields + summarize + risk assessment
    - `risk`     — focused risk and compliance analysis
    - `anomaly`  — detect inconsistencies and suspicious patterns
    - `comparison` — structured field extraction with summary
    """
    text = document_service.get_document_text(req.document_id)
    if text is None:
        raise HTTPException(status_code=404, detail=f"Document {req.document_id} not found")

    result = run_agent_analysis(
        document_text=text,
        analysis_type=req.analysis_type,
        session_id=session_id,
        doc_id=req.document_id,
    )

    # Parse key points from structured data if available
    key_points: list = []
    if result.structured_data:
        sd = result.structured_data
        key_points = sd.get("flags", []) or sd.get("key_parties", []) or []

    return AnalysisResponse(
        document_id=req.document_id,
        analysis_type=req.analysis_type,
        findings=result.answer,
        key_points=key_points[:5],
        confidence=result.confidence,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
    )


# ── Agent state / conversation history ────────────────────────────────────────

class ConversationState(BaseModel):
    session_id: str
    messages: List[dict]
    total_messages: int
    last_doc_id: Optional[str] = None


@router.get("/history/{session_id}", response_model=ConversationState)
def get_history(session_id: str):
    """Retrieve conversation history and state for a session (from checkpointer)."""
    messages = checkpointer.get_history(session_id)
    last_doc = checkpointer.get_last_doc_id(session_id)
    return ConversationState(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages),
        last_doc_id=last_doc,
    )


@router.delete("/history/{session_id}")
def clear_history(session_id: str):
    """Clear conversation state for a session."""
    checkpointer.clear(session_id)
    return {"success": True, "message": f"Session {session_id} cleared"}


@router.get("/sessions")
def list_sessions():
    """List all active conversation sessions."""
    sessions = checkpointer.list_sessions()
    return {"sessions": sessions, "total": len(sessions)}


# ── Agent info endpoint ────────────────────────────────────────────────────────

@router.get("/agent/info")
def agent_info():
    """Return information about the active agent configuration."""
    import os
    has_key = bool(os.getenv("OPENAI_API_KEY", ""))
    return {
        "agent_type": "LangChain ReAct Agent",
        "model": "gpt-4o-mini",
        "framework": "LangChain",
        "tools": [
            "extract_invoice_fields",
            "summarize_document",
            "analyze_risks",
            "answer_question",
        ],
        "state_persistence": "In-memory checkpointer",
        "structured_outputs": "Pydantic v2",
        "agent_ready": has_key,
        "status": "active" if has_key else "requires OPENAI_API_KEY",
    }
