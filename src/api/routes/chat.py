"""Chat and Q&A endpoints — fully async, backed by LangChain ReAct agent + RAG."""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logger import get_logger
from src.db.database import get_db
from src.db.repository import ConversationRepository
from src.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ChatRequest,
    ChatResponse,
)
from src.services import document_service
from src.services.agent_service import (
    checkpointer,
    run_agent,
    run_agent_analysis,
)

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


# ── Ask endpoint (RAG-backed Q&A) ─────────────────────────────────────────────

@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    req: ChatRequest,
    session_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    """
    Ask the LangChain ReAct agent a question about one or all documents.

    The agent autonomously selects the right tool:
    - answer_question: RAG-backed (retrieves relevant chunks from ChromaDB first)
    - extract_invoice_fields: structured field extraction
    - summarize_document: document overview
    - analyze_risks: risk & compliance assessment
    - search_similar_documents: cross-document semantic search

    Parameters
    ----------
    document_id : target a specific document (leave empty for all documents)
    session_id  : query param — maintains stateful multi-turn conversation
    """
    if req.document_id:
        text = await document_service.get_document_text(req.document_id, db)
        if text is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document {req.document_id} not found",
            )
        doc_info = await document_service.get_document(req.document_id, db)
        source = doc_info.filename if doc_info else req.document_id
    else:
        all_docs = await document_service.get_all_documents_for_qa(db)
        if not all_docs:
            return ChatResponse(
                answer="No documents uploaded yet. Please upload a PDF first.",
                confidence=0.0,
                source_document=None,
                mode=req.mode,
                model_used="none",
                latency_ms=0.0,
            )
        # Combine text from all docs for cross-document Q&A
        text = "\n\n---\n\n".join(
            f"[Document: {d['filename']}]\n{d['text']}" for d in all_docs
        )
        source = "all documents"

    # Run LangChain agent off the event loop (sync LangChain call)
    result = await asyncio.to_thread(
        run_agent,
        req.question,
        text,
        session_id,
        req.document_id,
    )

    # Persist to DB asynchronously
    conv_repo = ConversationRepository(db)
    await conv_repo.save_message(
        session_id=session_id,
        role="user",
        content=req.question,
        doc_id=req.document_id,
    )
    await conv_repo.save_message(
        session_id=session_id,
        role="assistant",
        content=result.answer,
        doc_id=req.document_id,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
    )

    return ChatResponse(
        answer=result.answer,
        confidence=result.confidence,
        source_document=source,
        mode=req.mode,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
    )


# ── Analyze endpoint ───────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    req: AnalysisRequest,
    session_id: str = "default",
    db: AsyncSession = Depends(get_db),
):
    """
    Deep agentic analysis of a specific document.

    analysis_type options:
    - full      — extract fields + summarize + risk assessment (3 tool calls)
    - risk      — focused risk and compliance analysis
    - anomaly   — detect inconsistencies and suspicious patterns
    - comparison — structured field extraction with summary
    """
    text = await document_service.get_document_text(req.document_id, db)
    if text is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document {req.document_id} not found",
        )

    result = await asyncio.to_thread(
        run_agent_analysis,
        text,
        req.analysis_type,
        session_id,
        req.document_id,
    )

    # Persist analysis interaction to DB
    conv_repo = ConversationRepository(db)
    await conv_repo.save_message(
        session_id=session_id,
        role="user",
        content=f"[{req.analysis_type} analysis]",
        doc_id=req.document_id,
    )
    await conv_repo.save_message(
        session_id=session_id,
        role="assistant",
        content=result.answer,
        doc_id=req.document_id,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
    )

    key_points: list = []
    if result.structured_data:
        sd = result.structured_data
        key_points = (
            sd.get("flags", [])
            or sd.get("key_parties", [])
            or sd.get("anomalies", [])
        )

    return AnalysisResponse(
        document_id=req.document_id,
        analysis_type=req.analysis_type,
        findings=result.answer,
        key_points=key_points[:5],
        confidence=result.confidence,
        model_used=result.model_used,
        latency_ms=result.latency_ms,
    )


# ── Conversation state ────────────────────────────────────────────────────────

class ConversationState(BaseModel):
    session_id: str
    messages: List[dict]
    total_messages: int
    last_doc_id: Optional[str] = None


@router.get("/history/{session_id}", response_model=ConversationState)
async def get_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    source: str = "db",
):
    """
    Retrieve conversation history for a session.

    source=db   — from persistent SQLAlchemy store (survives restarts)
    source=mem  — from in-memory checkpointer (current session only)
    """
    if source == "mem":
        messages = checkpointer.get_history(session_id)
        last_doc = checkpointer.get_last_doc_id(session_id)
    else:
        conv_repo = ConversationRepository(db)
        db_msgs = await conv_repo.get_session_history(session_id)
        messages = [
            {"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in db_msgs
        ]
        last_doc = db_msgs[-1].doc_id if db_msgs else None

    return ConversationState(
        session_id=session_id,
        messages=messages,
        total_messages=len(messages),
        last_doc_id=last_doc,
    )


@router.delete("/history/{session_id}")
async def clear_history(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Clear conversation state from both memory and DB."""
    checkpointer.clear(session_id)
    conv_repo = ConversationRepository(db)
    deleted = await conv_repo.delete_session(session_id)
    return {
        "success": True,
        "message": f"Session {session_id} cleared",
        "db_messages_deleted": deleted,
    }


@router.get("/sessions")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    """List all conversation sessions with history in the DB."""
    conv_repo = ConversationRepository(db)
    sessions = await conv_repo.list_sessions()
    return {"sessions": sessions, "total": len(sessions)}


# ── Agent info ────────────────────────────────────────────────────────────────

@router.get("/agent/info")
async def agent_info():
    """Return the active agent configuration and capability summary."""
    import os
    from src.services.vector_store import get_vector_store
    api_key = (
        __import__("src.core.config", fromlist=["settings"]).settings.OPENAI_API_KEY
        or os.getenv("OPENAI_API_KEY", "")
    )
    vs = get_vector_store()
    return {
        "agent_type": "LangChain ReAct Agent",
        "model": "gpt-4o-mini",
        "framework": "LangChain",
        "tools": [
            "extract_invoice_fields",
            "summarize_document",
            "analyze_risks",
            "answer_question (RAG-backed)",
            "search_similar_documents",
        ],
        "rag": {
            "enabled": vs.is_available,
            "vector_store": "ChromaDB",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunks_indexed": vs.document_count() if vs.is_available else 0,
        },
        "state_persistence": {
            "in_memory": "ConversationCheckpointer",
            "persistent": "SQLAlchemy (SQLite/Postgres)",
        },
        "structured_outputs": "Pydantic v2",
        "agent_ready": bool(api_key),
        "status": "active" if api_key else "requires OPENAI_API_KEY",
    }
