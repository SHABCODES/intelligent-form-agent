"""
AI Service — thin facade over the LangChain agent.

This module previously contained HuggingFace pipeline wrappers.
Now it delegates to agent_service which runs LangChain + GPT-4o-mini.

Keeping this module preserves the import path used by app.py startup
and makes the dependency direction explicit:
  routes → document_service → ai_service → agent_service → LangChain/OpenAI
"""

from __future__ import annotations

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


def preload_models() -> None:
    """
    Warm up the LangChain agent on startup.
    No HuggingFace models to download — this just validates the API key
    and logs the agent status.
    """
    import os
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if api_key:
        log.info("OpenAI API key present — LangChain agent will initialize on first request.")
    else:
        log.warning(
            "OPENAI_API_KEY not set. "
            "The agentic AI features will return a fallback message. "
            "Set OPENAI_API_KEY in your .env file to enable full functionality."
        )

    # Warm up ChromaDB vector store
    try:
        from src.services.vector_store import get_vector_store
        vs = get_vector_store()
        if vs.is_available:
            log.info("ChromaDB vector store ready — %d chunks indexed", vs.document_count())
        else:
            log.warning("ChromaDB not available — RAG will fall back to raw text")
    except Exception as exc:
        log.warning("Vector store warmup failed: %s", exc)


def summarize_document(text: str) -> str:
    """
    Summarize a document using the LangChain agent.
    Kept for backward compatibility — document_service calls this directly.
    """
    if not text or len(text.strip()) < 20:
        return "Document too short for summary."
    try:
        from langchain_openai import ChatOpenAI
        import os
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return _basic_summary(text)
        llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0, api_key=api_key)
        prompt = f"Provide a concise 2-3 sentence summary of this document:\n\n{text[:3000]}"
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as exc:
        log.warning("AI summarization failed: %s", exc)
        return _basic_summary(text)


def _basic_summary(text: str) -> str:
    """Fallback summary when AI is unavailable."""
    clean = " ".join(text.split())
    return clean[:300] + ("..." if len(clean) > 300 else "")
