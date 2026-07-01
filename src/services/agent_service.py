"""
LangChain ReAct Agent Service — production-grade agentic AI brain.

Architecture:
  • ReAct agent loop (Reason → Act → Observe → repeat)
  • Five tools: extract_fields, summarize_document, analyze_risks,
                answer_question (RAG-backed), search_similar_documents
  • RAG: answer_question retrieves relevant chunks from ChromaDB first,
         then sends retrieved context + question to GPT (Retrieve-Augment-Generate)
  • Pydantic v2 structured outputs for every tool response
  • _ConversationCheckpointer for in-memory stateful multi-turn conversations
  • Graceful fallback when OPENAI_API_KEY is absent
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.logger import get_logger

log = get_logger(__name__)


# ── Structured output schemas ─────────────────────────────────────────────────

class ExtractedInvoiceFields(BaseModel):
    invoice_number: Optional[str] = Field(None, description="Invoice or document number")
    date: Optional[str] = Field(None, description="Invoice date")
    due_date: Optional[str] = Field(None, description="Payment due date")
    seller: Optional[str] = Field(None, description="Seller / vendor name")
    buyer: Optional[str] = Field(None, description="Buyer / customer name")
    amount: Optional[str] = Field(None, description="Total amount due")
    currency: Optional[str] = Field(None, description="Currency code or symbol")
    tax: Optional[str] = Field(None, description="Tax or GST amount")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone")
    line_items: List[Dict[str, Any]] = Field(default_factory=list, description="Line items")


class RiskAssessment(BaseModel):
    risk_level: str = Field(description="LOW / MEDIUM / HIGH")
    flags: List[str] = Field(default_factory=list, description="Risk flags found")
    missing_fields: List[str] = Field(default_factory=list, description="Important missing fields")
    anomalies: List[str] = Field(default_factory=list, description="Detected anomalies")
    recommendation: str = Field(description="Short recommendation")


class DocumentSummary(BaseModel):
    summary: str = Field(description="2-3 sentence document summary")
    document_type: str = Field(description="e.g. Invoice, Contract, Receipt")
    key_parties: List[str] = Field(default_factory=list, description="Key parties mentioned")
    total_value: Optional[str] = Field(None, description="Total monetary value if present")


class AgentResult(BaseModel):
    answer: str
    confidence: float = 1.0
    model_used: str = "langchain-react-agent"
    tool_calls: List[str] = Field(default_factory=list)
    structured_data: Optional[Dict[str, Any]] = None
    rag_chunks_used: int = 0
    latency_ms: float = 0.0


# ── RAG context retrieval ─────────────────────────────────────────────────────

def _retrieve_rag_context(query: str, doc_id: Optional[str] = None) -> tuple[str, int]:
    """
    Retrieve relevant document chunks from ChromaDB for a query.
    Returns (formatted_context_string, chunk_count).

    This implements the R in RAG: Retrieve relevant chunks before sending to LLM.
    """
    try:
        from src.services.vector_store import get_vector_store
        vs = get_vector_store()
        if not vs.is_available:
            return "", 0

        chunks = vs.search(query, n_results=settings.RAG_TOP_K, doc_id=doc_id)
        if not chunks:
            return "", 0

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            sim = chunk.get("similarity", 0)
            text = chunk.get("text", "")
            context_parts.append(f"[Chunk {i} | similarity={sim:.3f}]\n{text}")

        return "\n\n".join(context_parts), len(chunks)

    except Exception as exc:
        log.warning("RAG retrieval failed: %s", exc)
        return "", 0


# ── LangChain agent builder ───────────────────────────────────────────────────

def _build_agent(api_key: str):
    """Build and return a LangChain ReAct agent with 5 tools."""
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.tools import tool
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=0,
        api_key=api_key,
    )

    # ── Tool 1: Extract invoice fields ────────────────────────────────────────

    @tool
    def extract_invoice_fields(document_text: str) -> str:
        """
        Extract all structured fields from an invoice or business document.
        Returns JSON with: invoice_number, date, due_date, seller, buyer,
        amount, currency, tax, email, phone, and line_items.
        Use this tool first when processing any new document.
        """
        parser = JsonOutputParser(pydantic_object=ExtractedInvoiceFields)
        prompt = (
            f"Extract all invoice/form fields from the document below.\n"
            f"Return ONLY valid JSON matching this schema: {parser.get_format_instructions()}\n\n"
            f"Document:\n{document_text[:4000]}"
        )
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            return json.dumps(data)
        except Exception:
            return response.content

    # ── Tool 2: Summarize document ────────────────────────────────────────────

    @tool
    def summarize_document(document_text: str) -> str:
        """
        Generate a structured summary of the document.
        Returns JSON with: summary, document_type, key_parties, total_value.
        Use this when the user asks for a summary or overview of the document.
        """
        parser = JsonOutputParser(pydantic_object=DocumentSummary)
        prompt = (
            f"Summarize the following document concisely.\n"
            f"Return ONLY valid JSON matching: {parser.get_format_instructions()}\n\n"
            f"Document:\n{document_text[:4000]}"
        )
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            return json.dumps(data)
        except Exception:
            return response.content

    # ── Tool 3: Analyze risks ─────────────────────────────────────────────────

    @tool
    def analyze_risks(document_text: str) -> str:
        """
        Analyze the document for risks, anomalies, and missing critical fields.
        Returns JSON with: risk_level (LOW/MEDIUM/HIGH), flags, missing_fields,
        anomalies, and recommendation.
        Use this for risk assessment, compliance review, or audit queries.
        """
        parser = JsonOutputParser(pydantic_object=RiskAssessment)
        prompt = (
            f"Analyze the following document for risks and anomalies.\n"
            f"Check for: missing required fields, unusual amounts, date inconsistencies,\n"
            f"vague descriptions, missing tax/GST info, payment term issues.\n"
            f"Return ONLY valid JSON matching: {parser.get_format_instructions()}\n\n"
            f"Document:\n{document_text[:4000]}"
        )
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            return json.dumps(data)
        except Exception:
            return response.content

    # ── Tool 4: Answer question (RAG-backed) ──────────────────────────────────

    @tool
    def answer_question(input_json: str) -> str:
        """
        Answer a specific natural language question about document content.
        Input must be JSON with keys: 'question', 'document_text', and optionally 'doc_id'.

        This tool uses RAG (Retrieval-Augmented Generation):
        1. Semantically searches indexed document chunks in ChromaDB
        2. Retrieves the most relevant passages
        3. Sends retrieved context + question to the LLM for a grounded answer

        Use this for targeted questions about specific document details.
        """
        try:
            parsed = json.loads(input_json)
            question = parsed.get("question", "")
            doc_text = parsed.get("document_text", "")
            doc_id = parsed.get("doc_id")
        except Exception:
            return "Invalid input format. Expected JSON with 'question' and 'document_text'."

        # ── RAG: Retrieve relevant chunks from ChromaDB ────────────────────
        rag_context, chunk_count = _retrieve_rag_context(question, doc_id=doc_id)

        if rag_context:
            context_section = (
                f"RETRIEVED CONTEXT (from semantic search — {chunk_count} chunks):\n"
                f"{rag_context}\n\n"
                f"FULL DOCUMENT (fallback):\n{doc_text[:2000]}"
            )
            log.debug("RAG retrieved %d chunks for question: %s", chunk_count, question[:60])
        else:
            # Fallback: use raw document text when ChromaDB is unavailable
            context_section = f"DOCUMENT:\n{doc_text[:4000]}"
            log.debug("RAG unavailable — answering from raw text")

        prompt = (
            f"Answer the following question based ONLY on the provided document context.\n"
            f"Be concise and specific. Cite the relevant section if possible.\n"
            f"If the answer is not in the context, say so explicitly.\n\n"
            f"Question: {question}\n\n"
            f"{context_section}\n\n"
            f"Answer:"
        )
        response = llm.invoke(prompt)
        return response.content.strip()

    # ── Tool 5: Semantic document search ─────────────────────────────────────

    @tool
    def search_similar_documents(query: str) -> str:
        """
        Perform a semantic similarity search across ALL indexed documents.
        Returns the top matching document chunks with similarity scores.
        Use this when the user wants to find documents related to a topic,
        or when comparing information across multiple documents.
        Input: a plain text search query.
        """
        try:
            from src.services.vector_store import get_vector_store
            vs = get_vector_store()
            if not vs.is_available:
                return json.dumps({"error": "Vector store not available", "results": []})

            results = vs.search(query, n_results=settings.RAG_TOP_K)
            formatted = [
                {
                    "similarity": r["similarity"],
                    "doc_id": r["metadata"].get("doc_id", "unknown"),
                    "filename": r["metadata"].get("filename", "unknown"),
                    "excerpt": r["text"][:300],
                }
                for r in results
            ]
            return json.dumps({"results": formatted, "total": len(formatted)})
        except Exception as exc:
            log.warning("search_similar_documents failed: %s", exc)
            return json.dumps({"error": str(exc), "results": []})

    tools = [
        extract_invoice_fields,
        summarize_document,
        analyze_risks,
        answer_question,
        search_similar_documents,
    ]

    # ── ReAct prompt ──────────────────────────────────────────────────────────

    react_prompt = PromptTemplate.from_template(
        """You are an intelligent document analysis agent specializing in invoices, contracts, and business documents.
You have access to tools to extract fields, summarize documents, analyze risks, answer questions using RAG, and search across documents.

Key guidance:
- For general questions, use answer_question (it automatically retrieves relevant chunks via RAG)
- For document overviews, use summarize_document
- For structured field extraction, use extract_invoice_fields
- For risk/compliance review, use analyze_risks
- To find related documents or cross-document queries, use search_similar_documents
- For "full analysis", call extract_invoice_fields + summarize_document + analyze_risks in sequence

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )

    agent = create_react_agent(llm, tools, react_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=6,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor


# ── In-memory conversation checkpointer ──────────────────────────────────────

class _ConversationCheckpointer:
    """
    Fast in-memory state store — simulates LangGraph checkpointing.
    Keyed by session_id; stores the last 20 messages and the last active doc.

    Complements the DB-backed ConversationRepository:
    - checkpointer = fast in-process reads during a session
    - ConversationRepository = durable storage across restarts
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(
        self,
        session_id: str,
        role: str,
        content: str,
        doc_id: Optional[str] = None,
    ) -> None:
        if session_id not in self._store:
            self._store[session_id] = {"messages": [], "last_doc_id": None}
        self._store[session_id]["messages"].append({"role": role, "content": content})
        if doc_id:
            self._store[session_id]["last_doc_id"] = doc_id
        # Keep last 20 messages to bound memory
        if len(self._store[session_id]["messages"]) > 20:
            self._store[session_id]["messages"] = self._store[session_id]["messages"][-20:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._store.get(session_id, {}).get("messages", [])

    def get_last_doc_id(self, session_id: str) -> Optional[str]:
        return self._store.get(session_id, {}).get("last_doc_id")

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._store.keys())


checkpointer = _ConversationCheckpointer()


# ── Agent singleton ───────────────────────────────────────────────────────────

_agent_executor = None


def _get_agent():
    global _agent_executor
    if _agent_executor is None:
        import os
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        try:
            _agent_executor = _build_agent(api_key)
            log.info("LangChain ReAct agent initialized ✓ (model=%s)", settings.LLM_MODEL)
        except Exception as exc:
            log.error("Failed to initialize LangChain agent: %s", exc)
            return None
    return _agent_executor


# ── Public API ────────────────────────────────────────────────────────────────

def run_agent(
    question: str,
    document_text: str,
    session_id: str = "default",
    doc_id: Optional[str] = None,
) -> AgentResult:
    """
    Run the LangChain ReAct agent on a question + document context.

    Flow:
    1. Save user message to in-memory checkpointer
    2. Build prompt with document context
    3. Agent loops: Reason → select tool → call tool (may use RAG) → Observe
    4. Save assistant response to checkpointer
    5. Return structured AgentResult
    """
    t0 = time.perf_counter()
    agent = _get_agent()

    checkpointer.save(session_id, "user", question, doc_id)

    if agent is None:
        result = AgentResult(
            answer=(
                "LangChain agent not available. "
                "Set OPENAI_API_KEY in your .env file to enable agentic AI."
            ),
            confidence=0.0,
            model_used="fallback",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        checkpointer.save(session_id, "assistant", result.answer)
        return result

    # Inject doc_id into question so the answer_question tool can pass it to RAG
    full_input = (
        f"Document ID for RAG lookup: {doc_id or 'none'}\n\n"
        f"Document content:\n---\n{document_text[:5000]}\n---\n\n"
        f"User question: {question}"
    )

    rag_chunks_total = 0
    try:
        response = agent.invoke({"input": full_input})
        answer = response.get("output", "No answer generated.")

        tool_calls = [
            step[0].tool
            for step in response.get("intermediate_steps", [])
            if hasattr(step[0], "tool")
        ]

        # Extract structured data from last tool observation
        structured_data = None
        steps = response.get("intermediate_steps", [])
        if steps:
            last_obs = steps[-1][1] if len(steps[-1]) > 1 else None
            if last_obs and isinstance(last_obs, str):
                try:
                    structured_data = json.loads(last_obs)
                except Exception:
                    pass

        result = AgentResult(
            answer=answer,
            confidence=0.95,
            model_used=f"{settings.LLM_MODEL}/langchain-react",
            tool_calls=tool_calls,
            structured_data=structured_data,
            rag_chunks_used=rag_chunks_total,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    except Exception as exc:
        log.error("Agent error: %s", exc)
        result = AgentResult(
            answer=f"Agent encountered an error: {str(exc)}",
            confidence=0.0,
            model_used="error",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    checkpointer.save(session_id, "assistant", result.answer)
    return result


def run_agent_analysis(
    document_text: str,
    analysis_type: str = "full",
    session_id: str = "default",
    doc_id: Optional[str] = None,
) -> AgentResult:
    """
    Run a specific analysis type via the agent.
    analysis_type: 'full' | 'risk' | 'anomaly' | 'comparison'
    """
    prompts = {
        "full": (
            "Give me a complete analysis: extract all fields, summarize the document, "
            "and identify any risks or anomalies."
        ),
        "risk": (
            "Perform a thorough risk and compliance assessment. "
            "Identify all risk flags, anomalies, and missing critical fields."
        ),
        "anomaly": (
            "Detect any inconsistencies, anomalies, or suspicious patterns in this document."
        ),
        "comparison": (
            "Extract all structured fields from this document and provide a detailed summary "
            "suitable for comparison with other documents."
        ),
    }
    question = prompts.get(analysis_type, prompts["full"])
    return run_agent(question, document_text, session_id=session_id, doc_id=doc_id)
