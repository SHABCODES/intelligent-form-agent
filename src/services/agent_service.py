"""
LangChain ReAct Agent Service — the new AI brain.

Replaces the HuggingFace pipeline approach with a proper agentic system:
  • ReAct agent loop (Reason + Act)
  • Four tools: extract_fields, summarize_document, analyze_risks, answer_question
  • Pydantic structured outputs for every tool
  • In-memory checkpointer for stateful multi-turn conversations
  • Falls back gracefully if OPENAI_API_KEY is not set
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
    flags: List[str] = Field(default_factory=list, description="List of risk flags found")
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
    latency_ms: float = 0.0


# ── LangChain agent builder ───────────────────────────────────────────────────

def _build_agent(api_key: str):
    """Build and return a LangChain ReAct agent with tools."""
    from langchain_openai import ChatOpenAI
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.tools import tool
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    # ── Tool definitions ──────────────────────────────────────────────────

    @tool
    def extract_invoice_fields(document_text: str) -> str:
        """
        Extract structured fields from an invoice or form document.
        Returns JSON with: invoice_number, date, due_date, seller, buyer,
        amount, currency, tax, email, phone, and line_items.
        Use this tool first when processing any document.
        """
        parser = JsonOutputParser(pydantic_object=ExtractedInvoiceFields)
        prompt = f"""Extract all invoice/form fields from the document below.
Return ONLY valid JSON matching this schema: {parser.get_format_instructions()}

Document:
{document_text[:4000]}
"""
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            return json.dumps(data)
        except Exception:
            return response.content

    @tool
    def summarize_document(document_text: str) -> str:
        """
        Generate a structured summary of the document.
        Returns JSON with: summary, document_type, key_parties, total_value.
        Use this when the user asks for a summary or overview.
        """
        parser = JsonOutputParser(pydantic_object=DocumentSummary)
        prompt = f"""Summarize the following document concisely.
Return ONLY valid JSON matching: {parser.get_format_instructions()}

Document:
{document_text[:4000]}
"""
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            return json.dumps(data)
        except Exception:
            return response.content

    @tool
    def analyze_risks(document_text: str) -> str:
        """
        Analyze the document for risks, anomalies, and missing critical fields.
        Returns JSON with: risk_level, flags, missing_fields, anomalies, recommendation.
        Use this for risk assessment, anomaly detection, or audit queries.
        """
        parser = JsonOutputParser(pydantic_object=RiskAssessment)
        prompt = f"""Analyze the following document for risks and anomalies.
Check for: missing required fields, unusual amounts, date inconsistencies,
vague descriptions, missing tax info, payment term issues.
Return ONLY valid JSON matching: {parser.get_format_instructions()}

Document:
{document_text[:4000]}
"""
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            return json.dumps(data)
        except Exception:
            return response.content

    @tool
    def answer_question(input_json: str) -> str:
        """
        Answer a specific natural language question about document content.
        Input must be JSON with keys: 'question' and 'document_text'.
        Use this for targeted questions about specific document details.
        """
        try:
            parsed = json.loads(input_json)
            question = parsed.get("question", "")
            doc_text = parsed.get("document_text", "")
        except Exception:
            return "Invalid input format. Expected JSON with 'question' and 'document_text'."

        prompt = f"""Answer the following question based ONLY on the document provided.
Be concise and specific. If the answer is not in the document, say so clearly.

Question: {question}

Document:
{doc_text[:4000]}

Answer:"""
        response = llm.invoke(prompt)
        return response.content.strip()

    tools = [extract_invoice_fields, summarize_document, analyze_risks, answer_question]

    # ── ReAct prompt ──────────────────────────────────────────────────────
    react_prompt = PromptTemplate.from_template("""You are an intelligent document analysis agent specializing in invoices, contracts, and forms.
You have access to tools to extract fields, summarize documents, analyze risks, and answer questions.

Always use the most appropriate tool for the task. For general questions, use answer_question.
For overviews, use summarize_document. For structured extraction, use extract_invoice_fields.

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
Thought:{agent_scratchpad}""")

    agent = create_react_agent(llm, tools, react_prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    return executor


# ── Checkpointer (in-memory state persistence) ────────────────────────────────

class _ConversationCheckpointer:
    """
    Simple in-memory state store — simulates LangGraph checkpointing.
    Keyed by session_id, stores message history and last agent context.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def save(self, session_id: str, role: str, content: str, doc_id: Optional[str] = None):
        if session_id not in self._store:
            self._store[session_id] = {"messages": [], "last_doc_id": None}
        self._store[session_id]["messages"].append({"role": role, "content": content})
        if doc_id:
            self._store[session_id]["last_doc_id"] = doc_id
        # Keep last 20 messages
        if len(self._store[session_id]["messages"]) > 20:
            self._store[session_id]["messages"] = self._store[session_id]["messages"][-20:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._store.get(session_id, {}).get("messages", [])

    def get_last_doc_id(self, session_id: str) -> Optional[str]:
        return self._store.get(session_id, {}).get("last_doc_id")

    def clear(self, session_id: str):
        self._store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self._store.keys())


checkpointer = _ConversationCheckpointer()


# ── Public API ────────────────────────────────────────────────────────────────

_agent_executor = None


def _get_agent():
    global _agent_executor
    if _agent_executor is None:
        import os
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        try:
            _agent_executor = _build_agent(api_key)
            log.info("LangChain ReAct agent initialized ✓")
        except Exception as e:
            log.error("Failed to initialize LangChain agent: %s", e)
            return None
    return _agent_executor


def run_agent(
    question: str,
    document_text: str,
    session_id: str = "default",
    doc_id: Optional[str] = None,
) -> AgentResult:
    """
    Run the LangChain ReAct agent on a question + document.
    Saves conversation state to the checkpointer.
    """
    t0 = time.perf_counter()
    agent = _get_agent()

    # Save user message to state
    checkpointer.save(session_id, "user", question, doc_id)

    if agent is None:
        # Graceful fallback — agent unavailable (no API key)
        result = AgentResult(
            answer="LangChain agent not available. Set OPENAI_API_KEY in your .env file to enable agentic AI.",
            confidence=0.0,
            model_used="fallback",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        checkpointer.save(session_id, "assistant", result.answer)
        return result

    # Inject document context into the question
    full_input = f"""Document content:
---
{document_text[:5000]}
---

User question: {question}"""

    try:
        response = agent.invoke({"input": full_input})
        answer = response.get("output", "No answer generated.")
        tool_calls = [
            step[0].tool
            for step in response.get("intermediate_steps", [])
            if hasattr(step[0], "tool")
        ]

        # Try to extract structured data from last tool observation
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
            model_used="gpt-4o-mini/langchain-react",
            tool_calls=tool_calls,
            structured_data=structured_data,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    except Exception as e:
        log.error("Agent error: %s", e)
        result = AgentResult(
            answer=f"Agent encountered an error: {str(e)}",
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
    Run a specific analysis type using the agent.
    analysis_type: 'full' | 'risk' | 'anomaly' | 'comparison'
    """
    prompts = {
        "full": "Give me a complete analysis of this document: extract all fields, summarize it, and identify any risks.",
        "risk": "Perform a thorough risk assessment of this document. Identify all risk flags, anomalies, and missing critical fields.",
        "anomaly": "Detect any inconsistencies, anomalies, or suspicious patterns in this document.",
        "comparison": "Extract all structured fields from this document and provide a detailed summary.",
    }
    question = prompts.get(analysis_type, prompts["full"])
    return run_agent(question, document_text, session_id=session_id, doc_id=doc_id)
