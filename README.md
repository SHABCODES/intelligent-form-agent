# Intelligent Form Agent 🤖

A **LangChain ReAct agent** that autonomously reads, understands, and analyzes invoices and business documents. Upload a PDF — the agent decides which tools to use, extracts structured data, assesses risks, and answers natural language questions.

Built to demonstrate agentic AI architecture: orchestrator → tools → structured outputs → stateful conversations.

---

## Architecture

```
PDF Upload → Text Extraction (PyMuPDF/OCR)
                    ↓
         LangChain ReAct Agent Loop
         ┌──────────────────────────┐
         │  Reason → Act → Observe  │
         │                          │
         │  Tools:                  │
         │  • extract_invoice_fields│  ← Pydantic structured output
         │  • summarize_document    │  ← Pydantic structured output
         │  • analyze_risks         │  ← Pydantic structured output
         │  • answer_question       │  ← Natural language
         └──────────────────────────┘
                    ↓
         In-Memory Checkpointer (stateful multi-turn)
                    ↓
         FastAPI REST API  ←→  Frontend UI
```

**Key concepts demonstrated:**
- **ReAct agent loop** — the agent reasons about which tool to call, calls it, observes the result, and repeats until it has a final answer
- **Tool-based architecture** — each capability is a discrete tool with a clear interface
- **Pydantic structured outputs** — every tool returns validated, typed data
- **State persistence** — `_ConversationCheckpointer` stores session history across turns
- **Multi-step workflows** — complex queries (e.g. "full analysis") trigger multiple tool calls automatically

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent framework | LangChain (ReAct agent) |
| LLM | GPT-4o-mini via OpenAI API |
| Structured outputs | Pydantic v2 |
| State persistence | In-memory checkpointer |
| Backend API | FastAPI + Uvicorn |
| PDF processing | PyMuPDF + Tesseract OCR |
| Containerization | Docker + docker-compose |

---

## Quick Start

### 1. Clone & configure
```bash
git clone https://github.com/SHABCODES/intelligent-form-agent.git
cd intelligent-form-agent
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run with Docker (recommended)
```bash
docker-compose up --build
```

### 3. Or run locally
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

Visit **http://localhost:8000** for the UI, **http://localhost:8000/docs** for the API.

---

## API Endpoints

### Upload a document
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@invoice.pdf"
```

### Ask the agent a question
```bash
curl -X POST http://localhost:8000/api/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total amount and who is the seller?", "document_id": "<doc_id>"}'
```

### Run a full agentic analysis
```bash
curl -X POST http://localhost:8000/api/chat/analyze \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<doc_id>", "analysis_type": "risk"}'
```

### Check conversation state
```bash
curl http://localhost:8000/api/chat/history/my-session
```

### Agent info
```bash
curl http://localhost:8000/api/chat/agent/info
```

---

## Analysis Types

| Type | What the agent does |
|------|-------------------|
| `full` | Calls all 3 tools: extract fields + summarize + risk assessment |
| `risk` | Focused risk analysis — flags missing fields, anomalies, compliance issues |
| `anomaly` | Detects inconsistencies and suspicious patterns |
| `comparison` | Structured extraction + summary for comparison across docs |

---

## Structured Output Schemas

The agent returns validated Pydantic models:

```python
class ExtractedInvoiceFields(BaseModel):
    invoice_number: Optional[str]
    date: Optional[str]
    due_date: Optional[str]
    seller: Optional[str]
    buyer: Optional[str]
    amount: Optional[str]
    currency: Optional[str]
    tax: Optional[str]
    line_items: List[Dict[str, Any]]

class RiskAssessment(BaseModel):
    risk_level: str           # LOW / MEDIUM / HIGH
    flags: List[str]
    missing_fields: List[str]
    anomalies: List[str]
    recommendation: str

class DocumentSummary(BaseModel):
    summary: str
    document_type: str
    key_parties: List[str]
    total_value: Optional[str]
```

---

## Project Structure

```
intelligent-form-agent/
├── src/
│   ├── api/
│   │   ├── app.py                  # FastAPI factory
│   │   └── routes/
│   │       ├── chat.py             # Agent chat & analysis endpoints
│   │       ├── documents.py        # Upload, list, export endpoints
│   │       └── health.py           # Health check
│   ├── core/
│   │   ├── config.py               # Settings (pydantic-settings)
│   │   ├── exceptions.py
│   │   └── logger.py
│   ├── models/
│   │   └── schemas.py              # API request/response schemas
│   ├── services/
│   │   ├── agent_service.py        # ★ LangChain ReAct agent + checkpointer
│   │   ├── ai_service.py           # Thin wrapper → agent_service
│   │   ├── document_service.py     # Document storage & retrieval
│   │   ├── extraction_service.py   # Regex field extraction (fallback)
│   │   └── cache_service.py        # In-memory cache
│   └── utils/
│       ├── pdf_utils.py            # PyMuPDF + OCR text extraction
│       └── text_utils.py
├── data/                           # Sample invoice PDFs
├── frontend/                       # Single-page UI
├── tests/                          # Pytest test suite
├── .env.example                    # Environment config template
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── server.py                       # Entry point
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT
