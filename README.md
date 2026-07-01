# Intelligent Document AI Platform 🤖

A **production-grade** agentic AI platform for intelligent PDF document processing, RAG-backed Q&A, and structured data extraction.

Upload a PDF → the system extracts structured fields, generates an AI summary, indexes chunks into ChromaDB, and persists everything to a SQLAlchemy database. Then ask questions in natural language — the LangChain ReAct agent retrieves relevant passages via RAG and answers with full traceability.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Frontend (Dark-mode SPA)                   │
│    Upload │ Library │ Chat │ Analytics                  │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP / REST  (async FastAPI)
┌───────────────────────▼─────────────────────────────────┐
│                   FastAPI (async)                       │
│  /documents  │  /chat  │  /health  │  /metrics          │
│         CORS · Request timing middleware               │
└───────┬───────────────────────┬─────────────────────────┘
        │                       │
┌───────▼────────┐   ┌──────────▼──────────────────────┐
│ Document       │   │  LangChain ReAct Agent           │
│ Service        │   │                                  │
│ (async)        │   │  5 Tools:                        │
│                │   │  ① extract_invoice_fields        │
│ ┌────────────┐ │   │  ② summarize_document            │
│ │SQLAlchemy  │ │   │  ③ analyze_risks                 │
│ │(aiosqlite) │ │   │  ④ answer_question ← RAG-backed  │
│ └────────────┘ │   │  ⑤ search_similar_documents      │
│                │   │                                  │
│ ┌────────────┐ │◄──┤  _ConversationCheckpointer       │
│ │ ChromaDB   │ │   │  ConversationRepository (DB)     │
│ │  (RAG)     │ │   └──────────────────────────────────┘
│ └────────────┘ │
│                │
│ ┌────────────┐ │
│ │ TTL Cache  │ │
│ │ (in-mem /  │ │
│ │  Redis)    │ │
│ └────────────┘ │
│                │
│ PyMuPDF + OCR  │
└────────────────┘
```

### RAG Pattern (Retrieve-Augment-Generate)

When you ask a question, `answer_question` tool:
1. **Retrieve** — semantic search in ChromaDB (sentence-transformer embeddings)
2. **Augment** — top-k chunks injected as context into the LLM prompt
3. **Generate** — GPT-4o-mini produces a grounded, citation-aware answer

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangChain (ReAct agent) |
| LLM | GPT-4o-mini via OpenAI API |
| Structured outputs | Pydantic v2 |
| Database (async) | SQLAlchemy 2.0 + aiosqlite (SQLite → Postgres-upgradeable) |
| Vector store / RAG | ChromaDB + sentence-transformers/all-MiniLM-L6-v2 |
| State persistence | In-memory checkpointer + SQLAlchemy ConversationRepository |
| Backend API | FastAPI (async) + Uvicorn |
| PDF processing | PyMuPDF + Tesseract OCR fallback |
| Cache | In-memory TTL cache (Redis-compatible interface) |
| Testing | pytest + pytest-asyncio (35+ tests) |
| Containerization | Docker + docker-compose |

---

## Quick Start

### 1. Clone & configure

```bash
git clone https://github.com/SHABCODES/intelligent-form-agent.git
cd intelligent-form-agent
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

### 2. Docker (recommended)

```bash
docker-compose up --build
```

### 3. Local dev

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
python server.py
```

Visit **http://localhost:8000** for the UI, **http://localhost:8000/docs** for Swagger.

---

## API Reference

### Upload document
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@invoice.pdf"
```

### Ask the agent (RAG-backed)
```bash
curl -X POST "http://localhost:8000/api/chat/ask?session_id=my-session" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total amount?", "document_id": "<doc_id>"}'
```

### Full agentic analysis
```bash
curl -X POST http://localhost:8000/api/chat/analyze \
  -H "Content-Type: application/json" \
  -d '{"document_id": "<doc_id>", "analysis_type": "risk"}'
```

### Semantic search (ChromaDB)
```bash
curl "http://localhost:8000/api/documents/search/semantic?q=cloud+hosting+invoice"
```

### Get conversation history (DB-persisted)
```bash
curl http://localhost:8000/api/chat/history/my-session
```

### Export extracted data
```bash
curl http://localhost:8000/api/documents/<doc_id>/export/json
curl http://localhost:8000/api/documents/<doc_id>/export/csv
```

---

## Analysis Types

| Type | Agent behaviour |
|---|---|
| `full` | Calls 3 tools: extract fields + summarize + risk assessment |
| `risk` | Focused risk/compliance analysis — flags missing fields, anomalies |
| `anomaly` | Detects inconsistencies and suspicious patterns |
| `comparison` | Structured extraction + summary for multi-document comparison |

---

## Pydantic Output Schemas

Every tool returns a validated Pydantic v2 model:

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
    email: Optional[str]
    phone: Optional[str]
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
│   │   ├── app.py                   # FastAPI factory + startup (DB init)
│   │   └── routes/
│   │       ├── chat.py              # Agent Q&A, analysis, session management
│   │       ├── documents.py         # Upload, list, search, export endpoints
│   │       └── health.py            # Health check + metrics
│   ├── core/
│   │   ├── config.py                # pydantic-settings (DATABASE_URL, OPENAI_API_KEY, etc.)
│   │   ├── exceptions.py
│   │   └── logger.py
│   ├── db/                          ← SQLAlchemy layer
│   │   ├── database.py              # Async engine, session factory, get_db()
│   │   ├── models.py                # Document, ExtractedField, ConversationMessage ORM
│   │   └── repository.py           # DocumentRepository, ConversationRepository
│   ├── models/
│   │   └── schemas.py               # API request/response Pydantic schemas
│   ├── services/
│   │   ├── agent_service.py         # ★ LangChain ReAct agent + 5 tools + RAG
│   │   ├── ai_service.py            # Thin facade (preload, summarize)
│   │   ├── document_service.py      # Async processing pipeline
│   │   ├── extraction_service.py    # Regex field extraction
│   │   ├── cache_service.py         # In-memory TTL cache / Redis
│   │   └── vector_store.py          # ChromaDB wrapper (add, search, delete)
│   └── utils/
│       ├── pdf_utils.py             # PyMuPDF + Tesseract OCR
│       └── text_utils.py            # chunk_text, clean_text, extract_currency
├── tests/
│   ├── conftest.py                  # Fixtures: in-memory DB, async client, mocks
│   ├── test_services.py             # Unit: extraction, cache, text utils, vector store
│   ├── test_db.py                   # Async repository CRUD tests
│   ├── test_agent.py                # Agent schemas, checkpointer, RAG, fallback
│   └── test_api.py                  # Integration: full HTTP lifecycle
├── frontend/
│   └── index.html                  # Dark-mode SPA (Upload/Library/Chat/Analytics)
├── data/                            # SQLite DB persisted here (gitignored)
├── chroma_db/                       # ChromaDB index (gitignored)
├── uploads/                         # Uploaded PDFs (gitignored)
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                   # pytest config (asyncio_mode=auto)
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

35+ tests — no OpenAI API key required. ChromaDB is mocked. Uses in-memory SQLite.

---

## Upgrading to PostgreSQL

Change one line in `.env`:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/docai
```

Install the asyncpg driver:
```bash
pip install asyncpg
```

Zero application code changes needed.

---

## License

MIT
