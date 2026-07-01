"""
Shared test fixtures for the entire test suite.

Key design decisions:
- In-memory SQLite for all DB tests (no file I/O, no cleanup needed)
- TestClient wraps the full FastAPI app (integration tests hit real routes)
- LangChain FakeListLLM mocks the OpenAI API so no API key is needed
- ChromaDB is mocked at the VectorStore layer
"""

from __future__ import annotations

import asyncio
import os
import pytest
import pytest_asyncio

from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from unittest.mock import AsyncMock, MagicMock, patch

# ── Environment setup (must happen before any imports that read settings) ─────
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-testing")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("USE_REDIS", "false")

from src.db.database import Base, get_db
from src.api.app import create_app

# ── In-memory test DB ─────────────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False,
                                  connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        # Import all models so Base knows about them
        import src.db.models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    """Fresh async session per test — rolls back after each test."""
    TestSessionLocal = async_sessionmaker(
        bind=test_engine, class_=AsyncSession,
        autocommit=False, autoflush=False, expire_on_commit=False,
    )
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()


# ── FastAPI test client ────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="session")
async def test_app(test_engine):
    """Full FastAPI app with DB overridden to in-memory test DB."""
    app = create_app()
    TestSessionLocal = async_sessionmaker(
        bind=test_engine, class_=AsyncSession,
        autocommit=False, autoflush=False, expire_on_commit=False,
    )

    async def override_get_db():
        async with TestSessionLocal() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    yield app


@pytest_asyncio.fixture
async def client(test_app):
    """Async HTTP test client."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as ac:
        yield ac


# ── Sample document text ───────────────────────────────────────────────────────

@pytest.fixture
def sample_invoice_text():
    return """
    TAX INVOICE
    Invoice Number: INV/2024/001
    Date: 15-03-2024
    Due Date: 30-03-2024

    Seller: Acme Technologies Pvt Ltd
    GSTIN: 27AAPCA1234H1Z5
    Email: billing@acme.com
    Phone: +91 98765 43210

    Bill To: TechCorp Solutions
    Customer Name: Rajesh Kumar

    Items:
    Cloud Hosting Services  12  5,000.00  60,000.00
    Support & Maintenance   1   15,000.00 15,000.00

    Subtotal: ₹75,000.00
    GST (18%): ₹13,500.00
    Grand Total: ₹88,500.00
    """


@pytest.fixture(scope="session", autouse=True)
def mock_vector_store():
    """Mock ChromaDB vector store singleton globally — avoids requiring ChromaDB/HF model downloads."""
    import src.services.vector_store
    mock_vs = MagicMock()
    mock_vs.is_available = True
    mock_vs.document_count.return_value = 0
    mock_vs.add_document.return_value = None
    mock_vs.search.return_value = [
        {
            "text": "Invoice Number: INV/2024/001\nGrand Total: ₹88,500.00",
            "metadata": {"doc_id": "test-doc-id", "filename": "invoice.pdf", "chunk_index": 0},
            "similarity": 0.92,
        }
    ]
    mock_vs.delete_document.return_value = None
    
    # Override singleton instance directly to bypass namespace-import mocking traps
    src.services.vector_store._vector_store_instance = mock_vs
    yield mock_vs

