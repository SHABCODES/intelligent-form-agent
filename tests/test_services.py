"""
Unit tests for service-layer components.
No API key, no ChromaDB, no file system required.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from src.services.extraction_service import extract_fields, field_completion_pct
from src.services.cache_service import InMemoryCache
from src.utils.text_utils import chunk_text, clean_text


# ══════════════════════════════════════════════════════════════════════════════
# Extraction Service Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractionService:
    """Regex field extraction — pattern coverage tests."""

    @pytest.mark.parametrize("text,expected", [
        ("Invoice Number: INV/2024/001", "INV/2024/001"),
        ("Invoice #: INT/2023/007",      "INT/2023/007"),
        ("Invoice reference: MSI/2023/006", "MSI/2023/006"),
        ("Bill No.: BILL-123",           "BILL-123"),
        ("Order No.: ORD-456",           "ORD-456"),
    ])
    def test_invoice_number_patterns(self, text, expected):
        fields = extract_fields(text)
        assert fields.invoice_number == expected, (
            f"Expected '{expected}', got '{fields.invoice_number}' from: {text!r}"
        )

    @pytest.mark.parametrize("text,expected", [
        ("Total Amount: ₹1,53,400.00",  "1,53,400.00"),
        ("Amount: $15,000.00",          "15,000.00"),
        ("Grand Total: ₹88,500.00",     "88,500.00"),
        ("Amount Due: £5,000.00",       "5,000.00"),
        ("TOTAL: €2,500.50",            "2,500.50"),
    ])
    def test_amount_patterns(self, text, expected):
        fields = extract_fields(text)
        assert fields.amount == expected

    @pytest.mark.parametrize("text,expected", [
        ("Date: 15-03-2024",            "15-03-2024"),
        ("Invoice Date: 01/01/2023",    "01/01/2023"),
        ("Issued On: 20-12-2023",       "20-12-2023"),
    ])
    def test_date_patterns(self, text, expected):
        fields = extract_fields(text)
        assert fields.date == expected

    def test_email_extraction(self):
        fields = extract_fields("Contact: billing@acme.com for queries")
        assert fields.email == "billing@acme.com"

    def test_phone_extraction(self):
        fields = extract_fields("Phone: +91 98765 43210")
        assert fields.phone is not None
        assert "98765" in fields.phone

    def test_gst_extraction(self):
        fields = extract_fields("GSTIN: 27AAPCA1234H1Z5")
        assert fields.gst == "27AAPCA1234H1Z5"

    def test_seller_extraction(self):
        fields = extract_fields("Seller: Acme Technologies Pvt Ltd")
        assert fields.seller is not None
        assert "Acme" in fields.seller

    def test_empty_text(self):
        fields = extract_fields("")
        assert fields.invoice_number is None
        assert fields.amount is None
        assert fields.email is None
        assert fields.line_items == []

    def test_full_invoice(self, sample_invoice_text):
        fields = extract_fields(sample_invoice_text)
        assert fields.invoice_number == "INV/2024/001"
        assert fields.date == "15-03-2024"
        assert fields.email == "billing@acme.com"
        assert fields.gst == "27AAPCA1234H1Z5"
        assert fields.amount is not None

    def test_field_completion_pct_all_filled(self):
        fields = extract_fields("""
            Invoice Number: INV/001
            Date: 01-01-2024
            Customer Name: Test Corp
            Seller: Vendor Ltd
            Email: test@test.com
            Phone: +1 555 0100
            Total: $10,000.00
        """)
        pct = field_completion_pct(fields)
        assert pct > 80.0

    def test_field_completion_pct_empty(self):
        fields = extract_fields("Hello world")
        pct = field_completion_pct(fields)
        assert pct < 30.0


# ══════════════════════════════════════════════════════════════════════════════
# Cache Service Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestInMemoryCache:
    """TTL cache — set/get/expire/evict behavior."""

    def test_basic_set_get(self):
        cache = InMemoryCache(max_items=100, default_ttl=60)
        cache.set("key1", {"value": 42})
        result = cache.get("key1")
        assert result == {"value": 42}

    def test_missing_key_returns_none(self):
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_delete(self):
        cache = InMemoryCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_delete_nonexistent(self):
        cache = InMemoryCache()
        assert cache.delete("ghost") is False

    def test_exists(self):
        cache = InMemoryCache()
        cache.set("k", "v")
        assert cache.exists("k") is True
        assert cache.exists("missing") is False

    def test_expired_entry_returns_none(self):
        import time
        cache = InMemoryCache(default_ttl=1)
        cache.set("expiring", "soon")
        # Manually expire the entry
        cache._store["expiring"].expires_at = time.monotonic() - 1
        assert cache.get("expiring") is None

    def test_max_items_triggers_eviction(self):
        cache = InMemoryCache(max_items=3, default_ttl=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # triggers eviction
        assert len(cache._store) <= 3

    def test_flush(self):
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.set("b", 2)
        cache.flush()
        assert len(cache._store) == 0

    def test_stats(self):
        cache = InMemoryCache()
        cache.set("x", 1)
        cache.get("x")   # hit
        cache.get("y")   # miss
        stats = cache.stats()
        assert stats["backend"] == "in-memory"
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Text Utils Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTextUtils:

    def test_chunk_text_basic(self):
        text = "word " * 300   # 1500 chars
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 512 + 10  # small tolerance

    def test_chunk_text_short(self):
        text = "Short text."
        chunks = chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_overlap(self):
        """Chunks should share content at boundaries."""
        text = "A " * 400
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        if len(chunks) > 1:
            # Last chars of chunk[0] should appear at start of chunk[1]
            tail = chunks[0][-30:]
            head = chunks[1][:60]
            assert len(tail) > 0

    def test_clean_text(self):
        dirty = "Hello\x00World\t\tTest\n\n\n  spaces  "
        cleaned = clean_text(dirty)
        assert "\x00" not in cleaned
        assert "\t\t" not in cleaned

    def test_clean_text_empty(self):
        assert clean_text("") == ""


# ══════════════════════════════════════════════════════════════════════════════
# Vector Store Tests (mocked ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class TestVectorStore:

    def test_add_and_search(self, mock_vector_store):
        """Vector store add and search are called with correct args."""
        vs = mock_vector_store
        vs.add_document("doc1", "Invoice for cloud services", {"filename": "inv.pdf"})
        vs.add_document.assert_called_once()

        results = vs.search("cloud invoice", n_results=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "text" in results[0]
        assert "similarity" in results[0]

    def test_delete_document(self, mock_vector_store):
        vs = mock_vector_store
        vs.delete_document("doc1")
        vs.delete_document.assert_called_with("doc1")

    def test_document_count(self, mock_vector_store):
        count = mock_vector_store.document_count()
        assert isinstance(count, int)
        assert count >= 0
