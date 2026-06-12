"""
Tests for extraction_service and text_utils.
These are pure unit tests with no model loading.
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.extraction_service import (
    extract_fields,
    field_completion_pct,
    parse_amount_value,
)
from src.utils.text_utils import clean_text, chunk_text, extract_currency


# ── Field extraction tests ─────────────────────────────────────────────────────

class TestFieldExtraction:

    def test_invoice_number_slash_format(self):
        text = "Invoice Number: INV/2024/001"
        fields = extract_fields(text)
        assert fields.invoice_number == "INV/2024/001"

    def test_invoice_number_hash_format(self):
        text = "Invoice #: INT/2024/007"
        fields = extract_fields(text)
        assert fields.invoice_number == "INT/2024/007"

    def test_invoice_number_reference(self):
        text = "Invoice reference: MSI/2023/006"
        fields = extract_fields(text)
        assert fields.invoice_number == "MSI/2023/006"

    def test_date_extraction(self):
        text = "Date: 15-01-2024"
        fields = extract_fields(text)
        assert fields.date == "15-01-2024"

    def test_invoice_date_extraction(self):
        text = "Invoice Date: 25/01/2024"
        fields = extract_fields(text)
        assert fields.date is not None

    def test_amount_inr(self):
        text = "Total Amount: ₹1,53,400.00"
        fields = extract_fields(text)
        assert fields.amount == "1,53,400.00"

    def test_amount_usd(self):
        text = "Amount: $15,000.00"
        fields = extract_fields(text)
        assert fields.amount == "15,000.00"

    def test_amount_total_keyword(self):
        text = "Total: ₹75,000.00"
        fields = extract_fields(text)
        assert fields.amount == "75,000.00"

    def test_customer_name_bill_to(self):
        text = "Bill To: Sharma Enterprises"
        fields = extract_fields(text)
        assert fields.name is not None

    def test_email_extraction(self):
        text = "Email: rajesh.sharma@sharmaenterprises.com"
        fields = extract_fields(text)
        assert fields.email == "rajesh.sharma@sharmaenterprises.com"

    def test_phone_extraction(self):
        text = "Phone: +91 98765 43210"
        fields = extract_fields(text)
        assert fields.phone is not None

    def test_gst_extraction(self):
        text = "GSTIN: 27AAAAA0000A1Z5"
        fields = extract_fields(text)
        assert fields.gst == "27AAAAA0000A1Z5"

    def test_currency_inr(self):
        text = "Total: ₹75,000.00"
        fields = extract_fields(text)
        assert fields.currency == "INR"

    def test_currency_usd(self):
        text = "Amount: $15,000.00"
        fields = extract_fields(text)
        assert fields.currency == "USD"

    def test_empty_text(self):
        fields = extract_fields("")
        assert fields.invoice_number is None
        assert fields.amount is None
        assert fields.email is None

    def test_full_invoice(self, sample_invoice_text):
        fields = extract_fields(sample_invoice_text)
        assert fields.invoice_number is not None
        assert fields.date is not None
        assert fields.email is not None
        assert fields.gst is not None
        assert fields.amount is not None


# ── Completion scoring ─────────────────────────────────────────────────────────

class TestCompletionScoring:

    def test_full_completion(self, sample_invoice_text):
        fields = extract_fields(sample_invoice_text)
        pct = field_completion_pct(fields)
        assert pct > 50.0

    def test_minimal_invoice(self, minimal_invoice_text):
        fields = extract_fields(minimal_invoice_text)
        pct = field_completion_pct(fields)
        assert 0.0 <= pct <= 100.0

    def test_empty_document(self):
        fields = extract_fields("")
        pct = field_completion_pct(fields)
        assert pct == 0.0


# ── Amount parsing ─────────────────────────────────────────────────────────────

class TestAmountParsing:

    def test_comma_formatted(self):
        from src.models.schemas import ExtractedFields
        f = ExtractedFields(amount="1,53,400.00")
        val = parse_amount_value(f)
        assert val == 153400.0

    def test_plain_number(self):
        from src.models.schemas import ExtractedFields
        f = ExtractedFields(amount="75000.00")
        val = parse_amount_value(f)
        assert val == 75000.0

    def test_none_amount(self):
        from src.models.schemas import ExtractedFields
        f = ExtractedFields(amount=None)
        val = parse_amount_value(f)
        assert val is None


# ── Text utilities ─────────────────────────────────────────────────────────────

class TestTextUtils:

    def test_clean_text_removes_extra_spaces(self):
        text = "hello   world\n\n\nfoo"
        cleaned = clean_text(text)
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_chunk_text_basic(self):
        words = " ".join(["word"] * 200)
        chunks = chunk_text(words, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_empty(self):
        assert chunk_text("") == []

    def test_extract_currency_inr(self):
        assert extract_currency("Total: ₹75,000") == "INR"

    def test_extract_currency_usd(self):
        assert extract_currency("Total: $1,000") == "USD"

    def test_extract_currency_none(self):
        assert extract_currency("No money here") is None
