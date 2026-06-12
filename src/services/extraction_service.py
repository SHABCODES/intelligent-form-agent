"""
Regex-based field extraction service.

Handles: invoice numbers, dates, names, sellers, amounts,
         GST/tax IDs, emails, phones, line items.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional

from src.core.logger import get_logger
from src.models.schemas import ExtractedFields
from src.utils.text_utils import extract_currency

log = get_logger(__name__)


# ── Pattern banks ─────────────────────────────────────────────────────────────

_INVOICE_PATTERNS = [
    r"Invoice\s*Number\s*[:]?\s*([A-Za-z0-9\/\-]+)",
    r"Invoice\s*#\s*[:]?\s*([A-Za-z0-9\/\-]+)",
    r"Invoice\s*No\.?\s*[:]?\s*([A-Za-z0-9\/\-]+)",
    r"Invoice\s*reference\s*[:]?\s*([A-Za-z0-9\/\-]+)",
    r"INV[-/](\w+)",
    r"INV[/](\d{4}[-]\d+[/]\d+)",
    r"Bill\s*No\.?\s*[:]?\s*([A-Za-z0-9\/\-]+)",
    r"Order\s*No\.?\s*[:]?\s*([A-Za-z0-9\/\-]+)",
]

_DATE_PATTERNS = [
    r"Invoice\s*Date\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    r"Date\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    r"Date\s*[:]?\s*(\d{1,2}\s+\w+\s+\d{4})",
    r"Issued\s*On\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
]

_DUE_DATE_PATTERNS = [
    r"Due\s*Date\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    r"Payment\s*Due\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    r"Due\s*On\s*[:]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
]

_NAME_PATTERNS = [
    r"Bill\s*To\s*[:]?\s*([^\n\r]+)",
    r"Customer\s*Name\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"Customer\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"Client\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"Name\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
]

_SELLER_PATTERNS = [
    r"Seller\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"From\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"Company\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"Supplier\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"Vendor\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
    r"Issued\s*By\s*[:]?\s*([A-Z][a-zA-Z\s\.&]+)",
]

_AMOUNT_PATTERNS = [
    r"Grand\s*Total\s*[:]?\s*[₹$€£]?\s*([\d,]+\.?\d*)",
    r"Total\s*Amount\s*[:]?\s*[₹$€£]?\s*([\d,]+\.?\d*)",
    r"Amount\s*Due\s*[:]?\s*[₹$€£]?\s*([\d,]+\.?\d*)",
    r"Total\s*[:]?\s*[₹$€£]?\s*([\d,]+\.?\d*)",
    r"TOTAL\s*[:]?\s*[₹$€£]?\s*([\d,]+\.?\d*)",
    r"due\s*is\s*[₹$€£]?\s*([\d,]+\.?\d*)",
    r"Amount\s*[:]?\s*[₹$€£]?\s*([\d,]+\.?\d*)",
    r"[₹]\s*([\d,]+\.?\d*)",
    r"\$\s*([\d,]+\.?\d*)",
]

_GST_PATTERNS = [
    r"GSTIN\s*[#]?\s*[:]?\s*([A-Z0-9]{15})",
    r"GST\s*[:]?\s*([A-Z0-9]{15})",
    r"Tax\s*ID\s*[:]?\s*([A-Z0-9\-]+)",
    r"VAT\s*[:]?\s*([A-Z0-9\-]+)",
]

_EMAIL_PATTERN = r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
_PHONE_PATTERN = r"(\+?\d[\d\s\-\(\)]{7,}\d)"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _search(text: str, patterns: List[str]) -> Optional[str]:
    for pattern in patterns:
        try:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return (m.group(1) if m.lastindex and m.lastindex >= 1
                        else m.group(0)).strip()
        except Exception:
            continue
    return None


def _extract_email(text: str) -> Optional[str]:
    m = re.search(_EMAIL_PATTERN, text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> Optional[str]:
    m = re.search(_PHONE_PATTERN, text)
    return m.group(1) if m else None


def _parse_amount(amount_str: Optional[str]) -> Optional[float]:
    if not amount_str:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", str(amount_str)))
    except ValueError:
        return None


def _extract_line_items(text: str) -> List[Dict[str, Any]]:
    """
    Heuristic line-item extractor.
    Looks for rows of: description + quantity + unit price + total.
    """
    items: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"(?P<desc>[A-Za-z][A-Za-z\s\-\.]{3,40})\s+"
        r"(?P<qty>\d+(?:\.\d+)?)\s+"
        r"(?P<unit>[₹$€£]?[\d,]+\.?\d*)\s+"
        r"(?P<total>[₹$€£]?[\d,]+\.?\d*)",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        items.append({
            "description": m.group("desc").strip(),
            "quantity": m.group("qty"),
            "unit_price": m.group("unit"),
            "total": m.group("total"),
        })
    return items[:20]   # cap for safety


# ── Public API ────────────────────────────────────────────────────────────────

def extract_fields(text: str) -> ExtractedFields:
    """
    Run all extraction patterns against *text* and return an
    ``ExtractedFields`` model with all matched values.
    """
    return ExtractedFields(
        invoice_number=_search(text, _INVOICE_PATTERNS),
        date=_search(text, _DATE_PATTERNS),
        due_date=_search(text, _DUE_DATE_PATTERNS),
        name=_search(text, _NAME_PATTERNS),
        seller=_search(text, _SELLER_PATTERNS),
        email=_extract_email(text),
        phone=_extract_phone(text),
        amount=_search(text, _AMOUNT_PATTERNS),
        gst=_search(text, _GST_PATTERNS),
        currency=extract_currency(text),
        line_items=_extract_line_items(text),
    )


def field_completion_pct(fields: ExtractedFields) -> float:
    """Return the percentage of key fields that were successfully extracted."""
    key_fields = [
        fields.invoice_number, fields.date, fields.name,
        fields.seller, fields.email, fields.phone, fields.amount,
    ]
    filled = sum(1 for f in key_fields if f is not None)
    return round((filled / len(key_fields)) * 100, 1)


def parse_amount_value(fields: ExtractedFields) -> Optional[float]:
    return _parse_amount(fields.amount)
