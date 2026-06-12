"""Text preprocessing and chunking utilities."""

from __future__ import annotations
import re
from typing import List


def clean_text(text: str) -> str:
    """Normalise whitespace and remove control characters."""
    text = re.sub(r"[\r\n]+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n₹$€£¥]", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split text into overlapping word-based chunks for embedding.
    
    Parameters
    ----------
    chunk_size : approximate word count per chunk
    overlap    : word overlap between adjacent chunks
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def truncate_for_model(text: str, max_chars: int = 3000) -> str:
    """Safely truncate text to fit within model context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def extract_currency(text: str) -> str | None:
    """Detect the primary currency symbol / code in the text."""
    mapping = {
        "₹": "INR", "$": "USD", "€": "EUR", "£": "GBP",
        "¥": "JPY", "USD": "USD", "EUR": "EUR", "INR": "INR",
    }
    for symbol, code in mapping.items():
        if symbol in text:
            return code
    return None
