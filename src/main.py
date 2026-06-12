"""
Enhanced main.py — backward-compatible with original CLI usage.
Now wraps the new service layer for clean architecture.
"""

from __future__ import annotations
import os
import sys

# Add parent to path so 'src' is importable when called directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import pipeline

from src.core.logger import get_logger
from src.services.extraction_service import extract_fields, field_completion_pct
from src.utils.pdf_utils import extract_document_text
from src.utils.text_utils import clean_text

log = get_logger(__name__)


class IntelligentFormAgent:
    """
    Original API surface — preserved for backward compatibility.
    Internally uses the new service layer.
    """

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.processed_forms = []
        self.last_result = None
        self._initialize_ai_models()

    def _initialize_ai_models(self):
        log.info("Loading AI models...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            device=self.device,
        )
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=self.device,
        )
        log.info("AI models loaded ✓")

    # ── Text extraction ───────────────────────────────────────────────────

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            import fitz
            doc = fitz.open(file_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            log.warning("PDF extraction error: %s", e)
            return ""

    def extract_text_with_ocr(self, file_path: str) -> str:
        try:
            from pdf2image import convert_from_path
            import pytesseract
            pages = convert_from_path(file_path, dpi=200)
            return "\n".join(pytesseract.image_to_string(p) for p in pages)
        except Exception as e:
            log.warning("OCR error: %s", e)
            return ""

    def extract_document_text(self, file_path: str) -> str:
        text, _ = extract_document_text(file_path)
        return text

    # ── Field extraction ──────────────────────────────────────────────────

    def extract_form_fields(self, text_content: str) -> dict:
        fields = extract_fields(text_content)
        return fields.model_dump(exclude={"line_items"})

    # ── Q&A ───────────────────────────────────────────────────────────────

    def answer_question(self, text_content: str, question: str) -> dict:
        try:
            context = text_content[:3000]
            result = self.qa_pipeline(question=question, context=context)
            return {"answer": result["answer"], "confidence": round(result["score"], 3)}
        except Exception:
            return {"answer": "Unable to process question", "confidence": 0.0}

    def summarize_document(self, text_content: str) -> str:
        if len(text_content) < 100:
            return "Document too short for summary"
        try:
            txt = text_content[:1500]
            result = self.summarizer(txt, max_length=150, min_length=50, do_sample=False)
            return result[0]["summary_text"]
        except Exception:
            return "Summary generation failed"

    # ── Document processing ───────────────────────────────────────────────

    def process_document(self, file_path: str) -> dict | None:
        text, _ = extract_document_text(file_path)
        if not text.strip():
            return None
        text = clean_text(text)
        form_info = self.extract_form_fields(text)
        summary = self.summarize_document(text)
        form_data = {
            "filename": os.path.basename(file_path),
            "text": text[:1000] + "..." if len(text) > 1000 else text,
            "info": form_info,
            "summary": summary,
        }
        self.processed_forms.append(form_data)
        self.last_result = form_data
        return form_data

    def process_multiple_documents(self, file_paths: list) -> list:
        self.processed_forms = []
        for fp in file_paths:
            self.process_document(fp)
        return self.processed_forms

    def ask_question_about_forms(self, question: str) -> dict:
        if not self.processed_forms:
            return {"answer": "No forms available", "confidence": 0.0, "source": None}
        best, best_conf, source = None, 0.0, None
        for form in self.processed_forms:
            res = self.answer_question(form["text"], question)
            if res["confidence"] > best_conf:
                best, best_conf, source = res, res["confidence"], form["filename"]
        if best and best_conf > 0.1:
            return {"answer": best["answer"], "confidence": best_conf, "source": source}
        return {"answer": "No reliable answer found", "confidence": 0.0, "source": None}

    def analyze_form_collection(self) -> dict:
        if not self.processed_forms:
            return {}
        all_fields = ["invoice_number", "date", "name", "email", "phone", "amount", "seller", "gst"]
        field_counts = {f: 0 for f in all_fields}
        total_amount, forms_with_amount = 0.0, 0

        for form in self.processed_forms:
            info = form["info"]
            for field in all_fields:
                if info.get(field):
                    field_counts[field] += 1
            if info.get("amount"):
                forms_with_amount += 1
                val = self._parse_amount(info["amount"])
                if val:
                    total_amount += val

        n = len(self.processed_forms)
        return {
            "total_forms": n,
            "forms_with_amounts": forms_with_amount,
            "total_amount": round(total_amount, 2),
            "average_amount": round(total_amount / forms_with_amount, 2) if forms_with_amount else 0,
            "field_completion": {f: round((c / n) * 100, 1) for f, c in field_counts.items()},
        }

    def _parse_amount(self, amount_str) -> float | None:
        if not amount_str:
            return None
        import re
        try:
            return float(re.sub(r"[^\d.]", "", str(amount_str)))
        except ValueError:
            return None
