"""Pydantic request / response schemas for the entire API."""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Extracted Fields ──────────────────────────────────────────────────────────

class ExtractedFields(BaseModel):
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    due_date: Optional[str] = None
    name: Optional[str] = None
    seller: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    amount: Optional[str] = None
    gst: Optional[str] = None
    currency: Optional[str] = None
    line_items: List[Dict[str, Any]] = Field(default_factory=list)


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_size_bytes: int
    page_count: int
    extracted_fields: ExtractedFields
    summary: str
    text_preview: str
    field_completion_pct: float
    processed_at: datetime
    processing_time_ms: float


class DocumentListItem(BaseModel):
    id: str
    filename: str
    processed_at: datetime
    field_completion_pct: float
    amount: Optional[str] = None
    invoice_number: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentListItem]
    total: int


class UploadResponse(BaseModel):
    success: bool
    document: Optional[DocumentInfo] = None
    error: Optional[str] = None


class BatchUploadResponse(BaseModel):
    total_uploaded: int
    successful: int
    failed: int
    documents: List[UploadResponse]


class ExportData(BaseModel):
    document_id: str
    filename: str
    extracted_fields: Dict[str, Any]
    summary: str
    exported_at: datetime


# ── Chat / Q&A ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)
    document_id: Optional[str] = None   # None = search all documents
    mode: str = Field(default="qa", description="'qa' or 'analyze'")


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    source_document: Optional[str] = None
    mode: str
    model_used: str
    latency_ms: float


class AnalysisRequest(BaseModel):
    document_id: str
    analysis_type: str = Field(
        default="full",
        description="'full' | 'risk' | 'comparison' | 'anomaly'"
    )


class AnalysisResponse(BaseModel):
    document_id: str
    analysis_type: str
    findings: str
    key_points: List[str]
    confidence: float
    model_used: str
    latency_ms: float


class ConversationMessage(BaseModel):
    role: str          # "user" | "assistant"
    content: str
    timestamp: datetime
    document_id: Optional[str] = None


# ── Analytics ─────────────────────────────────────────────────────────────────

class FieldCompletionStats(BaseModel):
    field: str
    completion_pct: float
    count: int


class CollectionAnalytics(BaseModel):
    total_documents: int
    documents_with_amounts: int
    total_amount: float
    average_amount: float
    currency_breakdown: Dict[str, int]
    field_stats: List[FieldCompletionStats]
    recent_uploads: List[DocumentListItem]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthStatus(BaseModel):
    status: str          # "healthy" | "degraded" | "unhealthy"
    version: str
    uptime_seconds: float
    models_loaded: bool
    vector_store_ok: bool
    cache_ok: bool
    total_documents: int
    timestamp: datetime
