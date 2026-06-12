"""
ChromaDB vector store service for semantic document search (RAG).

Documents are chunked, embedded with sentence-transformers/all-MiniLM-L6-v2,
and stored in a local persistent ChromaDB collection.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.core.config import settings
from src.core.logger import get_logger
from src.core.exceptions import VectorStoreError
from src.utils.text_utils import chunk_text

log = get_logger(__name__)


class VectorStore:
    """Thin wrapper around ChromaDB with sentence-transformer embeddings."""

    def __init__(self) -> None:
        self._client = None
        self._collection = None
        self._embed_fn = None
        self._ready = False

    # ── Lazy initialisation ───────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            from chromadb.utils import embedding_functions

            persist_dir = str(settings.CHROMA_DIR)
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(path=persist_dir)

            self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )

            self._collection = self._client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )

            self._ready = True
            log.info(
                "ChromaDB ready — collection '%s' has %d docs",
                settings.CHROMA_COLLECTION,
                self._collection.count(),
            )
        except Exception as exc:
            log.error("ChromaDB initialisation failed: %s", exc)
            self._ready = False

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        try:
            self._ensure_ready()
            return self._ready
        except Exception:
            return False

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ensure_ready()
        if not self._ready:
            log.warning("VectorStore not ready — skipping indexing for %s", doc_id)
            return

        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            return

        ids = [f"{doc_id}__chunk_{i}" for i in range(len(chunks))]
        meta = metadata or {}
        metadatas = [{**meta, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        try:
            # Delete existing chunks for this document (re-index = upsert)
            existing = self._collection.get(where={"doc_id": doc_id})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])

            self._collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
            log.info("Indexed %d chunks for document %s", len(chunks), doc_id)
        except Exception as exc:
            raise VectorStoreError(f"Failed to add document {doc_id}", str(exc))

    def search(
        self,
        query: str,
        n_results: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_ready()
        if not self._ready:
            return []

        where: Optional[Dict] = {"doc_id": doc_id} if doc_id else None

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n_results, max(1, self._collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.warning("VectorStore search failed: %s", exc)
            return []

        output: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "similarity": round(1 - dist, 4),
            })
        return output

    def delete_document(self, doc_id: str) -> None:
        self._ensure_ready()
        if not self._ready:
            return
        try:
            existing = self._collection.get(where={"doc_id": doc_id})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                log.info("Deleted %d chunks for document %s", len(existing["ids"]), doc_id)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document {doc_id}", str(exc))

    def document_count(self) -> int:
        self._ensure_ready()
        if not self._ready:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0


# ── Singleton ─────────────────────────────────────────────────────────────────

_vector_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
