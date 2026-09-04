"""Retrieve relevant chunks for a query (the R in RAG)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.config import get_settings
from app.memory.chroma_store import ChromaStore
from app.memory.embedding_service import EmbeddingService


class Retriever(Protocol):
    """Retrieval arayüzü — `RetrievalService`, `RerankingRetriever`, test stub'ları
    ve diğer sarmalayıcılar bu protokolü uygular. RAG/eval kodunun somut sınıfa
    değil arayüze bağlanmasını sağlar."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]: ...


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    text: str
    page_number: int | None
    section_name: str | None
    title: str | None
    distance: float | None

    @property
    def citation(self) -> str:
        page = f", s.{self.page_number}" if self.page_number and self.page_number > 0 else ""
        return f"[{self.paper_id}:{self.chunk_id}{page}]"


class RetrievalService:
    def __init__(
        self,
        chroma: ChromaStore | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.chroma = chroma or ChromaStore()
        self.embedder = embedder or EmbeddingService()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.settings.rag_top_k
        q_emb = self.embedder.embed_one(query)
        hits = self.chroma.query(q_emb, top_k=k)
        out: list[RetrievedChunk] = []
        for h in hits:
            meta = h.get("metadata", {})
            out.append(
                RetrievedChunk(
                    chunk_id=h["chunk_id"],
                    paper_id=meta.get("paper_id", "?"),
                    text=h.get("document", ""),
                    page_number=meta.get("page_number"),
                    section_name=meta.get("section_name") or None,
                    title=meta.get("title") or None,
                    distance=h.get("distance"),
                )
            )
        return out
