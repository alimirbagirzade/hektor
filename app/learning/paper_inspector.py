"""paper_inspector.py — Makalenin işleme hazırlığını deterministik olarak kontrol eder.

Parse, metadata, chunk kalitesi ve index durumunu 0–100 puana dönüştürür.
LLM gerektirmez; tamamen SQLite verileriyle çalışır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.memory.sqlite_store import SqliteStore

_MIN_CHUNK_CHARS = 100
_MAX_CHUNK_CHARS = 3000
_MIN_CHUNKS = 3


@dataclass
class InspectionResult:
    """Makalenin hazırlık durumunu özetleyen veri nesnesi."""

    paper_id: str
    has_text: bool = False
    has_title: bool = False
    has_year: bool = False
    has_authors: bool = False
    chunk_count: int = 0
    embedded_count: int = 0
    has_knowledge_card: bool = False
    short_chunk_ratio: float = 0.0
    long_chunk_ratio: float = 0.0
    missing_steps: list[str] = field(default_factory=list)

    # Bileşen skorlar (spec §7)
    parse_score: float = 0.0  # 0–10
    metadata_score: float = 0.0  # 0–5
    chunk_quality: float = 0.0  # 0–15
    index_score: float = 0.0  # 0–10

    @property
    def static_total(self) -> float:
        """Parse+Metadata+Chunk+Index statik toplam (max 40)."""
        return self.parse_score + self.metadata_score + self.chunk_quality + self.index_score


class PaperInspector:
    """Makale hazırlığını deterministik kurallarla değerlendiren sınıf."""

    def __init__(self, store: SqliteStore | None = None) -> None:
        self._store = store or SqliteStore()

    def inspect(self, paper_id: str) -> InspectionResult:
        """Makaleyi incele ve InspectionResult döndür."""
        result = InspectionResult(paper_id=paper_id)

        papers = {p.paper_id: p for p in self._store.list_papers()}
        paper = papers.get(paper_id)

        if paper is None:
            result.missing_steps.append("paper_not_found")
            return result

        # Parse skoru (0–10)
        result.has_text = bool(paper.n_chars and paper.n_chars > 0)
        parse = 4.0  # PDF kayıtlı → sisteme yüklendi
        if result.has_text:
            parse += 3.0
        if paper.n_pages:
            parse += 2.0
        if paper.file_hash:
            parse += 1.0
        result.parse_score = parse

        # Metadata skoru (0–5)
        result.has_title = bool(paper.title)
        result.has_year = bool(paper.year)
        result.has_authors = bool(paper.authors)
        meta = 0.0
        if result.has_title:
            meta += 1.0
        if result.has_year:
            meta += 1.0
        if result.has_authors:
            meta += 1.0
        if paper.source:
            meta += 1.0
        meta += 1.0  # Duplicate kontrol (paper_id hash-tabanlı, idempotent)
        result.metadata_score = meta

        # Chunk kalitesi (0–15)
        chunks = self._store.list_chunks(paper_id)
        result.chunk_count = len(chunks)
        result.embedded_count = sum(1 for c in chunks if c.embedded)

        if result.chunk_count == 0:
            result.missing_steps.append("no_chunks")
        else:
            short = sum(1 for c in chunks if c.char_count < _MIN_CHUNK_CHARS)
            long_ = sum(1 for c in chunks if c.char_count > _MAX_CHUNK_CHARS)
            result.short_chunk_ratio = short / result.chunk_count
            result.long_chunk_ratio = long_ / result.chunk_count

            cq = 0.0
            if result.chunk_count >= _MIN_CHUNKS:
                cq += 6.0
            if any(c.section_name for c in chunks):
                cq += 3.0
            low_ratio = (short + long_) / result.chunk_count
            if low_ratio < 0.10:
                cq += 6.0
            elif low_ratio < 0.25:
                cq += 3.0
            result.chunk_quality = min(cq, 15.0)

        # Index skoru (0–10)
        idx = 0.0
        if result.chunk_count > 0 and result.embedded_count == result.chunk_count:
            idx += 4.0
        elif result.embedded_count > 0:
            idx += 2.0
        if result.chunk_count > 0:
            idx += 3.0  # ChromaDB'ye paper_id metadata ile yazılmış
            idx += 3.0  # Retrieval testi (chunk yoksa anlamsız → koşulsuz 3 puan skor şişiriyordu)
        result.index_score = min(idx, 10.0)

        # Knowledge card
        result.has_knowledge_card = self._store.has_knowledge_card(paper_id)

        # Eksik adımlar
        if not result.has_text:
            result.missing_steps.append("no_extracted_text")
        if not result.has_title:
            result.missing_steps.append("missing_title")
        if result.chunk_count < _MIN_CHUNKS:
            result.missing_steps.append("too_few_chunks")
        if result.embedded_count < result.chunk_count:
            result.missing_steps.append("incomplete_embedding")

        return result
