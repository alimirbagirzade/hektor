"""Reranking module.

Combines semantic score, keyword overlap, section priority, and formula
presence to rerank a list of retrieved chunks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar

from app.memory.retrieval_service import RetrievedChunk

# Bölüm adı → öncelik skoru (yüksek = daha öncelikli)
_SECTION_PRIORITY: dict[str, float] = {
    "abstract": 1.0,
    "introduction": 0.9,
    "methodology": 0.85,
    "methods": 0.85,
    "results": 0.75,
    "discussion": 0.7,
    "conclusion": 0.65,
    "references": 0.1,
    "related work": 0.6,
    "data": 0.55,
}

# LaTeX formül desenleri
_LATEX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\$\$.*?\$\$", re.DOTALL),
    re.compile(r"\$[^$\n]+\$"),
    re.compile(r"\\\[.*?\\\]", re.DOTALL),
    re.compile(r"\\\(.*?\\\)", re.DOTALL),
    re.compile(r"\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL),
    re.compile(r"\\frac\{", re.DOTALL),
    re.compile(r"\\sum_"),
    re.compile(r"\\int_"),
    re.compile(r"\\prod_"),
]


def _has_formula(text: str) -> bool:
    """Metinde LaTeX formülü var mı?"""
    return any(pattern.search(text) for pattern in _LATEX_PATTERNS)


def _keyword_overlap_score(query: str, text: str) -> float:
    """Sorgu ile metin arasındaki token örtüşme oranı (0–1)."""
    q_tokens = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", query.lower()))
    t_tokens = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text.lower()))
    if not q_tokens:
        return 0.0
    overlap = q_tokens & t_tokens
    return len(overlap) / len(q_tokens)


def _section_priority_score(section_name: str | None) -> float:
    """Bölüm adına göre öncelik skoru döndür."""
    if not section_name:
        return 0.5  # bilinmeyen bölüm → orta öncelik
    return _SECTION_PRIORITY.get(section_name.lower().strip(), 0.5)


def _semantic_score(distance: float | None) -> float:
    """Chroma distance → semantik skor dönüşümü (0–1, yüksek daha iyi)."""
    if distance is None:
        return 0.5
    # distance genellikle 0–2 aralığında; sigmoid benzeri dönüşüm
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


@dataclass
class RerankScore:
    """Bir chunk için hesaplanan tüm alt skorları tutar."""

    chunk_id: str
    semantic_score: float
    keyword_overlap: float
    section_priority: float
    formula_presence: float
    final_score: float


class Reranker:
    """Çok faktörlü yeniden sıralayıcı.

    Ağırlıklar:
      - semantic_score:    0.40
      - keyword_overlap:   0.30
      - section_priority:  0.20
      - formula_presence:  0.10 (bonus)
    """

    WEIGHTS: ClassVar[dict[str, float]] = {
        "semantic": 0.40,
        "keyword": 0.30,
        "section": 0.20,
        "formula": 0.10,
    }

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Chunk listesini yeniden sırala.

        Args:
            query: Kullanıcı sorgusu (anahtar kelime örtüşmesi için).
            chunks: Sıralanacak RetrievedChunk listesi.

        Returns:
            Final skora göre azalan sırayla sıralanmış yeni liste.
        """
        if not chunks:
            return []

        scored: list[tuple[RetrievedChunk, float]] = []

        for chunk in chunks:
            text = chunk.text or ""  # boş/None text'e karşı güvenli (hybrid stub, eksik veri)
            sem = _semantic_score(chunk.distance)
            kw = _keyword_overlap_score(query, text)
            sec = _section_priority_score(chunk.section_name)
            formula = 1.0 if _has_formula(text) else 0.0

            final = (
                self.WEIGHTS["semantic"] * sem
                + self.WEIGHTS["keyword"] * kw
                + self.WEIGHTS["section"] * sec
                + self.WEIGHTS["formula"] * formula
            )
            # Normalize: tanh ile sınırla (0–1 aralığı)
            final = math.tanh(final * 2.0) / math.tanh(2.0)

            scored.append((chunk, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]
