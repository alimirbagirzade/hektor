"""Sorgu yönlendirici + konveks füzyon — derin araştırma (Zincir 2) uygulaması.

ÖLÇÜM (reports/rag_retrieval_ab_findings.md): uzun/semantik sorgularda dense-only
hibridi yener; ama literatür (BEIR, Bruch et al. 2210.11934) net: hibrit KISA keyword/
entity sorgularda (ticker, "Sharpe ratio", kısaltma, sayı, tırnaklı ifade) kazanır.
Doğru cevap "global hibrit vs dense" DEĞİL → **sorgu-tipine göre yönlendir** + füzyonu
"union + sezgisel rerank" (doğru dense isabetini düşüren ANTİ-DESEN) yerine **min-max
normalize KONVEKS kombinasyon** (α·dense + (1−α)·bm25, dense-ağırlıklı) yap.

Hepsi DETERMİNİSTİK, LLM'siz, ucuz (CPU-dostu). Yalnız `rag_router` açıkken devreye girer.
"""

from __future__ import annotations

import re

# Kısa + "exact-term sinyali" taşıyan sorgular lexical (BM25 katkısı) ister.
# Sinyaller: BÜYÜK-harf kısaltma (ATR, RSI, GARCH), rakam/sayı, tırnaklı ifade.
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,6}\b")
_DIGIT_RE = re.compile(r"\d")
_QUOTED_RE = re.compile(r"[\"'][^\"']{2,}[\"']")
# Kısalık eşiği: kelime sayısı. Uzun doğal-dil sorguları semantiktir (dense kazanır).
_SHORT_MAX_WORDS = 6


def classify_query(query: str) -> str:
    """Sorguyu 'lexical' (BM25 yardımcı) veya 'semantic' (dense-only) olarak sınıfla.

    Deterministik, ucuz (regex). Lexical = KISA ve exact-term sinyali var (kısaltma/
    rakam/tırnak), VEYA çok kısa (1-2 kelime, muhtemelen entity/terim). Aksi semantic.
    """
    q = (query or "").strip()
    if not q:
        return "semantic"
    words = q.split()
    n = len(words)
    if n <= 2:
        return "lexical"  # tek terim/entity → keyword araması
    if n <= _SHORT_MAX_WORDS and (
        _ACRONYM_RE.search(q) or _DIGIT_RE.search(q) or _QUOTED_RE.search(q)
    ):
        return "lexical"
    return "semantic"


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Skorları [0,1]'e min-max normalize et (boş/tek-eleman güvenli)."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return dict.fromkeys(scores, 1.0)  # hepsi eşit → 1.0
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def convex_fuse(dense: dict[str, float], bm25: dict[str, float], alpha: float) -> list[str]:
    """α·norm(dense) + (1−α)·norm(bm25) → azalan skora göre id listesi (deterministik).

    Bruch et al.: konveks kombinasyon RRF'i (skoru atan) yener, OOD'de bile; tek param.
    `dense`/`bm25`: id→ham skor (dense: benzerlik=1−distance; bm25: BM25 skoru). α∈[0,1],
    dense-ağırlıklı (~0.7-0.9 önerilir). Eşit skorda id'ye göre kararlı sıralama.
    """
    nd = _minmax(dense)
    nb = _minmax(bm25)
    ids = set(nd) | set(nb)
    fused = {cid: alpha * nd.get(cid, 0.0) + (1.0 - alpha) * nb.get(cid, 0.0) for cid in ids}
    return sorted(fused, key=lambda c: (-fused[c], c))
