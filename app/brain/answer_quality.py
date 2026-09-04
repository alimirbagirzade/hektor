"""Cevap-kalitesi yardımcıları — deterministik, LLM'siz (CPU-dostu).

Derin araştırma (reports/rag_deep_research_roadmap.md) bulguları: 4B yerel modelde
en yüksek ROI okuma-kalitesi kazançları HEPSİ deterministik / $0 ekstra-LLM:

1. CRAG-lite güven kapısı: retrieval zayıfsa (en iyi benzerlik düşük / top-1↔top-2 marjı
   küçük) cevap üretmeden ABSTAIN ("yetersiz dayanak") → CLAUDE.md Kural 7 (uydurma yok).
2. "Lost in the middle" yeniden-sıralama: en alakalı chunk'ları bağlamın BAŞINA ve SONUNA,
   en az alakalıyı ortaya koy (LLM'ler U-biçimli; orta konum unutulur — arXiv 2307.03172).

Mesafe: ChromaDB cosine → distance ∈ [0,2], benzerlik = 1 − distance (yüksek = iyi).
HyDE / multi-query / Self-RAG BİLİNÇLİ ATLANDI (4B'de net-zararlı/çok pahalı; araştırma).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.memory.retrieval_service import RetrievedChunk

# Satır-içi atıf deseni: [paper_id:chunk_id] veya [paper_id:chunk_id, s.3]
_CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]+):([A-Za-z0-9_\-]+)(?:,[^\]]*)?\]")


def similarity(distance: float | None) -> float:
    """Cosine distance → [0,1] benzerlik. None/aralık-dışı güvenli."""
    if distance is None:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(distance)))


@dataclass
class RetrievalConfidence:
    """Retrieval güven sinyalleri (deterministik, mesafe-tabanlı)."""

    best_similarity: float  # en iyi chunk benzerliği [0,1]
    margin: float  # top-1 ile top-2 benzerlik farkı (ayırt edicilik)
    n: int

    @property
    def weak(self) -> bool:  # eşik kontrolü çağıranda (config'li) yapılır
        return self.n == 0


def assess_confidence(chunks: list[RetrievedChunk]) -> RetrievalConfidence:
    """Retrieve edilmiş chunk'lardan güven sinyallerini çıkar (sıra: en iyi önce)."""
    if not chunks:
        return RetrievalConfidence(best_similarity=0.0, margin=0.0, n=0)
    sims = [similarity(c.distance) for c in chunks]
    best = sims[0]
    second = sims[1] if len(sims) > 1 else 0.0
    return RetrievalConfidence(best_similarity=best, margin=best - second, n=len(chunks))


def is_weak_retrieval(conf: RetrievalConfidence, min_similarity: float, min_margin: float) -> bool:
    """CRAG-lite: retrieval zayıf mı (abstain edilmeli mi)?

    Zayıf = hiç chunk yok, VEYA en iyi benzerlik tabanın altında (alakasız sorgu),
    VEYA top-1↔top-2 marjı çok küçük (belirsiz — hiçbir kaynak net öne çıkmıyor).
    Eşikler config'ten gelir; KORUNMA için varsayılan muhafazakâr (yalnız belirgin
    alakasızlıkta tetiklenir — meşru sorguları abstain ETMEZ; over-abstain riski düşük).
    """
    if conf.n == 0:
        return True
    if conf.best_similarity < min_similarity:
        return True
    # marj kontrolü yalnız >1 chunk varken anlamlı; tek-chunk durumunda atla
    return conf.n > 1 and conf.margin < min_margin and conf.best_similarity < (min_similarity * 1.5)


def reorder_lost_in_middle(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """En alakalı chunk'ları başa ve sona, en azı ortaya koy (LLM U-biçim mitigasyonu).

    Girdi sıralı varsayılır (en alakalı önce). Çıktı aynı chunk kümesi, yeniden dizilmiş:
    rank0 → bir uca, rank1 → diğer uca, ... en zayıflar ortada toplanır. Deterministik,
    chunk EKLEMEZ/ÇIKARMAZ (yalnız sıra). Kaynak: arXiv 2307.03172.
    """
    if len(chunks) <= 2:
        return list(chunks)
    head: list[RetrievedChunk] = []
    tail: list[RetrievedChunk] = []
    for i, c in enumerate(chunks):
        (head if i % 2 == 0 else tail).append(c)
    # head: rank0,2,4… (en güçlü başta); tail tersine → rank1,3,5 sona doğru güçlenir
    return head + tail[::-1]


@dataclass
class CitationCheck:
    """Satır-içi atıf doğrulama sonucu (deterministik, LLM'siz)."""

    n_cited: int  # cevapta toplam atıf sayısı
    unsupported: list[str] = field(default_factory=list)  # retrieve edilmeyene atıflar

    @property
    def has_unsupported(self) -> bool:
        return bool(self.unsupported)


def verify_citations(answer: str, chunks: list[RetrievedChunk]) -> CitationCheck:
    """Cevaptaki [paper_id:chunk_id] atıflarını retrieve edilen kaynaklarla doğrula.

    Citation-forcing prompt yine de UYDURMA atıf üretebilir ("correctness ≠ faithfulness",
    araştırma) → deterministik son-kontrol: chunk_id'si VEYA paper_id'si getirilen kaynak
    kümesinde OLMAYAN atıflar 'unsupported' (dayanaksız). LLM çağrısı yok. (Kural 7)
    """
    if not answer or not chunks:
        return CitationCheck(n_cited=0)
    chunk_ids = {c.chunk_id for c in chunks}
    paper_ids = {c.paper_id for c in chunks}
    seen: set[str] = set()
    unsupported: list[str] = []
    n = 0
    for m in _CITE_RE.finditer(answer):
        pid, cid = m.group(1), m.group(2)
        n += 1
        key = f"{pid}:{cid}"
        if key in seen:
            continue
        seen.add(key)
        # chunk_id getirilenlerde varsa desteklenir; değilse paper_id eşleşmesi de kabul
        # (LLM bazen sayfa/parça id'sini yaklaşık verir ama doğru makaleyi gösterir).
        if cid not in chunk_ids and pid not in paper_ids:
            unsupported.append(key)
    return CitationCheck(n_cited=n, unsupported=unsupported)


def citation_warning(check: CitationCheck) -> str:
    """Dayanaksız atıf varsa eklenecek deterministik uyarı metni (yoksa boş)."""
    if not check.has_unsupported:
        return ""
    ids = ", ".join(check.unsupported[:8])
    return (
        "\n\n---\n⚠ DİKKAT (otomatik doğrulama): aşağıdaki atıf(lar) getirilen kaynaklarda "
        f"YOK — dayanaksız olabilir, doğrulayın: {ids}\n"
        "(Auto-check: the cited source(s) above were not in the retrieved set — verify.)"
    )
