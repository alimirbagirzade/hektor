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

    n_cited: int  # cevapta toplam atıf sayısı (tekrarlar dahil)
    unsupported: list[str] = field(default_factory=list)  # retrieve edilmeyene atıflar
    n_unique: int = 0  # farklı (paper_id, chunk_id) çifti sayısı
    malformed: list[str] = field(default_factory=list)  # çözümlenemeyen kaynak-benzeri köşeli

    @property
    def has_unsupported(self) -> bool:
        return bool(self.unsupported)


# Köşeli parantez içeriği; içte `,`/`;` ile ayrılmış birden çok `id:id` olabilir (Kademe 2
# F3-3, 2026-10-01: eski desen "[a:c1, b:c9]"nin ikinci kimliğini sayfa eki sanıp yutuyordu).
_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,400})\]")
# Her iki kimlik harfle başlar ve ≥2 karakterdir: "[0:1]", "[09:30]", "x[t:T]" gibi matematik /
# saat ifadeleri atıf sayılmaz (Kademe 2 F1-8).
_CITE_TOKEN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_\-]+)\s*:\s*([A-Za-z][A-Za-z0-9_\-]+)$")
_PAGE_TOKEN_RE = re.compile(r"^(?:s|p|pp|sayfa|page)\.?\s*\d+(?:\s*[-–]\s*\d+)?$", re.I)
_SOURCE_LIKE_RE = re.compile(r"\b(?:paper|card|source)_\w+", re.I)


def parse_citations(answer: str) -> tuple[list[tuple[str, str]], list[str]]:
    """(atıf çiftleri — sırayla, tekrarlar dahil; kaynak-benzeri ama çözümlenemeyen köşeliler)."""
    cites: list[tuple[str, str]] = []
    malformed: list[str] = []
    for m in _BRACKET_RE.finditer(answer):
        body = m.group(1)
        found: list[tuple[str, str]] = []
        odd = False
        for tok in (t.strip() for t in re.split(r"[;,]", body)):
            cm = _CITE_TOKEN_RE.match(tok)
            if cm:
                found.append((cm.group(1), cm.group(2)))
            elif tok and not _PAGE_TOKEN_RE.match(tok) and _SOURCE_LIKE_RE.search(tok):
                odd = True
        if found and _CITE_TOKEN_RE.match(body.split(",")[0].split(";")[0].strip()) is None:
            odd = True  # köşeli bir atıfla başlamıyor (ör. "[bkz. paper_x:c1]")
        cites.extend(found)
        if odd or (not found and _SOURCE_LIKE_RE.search(body)):
            malformed.append(m.group(0))
    return cites, malformed


def verify_citations(
    answer: str, chunks: list[RetrievedChunk], *, strict: bool = False
) -> CitationCheck:
    """Cevaptaki [paper_id:chunk_id] atıflarını retrieve edilen kaynaklarla doğrula.

    Citation-forcing prompt yine de UYDURMA atıf üretebilir ("correctness ≠ faithfulness",
    araştırma) → deterministik son-kontrol. LLM çağrısı yok. (Kural 7)

    Varsayılan (canlı uyarı): chunk_id'si VEYA paper_id'si getirilen kümede olan atıf
    desteklenir (LLM bazen parça id'sini yaklaşık verir ama doğru makaleyi gösterir).
    ``strict=True`` (eğitim verisi kapısı, Kademe 2 F3-2): (paper_id, chunk_id) ÇİFTİ birebir
    getirilenlerde olmalı — uydurma parça kimliği / yanlış eşleşmiş çift eğitime girmez.
    """
    cites, malformed = parse_citations(answer or "")
    if not answer or not chunks:
        return CitationCheck(n_cited=0, malformed=malformed)
    pairs = {(c.paper_id, c.chunk_id) for c in chunks}
    chunk_ids = {c.chunk_id for c in chunks}
    paper_ids = {c.paper_id for c in chunks}
    seen: set[tuple[str, str]] = set()
    unsupported: list[str] = []
    for pid, cid in cites:
        if (pid, cid) in seen:
            continue
        seen.add((pid, cid))
        ok = (pid, cid) in pairs if strict else (cid in chunk_ids or pid in paper_ids)
        if not ok:
            unsupported.append(f"{pid}:{cid}")
    return CitationCheck(
        n_cited=len(cites), unsupported=unsupported, n_unique=len(seen), malformed=malformed
    )


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
