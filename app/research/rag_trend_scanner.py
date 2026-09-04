"""RAG trend tarayıcı — projeye yerleşik periyodik 'tarama ajanı'.

arXiv'de RAG/retrieval/eğitim **yöntem** aramaları yapar (trading içeriği değil),
yeni adayları izleme listesine (`docs/egitim/rag-watchlist.md`) işler. Bu, güncel-RAG
döngüsünün **ucuz** katmanıdır: Claude/kotası gerektirmez (yalnız arXiv + heuristik
alaka skoru). Pahalı entegrasyon (kod yazma + doküman sürümleme + push) bir kodlama
ajanı gerektirir; bu modül onu YAPMAZ, yalnız aday biriktirir.

Çevrimdışı/test: `searcher` enjekte edilebilir; ağ yoksa graceful (boş döner, çökmez).
Determinizm: skor anahtar-kelime sayımıdır, sıralama (skor, tarih, id) sabittir.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.arxiv_fetcher import ArxivEntry, search_arxiv

logger = logging.getLogger(__name__)

#: RAG/retrieval/eğitim YÖNTEM arama sorguları (trading konusu değil).
DEFAULT_QUERIES: tuple[str, ...] = (
    "retrieval augmented generation reranking",
    "RAG chunking late chunking contextual retrieval",
    "corrective RAG self-RAG adaptive retrieval",
    "query rewriting HyDE reciprocal rank fusion",
    "GraphRAG knowledge graph retrieval",
    "RAG evaluation faithfulness groundedness hallucination",
    "retrieval augmented fine-tuning RAFT distractor",
    "dense retrieval embedding cross-encoder reranker",
)

#: Heuristik alaka için anahtar kelimeler (offline, deterministik skor).
_RELEVANCE_TERMS: tuple[str, ...] = (
    "retriev",
    "rag",
    "rerank",
    "chunk",
    "embedding",
    "dense",
    "bm25",
    "hybrid",
    "fusion",
    "graphrag",
    "hyde",
    "faithful",
    "ground",
    "raft",
    "distractor",
    "passage",
    "context",
    "knowledge graph",
    "vector",
)

#: arXiv id deseni (yeni stil 2401.12345 + eski stil hep-th/9901001).
_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})\b")

_WATCHLIST_REL = ("docs", "egitim", "rag-watchlist.md")
_AUTO_SECTION = "## Otomatik tarama adayları (rag-scan)"


@dataclass
class TrendCandidate:
    """Tarama turunda bulunan bir aday makale."""

    arxiv_id: str
    title: str
    published: str
    score: int
    query: str
    abstract: str = ""


def repo_root() -> Path:
    """Repo kökü (app/research/rag_trend_scanner.py → ../../)."""
    return Path(__file__).resolve().parent.parent.parent


def watchlist_path() -> Path:
    """`docs/egitim/rag-watchlist.md` yolu."""
    return repo_root().joinpath(*_WATCHLIST_REL)


def _score(entry: ArxivEntry, terms: Sequence[str] = _RELEVANCE_TERMS) -> int:
    """Başlık+özette geçen ayırt edici anahtar kelime sayısı (deterministik).

    `terms` konu paketine göre değişir (literature_scout); varsayılan RAG terimleridir.
    """
    haystack = f"{entry.title}\n{entry.abstract}".lower()
    return sum(1 for term in terms if term in haystack)


def scan_rag_trends(
    queries: Sequence[str] = DEFAULT_QUERIES,
    max_per_query: int = 8,
    min_score: int = 2,
    searcher: Callable[[str, int], list[ArxivEntry]] = search_arxiv,
    terms: Sequence[str] = _RELEVANCE_TERMS,
) -> list[TrendCandidate]:
    """arXiv'de yöntem ara → eşik üstü adayları (deterministik sıralı) döndür.

    Args:
        queries: arXiv arama sorguları.
        max_per_query: Sorgu başına maksimum sonuç.
        min_score: Minimum heuristik alaka skoru (eşik).
        searcher: `(query, max_results) -> list[ArxivEntry]`; test/çevrimdışı için enjekte edilir.
        terms: Alaka skorunda aranan anahtar kelimeler (konu paketi başına farklı).

    Returns:
        Skora göre azalan (eşitlikte tarih, sonra id) benzersiz aday listesi. Ağ hatası
        veya sonuç yoksa boş liste (çökmez).
    """
    seen: dict[str, TrendCandidate] = {}
    for q in queries:
        try:
            entries = searcher(q, max_per_query)
        except Exception as exc:  # ağ/parse hatası bir sorguyu atlatır, turu çökertmez
            logger.warning("arXiv arama hatası (%s): %s", q, exc)
            continue
        for e in entries:
            if not e.arxiv_id:
                continue
            sc = _score(e, terms)
            if sc < min_score:
                continue
            prev = seen.get(e.arxiv_id)
            if prev is None or sc > prev.score:
                seen[e.arxiv_id] = TrendCandidate(
                    arxiv_id=e.arxiv_id,
                    title=e.title,
                    published=e.published,
                    score=sc,
                    query=q,
                    abstract=e.abstract,
                )
    return sorted(
        seen.values(),
        key=lambda c: (c.score, c.published, c.arxiv_id),
        reverse=True,
    )


def existing_ids(watchlist_text: str) -> set[str]:
    """Watchlist metnindeki tüm arXiv id'lerini çıkar (dedup için)."""
    return set(_ARXIV_ID_RE.findall(watchlist_text))


def _cell(text: str) -> str:
    """Markdown tablo hücresi için boru/satırsonu temizle."""
    return text.replace("|", "/").replace("\n", " ").strip()


def append_candidates(
    path: Path,
    candidates: Sequence[TrendCandidate],
    today: str | None = None,
) -> int:
    """Yeni adayları (watchlist'te id'si olmayanları) otomatik tarama tablosuna ekle.

    Idempotent: zaten id'si bulunan aday atlanır. Hiç yeni aday yoksa dosya yazılmaz.

    Returns:
        Eklenen yeni satır sayısı.
    """
    if today is None:
        today = dt.date.today().isoformat()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    known = existing_ids(text)
    fresh = [c for c in candidates if c.arxiv_id not in known]
    if not fresh:
        return 0

    if _AUTO_SECTION not in text:
        text = (
            text.rstrip()
            + "\n\n"
            + _AUTO_SECTION
            + "\n\n> `rag-scan` ajanının otomatik eklediği arXiv adayları (heuristik skorlu).\n"
            + "> Entegrasyon turu bunları değerlendirip yukarıdaki ana tabloya taşır.\n\n"
            + "| Eklendi | arXiv | Skor | Başlık | Sorgu |\n"
            + "|---|---|---|---|---|\n"
        )
    rows = "".join(
        f"| {today} | {c.arxiv_id} | {c.score} | {_cell(c.title)[:90]} | {_cell(c.query)} |\n"
        for c in fresh
    )
    text = text.rstrip() + "\n" + rows
    path.write_text(text, encoding="utf-8")
    return len(fresh)
