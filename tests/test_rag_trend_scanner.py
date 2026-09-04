"""RAG trend tarayıcı (projeye yerleşik tarama ajanı) testleri — çevrimdışı, deterministik."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.arxiv_fetcher import ArxivEntry
from app.research.rag_trend_scanner import (
    append_candidates,
    existing_ids,
    scan_rag_trends,
)


def _entry(
    arxiv_id: str, title: str, abstract: str = "", published: str = "2026-06-01"
) -> ArxivEntry:
    return ArxivEntry(
        arxiv_id=arxiv_id,
        title=title,
        authors=["A"],
        abstract=abstract,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        published=published,
    )


def test_scan_filters_by_min_score_and_dedups() -> None:
    """Eşik altı elenir; aynı id farklı sorgulardan gelirse tek kalır (yüksek skor korunur)."""
    by_query = {
        "q1": [
            _entry("2401.00001", "A reranking method for retrieval augmented generation"),
            _entry("2401.00002", "Unrelated paper about cooking recipes"),  # skor 0 → elenir
        ],
        "q2": [
            _entry(
                "2401.00001", "RAG reranking dense embedding hybrid fusion"
            ),  # aynı id, yüksek skor
        ],
    }

    def fake_search(query: str, max_results: int) -> list[ArxivEntry]:
        return by_query.get(query, [])

    out = scan_rag_trends(queries=["q1", "q2"], min_score=2, searcher=fake_search)
    ids = [c.arxiv_id for c in out]
    assert ids == ["2401.00001"]  # cooking elendi, id tekilleşti
    assert out[0].score >= 3  # q2'deki yüksek skorlu varyant korundu


def test_scan_deterministic_order() -> None:
    """Sıralama (skor desc) deterministik."""

    def fake_search(query: str, max_results: int) -> list[ArxivEntry]:
        return [
            _entry("2401.00010", "retrieval dense", abstract="rag"),  # skor 3
            _entry(
                "2401.00011", "reranking chunk embedding bm25 hybrid fusion graphrag"
            ),  # daha yüksek
        ]

    out = scan_rag_trends(queries=["q"], min_score=1, searcher=fake_search)
    assert out[0].arxiv_id == "2401.00011"  # en yüksek skor başta
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)  # skor azalan (deterministik)


def test_scan_graceful_on_search_error() -> None:
    """Bir sorgu hata atarsa tur çökmez, diğerlerine devam eder."""

    def flaky_search(query: str, max_results: int) -> list[ArxivEntry]:
        if query == "boom":
            raise RuntimeError("ağ yok")
        return [_entry("2401.00020", "retrieval augmented rag reranking")]

    out = scan_rag_trends(queries=["boom", "ok"], min_score=1, searcher=flaky_search)
    assert [c.arxiv_id for c in out] == ["2401.00020"]


def test_existing_ids_extracts_arxiv_ids() -> None:
    text = "satır 2401.10131 ve 2507.09554 ile hep-th/9901001 burada."
    found = existing_ids(text)
    assert "2401.10131" in found
    assert "2507.09554" in found


def test_append_candidates_idempotent(tmp_path: Path) -> None:
    """Var olan id atlanır; yeni id eklenir; iki kez çalıştırınca tekrar eklenmez."""
    wl = tmp_path / "rag-watchlist.md"
    wl.write_text("# Watchlist\n\n| x | 2401.99999 | url | aday | mevcut |\n", encoding="utf-8")

    out = scan_rag_trends(
        queries=["q"],
        min_score=1,
        searcher=lambda q, n: [
            _entry("2401.99999", "retrieval rag"),  # zaten var → atlanır
            _entry("2402.00001", "reranking dense embedding fusion"),  # yeni
        ],
    )
    added = append_candidates(wl, out, today="2026-06-17")
    assert added == 1
    text = wl.read_text(encoding="utf-8")
    assert "2402.00001" in text
    assert "Otomatik tarama adayları" in text

    # İkinci kez: yeni aday yok → 0 eklenir, dosya büyümez.
    again = append_candidates(wl, out, today="2026-06-18")
    assert again == 0


def test_append_candidates_escapes_pipes(tmp_path: Path) -> None:
    """Başlıktaki boru karakteri tabloyu bozmasın."""
    wl = tmp_path / "rag-watchlist.md"
    wl.write_text("# Watchlist\n", encoding="utf-8")
    from app.research.rag_trend_scanner import TrendCandidate

    c = TrendCandidate(
        arxiv_id="2403.00001",
        title="A | B | C reranking",
        published="2026-06-01",
        score=2,
        query="q|x",
    )
    added = append_candidates(wl, [c], today="2026-06-17")
    assert added == 1
    text = wl.read_text(encoding="utf-8")
    assert "A / B / C reranking" in text  # boru → eğik çizgi
