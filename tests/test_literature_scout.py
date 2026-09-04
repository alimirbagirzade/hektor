"""Literatür keşif ajanı — çevrimdışı testler.

Ağ YOK (searcher enjekte), indirme YOK (downloader enjekte), Desktop'a YAZILMAZ
(inbox tmp_path). Determinizm: sabit tarih damgası + sabit sıralama.

Sözleşmeler (bozulursa test kırılır):
  - Ajan RAG'a ingest ETMEZ / eğitim BAŞLATMAZ (Kural 8) — modülde o çağrılar olmamalı.
  - Bozuk/PDF-olmayan içerik diske YAZILMAZ.
  - Idempotent: aynı tur iki kez koşarsa dosya yeniden indirilmez.
  - Ağ patlarsa tur çökmez (graceful).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.arxiv_fetcher import ArxivEntry
from app.research.literature_scout import (
    TOPIC_PACKS,
    download_candidate,
    resolve_topics,
    run_scout,
)
from app.research.rag_trend_scanner import TrendCandidate

_PDF = b"%PDF-1.7\nsahte icerik"


def _entry(aid: str, title: str, abstract: str) -> ArxivEntry:
    return ArxivEntry(
        arxiv_id=aid,
        title=title,
        abstract=abstract,
        published="2026-07-01",
        pdf_url=f"https://arxiv.org/pdf/{aid}.pdf",
    )


def _searcher_for(topic_terms: str):
    """Her sorguya, o konunun terimlerini içeren tek bir sonuç döndüren sahte arayıcı."""

    def _search(query: str, max_results: int = 8) -> list[ArxivEntry]:
        aid = f"2607.{abs(hash(query)) % 90000 + 10000}"
        return [_entry(aid, f"Paper for {query[:20]}", topic_terms)]

    return _search


# ── konu paketleri ───────────────────────────────────────────────────────────
def test_four_topic_packs_present() -> None:
    assert set(TOPIC_PACKS) == {"rag", "lora", "rlm", "math-physics"}
    for pack in TOPIC_PACKS.values():
        assert pack.queries and pack.terms, f"{pack.key}: sorgu/terim boş olamaz"


def test_resolve_topics_all_and_unknown() -> None:
    assert [p.key for p in resolve_topics(None)] == sorted(TOPIC_PACKS)  # stabil sıra
    assert [p.key for p in resolve_topics(["lora"])] == ["lora"]
    with pytest.raises(ValueError):
        resolve_topics(["bilinmeyen-konu"])


# ── indirme davranışı ────────────────────────────────────────────────────────
def test_download_writes_pdf_and_is_idempotent(tmp_path: Path) -> None:
    cand = TrendCandidate(
        arxiv_id="2607.11111", title="T", published="2026-07-01", score=3, query="q"
    )
    calls: list[str] = []

    def dl(url: str) -> bytes:
        calls.append(url)
        return _PDF

    path, was_new = download_candidate(cand, tmp_path, downloader=dl)
    assert path is not None and path.exists() and was_new is True
    # ikinci çağrı: dosya var → yeniden İNDİRMEZ
    path2, was_new2 = download_candidate(cand, tmp_path, downloader=dl)
    assert path2 == path and was_new2 is False
    assert len(calls) == 1, "idempotent değil — ikinci kez indirdi"


def test_non_pdf_content_is_not_written(tmp_path: Path) -> None:
    """Anti-bot HTML sayfası vs. diske YAZILMAMALI."""
    cand = TrendCandidate(
        arxiv_id="2607.22222", title="T", published="2026-07-01", score=3, query="q"
    )
    path, was_new = download_candidate(cand, tmp_path, downloader=lambda u: b"<html>nope</html>")
    assert path is None and was_new is False
    assert list(tmp_path.glob("*.pdf")) == []


def test_download_error_does_not_crash(tmp_path: Path) -> None:
    cand = TrendCandidate(
        arxiv_id="2607.33333", title="T", published="2026-07-01", score=3, query="q"
    )

    def boom(url: str) -> bytes:
        raise RuntimeError("ağ yok")

    path, was_new = download_candidate(cand, tmp_path, downloader=boom)
    assert path is None and was_new is False


# ── tur akışı ────────────────────────────────────────────────────────────────
def test_run_scout_downloads_and_writes_manifest(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    report = run_scout(
        topics=["lora"],
        top_n_download=2,
        inbox=inbox,
        watchlist_dir=tmp_path,
        searcher=_searcher_for("lora low-rank adapter fine-tuning peft"),
        downloader=lambda u: _PDF,
        today="2026-07-21",
    )
    assert report.ran_at == "2026-07-21"
    assert report.found, "aday bulunamadı"
    assert all(f.topic == "lora" for f in report.found)
    assert report.downloaded_count >= 1
    # PDF'ler konu alt klasörüne indi
    assert list((inbox / "lora").glob("*.pdf"))
    # insan-okur özet yazıldı ve Kural-8 uyarısını içeriyor
    manifest = (inbox / "BULUNANLAR.md").read_text(encoding="utf-8")
    assert "Kural 8" in manifest
    assert "2026-07-21" in manifest


def test_no_download_mode_writes_nothing(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    report = run_scout(
        topics=["rlm"],
        inbox=inbox,
        watchlist_dir=tmp_path,
        download=False,
        searcher=_searcher_for("reasoning claim verification grounding evidence"),
        downloader=lambda u: pytest.fail("download=False iken indirici çağrılmamalı"),
        today="2026-07-21",
    )
    assert report.downloaded_count == 0
    assert not list(inbox.rglob("*.pdf"))


def test_network_failure_is_graceful(tmp_path: Path) -> None:
    def dead(query: str, max_results: int = 8) -> list[ArxivEntry]:
        raise RuntimeError("ağ yok")

    report = run_scout(
        topics=["rag"],
        inbox=tmp_path,
        watchlist_dir=tmp_path,
        searcher=dead,
        downloader=lambda u: _PDF,
        today="2026-07-21",
    )
    assert report.found == []  # çökmedi


def test_scout_never_ingests_or_trains() -> None:
    """KURAL 8 SÖZLEŞMESİ: keşif ajanı ingest/eğitim yollarına DOKUNMAZ.

    Düz metin araması yanıltıcı olurdu (docstring'de "RAG'a ingest ETMEZ" geçiyor) →
    AST ile GERÇEK import ve çağrılara bakılır.
    """
    import ast

    src = Path("app/research/literature_scout.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)

    # eğitim/ingest alt sistemleri import EDİLMEMELİ
    for mod in imported:
        assert not mod.startswith(("app.training", "app.lora", "app.orchestration")), (
            f"keşif ajanı eğitim alt sistemini import etmemeli: {mod} (Kural 8)"
        )
    # ingest/eğitim fonksiyonları ÇAĞRILMAMALI
    forbidden_calls = {
        "index_paper",
        "index_pdf",
        "ingest_pdf",
        "fetch_arxiv_papers",
        "detached_launch",
        "run_training",
        "train",
    }
    assert not (called & forbidden_calls), (
        f"keşif ajanı şunları çağırmamalı (Kural 8): {sorted(called & forbidden_calls)}"
    )


def test_watchlist_is_injectable_repo_not_touched(tmp_path: Path) -> None:
    """Testler/koşular repo'daki docs/egitim'e YAZMAMALI — defter kökü enjekte edilebilir."""
    from app.research.literature_scout import watchlist_dir_default, watchlist_path_for

    # enjekte edilen kök kullanılır
    assert watchlist_path_for("lora", tmp_path) == tmp_path / "lora-watchlist.md"
    # rag geriye uyum: dosya adı korunur
    assert watchlist_path_for("rag", tmp_path).name == "rag-watchlist.md"
    # varsayilan kok repo docs/egitim
    assert watchlist_dir_default().name == "egitim"

    wl = tmp_path / "wl"
    run_scout(
        topics=["lora"],
        inbox=tmp_path / "ib",
        watchlist_dir=wl,
        searcher=_searcher_for("lora adapter fine-tuning peft"),
        downloader=lambda u: _PDF,
        today="2026-07-21",
    )
    assert (wl / "lora-watchlist.md").exists(), "defter enjekte edilen köke yazılmalı"
