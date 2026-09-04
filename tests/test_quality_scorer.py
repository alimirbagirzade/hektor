"""İçe-alım kalite skoru testleri (saf rubrik + DB toplama, offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.quality_scorer import (
    IngestionInputs,
    gather_inputs,
    score_all_papers,
    score_ingestion,
    score_paper,
)
from app.memory.sqlite_store import ChunkQualityFlag, SqliteStore


def _rich_inputs(**over: object) -> IngestionInputs:
    base: dict[str, object] = {
        "n_pages": 10,
        "n_chars": 20_000,
        "has_title": True,
        "has_authors": True,
        "has_year": True,
        "n_chunks": 12,
        "n_sections": 5,
        "n_formulas": 3,
        "n_tables": 2,
        "n_figure_captions": 2,
        "clean_text_score": 10.0,
    }
    base.update(over)
    return IngestionInputs(**base)  # type: ignore[arg-type]


def test_rich_paper_ready_for_rag() -> None:
    res = score_ingestion(_rich_inputs())
    assert res.total == pytest.approx(100.0)
    assert res.status == "ready_for_rag"


def test_text_only_paper_usable() -> None:
    # formül/tablo/figür yok ama parse iyi → nötr puanlar → ~usable
    res = score_ingestion(_rich_inputs(n_formulas=0, n_tables=0, n_figure_captions=0))
    assert res.status in ("usable", "slow_but_usable")
    assert res.components["formula"] == 7.5
    assert res.components["table"] == 7.5


def test_failed_parse_zero_chunks() -> None:
    res = score_ingestion(_rich_inputs(n_chunks=0))
    assert res.status in ("failed", "unstable")
    # parse başarısız → çıkarım bileşenleri 0, yalnız metadata kalır
    assert res.components["section"] == 0.0
    assert res.components["formula"] == 0.0
    assert res.components["metadata"] == 10.0


def test_thresholds_deterministic() -> None:
    a = score_ingestion(_rich_inputs())
    b = score_ingestion(_rich_inputs())
    assert a.to_dict() == b.to_dict()


def test_low_density_low_ocr() -> None:
    res = score_ingestion(_rich_inputs(n_chars=1000, n_pages=20))  # 50 char/sayfa
    assert res.components["ocr"] < 5.0
    assert res.components["parse"] < 5.0


# --- DB toplama yolu -------------------------------------------------------
@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "ing.db")


def test_score_paper_from_db(store: SqliteStore) -> None:
    store.upsert_paper(
        paper_id="p1",
        file_hash="h1",
        source_path="x.pdf",
        title="Volatilite ve Momentum",
        authors='["A. Yazar"]',
        year="2024",
        n_pages=8,
        n_chars=16_000,
    )
    chunks = [
        {
            "chunk_id": f"p1_{i}",
            "paper_id": "p1",
            "chunk_index": i,
            "section_name": sec,
            "text": "Bu bölüm momentum ve volatilite ilişkisini açıklar. Figure 1 sonucu gösterir. "
            * 6,
        }
        for i, sec in enumerate(["abstract", "introduction", "methods", "results", "conclusion"])
    ]
    store.add_chunks(chunks)
    # bir tablo bayrağı + bir formül
    with store.session() as s:
        s.add(ChunkQualityFlag(flag_id="f1", chunk_id="p1_2", has_table=1))
    res = score_paper(store, "p1")
    assert res.total > 60
    assert res.status in ("usable", "slow_but_usable", "ready_for_rag")


def test_gather_unknown_paper_raises(store: SqliteStore) -> None:
    with pytest.raises(ValueError):
        gather_inputs(store, "nope")


def test_record_updates_paper_and_runs(store: SqliteStore) -> None:
    store.upsert_paper(paper_id="p2", file_hash="h2", source_path="y.pdf", title="T", n_pages=4)
    store.add_chunks(
        [{"chunk_id": "p2_0", "paper_id": "p2", "chunk_index": 0, "text": "metin " * 80}]
    )
    res = score_paper(store, "p2")
    run_id = store.add_ingestion_run(
        paper_id="p2",
        status=res.status,
        quality_score=res.total,
        component_scores=res.components,
        n_chunks=1,
    )
    assert run_id.startswith("ing_")
    runs = store.list_ingestion_runs(paper_id="p2")
    assert len(runs) == 1 and runs[0]["quality_score"] == res.total
    # Paper alanları güncellendi
    papers = {p.paper_id: p for p in store.list_papers()}
    assert papers["p2"].quality_score == res.total
    assert papers["p2"].ingest_status == res.status


def test_legacy_paper_null_quality_not_blocking(store: SqliteStore) -> None:
    # Skorlanmamış makale: quality_score NULL, yine de listelenir/çalışır (retrieval engellenmez)
    store.upsert_paper(paper_id="p3", file_hash="h3", source_path="z.pdf", title="Eski")
    papers = {p.paper_id: p for p in store.list_papers()}
    assert papers["p3"].quality_score is None
    assert papers["p3"].ingest_status is None


def test_add_ingestion_run_unknown_paper_raises(store: SqliteStore) -> None:
    # Öksüz içe-alım koşusu yaratılmamalı (uygulama düzeyi referans bütünlüğü)
    with pytest.raises(ValueError, match="paper_id"):
        store.add_ingestion_run(paper_id="yok_makale", status="usable", quality_score=80.0)


# --- toplu korpus taraması -------------------------------------------------
def _add_paper(store: SqliteStore, pid: str, n_chunks: int) -> None:
    store.upsert_paper(
        paper_id=pid, file_hash=f"h_{pid}", source_path=f"{pid}.pdf", title="T", n_pages=4
    )
    if n_chunks:
        store.add_chunks(
            [
                {"chunk_id": f"{pid}_{i}", "paper_id": pid, "chunk_index": i, "text": "metin " * 60}
                for i in range(n_chunks)
            ]
        )


def test_score_all_papers_distribution(store: SqliteStore) -> None:
    _add_paper(store, "good", 5)
    _add_paper(store, "empty", 0)  # chunk yok → düşük/failed
    summary = score_all_papers(store, worst_n=5)
    assert summary.total == 2
    assert summary.scored == 2
    assert sum(summary.by_status.values()) == 2
    assert 0.0 <= summary.avg_score <= 100.0
    # en zayıf en başta (artan skor)
    assert summary.worst[0]["total"] <= summary.worst[-1]["total"]


def test_score_all_papers_record_persists(store: SqliteStore) -> None:
    _add_paper(store, "p1", 3)
    summary = score_all_papers(store, record=True)
    assert summary.scored == 1
    assert store.list_ingestion_runs(paper_id="p1")
    papers = {p.paper_id: p for p in store.list_papers()}
    assert papers["p1"].quality_score is not None


def test_score_all_papers_empty_corpus(store: SqliteStore) -> None:
    summary = score_all_papers(store)
    assert summary.total == 0 and summary.scored == 0 and summary.avg_score == 0.0
