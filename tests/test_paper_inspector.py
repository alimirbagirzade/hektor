"""PaperInspector birim testleri — çevrimdışı, gerçek DB ile."""

from __future__ import annotations

from pathlib import Path

from app.learning.paper_inspector import PaperInspector
from app.memory.sqlite_store import SqliteStore


def _store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "test.db")


def _seed_paper(store: SqliteStore, paper_id: str, n_chars: int = 5000) -> None:
    store.upsert_paper(
        paper_id=paper_id,
        file_hash=f"hash_{paper_id}",
        source_path=f"/tmp/{paper_id}.pdf",
        title="Test Makale",
        year="2024",
        authors='["A. Yazar"]',
        n_pages=10,
        n_chars=n_chars,
    )


def _seed_chunks(store: SqliteStore, paper_id: str, n: int = 10, embedded: bool = True) -> None:
    chunks = [
        {
            "chunk_id": f"{paper_id}_c{i:04d}",
            "paper_id": paper_id,
            "chunk_index": i,
            "section_name": "Introduction",
            "text": "Bu chunk makale içeriğidir. " * 10,
            "char_count": 280,
            "token_estimate": 70,
            "embedded": 1 if embedded else 0,
        }
        for i in range(n)
    ]
    store.add_chunks(chunks)


def test_paper_not_found(tmp_path: Path) -> None:
    insp = PaperInspector(store=_store(tmp_path))
    result = insp.inspect("ghost_paper")
    assert "paper_not_found" in result.missing_steps
    assert result.parse_score == 0.0


def test_paper_with_full_data(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p1")
    _seed_chunks(store, "p1", n=10)
    insp = PaperInspector(store=store)
    result = insp.inspect("p1")
    assert result.parse_score > 0
    assert result.metadata_score >= 3.0
    assert result.chunk_count == 10
    assert result.embedded_count == 10
    assert result.chunk_quality > 0


def test_missing_title_detected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert_paper(
        paper_id="p2",
        file_hash="h2",
        source_path="/tmp/p2.pdf",
        n_chars=1000,
    )
    _seed_chunks(store, "p2", n=5)
    insp = PaperInspector(store=store)
    result = insp.inspect("p2")
    assert "missing_title" in result.missing_steps
    assert result.has_title is False


def test_no_chunks_detected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p3")
    insp = PaperInspector(store=store)
    result = insp.inspect("p3")
    assert "no_chunks" in result.missing_steps
    assert result.chunk_count == 0


def test_incomplete_embedding_detected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p4")
    _seed_chunks(store, "p4", n=10, embedded=False)
    insp = PaperInspector(store=store)
    result = insp.inspect("p4")
    assert "incomplete_embedding" in result.missing_steps
    assert result.embedded_count == 0


def test_static_total_max_40(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _seed_paper(store, "p5")
    _seed_chunks(store, "p5", n=20)
    insp = PaperInspector(store=store)
    result = insp.inspect("p5")
    assert result.static_total <= 40.0
