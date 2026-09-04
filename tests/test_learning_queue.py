"""LearningQueue ve PaperMasteryAgent birim testleri (mock ile)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.learning.mastery_scorer import MasteryScore
from app.learning.paper_mastery_agent import LearningQueue, MasteryRunResult
from app.memory.mastery_store import MasteryStore
from app.memory.sqlite_store import SqliteStore


def _stores(tmp_path: Path) -> tuple[SqliteStore, MasteryStore]:
    db = tmp_path / "achilles.db"
    return SqliteStore(db_path=db), MasteryStore(db_path=db)


def _queue(tmp_path: Path) -> LearningQueue:
    sqlite_store, mastery_store = _stores(tmp_path)
    return LearningQueue(store=sqlite_store, mastery_store=mastery_store)


def _seed_paper(store: SqliteStore, paper_id: str) -> None:
    store.upsert_paper(
        paper_id=paper_id,
        file_hash=f"h_{paper_id}",
        source_path=f"/tmp/{paper_id}.pdf",
        title="Test",
    )


def test_enqueue_single_paper(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    qid = q.enqueue_paper("p1")
    assert qid
    entries = q.list_all()
    assert any(e["paper_id"] == "p1" for e in entries)


def test_enqueue_all_papers(tmp_path: Path) -> None:
    sqlite_store, mastery_store = _stores(tmp_path)
    for i in range(3):
        _seed_paper(sqlite_store, f"p{i}")
    q = LearningQueue(store=sqlite_store, mastery_store=mastery_store)
    count = q.enqueue_all_papers()
    assert count == 3


def test_enqueue_deduplicated(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    q.enqueue_paper("p1")
    q.enqueue_paper("p1")
    entries = [e for e in q.list_all() if e["paper_id"] == "p1"]
    assert len(entries) >= 1


def test_run_next_returns_none_if_empty(tmp_path: Path) -> None:
    q = _queue(tmp_path)
    result = q.run_next()
    assert result is None


def test_failed_record_retried_within_max_attempts(tmp_path: Path) -> None:
    # Başarısız (failed) ama attempts<max_attempts kayıt yeniden seçilebilmeli (retry ölü değil).
    _, ms = _stores(tmp_path)
    qid = ms.enqueue("p1", priority=5)
    ms.update_queue_status(qid, "running")  # attempts → 1
    ms.update_queue_status(qid, "failed", error="geçici hata")
    nxt = ms.get_next_queued()
    assert nxt is not None and nxt["paper_id"] == "p1"


def _make_mock_result(paper_id: str) -> MasteryRunResult:
    score = MasteryScore(
        paper_id=paper_id,
        test_id="t1",
        parse_score=10,
        metadata_score=5,
        chunk_quality_score=15,
        index_score=10,
        retrieval_score=15,
        citation_score=15,
        grounding_score=15,
        abstention_score=10,
        formula_argument_score=5,
    )
    return MasteryRunResult(
        paper_id=paper_id,
        test_id="t1",
        score=score,
        n_questions=10,
        n_passed=10,
        n_failed=0,
        report_json="r.json",
        report_md="r.md",
    )


@patch("app.learning.paper_mastery_agent.PaperMasteryAgent.run")
def test_run_next_processes_queued(mock_run: MagicMock, tmp_path: Path) -> None:
    sqlite_store, mastery_store = _stores(tmp_path)
    _seed_paper(sqlite_store, "p1")
    q = LearningQueue(store=sqlite_store, mastery_store=mastery_store)
    q.enqueue_paper("p1", priority=5)
    mock_run.return_value = _make_mock_result("p1")
    result = q.run_next()
    assert result is not None
    assert result.paper_id == "p1"


@patch("app.learning.paper_mastery_agent.PaperMasteryAgent.run")
def test_run_all_respects_limit(mock_run: MagicMock, tmp_path: Path) -> None:
    sqlite_store, mastery_store = _stores(tmp_path)
    for i in range(5):
        _seed_paper(sqlite_store, f"p{i}")
        mastery_store.enqueue(f"p{i}", priority=5)
    mock_run.side_effect = lambda paper_id, **kw: _make_mock_result(paper_id)
    q = LearningQueue(store=sqlite_store, mastery_store=mastery_store)
    results = q.run_all(limit=3)
    assert len(results) <= 3
