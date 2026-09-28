"""LearningQueue ve PaperMasteryAgent birim testleri (mock ile)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.learning.mastery_scorer import MasteryScore
from app.learning.paper_mastery_agent import LearningQueue, MasteryRunResult, PaperMasteryAgent
from app.memory.mastery_store import MasteryStore, PaperLearningQueue
from app.memory.sqlite_store import SqliteStore


def _stores(tmp_path: Path) -> tuple[SqliteStore, MasteryStore]:
    db = tmp_path / "hektor.db"
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


# ── Kademe 2 regresyonları: asılı 'running', retry sırası, başarısız test kapanışı ──


def _set_row(ms: MasteryStore, queue_id: str, **fields: object) -> None:
    with ms.session() as s:
        row = s.get(PaperLearningQueue, queue_id)
        assert row is not None
        for k, v in fields.items():
            setattr(row, k, v)


def _hours_ago(h: float) -> str:
    return (dt.datetime.now(dt.UTC) - dt.timedelta(hours=h)).isoformat()


@patch("app.learning.paper_mastery_agent.PaperMasteryAgent.run")
def test_run_next_exception_marks_failed_and_run_all_continues(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    # E1: ajan istisna fırlatırsa kayıt 'running'de asılı kalmamalı, run_all çökmemeli.
    sqlite_store, ms = _stores(tmp_path)
    ms.enqueue("p_bad", priority=9)
    ms.enqueue("p_ok", priority=1)

    def _run(paper_id: str, **kw: object) -> MasteryRunResult:
        if paper_id == "p_bad":
            raise RuntimeError("database is locked")
        return _make_mock_result(paper_id)

    mock_run.side_effect = _run
    q = LearningQueue(store=sqlite_store, mastery_store=ms)
    results = q.run_all(limit=2)
    assert [r.paper_id for r in results] == ["p_bad", "p_ok"]
    assert results[0].error and "database is locked" in results[0].error
    rows = {r["paper_id"]: r for r in ms.list_queue()}
    assert rows["p_bad"]["status"] == "failed"
    assert "database is locked" in (rows["p_bad"]["last_error"] or "")
    assert rows["p_ok"]["status"] == "done"
    assert not any(r["status"] == "running" for r in rows.values())


@patch("app.learning.paper_mastery_agent.PaperMasteryAgent.run")
def test_run_next_keyboard_interrupt_marks_failed_and_reraises(
    mock_run: MagicMock, tmp_path: Path
) -> None:
    sqlite_store, ms = _stores(tmp_path)
    ms.enqueue("p1")
    mock_run.side_effect = KeyboardInterrupt
    q = LearningQueue(store=sqlite_store, mastery_store=ms)
    with pytest.raises(KeyboardInterrupt):
        q.run_next()
    row = ms.list_queue()[0]
    assert row["status"] == "failed" and "KeyboardInterrupt" in (row["last_error"] or "")


def test_agent_create_test_failure_is_contained(tmp_path: Path) -> None:
    # create_test ajanın try'ı dışında — hata run_next'te yakalanıp kayıt kapanmalı.
    sqlite_store, ms = _stores(tmp_path)
    ms.enqueue("p1")
    q = LearningQueue(store=sqlite_store, mastery_store=ms)
    with patch.object(ms, "create_test", side_effect=RuntimeError("database is locked")):
        result = q.run_next()
    assert result is not None and result.error
    assert ms.list_queue()[0]["status"] == "failed"


def test_stale_running_row_is_reclaimed(tmp_path: Path) -> None:
    # E1: çöken süreçten kalan bayat 'running' kayıt yeniden seçilebilir (deneme sayılır).
    _, ms = _stores(tmp_path)
    qid = ms.enqueue("p1")
    ms.update_queue_status(qid, "running")  # attempts → 1
    _set_row(ms, qid, updated_at=_hours_ago(7))
    claimed = ms.claim_next_queued()
    assert claimed is not None and claimed["queue_id"] == qid
    assert claimed["attempts"] == 2
    assert ms.list_queue()[0]["attempts"] == 2


def test_fresh_running_row_is_not_stolen(tmp_path: Path) -> None:
    # Eşik dolmadan canlı süreçteki 'running' kayıt başka sürece verilmez.
    _, ms = _stores(tmp_path)
    qid = ms.enqueue("p1")
    ms.update_queue_status(qid, "running")
    _set_row(ms, qid, updated_at=_hours_ago(1))
    assert ms.get_next_queued() is None
    assert ms.claim_next_queued() is None


def test_stale_running_respects_max_attempts(tmp_path: Path) -> None:
    _, ms = _stores(tmp_path)
    qid = ms.enqueue("p1")
    _set_row(ms, qid, status="running", attempts=3, updated_at=_hours_ago(24))
    assert ms.get_next_queued() is None


def test_second_claim_does_not_take_running_row(tmp_path: Path) -> None:
    # Bir süreç kaydı aldıktan sonra ikinci talep aynı kaydı almaz.
    _, ms = _stores(tmp_path)
    ms.enqueue("p1")
    assert ms.claim_next_queued() is not None
    assert ms.claim_next_queued() is None
    assert ms.list_queue()[0]["attempts"] == 1


def test_claim_skips_row_changed_after_selection(tmp_path: Path) -> None:
    # Seçim ile işaretleme arasında başka süreç kaydı aldıysa koşullu UPDATE 0 satır etkiler.
    _, ms = _stores(tmp_path)
    qid = ms.enqueue("p1")
    real_get = ms.get_next_queued

    def _racy_get(**kw: object) -> dict | None:
        item = real_get()
        if item is not None and item["status"] == "pending":
            ms.update_queue_status(qid, "running")  # rakip süreç önce davrandı
        return item

    with patch.object(ms, "get_next_queued", side_effect=_racy_get):
        assert ms.claim_next_queued() is None
    assert ms.list_queue()[0]["attempts"] == 1  # yalnız rakibin denemesi


def test_pending_preferred_over_failed_retry(tmp_path: Path) -> None:
    # E2: başarısız kayıt, bekleyenler bitmeden art arda yeniden denenmez.
    _, ms = _stores(tmp_path)
    q_fail = ms.enqueue("p_fail", priority=5)
    ms.enqueue("p_pending", priority=5)
    ms.update_queue_status(q_fail, "running")
    ms.update_queue_status(q_fail, "failed", error="geçici")
    nxt = ms.get_next_queued()
    assert nxt is not None and nxt["paper_id"] == "p_pending"


def test_failed_retries_rotate_by_updated_at(tmp_path: Path) -> None:
    _, ms = _stores(tmp_path)
    a = ms.enqueue("p_a")
    b = ms.enqueue("p_b")
    _set_row(ms, a, status="failed", attempts=1, updated_at=_hours_ago(1))
    _set_row(ms, b, status="failed", attempts=1, updated_at=_hours_ago(2))
    nxt = ms.get_next_queued()
    assert nxt is not None and nxt["paper_id"] == "p_b"  # en uzun bekleyen önce


def test_done_clears_last_error(tmp_path: Path) -> None:
    # E3: yeniden denemede başarı → eski hata mesajı temizlenir.
    _, ms = _stores(tmp_path)
    qid = ms.enqueue("p1")
    ms.update_queue_status(qid, "running")
    ms.update_queue_status(qid, "failed", error="geçici")
    ms.update_queue_status(qid, "running")
    ms.update_queue_status(qid, "done")
    row = ms.list_queue()[0]
    assert row["status"] == "done" and row["last_error"] is None


def test_missing_paper_closes_test_as_failed(tmp_path: Path) -> None:
    # E3: başarısız koşu testi 'done' değil 'failed' kapatır (rapor üretimini gölgelemez).
    sqlite_store, ms = _stores(tmp_path)
    agent = PaperMasteryAgent(store=sqlite_store, mastery_store=ms)
    result = agent.run("olmayan_makale", question_count=2)
    assert result.error
    assert [t["status"] for t in ms.list_tests("olmayan_makale")] == ["failed"]
    assert ms.list_finished_tests() == []


def test_agent_exception_path_closes_test_as_failed(tmp_path: Path) -> None:
    sqlite_store, ms = _stores(tmp_path)
    _seed_paper(sqlite_store, "p1")
    agent = PaperMasteryAgent(store=sqlite_store, mastery_store=ms)
    with patch.object(agent._q_gen, "generate", side_effect=RuntimeError("patladı")):
        result = agent.run("p1", question_count=2)
    assert result.error == "patladı"
    assert [t["status"] for t in ms.list_tests("p1")] == ["failed"]


def test_agent_exception_cleanup_failure_does_not_mask_error(tmp_path: Path) -> None:
    sqlite_store, ms = _stores(tmp_path)
    _seed_paper(sqlite_store, "p1")
    agent = PaperMasteryAgent(store=sqlite_store, mastery_store=ms)
    with (
        patch.object(agent._q_gen, "generate", side_effect=RuntimeError("asıl hata")),
        patch.object(ms, "finish_test", side_effect=RuntimeError("database is locked")),
    ):
        result = agent.run("p1", question_count=2)
    assert result.error == "asıl hata"
