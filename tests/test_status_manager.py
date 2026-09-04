"""StatusManager birim testleri."""

from __future__ import annotations

from pathlib import Path

from app.learning.status_manager import StatusManager
from app.memory.mastery_store import MasteryStore


def _mgr(tmp_path: Path) -> StatusManager:
    store = MasteryStore(db_path=tmp_path / "mastery.db")
    return StatusManager(store=store)


def test_default_status_uploaded(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    status = mgr.get_current("p1")
    assert status == "uploaded"


def test_update_and_get_current(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.update("p1", "parsed", reason="parse OK")
    assert mgr.get_current("p1") == "parsed"


def test_history_tracks_transitions(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.update("p1", "parsed", reason="step1")
    mgr.update("p1", "chunked", reason="step2")
    history = mgr.get_history("p1")
    assert len(history) >= 2
    statuses = [h["new_status"] for h in history]
    assert "parsed" in statuses
    assert "chunked" in statuses


def test_status_from_score_thresholds(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    assert mgr.status_from_score(95.0) == "learned"
    assert mgr.status_from_score(80.0) == "usable_needs_review"
    assert mgr.status_from_score(65.0) == "partially_learned"
    assert mgr.status_from_score(45.0) == "needs_rechunking"
    assert mgr.status_from_score(20.0) == "failed"


def test_multiple_papers_independent(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    mgr.update("p1", "indexed")
    mgr.update("p2", "failed")
    assert mgr.get_current("p1") == "indexed"
    assert mgr.get_current("p2") == "failed"
