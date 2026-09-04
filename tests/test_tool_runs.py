"""Araç çalışma kaydı (tool_runs / tool_artifacts) round-trip testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.memory.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "tools.db")


def test_log_tool_run_ok(store: SqliteStore) -> None:
    with store.log_tool_run(tool_id="montecarlo", params={"n": 10}, seed=42) as run_id:
        store.set_tool_run_output(run_id, {"ruin_probability": 0.1})
    run = store.get_tool_run(run_id)
    assert run is not None
    assert run["status"] == "ok"
    assert run["seed"] == 42
    assert run["output_summary"]["ruin_probability"] == 0.1
    assert run["finished_at"] is not None


def test_log_tool_run_error_marks_status(store: SqliteStore) -> None:
    with pytest.raises(RuntimeError), store.log_tool_run(tool_id="stats", seed=1) as run_id:
        raise RuntimeError("patladı")
    run = store.get_tool_run(run_id)
    assert run is not None
    assert run["status"] == "error"
    assert "patladı" in (run["error"] or "")


def test_link_and_list_artifacts(store: SqliteStore) -> None:
    with store.log_tool_run(tool_id="montecarlo", seed=1) as run_id:
        store.link_tool_artifact(
            tool_run_id=run_id, artifact_type="json", description="mc özeti", content_hash="abc"
        )
    arts = store.list_tool_artifacts(run_id)
    assert len(arts) == 1
    assert arts[0]["description"] == "mc özeti"
    assert arts[0]["content_hash"] == "abc"


def test_list_tool_runs_filter_by_tool(store: SqliteStore) -> None:
    with store.log_tool_run(tool_id="montecarlo", seed=1):
        pass
    with store.log_tool_run(tool_id="stats-correlation", seed=1):
        pass
    mc = store.list_tool_runs(tool_id="montecarlo")
    assert len(mc) == 1
    assert mc[0]["tool_id"] == "montecarlo"


def test_link_artifact_unknown_run_raises(store: SqliteStore) -> None:
    # Öksüz artefakt yaratılmamalı (uygulama düzeyi referans bütünlüğü)
    with pytest.raises(ValueError, match="tool_run_id"):
        store.link_tool_artifact(tool_run_id="yok_böyle_run", description="x")
