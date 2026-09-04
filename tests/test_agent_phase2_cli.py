"""Phase 2 CLI testleri — task/approval/stop-all + train --run onay kapısı (offline).

Rich genişlik kırılganlığını önlemek için COLUMNS geniş; STOP_ALL dosya yolu ve
is_stop_all_active testlerde izole edilir (gerçek storage/ dokunulmaz).
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from app.main import app
from app.memory.sqlite_store import SqliteStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _test_tracker(tmp_path):
    from app.agents.runtime.tracker import RunTracker, set_tracker

    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


def test_task_create_and_list(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    r = runner.invoke(app, ["task-create", "--agent", "arxiv-fetcher", "--title", "çek"])
    assert r.exit_code == 0
    r2 = runner.invoke(app, ["tasks-list", "--limit", "10"])
    assert r2.exit_code == 0


def test_approvals_list(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    r = runner.invoke(app, ["approvals-list"])
    assert r.exit_code == 0


def test_approval_approve_unknown(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    r = runner.invoke(app, ["approval-approve", "apr_yok"])
    assert r.exit_code == 1


def test_stop_all_and_clear(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    # STOP_ALL'ı gerçek storage/ yerine tmp'ye yönlendir (hermetik)
    p = tmp_path / "STOP_ALL"
    monkeypatch.setattr("app.agents.runtime.supervisor._stop_all_path", lambda root=None: p)
    r = runner.invoke(app, ["stop-all", "--reason", "test"])
    assert r.exit_code == 0
    assert p.exists()
    r2 = runner.invoke(app, ["clear-stop-all"])
    assert r2.exit_code == 0
    assert not p.exists()


def test_train_run_blocked_without_approval(monkeypatch) -> None:
    """train --run taze onay olmadan eğitim BAŞLATMAZ (exit 3 + onay isteği)."""
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.delenv("ACHILLES_TRAIN_SUPERVISED", raising=False)
    # STOP_ALL etkisini ayır: bu test onay kapısını sınar
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda root=None: False)
    r = runner.invoke(
        app,
        ["train", "--run", "--backend", "peft", "--adapter-name", "phase2_smoke_block"],
    )
    assert r.exit_code == 3  # onay gerekli → eğitim başlamadı


def test_train_dry_run_not_gated(monkeypatch) -> None:
    """`train` (--run YOK) onay kapısına takılmaz — eski davranış korunur (dry-run)."""
    monkeypatch.setenv("COLUMNS", "300")
    # backend peft + dry-run: gerçek eğitim yok; yalnız komut/JSON üretir.
    r = runner.invoke(app, ["train", "--backend", "peft"])
    # dry-run yolu yazılım kurulu olmasa da exit 0 (eksik paket uyarısı basabilir)
    assert r.exit_code == 0
