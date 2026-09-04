"""Detached training stop fix (Phase 2) testleri — offline, izole (tmp root).

Gerçek eğitim BAŞLATILMAZ. Sonlandırma testi zararsız bir python uyku süreciyle
yapılır (çapraz platform).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from app.agents.runtime.tracker import RunTracker, set_tracker
from app.memory.sqlite_store import SqliteStore
from app.training import detached_launch


@pytest.fixture(autouse=True)
def _test_tracker(tmp_path):
    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


def test_stop_creates_stop_training_file(tmp_path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    # pid YOK → graceful stop_requested beklenir
    (storage / "train_status.json").write_text(
        json.dumps({"adapter": "x", "iterations": 5}), encoding="utf-8"
    )
    res = detached_launch.request_stop_detached_training(root=tmp_path)
    assert res["ok"] is True
    assert (storage / "STOP_TRAINING").exists()
    assert res["stopped"] is False
    assert "stop_requested" in res["detail"]


def test_stop_graceful_when_no_status(tmp_path) -> None:
    (tmp_path / "storage").mkdir()
    res = detached_launch.request_stop_detached_training(root=tmp_path)
    assert res["ok"] is True
    assert (tmp_path / "storage" / "STOP_TRAINING").exists()
    assert res["pid"] is None


def test_read_detached_training_status(tmp_path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "train_status.json").write_text(
        json.dumps({"adapter": "y", "pid": 123456}), encoding="utf-8"
    )
    info = detached_launch.read_detached_training_status(root=tmp_path)
    assert info["adapter"] == "y"
    assert info["pid"] == 123456


def test_is_detached_running_false_for_dead_pid(tmp_path) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    # neredeyse kesinlikle var olmayan bir pid
    (storage / "train_status.json").write_text(
        json.dumps({"adapter": "z", "pid": 2_147_480_000}), encoding="utf-8"
    )
    assert detached_launch.is_detached_training_running(root=tmp_path) is False


def test_terminate_real_process(tmp_path) -> None:
    """pid canlıysa süreç gerçekten sonlandırılır (zararsız uyku süreci)."""
    storage = tmp_path / "storage"
    storage.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (storage / "train_status.json").write_text(
        json.dumps({"adapter": "z", "pid": proc.pid}), encoding="utf-8"
    )
    try:
        res = detached_launch.request_stop_detached_training(root=tmp_path)
        assert res["ok"] is True
        # süreç birkaç saniye içinde ölmeli
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        assert proc.poll() is not None  # sonlandırıldı
    finally:
        if proc.poll() is None:
            proc.kill()
