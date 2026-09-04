"""Agent CLI (Phase 1) — agents-list / agents-runs / agents-log testleri (offline).

Rich tablo genişliğine bağlı kırılganlığı önlemek için COLUMNS geniş tutulur ve
içerik yerine ağırlıklı olarak exit_code doğrulanır.
"""

from __future__ import annotations

from typer.testing import CliRunner

from app.main import app

runner = CliRunner()


def test_agents_list_cli(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    result = runner.invoke(app, ["agents-list"])
    assert result.exit_code == 0
    assert result.stdout.strip()  # tablo basıldı


def test_agents_runs_cli(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    result = runner.invoke(app, ["agents-runs", "--limit", "5"])
    assert result.exit_code == 0


def test_agents_log_unknown_run(monkeypatch) -> None:
    monkeypatch.setenv("COLUMNS", "300")
    result = runner.invoke(app, ["agents-log", "arun_does_not_exist"])
    assert result.exit_code == 1
