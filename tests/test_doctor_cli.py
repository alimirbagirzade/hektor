"""`achilles doctor` — kurulum/sürüm sapması teşhisi (offline testler).

Tümü çevrimdışı: git yalnız yerel ref'leri okur, ağ/Ollama gerekmez.
Windows scheduled-task yolu `_task_path_matches` monkeypatch ile sahte tutulur,
böylece testler tüm platformlarda deterministiktir.
"""

from __future__ import annotations

from typer.testing import CliRunner

import app.main as m
from app.main import app

runner = CliRunner()
# Rich tablosu dar tty'de sarılmasın diye geniş sütun
_ENV = {"COLUMNS": "200"}


def test_doctor_runs_readonly() -> None:
    """Gerçek repoda salt-okuma çalışır; git mevcut → asla exit 1.

    Yakınsamışsa 0, sapma varsa 2 döner (ikisi de geçerli).
    """
    result = runner.invoke(app, ["doctor"], env=_ENV)
    assert result.exit_code in (0, 2)
    assert "Repo yolu" in result.stdout
    assert "Dal == main" in result.stdout


def test_doctor_no_git(monkeypatch) -> None:
    """git yoksa (rc=127) açık hata + exit 1."""
    monkeypatch.setattr(m, "_git_ro", lambda args, cwd: (127, ""))
    result = runner.invoke(app, ["doctor"], env=_ENV)
    assert result.exit_code == 1
    assert "git" in result.stdout.lower()


def _fake_git_converged(args: list[str], cwd: object) -> tuple[int, str]:
    table = {
        ("rev-parse", "--is-inside-work-tree"): (0, "true"),
        ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
        ("rev-parse", "--short", "HEAD"): (0, "abc1234"),
        ("rev-parse", "HEAD"): (0, "abc1234deadbeef"),
        ("rev-parse", "--short", "origin/main"): (0, "abc1234"),
        ("rev-parse", "origin/main"): (0, "abc1234deadbeef"),
        ("rev-list", "--left-right", "--count", "origin/main...HEAD"): (0, "0\t0"),
    }
    return table.get(tuple(args), (0, ""))


def test_doctor_converged(monkeypatch) -> None:
    """main + HEAD==origin/main + ahead/behind 0 → exit 0 (yakınsamış)."""
    monkeypatch.setattr(m, "_git_ro", _fake_git_converged)
    monkeypatch.setattr(m, "_task_path_matches", lambda task, repo: (None, None))
    result = runner.invoke(app, ["doctor"], env=_ENV)
    assert result.exit_code == 0
    assert "Yakınsamış" in result.stdout


def _fake_git_drift(args: list[str], cwd: object) -> tuple[int, str]:
    table = {
        ("rev-parse", "--is-inside-work-tree"): (0, "true"),
        ("rev-parse", "--abbrev-ref", "HEAD"): (0, "fix/parked"),
        ("rev-parse", "--short", "HEAD"): (0, "0009a33"),
        ("rev-parse", "HEAD"): (0, "0009a33feature"),
        ("rev-parse", "--short", "origin/main"): (0, "b77b559"),
        ("rev-parse", "origin/main"): (0, "b77b559mainmain"),
        ("rev-list", "--left-right", "--count", "origin/main...HEAD"): (0, "73\t15"),
    }
    return table.get(tuple(args), (0, ""))


def test_doctor_drift_parked_branch(monkeypatch) -> None:
    """Feature dalına parklanmış + HEAD!=origin/main → exit 2 (SAPMA)."""
    monkeypatch.setattr(m, "_git_ro", _fake_git_drift)
    monkeypatch.setattr(m, "_task_path_matches", lambda task, repo: (None, None))
    result = runner.invoke(app, ["doctor"], env=_ENV)
    assert result.exit_code == 2
    assert "SAPMA" in result.stdout
