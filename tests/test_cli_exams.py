"""CLI sınav komutları — çevrimdışı, deterministik (typer CliRunner).

LLM offline'a zorlanır (monkeypatch) → L3/L4 'skipped'; L5 LLM gerektirmez
(sentetik veri + evaluate). Tüm komutlar exit 0 dönmeli.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from app.main import app

runner = CliRunner()


@pytest.fixture
def _offline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.brain.local_llm.LocalLLM.available", lambda self: False)


def test_exam_l3_offline_skipped(_offline_llm: None) -> None:
    result = runner.invoke(app, ["exam-l3", "--indicator", "SMA", "--json"])
    assert result.exit_code == 0
    assert "skipped" in result.stdout


def test_understanding_score_offline_insufficient(_offline_llm: None) -> None:
    result = runner.invoke(app, ["understanding-score", "--json"])
    assert result.exit_code == 0
    assert "insufficient_data" in result.stdout


def test_exam_l5_synthetic_runs() -> None:
    # L5 LLM gerektirmez: sentetik veri + gerçek evaluate
    result = runner.invoke(app, ["exam-l5", "--json", "--seed", "42"])
    assert result.exit_code == 0
    assert "candidate" in result.stdout
    assert "verdict" in result.stdout
