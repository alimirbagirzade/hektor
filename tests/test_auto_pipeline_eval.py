"""Auto-LoRA eval gate testi — adapter base ile GERÇEKTEN karşılaştırılır.

v5 regresyon dersi: eski gate base modeli ölçüp körlemesine EVAL_PASSED damgalıyordu.
Artık evaluate_adapter ile base vs adapter kıyaslanır; regresyon → terfi YOK.
Tamamen çevrimdışı (evaluate_adapter monkeypatch'lenir, torch gerekmez).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.lora.auto_pipeline import AutoLoRAPipeline, PipelineStage


class _FakeRes:
    def __init__(self, regression: bool, verdict: str) -> None:
        self.regression = regression
        self.verdict = verdict

    def to_dict(self) -> dict:
        return {
            "regression": self.regression,
            "verdict": self.verdict,
            "n": 2,
            "adapter_score": 0.5,
        }


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoLoRAPipeline:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "core.jsonl").write_text(
        '{"question": "q?", "must_avoid": []}\n', encoding="utf-8"
    )
    (tmp_path / "storage").mkdir()
    p = AutoLoRAPipeline()

    async def _noop(name: str) -> None:
        return None

    monkeypatch.setattr(p, "_register_adapter", _noop)
    return p


def test_eval_rejects_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.training.adapter_eval.evaluate_adapter",
        lambda *a, **k: _FakeRes(regression=True, verdict="reject"),
    )
    asyncio.run(p._run_eval("adapter_x"))
    assert p._state.stage == PipelineStage.EVAL_FAILED


def test_eval_passes_when_adapter_better(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.training.adapter_eval.evaluate_adapter",
        lambda *a, **k: _FakeRes(regression=False, verdict="accept"),
    )
    asyncio.run(p._run_eval("adapter_x"))
    assert p._state.stage == PipelineStage.EVAL_PASSED


def test_eval_inconclusive_not_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "app.training.adapter_eval.evaluate_adapter",
        lambda *a, **k: _FakeRes(regression=False, verdict="inconclusive"),
    )
    asyncio.run(p._run_eval("adapter_x"))
    assert p._state.stage == PipelineStage.EVAL_SKIPPED


def test_eval_skipped_when_deps_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _setup(tmp_path, monkeypatch)

    def _raise(*a: object, **k: object) -> None:
        raise ImportError("torch kurulu değil")

    monkeypatch.setattr("app.training.adapter_eval.evaluate_adapter", _raise)
    asyncio.run(p._run_eval("adapter_x"))
    assert p._state.stage == PipelineStage.EVAL_SKIPPED
