"""Tool-use eğitim modülü birim testleri — çevrimdışı, fake embeddings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.memory.sqlite_store import SqliteStore
from app.training.tool_use_dataset_builder import build_tool_use_dataset, get_tool_use_stats
from app.training.tool_use_trainer import ToolUseSession, ToolUseStep, ToolUseTrainer


def _make_store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "test.db")


def _fake_metrics(*, n_trades: int, sharpe: float, ret: float):
    from app.trading.backtester import BacktestMetrics

    return BacktestMetrics(
        n_trades=n_trades,
        total_return_pct=ret,
        sharpe=sharpe,
        sortino=sharpe,
        max_drawdown_pct=-5.0,
        profit_factor=1.0,
        win_rate_pct=50.0,
    )


def _fake_synthesized(name: str = "FakeRSI"):
    from app.research.synthesis_engine import SynthesisResult

    ir_dict = {
        "name": name,
        "market": "BTCUSD",
        "timeframe": "1h",
        "indicators": [{"name": "RSI", "period": 14}],
        "entry_rules": ["rsi_14 < 30"],
        "exit_rules": ["rsi_14 > 70"],
    }
    return SynthesisResult(
        indicator_name=name,
        description="Test indikatörü",
        source_papers=[],
        formula_components=[],
        combination_reasoning="Test mantığı",
        expected_edge="Düşük RSI geri dönüş",
        failure_conditions=[],
        strategy_ir=ir_dict,
    )


def _seed_full_session(store: SqliteStore, session_id: str, verdict: str) -> None:
    steps = [
        ("think", "Hipotez", None, {}, {}),
        (
            "call",
            "backtest()",
            "run_backtest",
            {"n_bars": 200},
            {"verdict": verdict, "metrics": {"n_trades": 5, "sharpe_ratio": 1.2}},
        ),
        ("observe", f"Sonuç: {verdict.upper()}", None, {}, {}),
        ("conclude", f"Yansıma — {verdict}", None, {}, {}),
    ]
    for idx, (stype, content, tname, tinput, toutput) in enumerate(steps):
        store.save_tool_use_example(
            example_id=f"{session_id}_{idx}",
            session_id=session_id,
            question="Test sorusu?",
            step_index=idx,
            step_type=stype,
            content=content,
            tool_name=tname,
            tool_input=tinput,
            tool_output=toutput,
            verdict=verdict if stype in ("call", "observe", "conclude") else None,
        )


# ---- ToolUseStep ----


def test_tool_use_step_defaults() -> None:
    step = ToolUseStep(step_type="think", content="hipotez")
    assert step.tool_name is None
    assert step.tool_input == {}
    assert step.verdict is None


# ---- ToolUseSession.as_sft_example ----


def test_as_sft_example_structure() -> None:
    session = ToolUseSession(session_id="tu_test", question="RSI nasıl çalışır?")
    session.steps = [
        ToolUseStep("think", "Hipotez"),
        ToolUseStep(
            "call",
            "backtest()",
            tool_name="run_backtest",
            tool_input={"n_bars": 500},
            tool_output={"verdict": "fail", "metrics": {"n_trades": 2}},
            verdict="fail",
        ),
        ToolUseStep("observe", "Sonuç: FAIL", verdict="fail"),
        ToolUseStep("conclude", "Az işlem, eşiği gevşet.", verdict="fail"),
    ]
    session.final_verdict = "fail"
    ex = session.as_sft_example()
    assert "instruction" in ex and "input" in ex and "output" in ex
    assert "run_backtest" in ex["input"]
    assert ex["output"] == "Az işlem, eşiği gevşet."
    assert ex["metadata"]["final_verdict"] == "fail"
    assert ex["metadata"]["n_tool_calls"] == 1


# ---- SqliteStore tool_use helpers ----


def test_save_and_list_tool_use_examples(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    store.save_tool_use_example(
        example_id="ex_001",
        session_id="tu_abc",
        question="Test?",
        step_index=0,
        step_type="think",
        content="Hipotez",
    )
    rows = store.list_tool_use_examples()
    assert len(rows) == 1
    assert rows[0]["session_id"] == "tu_abc"
    assert rows[0]["tool_input"] == {}


def test_list_tool_use_examples_filter_by_session(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    for sid, idx in [("sess_a", 0), ("sess_a", 1), ("sess_b", 0)]:
        store.save_tool_use_example(
            example_id=f"{sid}_{idx}",
            session_id=sid,
            question="Q",
            step_index=idx,
            step_type="think",
            content="x",
        )
    assert len(store.list_tool_use_examples(session_id="sess_a")) == 2
    assert len(store.list_tool_use_examples(session_id="sess_b")) == 1


# ---- ToolUseTrainer (mocked synthesis + backtest) ----


@patch("app.training.tool_use_trainer.run_backtest")
@patch("app.training.tool_use_trainer.eval_strategy")
def test_run_session_records_four_steps(mock_eval, mock_bt, tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    trainer = ToolUseTrainer(store=store, n_bars=200, seed=1)

    trainer.synthesis = MagicMock()
    trainer.synthesis.synthesize.return_value = _fake_synthesized()
    mock_bt.return_value = MagicMock(metrics=_fake_metrics(n_trades=10, sharpe=1.5, ret=30.0))
    mock_eval.return_value = MagicMock(verdict="pass", reasons=["Sharpe yüksek"])

    session = trainer.run_session("RSI nasıl çalışır?", max_iterations=1)

    assert session.final_verdict == "pass"
    assert [s.step_type for s in session.steps] == ["think", "call", "observe", "conclude"]
    assert len(store.list_tool_use_examples(session_id=session.session_id)) == 4


@patch("app.training.tool_use_trainer.run_backtest")
@patch("app.training.tool_use_trainer.eval_strategy")
def test_run_session_fail_uses_reflection(mock_eval, mock_bt, tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    trainer = ToolUseTrainer(store=store, n_bars=200, seed=2)

    trainer.synthesis = MagicMock()
    trainer.synthesis.synthesize.return_value = _fake_synthesized("SlowEMA")
    mock_bt.return_value = MagicMock(metrics=_fake_metrics(n_trades=1, sharpe=-0.3, ret=-5.0))
    mock_eval.return_value = MagicMock(verdict="fail", reasons=["Az işlem"])
    trainer.reflection = MagicMock()
    trainer.reflection.reflect.return_value = "Entry kuralını gevşet."

    session = trainer.run_session("EMA stratejisi?", max_iterations=1)
    assert session.final_verdict == "fail"
    conclude = next(s for s in session.steps if s.step_type == "conclude")
    assert "gevşet" in conclude.content


# ---- build_tool_use_dataset ----


def test_build_returns_both_sessions(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_full_session(store, "tu_p", "pass")
    _seed_full_session(store, "tu_f", "fail")

    examples = build_tool_use_dataset(store=store)
    assert len(examples) == 2
    for ex in examples:
        assert {"instruction", "input", "output", "metadata"} <= ex.keys()


def test_build_verdict_filter(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_full_session(store, "tu_p", "pass")
    _seed_full_session(store, "tu_f", "fail")

    result = build_tool_use_dataset(store=store, only_verdict="pass")
    assert len(result) == 1
    assert result[0]["metadata"]["final_verdict"] == "pass"


def test_build_writes_jsonl(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_full_session(store, "tu_w", "pass")
    out = tmp_path / "out.jsonl"
    build_tool_use_dataset(store=store, output_path=out)
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["metadata"]["session_id"] == "tu_w"


def test_get_tool_use_stats(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    _seed_full_session(store, "s1", "pass")
    _seed_full_session(store, "s2", "fail")

    stats = get_tool_use_stats(store=store)
    assert stats["n_sessions"] == 2
    assert stats["n_steps"] == 8
    assert stats["sft_eligible"] == 2
    assert stats["verdict_distribution"]["pass"] == 1
    assert stats["verdict_distribution"]["fail"] == 1


def test_empty_db_returns_no_examples(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    assert build_tool_use_dataset(store=store) == []
    stats = get_tool_use_stats(store=store)
    assert stats["n_sessions"] == 0
