"""L5 CompositionGate testleri — Math + Novelty + Backtest kapıları + kombinasyon."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from app.trading.market_data_loader import generate_synthetic_ohlcv
from app.trading.strategy_ir import IndicatorSpec, StrategyIR, example_ir
from app.verification.exams.l5_composition import CompositionGate, _signature


def _gate_named(res, name):
    return next(g for g in res.gates if g.gate == name)


# ---------------------------------------------------------------- math gate
def test_math_gecerli_ir_pass() -> None:
    res = CompositionGate().evaluate_composition(example_ir())  # df yok
    assert _gate_named(res, "math").passed is True


def test_math_rsi_100_uzeri_fail() -> None:
    ir = StrategyIR(
        name="rsi_imkansiz",
        indicators=[IndicatorSpec(name="RSI", period=14), IndicatorSpec(name="EMA", period=20)],
        entry_rules=["rsi_14 > 100"],
        exit_rules=["ema_20 < rsi_14"],
    )
    res = CompositionGate().evaluate_composition(ir)
    math_gate = _gate_named(res, "math")
    assert math_gate.passed is False
    assert any("RSI" in d for d in math_gate.details)


def test_math_entropy_out_of_bounds_fail() -> None:
    # ENTROPY ∈ [0,1]; "entropy_4 > 5" mantıksal olarak imkansız → math kapısı reddetmeli.
    ir = StrategyIR(
        name="entropy_imkansiz",
        indicators=[IndicatorSpec(name="ENTROPY", period=4), IndicatorSpec(name="EMA", period=20)],
        entry_rules=["entropy_4 > 5"],
        exit_rules=["ema_20 < entropy_4"],
    )
    math_gate = _gate_named(CompositionGate().evaluate_composition(ir), "math")
    assert math_gate.passed is False
    assert any("ENTROPY" in d for d in math_gate.details)


def test_math_periyot_1_fail() -> None:
    ir = StrategyIR(
        name="periyot1",
        indicators=[IndicatorSpec(name="EMA", period=1), IndicatorSpec(name="RSI", period=14)],
        entry_rules=["ema_1 > rsi_14"],
        exit_rules=["ema_1 < rsi_14"],
    )
    assert _gate_named(CompositionGate().evaluate_composition(ir), "math").passed is False


# ---------------------------------------------------------------- novelty gate
def test_novelty_tek_tip_fail() -> None:
    ir = StrategyIR(
        name="ema_cross",
        indicators=[IndicatorSpec(name="EMA", period=20), IndicatorSpec(name="EMA", period=50)],
        entry_rules=["ema_20 > ema_50"],
        exit_rules=["ema_20 < ema_50"],
    )
    novelty = _gate_named(CompositionGate().evaluate_composition(ir), "novelty")
    assert novelty.passed is False


def test_novelty_iki_tip_pass() -> None:
    novelty = _gate_named(CompositionGate().evaluate_composition(example_ir()), "novelty")
    assert novelty.passed is True


def test_novelty_kopya_imza_fail() -> None:
    ir = example_ir()
    seen = {_signature(ir)}
    res = CompositionGate().evaluate_composition(ir, seen_signatures=seen)
    novelty = _gate_named(res, "novelty")
    assert novelty.passed is False
    assert any("Kopya" in d for d in novelty.details)


# ---------------------------------------------------------------- backtest gate + kombinasyon
def test_backtest_stub_fail_candidate_degil() -> None:
    def fail_eval(df, ir, **kw):
        return SimpleNamespace(verdict="fail", reasons=["stub fail"])

    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    res = CompositionGate(evaluator=fail_eval).evaluate_composition(example_ir(), df)
    assert _gate_named(res, "backtest").passed is False
    assert res.candidate is False
    assert res.verdict == "rejected"


def test_uc_kapi_pass_aday() -> None:
    def pass_eval(df, ir, **kw):
        return SimpleNamespace(verdict="pass", reasons=["stub pass"])

    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    res = CompositionGate(evaluator=pass_eval).evaluate_composition(example_ir(), df)
    assert res.candidate is True
    assert res.verdict == "candidate"
    assert all(g.passed for g in res.gates)
    assert res.to_dict()["candidate"] is True


def test_veri_yoksa_aday_degil() -> None:
    # df yok → backtest sertifikalanamaz → aday değil (test edilmeden hazır deme)
    res = CompositionGate().evaluate_composition(example_ir())
    assert res.candidate is False
    assert _gate_named(res, "backtest").passed is False


def test_gercek_evaluate_wiring_caliisir() -> None:
    # Gerçek evaluate() ile uçtan uca çalışır (pass/fail veri-bağımlı, sadece çökmesin).
    df = generate_synthetic_ohlcv(n=2000, seed=42)
    res = CompositionGate().evaluate_composition(example_ir(), df)
    backtest = _gate_named(res, "backtest")
    assert isinstance(res.candidate, bool)
    assert any("verdict=" in d for d in backtest.details)
