"""Üretim motoru ↔ bağımsız yeniden hesap: kontrollü bar senaryolarında birebir aynı işlemler.

Uç durumlar gerçek veride oluşmasını beklemeden, elle kurulmuş barlarla sınanır: gap'le stop
geçişi (long/short), aynı barda stop + hedef (stop önce), giriş barında stop, hedefin ötesinde
açılış (hedeften dolum), short stop, kural çıkışı ve dönem sonu kapanışı. Her senaryoda beklenen
çıkış nedeni ayrıca elle doğrulanır; bağımsız hesap ``tests/independent_backtest.py`` üretim
motorunun hiçbir fonksiyonunu içe aktarmaz.
"""

from __future__ import annotations

import pandas as pd
import pytest
from tests import independent_backtest as ib

from app.trading import event_engine as ee
from app.trading.strategy_spec import TestableStrategy

COSTS = {
    "commission_bps_per_side": 10,
    "slippage_bps_per_side": 2,
    "spread_bps": 4,
    "funding_bps_per_day": 5,
}


def _spec(direction: str, *, entry, exit_=None, stop=None, tp=None, sizing=None) -> dict:
    return {
        "name": "kontrol",
        "market": "TEST",
        "timeframe": "1h",
        "direction": direction,
        "indicators": [],
        "entry_rules": entry,
        "exit_rules": exit_ or [],
        "stop": stop or {"type": "fixed_pct", "value": 0.02},
        "take_profit": tp or {"type": "none"},
        "sizing": sizing or {"type": "fixed_fraction", "fraction": 1.0},
        "costs": COSTS,
    }


def _bars(rows: list[tuple[float, float, float, float]]) -> tuple[pd.DataFrame, ib.Bars]:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    times = [t.isoformat() for t in idx]
    b = ib.Bars(times, *[list(df[c]) for c in ("open", "high", "low", "close")])
    return df, b


def _both(spec: dict, rows) -> tuple[list[dict], list[ib.ITrade], dict]:
    df, b = _bars(rows)
    res = ee.run(df, TestableStrategy.model_validate(spec))
    ind = ib.run(b, spec, b.time[0], b.time[-1], bar_days=1 / 24)
    prod = res.summary()["trades"]
    cmp = ib.compare_trades(prod, ind.trades, n=100)
    assert cmp["same_structure"], cmp
    assert cmp["max_rel_diff"] < 1e-12, cmp
    assert abs(float(res.equity.iloc[-1]) - ind.final_equity) < 1e-12
    return prod, ind.trades, cmp


# Bar listeleri: (open, high, low, close). Sinyal "close > 100" (long) / "close < 100" (short).
FLAT = (99.0, 99.5, 98.5, 99.0)


def test_gap_through_stop_long() -> None:
    rows = [FLAT, (99, 101, 99, 101), (101, 101.5, 100.5, 101), (95, 96, 94, 95), FLAT]
    prod, _ind, _ = _both(_spec("long", entry=["close > 100"], exit_=["close < 90"]), rows)
    assert prod[0]["exit_reason"] == "gap_stop"
    assert prod[0]["exit_fill"] < 101.5 * 0.98  # açılıştan (stop'tan kötü) doldu


def test_same_bar_stop_and_target_stop_first() -> None:
    rows = [FLAT, (99, 101, 99, 101), (101, 105, 97, 100), FLAT]
    spec = _spec("long", entry=["close > 100.5"], tp={"type": "fixed_pct", "value": 0.02})
    prod, _ind, _ = _both(spec, rows)
    assert prod[0]["exit_reason"] == "stop"


def test_stop_inside_entry_bar() -> None:
    rows = [FLAT, (99, 101, 99, 101), (101, 101.2, 98.0, 98.5), FLAT, FLAT]
    prod, _ind, _ = _both(_spec("long", entry=["close > 100.5"]), rows)
    assert prod[0]["exit_reason"] == "stop" and prod[0]["entry_time"] == prod[0]["exit_time"]


def test_target_gap_fills_at_target_not_better_open() -> None:
    rows = [FLAT, (99, 101, 99, 101), (101, 101.5, 100.8, 101), (110, 111, 109, 110), FLAT]
    spec = _spec("long", entry=["close > 100.5"], tp={"type": "fixed_pct", "value": 0.03})
    prod, _ind, _ = _both(spec, rows)
    assert prod[0]["exit_reason"] == "hedef"
    assert prod[0]["exit_fill"] < 110  # daha iyi açılış kâr sayılmadı


def test_short_stop_and_gap() -> None:
    rows = [
        (101, 101.5, 100.5, 101),
        (101, 101, 99, 99),
        (99, 99.4, 98.6, 99),
        (99, 101.5, 98.9, 101),
    ]
    rows += [
        (101, 101.2, 100.8, 101),
        (101, 101, 99, 99),
        (99, 99.3, 98.7, 99),
        (106, 107, 105, 106),
    ]
    rows += [(106, 106.2, 105.8, 106)]
    spec = _spec("short", entry=["close < 100"], exit_=["close > 200"])
    prod, _ind, _ = _both(spec, rows)
    reasons = [t["exit_reason"] for t in prod]
    assert reasons[:2] == ["stop", "gap_stop"], reasons
    assert all(t["direction"] == "short" for t in prod)


def test_rule_exit_period_end_and_risk_sizing_r_target() -> None:
    rows = [FLAT, (99, 101, 99, 101), (101, 102, 100.5, 101.5), (101.5, 101.8, 101, 99.5)]
    rows += [(99.5, 99.8, 99, 99.2), (99.2, 101.5, 99, 101), (101, 101.6, 100.6, 101.2)]
    spec = _spec(
        "long",
        entry=["close > 100.5"],
        exit_=["close < 100"],
        sizing={"type": "risk_per_trade", "risk_pct": 0.01, "max_leverage": 2.0},
        tp={"type": "r_multiple", "value": 5.0},
    )
    prod, _ind, _ = _both(spec, rows)
    assert [t["exit_reason"] for t in prod] == ["kural", "donem_sonu"]


@pytest.mark.parametrize("direction", ["long", "short"])
def test_atr_stop_real_like_series_matches(direction) -> None:
    import numpy as np

    rng = np.random.default_rng(11)
    n = 600
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    o = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.002, n))
    hi = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    rows = list(zip(o, hi, lo, close, strict=True))
    op_in, op_out = (">", "<") if direction == "long" else ("<", ">")
    spec = _spec(
        direction,
        entry=[f"ema_10 {op_in} ema_30"],
        exit_=[f"ema_10 {op_out} ema_30"],
        stop={"type": "atr_initial", "value": 2.0, "atr_period": 14},
        tp={"type": "r_multiple", "value": 3.0},
        sizing={"type": "risk_per_trade", "risk_pct": 0.01, "max_leverage": 1.0},
    )
    spec["indicators"] = [{"name": "EMA", "period": 10}, {"name": "EMA", "period": 30}]
    _prod, _ind, cmp = _both(spec, rows)
    assert cmp["n_prod"] >= 10
