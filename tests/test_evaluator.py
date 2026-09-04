"""Strateji evaluatör ve overfit check birim testleri — çevrimdışı."""

from __future__ import annotations

import pandas as pd
import pytest

from app.trading.evaluator import evaluate
from app.trading.market_data_loader import generate_synthetic_ohlcv
from app.trading.overfit_checks import OverfitReport, in_out_of_sample, static_checks
from app.trading.strategy_ir import CostSpec, IndicatorSpec, RiskSpec, StrategyIR


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return generate_synthetic_ohlcv(n=1000, seed=7)


@pytest.fixture
def trend_ir() -> StrategyIR:
    return StrategyIR(
        name="trend_test",
        market="XAUUSD",
        timeframe="15m",
        indicators=[
            IndicatorSpec(name="EMA", period=20),
            IndicatorSpec(name="EMA", period=50),
            IndicatorSpec(name="RSI", period=14),
            IndicatorSpec(name="ATR", period=14),
        ],
        entry_rules=["ema_20 > ema_50", "rsi_14 > 55"],
        exit_rules=["ema_20 < ema_50", "rsi_14 < 45"],
        risk=RiskSpec(stop_loss="2 * ATR"),
        costs=CostSpec(),
    )


# ---------- static_checks ----------
def test_static_checks_few_trades() -> None:
    from app.trading.backtester import BacktestMetrics

    m = BacktestMetrics(
        n_trades=5,
        total_return_pct=10.0,
        sharpe=1.0,
        sortino=1.0,
        max_drawdown_pct=-10.0,
        profit_factor=1.5,
        win_rate_pct=55.0,
    )
    warnings = static_checks(m, min_trades=30)
    assert any("az işlem" in w for w in warnings)


def test_static_checks_suspicious_pf() -> None:
    from app.trading.backtester import BacktestMetrics

    m = BacktestMetrics(
        n_trades=50,
        total_return_pct=100.0,
        sharpe=2.0,
        sortino=2.0,
        max_drawdown_pct=-5.0,
        profit_factor=8.0,
        win_rate_pct=80.0,
    )
    warnings = static_checks(m, min_trades=30)
    assert any("profit factor" in w for w in warnings)


def test_static_checks_extreme_drawdown() -> None:
    from app.trading.backtester import BacktestMetrics

    m = BacktestMetrics(
        n_trades=100,
        total_return_pct=50.0,
        sharpe=1.0,
        sortino=1.0,
        max_drawdown_pct=-75.0,
        profit_factor=1.5,
        win_rate_pct=55.0,
    )
    warnings = static_checks(m, min_trades=30)
    assert any("drawdown" in w for w in warnings)


def test_static_checks_clean() -> None:
    from app.trading.backtester import BacktestMetrics

    m = BacktestMetrics(
        n_trades=60,
        total_return_pct=20.0,
        sharpe=1.2,
        sortino=1.5,
        max_drawdown_pct=-15.0,
        profit_factor=1.6,
        win_rate_pct=55.0,
    )
    assert static_checks(m) == []


# ---------- in_out_of_sample ----------
def test_oos_split_metrics(sample_df: pd.DataFrame, trend_ir: StrategyIR) -> None:
    report = in_out_of_sample(sample_df, trend_ir, split=0.7)
    assert report.in_sample is not None
    assert report.out_sample is not None


def test_oos_degradation_computed(sample_df: pd.DataFrame, trend_ir: StrategyIR) -> None:
    report = in_out_of_sample(sample_df, trend_ir)
    # degradation_pct sadece her ikisi de sıfır olmadığında None döner
    # değer float veya None olabilir — tip doğru olmalı
    deg = report.degradation_pct
    assert deg is None or isinstance(deg, float)


# ---------- evaluate ----------
def test_evaluate_returns_valid_verdict(sample_df: pd.DataFrame, trend_ir: StrategyIR) -> None:
    v = evaluate(sample_df, trend_ir)
    assert v.verdict in {"pass", "fail", "inconclusive"}
    assert isinstance(v.reasons, list)


def test_evaluate_fail_on_tiny_data(trend_ir: StrategyIR) -> None:
    tiny = generate_synthetic_ohlcv(n=100, seed=99)
    v = evaluate(tiny, trend_ir, min_trades=30)
    # Az veri → yeterli işlem olmaz → fail
    assert v.verdict in {"fail", "inconclusive"}


def test_evaluate_overfit_report_attached(sample_df: pd.DataFrame, trend_ir: StrategyIR) -> None:
    v = evaluate(sample_df, trend_ir)
    assert isinstance(v.overfit, OverfitReport)
    assert v.overfit.out_sample is not None
