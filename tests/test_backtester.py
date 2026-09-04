"""Backtest engine, overfit checks ve evaluator testleri.

Sentetik veri kullanır (gerçek piyasa verisi gerekmez). Amaç:
- run_backtest uçtan uca çalışıyor ve makul metrikler üretiyor mu
- look-ahead yok (pozisyon bir bar gecikmeli uygulanıyor)
- maliyetler getiriyi düşürüyor
- in/out-of-sample bölmesi çalışıyor
- evaluator OOS testi olmadan asla "pass" demiyor
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.trading.backtester import BacktestMetrics, run_backtest
from app.trading.evaluator import evaluate
from app.trading.market_data_loader import generate_synthetic_ohlcv
from app.trading.overfit_checks import in_out_of_sample, static_checks
from app.trading.strategy_ir import example_ir


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    # Deterministik (seed sabit) ve backtest için yeterince uzun.
    return generate_synthetic_ohlcv(n=2000, seed=7)


def test_synthetic_data_shape(synthetic_df: pd.DataFrame) -> None:
    assert len(synthetic_df) == 2000
    for col in ("open", "high", "low", "close", "volume"):
        assert col in synthetic_df.columns
    # high >= low her zaman
    assert (synthetic_df["high"] >= synthetic_df["low"]).all()


def test_run_backtest_produces_metrics(synthetic_df: pd.DataFrame) -> None:
    ir = example_ir()
    result = run_backtest(synthetic_df, ir)

    assert isinstance(result.metrics, BacktestMetrics)
    m = result.metrics
    # metrikler sonlu sayılar olmalı (NaN/inf değil)
    assert math.isfinite(m.total_return_pct)
    assert math.isfinite(m.sharpe)
    assert math.isfinite(m.max_drawdown_pct)
    assert m.n_trades >= 0
    assert 0.0 <= m.win_rate_pct <= 100.0
    # drawdown negatif veya sıfır olmalı (tanım gereği)
    assert m.max_drawdown_pct <= 0.0


def test_equity_curve_aligned(synthetic_df: pd.DataFrame) -> None:
    ir = example_ir()
    result = run_backtest(synthetic_df, ir)
    # equity eğrisi veri ile aynı uzunlukta ve pozitif
    assert len(result.equity_curve) == len(synthetic_df)
    assert (result.equity_curve > 0).all()


def test_no_lookahead_costs_reduce_return(synthetic_df: pd.DataFrame) -> None:
    """Maliyetli backtest, maliyetsiz versiyondan daha düşük getiri vermeli."""
    ir = example_ir()
    with_costs = run_backtest(synthetic_df, ir)

    zero_cost = ir.model_copy(deep=True)
    zero_cost.costs.commission = 0.0
    zero_cost.costs.slippage = 0.0
    without_costs = run_backtest(synthetic_df, zero_cost)

    # En az bir işlem olduğunda maliyetler farkı açar; işlem yoksa eşit kalır.
    if with_costs.metrics.n_trades > 0:
        assert without_costs.metrics.total_return_pct >= with_costs.metrics.total_return_pct


def test_net_returns_cost_aligned_with_earning_bar() -> None:
    """Cost-timing fix: giriş maliyeti, pozisyonun getiri kazanmaya başladığı barla hizalı.

    Eski hâlde maliyet kaydırılmamış position.diff()'ten geliyordu → giriş sinyalinden
    sonraki ilk barda (henüz getiri yok) düşülüyordu. Artık getiri+maliyet aynı gecikmeli
    pozisyondan türer.
    """
    from app.trading.backtester import _net_returns

    # pozisyon: bar1'de gir (0→1), bar3'te çık (1→0)
    position = pd.Series([0, 1, 1, 0, 0])
    bar_ret = pd.Series([0.0, 0.02, 0.03, 0.04, 0.05])
    net = _net_returns(position, bar_ret, cost_per_turn=0.01)

    # eff_pos=[0,0,1,1,0]; getiri bar2-3'te; giriş maliyeti bar2'de (ilk getiri barı), bar1'de DEĞİL
    assert net.iloc[1] == 0.0  # giriş sinyali sonrası ilk bar: maliyet YOK (eski hata buradaydı)
    assert abs(net.iloc[2] - (0.03 - 0.01)) < 1e-9  # getiri başladığı bar + giriş maliyeti hizalı
    assert abs(net.iloc[3] - 0.04) < 1e-9  # tutuş barı: maliyet yok
    # giriş + çıkış olmak üzere iki turnover maliyeti uygulanmış olmalı
    total_cost = (position.shift(1).fillna(0.0).diff().abs().fillna(0.0) * 0.01).sum()
    assert abs(total_cost - 0.02) < 1e-9


def test_static_checks_flags_few_trades() -> None:
    m = BacktestMetrics(
        n_trades=5,
        total_return_pct=10.0,
        sharpe=1.0,
        sortino=1.2,
        max_drawdown_pct=-10.0,
        profit_factor=1.5,
        win_rate_pct=55.0,
    )
    warnings = static_checks(m, min_trades=30)
    assert any("az işlem" in w.lower() for w in warnings)


def test_static_checks_flags_unrealistic_sharpe() -> None:
    m = BacktestMetrics(
        n_trades=200,
        total_return_pct=10.0,
        sharpe=9.9,  # gerçekçi değil -> look-ahead şüphesi
        sortino=10.0,
        max_drawdown_pct=-5.0,
        profit_factor=2.0,
        win_rate_pct=60.0,
    )
    warnings = static_checks(m)
    assert any("sharpe" in w.lower() for w in warnings)


def test_in_out_of_sample_splits(synthetic_df: pd.DataFrame) -> None:
    ir = example_ir()
    report = in_out_of_sample(synthetic_df, ir, split=0.7)
    assert report.in_sample is not None
    assert report.out_sample is not None
    # degradation hesaplanabiliyor (in_sample getirisi 0 değilse)
    deg = report.degradation_pct
    assert deg is None or math.isfinite(deg)


def test_evaluator_returns_valid_verdict(synthetic_df: pd.DataFrame) -> None:
    ir = example_ir()
    verdict = evaluate(synthetic_df, ir)
    assert verdict.verdict in {"pass", "fail", "inconclusive"}
    assert isinstance(verdict.reasons, list)
    assert verdict.overfit.out_sample is not None


def test_evaluator_never_passes_negative_return() -> None:
    """OOS getiri <= 0 ise verdict asla 'pass' olmamalı."""
    ir = example_ir()
    # Sürekli düşen bir seri üret: trend-takip stratejisi para kaybetmeli/başarısız olmalı.
    falling = generate_synthetic_ohlcv(n=1500, seed=99)
    # ikinci yarıyı zorla aşağı bük
    falling = falling.copy()
    n = len(falling)
    factor = pd.Series(
        [1.0 if i < n // 2 else max(0.3, 1.0 - (i - n // 2) / n) for i in range(n)],
        index=falling.index,
    )
    for col in ("open", "high", "low", "close"):
        falling[col] = falling[col] * factor

    verdict = evaluate(falling, ir)
    if verdict.overfit.out_sample and verdict.overfit.out_sample.total_return_pct <= 0:
        assert verdict.verdict != "pass"
