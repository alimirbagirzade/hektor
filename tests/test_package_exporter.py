"""Achilles Package exporter testleri (offline, Ollama gerektirmez)."""

from __future__ import annotations

import json

import pytest

from app.trading.backtester import _compute_columns, _position_series
from app.trading.package_exporter import (
    ACHPKG_VERSION,
    AchillesPackage,
    _ir_to_python,
    export_strategy,
)
from app.trading.strategy_ir import CostSpec, IndicatorSpec, StrategyIR, example_ir


@pytest.fixture()
def simple_ir() -> StrategyIR:
    return StrategyIR(
        name="test_strategy",
        market="BTCUSD",
        timeframe="1H",
        indicators=[
            IndicatorSpec(name="EMA", period=20),
            IndicatorSpec(name="RSI", period=14),
        ],
        entry_rules=["ema_20 > 100", "rsi_14 > 55"],
        exit_rules=["rsi_14 < 45"],
        costs=CostSpec(commission=0.001),
    )


def test_export_returns_package(simple_ir: StrategyIR) -> None:
    pkg = export_strategy(simple_ir)
    assert isinstance(pkg, AchillesPackage)
    assert pkg.name == "test_strategy"
    assert pkg.package_type == "strategy"
    assert pkg.source == "achilles_research"


def test_package_has_pine_code(simple_ir: StrategyIR) -> None:
    pkg = export_strategy(simple_ir)
    assert "//@version=5" in pkg.pine_code
    assert "strategy(" in pkg.pine_code
    assert "ema_20" in pkg.pine_code
    assert "rsi_14" in pkg.pine_code


def test_package_has_python_code(simple_ir: StrategyIR) -> None:
    pkg = export_strategy(simple_ir)
    assert "compute_signals" in pkg.python_code
    assert "entry_signal" in pkg.python_code
    assert "exit_signal" in pkg.python_code
    assert "shift(1)" in pkg.python_code  # look-ahead bias önlemi


def test_package_json_structure(simple_ir: StrategyIR) -> None:
    pkg = export_strategy(simple_ir)
    data = json.loads(pkg.to_json())
    assert data["achilles_package_version"] == ACHPKG_VERSION
    assert data["name"] == "test_strategy"
    assert data["type"] == "strategy"
    assert "pine" in data["code"]
    assert "python" in data["code"]
    assert data["code"]["pine"] == pkg.pine_code
    assert data["code"]["python"] == pkg.python_code


def test_export_with_backtest_metadata(simple_ir: StrategyIR) -> None:
    pkg = export_strategy(
        simple_ir,
        backtest_verdict="pass",
        backtest_metrics={"sharpe": 2.1, "total_return_pct": 150.0},
    )
    assert pkg.backtest_verdict == "pass"
    assert pkg.backtest_metrics["sharpe"] == 2.1
    data = pkg.to_dict()
    assert data["backtest_verdict"] == "pass"
    assert data["backtest_metrics"]["total_return_pct"] == 150.0


def test_python_code_has_all_indicators(simple_ir: StrategyIR) -> None:
    code = _ir_to_python(simple_ir)
    assert "ewm(span=20" in code  # EMA 20
    assert "rsi_14" in code  # RSI 14
    # RSI Wilder ewm ile üretilmeli (backtest ile aynı; SMA rolling değil).
    assert "ewm(alpha=1/14" in code
    assert "PACKAGE_NAME" in code


def test_python_code_all_indicator_types() -> None:
    """SMA, ATR, MACD, Bollinger kodlarını test et."""
    ir = StrategyIR(
        name="all_indicators",
        indicators=[
            IndicatorSpec(name="SMA", period=50),
            IndicatorSpec(name="ATR", period=14),
            IndicatorSpec(name="MACD", period=12),
            IndicatorSpec(name="BB", period=20),
        ],
        entry_rules=["sma_50 > 100"],
        exit_rules=["sma_50 < 100"],
    )
    code = _ir_to_python(ir)
    assert "rolling(50)" in code  # SMA
    assert "atr" in code.lower()  # ATR (tr satırı)
    assert "_e12" in code  # MACD
    assert "_mid" in code  # Bollinger


def test_package_save_and_load(tmp_path, simple_ir: StrategyIR) -> None:
    pkg = export_strategy(simple_ir)
    out = tmp_path / "test.achpkg"
    pkg.save(out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["name"] == "test_strategy"
    assert loaded["achilles_package_version"] == ACHPKG_VERSION


def test_example_ir_exports_cleanly() -> None:
    """example_ir() hatasız export edilmeli."""
    ir = example_ir()
    pkg = export_strategy(ir)
    assert len(pkg.pine_code) > 50
    assert len(pkg.python_code) > 100
    assert pkg.name == ir.name


# --- Dışa aktarım ↔ backtest tutarlılığı (üretilen kod GERÇEKTEN çalıştırılır) ---
#
# Eski testler üretilen kodu yalnız metin olarak kontrol ediyordu; bu yüzden MACD/BB
# kolon adlarının backtest ile sapması ve desteklenmeyen göstergelerin sessizce
# tanımsız kolona atıf yapması fark edilmemişti.


def _synthetic_ohlcv(n: int = 300):
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    px = pd.Series(100 + np.cumsum(np.random.default_rng(7).normal(0, 0.5, n)), index=idx)
    return pd.DataFrame(
        {"open": px, "high": px + 1.0, "low": px - 1.0, "close": px, "volume": 1000.0},
        index=idx,
    )


def _run_generated(code: str, df):
    ns: dict = {}
    # Üretilen modülü GERÇEKTEN çalıştır: bu, dışa aktarımın backtest ile aynı
    # kolonları hesapladığını doğrulayan tek yol (metin kontrolü bunu kaçırmıştı).
    exec(compile(code, "<generated>", "exec"), ns)
    return ns["compute_signals"](df.copy())


@pytest.mark.parametrize(
    ("ind_name", "period", "column"),
    [("EMA", 20, "ema_20"), ("SMA", 50, "sma_50"), ("RSI", 14, "rsi_14"), ("MACD", 12, "macd_12")],
)
def test_generated_python_runs_for_supported_indicators(ind_name, period, column) -> None:
    """Üretilen modül, kuralların atıf yaptığı kolonu gerçekten hesaplamalı."""
    ir = StrategyIR(
        name=f"{column}_smoke",
        indicators=[IndicatorSpec(name=ind_name, period=period)],
        entry_rules=[f"{column} > 0"],
        exit_rules=[f"{column} < 0"],
    )
    out = _run_generated(_ir_to_python(ir), _synthetic_ohlcv())
    assert column in out.columns
    assert set(out["entry_signal"].unique()) <= {0, 1}


def test_generated_python_bollinger_uses_canonical_mid_column() -> None:
    """BB kanonik kolonu = ORTA bant (compute_indicator ile aynı)."""
    ir = StrategyIR(
        name="bb_smoke",
        indicators=[IndicatorSpec(name="BB", period=20)],
        entry_rules=["close > bb_20"],
        exit_rules=["close < bb_20"],
    )
    out = _run_generated(_ir_to_python(ir), _synthetic_ohlcv())
    assert {"bb_20", "bb_20_upper", "bb_20_lower"} <= set(out.columns)


def test_generated_python_matches_backtester_position() -> None:
    """Dışa aktarılan sinyaller backtest'te doğrulanan pozisyonla aynı olmalı."""
    ir = StrategyIR(
        name="parity",
        indicators=[IndicatorSpec(name="EMA", period=10), IndicatorSpec(name="EMA", period=30)],
        entry_rules=["ema_10 > ema_30"],
        exit_rules=["ema_10 < ema_30"],
    )
    df = _synthetic_ohlcv()
    out = _run_generated(_ir_to_python(ir), df)

    enriched = _compute_columns(df, ir)
    entry = (enriched["ema_10"] > enriched["ema_30"]).shift(1).fillna(False).astype(int)
    assert out["entry_signal"].equals(entry.rename("entry_signal"))
    # Backtest pozisyonu da aynı sinyalden türer (blok mantığı).
    assert int(_position_series(enriched, ir).sum()) > 0


def test_export_rejects_rule_column_the_code_cannot_compute() -> None:
    """Desteklenmeyen gösterge → sessiz bozuk paket DEĞİL, açık hata."""
    ir = StrategyIR(
        name="entropy_v1",
        indicators=[IndicatorSpec(name="ENTROPY", period=14)],
        entry_rules=["entropy_14 < 50"],
        exit_rules=["entropy_14 > 80"],
    )
    with pytest.raises(ValueError, match="entropy_14"):
        _ir_to_python(ir)
    with pytest.raises(ValueError, match="entropy_14"):
        ir.to_pine()
    with pytest.raises(ValueError, match="entropy_14"):
        export_strategy(ir)


def test_pine_defines_every_rule_column() -> None:
    """MACD/BB Pine çıktısı kuralların atıf yaptığı değişkeni tanımlamalı."""
    ir = StrategyIR(
        name="pine_cols",
        indicators=[IndicatorSpec(name="MACD", period=12), IndicatorSpec(name="BB", period=20)],
        entry_rules=["macd_12 > 0", "close > bb_20"],
        exit_rules=["macd_12 < 0"],
    )
    pine = ir.to_pine()
    assert "[macd_12, macd_12_signal, macd_12_hist] = ta.macd" in pine
    assert "[bb_20_upper, bb_20, bb_20_lower] = ta.bb" in pine
    assert "???" not in pine


def test_pine_escapes_quotes_in_strategy_name() -> None:
    ir = StrategyIR(
        name='ema "breakout" v1',
        indicators=[IndicatorSpec(name="EMA", period=20)],
        entry_rules=["ema_20 > 100"],
        exit_rules=["ema_20 < 100"],
    )
    assert 'strategy("ema \\"breakout\\" v1"' in ir.to_pine()
