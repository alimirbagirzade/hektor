"""Achilles Package exporter testleri (offline, Ollama gerektirmez)."""

from __future__ import annotations

import json

import pytest

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
