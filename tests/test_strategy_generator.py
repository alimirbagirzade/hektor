"""Strateji üreteci birim testleri — çevrimdışı."""

from __future__ import annotations

from app.trading.strategy_generator import generate_from_hypothesis
from app.trading.strategy_ir import StrategyIR


def test_returns_strategy_ir() -> None:
    ir = generate_from_hypothesis("trend following with EMA crossover")
    assert isinstance(ir, StrategyIR)


def test_trend_hypothesis_gives_ema_indicators() -> None:
    ir = generate_from_hypothesis("trend following momentum strategy")
    names = {ind.name.upper() for ind in ir.indicators}
    assert "EMA" in names


def test_mean_reversion_keyword_english() -> None:
    ir = generate_from_hypothesis("mean reversion RSI oversold bounce")
    # mean-reversion şablonu RSI < 35 içermeli
    assert any("rsi" in r and "<" in r for r in ir.entry_rules)


def test_mean_reversion_keyword_turkish() -> None:
    ir = generate_from_hypothesis("ortalamaya dön stratejisi")
    assert any("rsi" in r and "<" in r for r in ir.entry_rules)


def test_oversold_keyword() -> None:
    ir = generate_from_hypothesis("buy when oversold")
    assert any("rsi" in r for r in ir.entry_rules)


def test_custom_name_preserved() -> None:
    ir = generate_from_hypothesis("trend hypothesis", name="my_strat")
    assert ir.name == "my_strat"


def test_default_name() -> None:
    ir = generate_from_hypothesis("some hypothesis")
    assert ir.name  # boş olmamalı


def test_custom_market_and_timeframe() -> None:
    ir = generate_from_hypothesis("trend", market="BTCUSD", timeframe="1h")
    assert ir.market == "BTCUSD"
    assert ir.timeframe == "1h"


def test_entry_exit_rules_valid() -> None:
    ir = generate_from_hypothesis("momentum trend entry")
    # Tüm kurallar StrategyIR validator'ından geçmeli (zaten nesne oluştu)
    assert len(ir.entry_rules) > 0
    assert len(ir.exit_rules) > 0


def test_costs_included() -> None:
    ir = generate_from_hypothesis("any hypothesis")
    assert ir.costs.commission > 0
    assert ir.costs.slippage > 0


def test_unknown_hypothesis_defaults_to_trend() -> None:
    ir = generate_from_hypothesis("quantum neural network alpha signal X7")
    ema_indicators = [i for i in ir.indicators if i.name.upper() == "EMA"]
    assert len(ema_indicators) >= 2
