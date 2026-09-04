"""Generate candidate Strategy IRs from hypotheses.

MVP: a deterministic, template-driven generator that maps common hypothesis
keywords (trend, momentum, mean-reversion, volatility) to a sensible starter
IR. An LLM-driven generator can be added later via the
``strategy_hypothesis`` prompt — but every generated strategy MUST still go
through the backtester + evaluator before it is trusted.
"""

from __future__ import annotations

from app.config import get_settings
from app.trading.strategy_ir import CostSpec, IndicatorSpec, RiskSpec, StrategyIR


def _trend_template(name: str, market: str, tf: str) -> StrategyIR:
    return StrategyIR(
        name=name,
        market=market,
        timeframe=tf,
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


def _mean_reversion_template(name: str, market: str, tf: str) -> StrategyIR:
    return StrategyIR(
        name=name,
        market=market,
        timeframe=tf,
        indicators=[
            IndicatorSpec(name="RSI", period=14),
            IndicatorSpec(name="EMA", period=50),
            IndicatorSpec(name="ATR", period=14),
        ],
        entry_rules=["rsi_14 < 35", "ema_50 > 0"],
        exit_rules=["rsi_14 > 55"],
        risk=RiskSpec(stop_loss="2 * ATR"),
        costs=CostSpec(),
    )


def generate_from_hypothesis(
    hypothesis: str,
    *,
    name: str | None = None,
    market: str | None = None,
    timeframe: str | None = None,
) -> StrategyIR:
    settings = get_settings()
    market = market or settings.default_market
    timeframe = timeframe or settings.default_timeframe
    h = hypothesis.lower()
    base_name = name or "generated_v1"

    if any(k in h for k in ("mean revers", "ortalamaya dön", "aşırı sat", "oversold")):
        return _mean_reversion_template(base_name, market, timeframe)
    # default: trend / momentum
    return _trend_template(base_name, market, timeframe)
