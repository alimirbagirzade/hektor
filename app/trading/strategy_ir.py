"""Strategy Intermediate Representation (IR).

A JSON-serializable description of a strategy: indicators, entry/exit rules,
risk, and cost model. Rules are simple comparison expressions over computed
indicator columns (e.g. "ema_20 > ema_50", "rsi_14 > 55").

The IR is intentionally restrictive and declarative so it is safe to evaluate
(no arbitrary code execution) and easy to translate to Pine/MQL5 later.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_RULE_RE = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(<|<=|>|>=|==|!=)\s*([a-zA-Z_][a-zA-Z0-9_]*|-?\d+(?:\.\d+)?)\s*$"
)


class IndicatorSpec(BaseModel):
    name: str
    period: int = 14

    @property
    def column(self) -> str:
        return f"{self.name.lower()}_{self.period}"


class RiskSpec(BaseModel):
    stop_loss: str | None = "2 * ATR"
    take_profit: str | None = None
    position_size: str = "fixed_fractional"


class CostSpec(BaseModel):
    commission: float = 0.0005
    slippage: float = 0.0005


class StrategyIR(BaseModel):
    name: str
    market: str = "XAUUSD"
    timeframe: str = "15m"
    indicators: list[IndicatorSpec] = Field(default_factory=list)
    entry_rules: list[str] = Field(default_factory=list)
    exit_rules: list[str] = Field(default_factory=list)
    risk: RiskSpec = Field(default_factory=RiskSpec)
    costs: CostSpec = Field(default_factory=CostSpec)

    @field_validator("entry_rules", "exit_rules")
    @classmethod
    def _validate_rules(cls, rules: list[str]) -> list[str]:
        for r in rules:
            if not _RULE_RE.match(r):
                raise ValueError(
                    f"Geçersiz kural: {r!r}. Biçim: '<col> <op> <col|sayı>' örn. 'ema_20 > ema_50'"
                )
        return rules

    def required_columns(self) -> set[str]:
        cols = {ind.column for ind in self.indicators}
        for rule in [*self.entry_rules, *self.exit_rules]:
            m = _RULE_RE.match(rule)
            if m:
                lhs, _, rhs = m.groups()
                cols.add(lhs)
                if not re.match(r"^-?\d", rhs):
                    cols.add(rhs)
        return cols

    def to_pine(self) -> str:
        """StrategyIR → TradingView Pine Script v5 (taslak)."""
        # commission_value in Pine expects percent; strip trailing zeros for readability
        commission_pct = self.costs.commission * 100
        commission_str = f"{commission_pct:.4f}".rstrip("0").rstrip(".")
        lines: list[str] = [
            "//@version=5",
            (
                f'strategy("{self.name}", overlay=true,'
                f" commission_type=strategy.commission.percent,"
                f" commission_value={commission_str})"
            ),
            "",
        ]
        # Indikatör tanımları
        ind_map: dict[str, str] = {}
        for ind in self.indicators:
            col = ind.column
            n = ind.name.upper()
            if n == "EMA":
                lines.append(f"{col} = ta.ema(close, {ind.period})")
            elif n == "SMA":
                lines.append(f"{col} = ta.sma(close, {ind.period})")
            elif n == "RSI":
                lines.append(f"{col} = ta.rsi(close, {ind.period})")
            elif n == "ATR":
                lines.append(f"{col} = ta.atr({ind.period})")
            elif n == "MACD":
                lines.append(f"{col}_line = ta.macd(close, 12, 26, 9).macd")
            elif n in ("BB", "BOLLINGER"):
                lines.append(
                    f"[{col}_upper, {col}_mid, {col}_lower] = ta.bb(close, {ind.period}, 2)"
                )
            elif n in ("STOCH", "STOCHASTIC"):
                # IndicatorSpec yalnız `period` taşır; %K/%D yumuşatma Pine varsayılanı 3.
                lines.append(f"{col}_k = ta.sma(ta.stoch(close, high, low, {ind.period}), 3)")
                lines.append(f"{col}_d = ta.sma({col}_k, 3)")
            elif n == "VWAP":
                lines.append(f"{col} = ta.vwap(hlc3)")
            elif n in ("SUPERTREND", "ST"):
                # IndicatorSpec çarpan taşımaz; Pine varsayılanı 3.0 sabit.
                lines.append(f"[{col}, {col}_dir] = ta.supertrend(3.0, {ind.period})")
            else:
                lines.append(f"// {col} = ???  /* {n} desteklenmiyor */")
            ind_map[col] = col
        lines.append("")

        def _rule_to_pine(rule: str) -> str:
            lhs, op, rhs = parse_rule(rule)
            return f"{lhs} {op} {rhs}"

        entry_cond = " and ".join(_rule_to_pine(r) for r in self.entry_rules) or "false"
        exit_cond = " and ".join(_rule_to_pine(r) for r in self.exit_rules) or "false"

        lines += [
            f"entryCondition = {entry_cond}",
            f"exitCondition  = {exit_cond}",
            "",
            "if entryCondition",
            '    strategy.entry("Long", strategy.long)',
            "if exitCondition",
            '    strategy.close("Long")',
        ]
        return "\n".join(lines)


def parse_rule(rule: str) -> tuple[str, str, str]:
    m = _RULE_RE.match(rule)
    if not m:
        raise ValueError(f"Geçersiz kural: {rule!r}")
    return m.group(1), m.group(2), m.group(3)


def example_ir() -> StrategyIR:
    return StrategyIR(
        name="ema_rsi_trend_filter_v1",
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
        risk=RiskSpec(stop_loss="2 * ATR", position_size="fixed_fractional"),
        costs=CostSpec(commission=0.0005, slippage=0.0005),
    )
