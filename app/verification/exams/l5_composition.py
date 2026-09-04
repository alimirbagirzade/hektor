"""L5 CompositionGate — KOMPOZİSYON sınavı (yeni formül = nihai anlama kanıtı).

Önerilen bir indikatör kompozisyonunu (StrategyIR) üç ardışık kapıdan geçirir:

  1. MathValidityGate — tüm kolonlar registry'den hesaplanabiliyor mu, periyotlar
     geçerli mi (>1), kurallar mantıklı aralıkta mı (ör. RSI ∈ [0,100]; 'rsi>100' reddi).
  2. NoveltyGate — en az 2 FARKLI gösterge tipi mi (tek-primitive değil), ve daha önce
     görülen adaylarla aynı imza değil mi (kopya reddi).
  3. BacktestGate — mevcut ``evaluate()`` ile maliyet-dahil + OOS verdict; yalnız
     verdict='pass' geçer.

Üçü de geçerse 'candidate' (ADAY); biri düşerse dürüst 'rejected' + neden.
Veri yoksa backtest sertifikalanamaz → aday DEĞİL ("test edilmeden hazır deme").
eval/exec YOK; her şey mevcut güvenli StrategyIR/evaluate üzerinden.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from app.trading.evaluator import evaluate
from app.trading.strategy_ir import StrategyIR, parse_rule

__all__ = ["CompositionGate", "CompositionResult", "GateResult"]

_REGISTRY = {"EMA", "SMA", "RSI", "ATR", "MACD", "ENTROPY", "PERMENTROPY"}
_OHLC = {"open", "high", "low", "close", "volume"}


@dataclass
class GateResult:
    gate: str  # "math" | "novelty" | "backtest"
    passed: bool
    details: list[str] = field(default_factory=list)


@dataclass
class CompositionResult:
    name: str
    candidate: bool
    verdict: str  # "candidate" | "rejected"
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _signature(ir: StrategyIR) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((ind.name.upper(), ind.period) for ind in ir.indicators))


def _parse_col(col: str) -> tuple[str, int | None]:
    if "_" not in col:
        return col.upper(), None
    name, _, per = col.rpartition("_")
    try:
        return name.upper(), int(per)
    except ValueError:
        return col.upper(), None


class CompositionGate:
    def __init__(self, evaluator: Callable[..., Any] | None = None) -> None:
        # evaluator enjeksiyonu: test/determinizm için (varsayılan gerçek evaluate).
        self._evaluate = evaluator or evaluate

    def evaluate_composition(
        self,
        ir: StrategyIR,
        df: pd.DataFrame | None = None,
        *,
        seen_signatures: set[tuple[tuple[str, int], ...]] | None = None,
        min_trades: int = 30,
    ) -> CompositionResult:
        gates: list[GateResult] = []

        math_gate = self._math_validity(ir)
        gates.append(math_gate)

        gates.append(self._novelty(ir, seen_signatures or set()))

        if df is None:
            gates.append(
                GateResult(
                    "backtest", False, ["veri yok — backtest sertifikalanamadı (aday değil)"]
                )
            )
        elif not math_gate.passed:
            gates.append(
                GateResult("backtest", False, ["matematik kapısı geçilmedi — backtest atlandı"])
            )
        else:
            gates.append(self._backtest(df, ir, min_trades))

        candidate = all(g.passed for g in gates)
        return CompositionResult(
            ir.name, candidate, "candidate" if candidate else "rejected", gates
        )

    # ---------------------------------------------------------------- gate 1
    def _math_validity(self, ir: StrategyIR) -> GateResult:
        problems: list[str] = []

        for ind in ir.indicators:
            if ind.name.upper() not in _REGISTRY:
                problems.append(f"{ind.name}: registry'de yok (hesaplanamaz)")
            if ind.period <= 1:
                problems.append(f"{ind.name}: periyot ≤ 1 ({ind.period}) — geçersiz")

        for col in ir.required_columns():
            if col in _OHLC:
                continue
            name, period = _parse_col(col)
            if name not in _REGISTRY:
                problems.append(f"Kolon {col!r}: registry-dışı/çözülemeyen gösterge")
            if period is not None and period <= 1:
                problems.append(f"Kolon {col!r}: periyot ≤ 1")

        for rule in [*ir.entry_rules, *ir.exit_rules]:
            problems.extend(_rule_bound_problems(rule))

        return GateResult("math", not problems, problems or ["matematik geçerli"])

    # ---------------------------------------------------------------- gate 2
    def _novelty(self, ir: StrategyIR, seen: set[tuple[tuple[str, int], ...]]) -> GateResult:
        problems: list[str] = []
        distinct_types = {ind.name.upper() for ind in ir.indicators}
        if len(distinct_types) < 2:
            problems.append(
                f"Yetersiz yenilik: {len(distinct_types)} farklı gösterge tipi "
                f"(en az 2 gerekir — tek-primitive 'yeni' sayılmaz)"
            )
        sig = _signature(ir)
        if sig in seen:
            problems.append("Kopya: bu kompozisyon imzası daha önce üretilmiş")
        return GateResult("novelty", not problems, problems or ["yeni kombinasyon"])

    # ---------------------------------------------------------------- gate 3
    def _backtest(self, df: pd.DataFrame, ir: StrategyIR, min_trades: int) -> GateResult:
        try:
            verdict = self._evaluate(df, ir, min_trades=min_trades)
        except Exception as exc:
            return GateResult("backtest", False, [f"backtest hatası: {exc}"])
        passed = getattr(verdict, "verdict", None) == "pass"
        reasons = list(getattr(verdict, "reasons", []))
        return GateResult(
            "backtest", passed, [f"verdict={getattr(verdict, 'verdict', '?')}", *reasons]
        )


# Sınırlı (bounded) göstergeler: kural sabiti bu aralık dışındaysa kural imkansızdır.
_BOUNDED_INDICATORS: dict[str, tuple[float, float]] = {
    "RSI": (0.0, 100.0),
    "ENTROPY": (0.0, 1.0),
    "PERMENTROPY": (0.0, 1.0),
}


def _rule_bound_problems(rule: str) -> list[str]:
    """Mantıksal olarak imkansız kuralları yakala (ör. RSI ∈ [0,100], ENTROPY ∈ [0,1] dışı)."""
    lhs, op, rhs = parse_rule(rule)
    if not re.match(r"^-?\d", rhs):
        return []  # rhs bir kolon; sayısal sınır kontrolü yok
    val = float(rhs)
    name, _ = _parse_col(lhs)
    bounds = _BOUNDED_INDICATORS.get(name)
    if bounds is None:
        return []
    lo, hi = bounds
    if val < lo or val > hi:
        return [f"{name} kuralı aralık dışı: {rule!r} ({name} ∈ [{lo:g},{hi:g}])"]
    if op in (">", ">=") and val >= hi:
        return [f"{name} kuralı asla doğru olamaz: {rule!r}"]
    if op in ("<", "<=") and val <= lo:
        return [f"{name} kuralı asla doğru olamaz: {rule!r}"]
    return []
