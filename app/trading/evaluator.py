"""Strategy evaluator: turn metrics + overfit checks into a verdict.

Verdict policy (conservative by design):
- 'fail'         : any hard red flag (extreme drawdown, OOS negative, too few trades)
- 'inconclusive' : passes hard flags but has soft warnings or weak edge
- 'pass'         : positive OOS, acceptable drawdown, enough trades, no red flags

Nothing is ever declared 'successful' without an out-of-sample test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.trading.overfit_checks import OverfitReport, in_out_of_sample
from app.trading.strategy_ir import StrategyIR


@dataclass
class Verdict:
    verdict: str  # pass | fail | inconclusive
    reasons: list[str]
    overfit: OverfitReport


def evaluate(df: pd.DataFrame, ir: StrategyIR, *, min_trades: int = 30) -> Verdict:
    # min_trades hem hard-fail hem 'çok az işlem' uyarısı için tutarlı kullanılmalı.
    report = in_out_of_sample(df, ir, min_trades=min_trades)
    reasons: list[str] = list(report.warnings)

    oos = report.out_sample
    if oos is None:
        # Yetersiz veri (IS/OOS bölünemedi) → karar verilemez; çökme yerine dürüst sonuç.
        reasons.append("Örneklem-dışı bölme için yeterli veri yok.")
        return Verdict("inconclusive", reasons, report)

    hard_fail = oos.n_trades < min_trades or oos.max_drawdown_pct < -50 or oos.total_return_pct <= 0

    if hard_fail:
        if oos.total_return_pct <= 0:
            reasons.append("Örneklem-dışı getiri pozitif değil.")
        return Verdict("fail", reasons, report)

    if report.warnings:
        return Verdict("inconclusive", reasons, report)

    if oos.sharpe < 0.5:
        reasons.append("Örneklem-dışı Sharpe zayıf (<0.5).")
        return Verdict("inconclusive", reasons, report)

    reasons.append("Örneklem-dışı pozitif, drawdown kabul edilebilir, yeterli işlem.")
    return Verdict("pass", reasons, report)
