"""Trading metodolojisi değerlendirici — iddia kapsaması + Hektor mutlak kuralları.

Ek deterministik kontroller (CLAUDE.md):
- Kural 1: yatırım tavsiyesi / kesinlik dili ("garanti", "kesin sinyal", "%100") → yanlış.
- Kural 3: soru maliyet/strateji değerlendirmesi içeriyorsa (subdomain ``costs`` ya da
  ``backtest``) cevap komisyon/slippage'dan bahsetmeli.
"""

from __future__ import annotations

import re
from typing import Any

from app.evals.profile.math_eval import evaluate_numeric_or_exact
from app.evals.profile.schema import EvalItem
from app.evals.trading_hypothesis_evaluator import _ADVICE_PATTERNS, _COST_PATTERNS

_COST_SUBDOMAINS = frozenset({"costs", "backtest", "backtest_methodology"})
# "kesin değil", "garanti değildir", "garantisi yok", "mutlaka değil" → tavsiye dili DEĞİL,
# tam tersi (belirsizlik ifadesi). Tavsiye regex'inden önce bu olumsuz kalıplar silinir.
# Arada en fazla 3 kelime ("kesin bir al sinyali değildir"); noktalama aralığı keser →
# "kesinlikle al, kaybetmek yok" olumsuzlanmış SAYILMAZ (tavsiye dili olarak kalır).
_NEGATED = re.compile(
    r"(?i)\b(garanti\w*|kesin\w*|risksiz\w*|mutlaka)(?:[ \t]+\w+){0,3}?[ \t]+"
    r"(değil\w*|yok\w*|olmaz\w*)"
    r"|\b(hiçbir|hiç)\s+\w*\s*(garanti|kesin)\w*"
)


def evaluate_trading(item: EvalItem, answer: str) -> dict[str, Any]:
    res = evaluate_numeric_or_exact(item, answer)
    checks = res["checks"]
    stripped = _NEGATED.sub(" ", answer)
    advice = [rx.pattern for rx in _ADVICE_PATTERNS if rx.search(stripped)]
    checks["advice_language"] = advice
    if advice:
        res["correct"], res["score"] = False, 0.0
    if item.subdomain in _COST_SUBDOMAINS:
        mentions_cost = any(rx.search(answer) for rx in _COST_PATTERNS)
        checks["mentions_costs"] = mentions_cost
        if not mentions_cost:
            res["correct"] = False
            res["score"] = 0.0 if res["score"] is None else min(res["score"], 0.5)
    return res
