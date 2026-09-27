"""İstatistik değerlendirici — sayısal tolerans + zorunlu kavram iddiaları.

Sayısal cevap doğru olsa bile ``required_claims`` (ör. "örneklem küçük", "bağımsızlık
varsayımı") eksikse kısmi skor alır; doğru sayılması için hem sayı hem iddialar tam olmalı.
"""

from __future__ import annotations

from typing import Any

from app.evals.profile.math_eval import evaluate_numeric_or_exact
from app.evals.profile.schema import EvalItem


def evaluate_statistics(item: EvalItem, answer: str) -> dict[str, Any]:
    res = evaluate_numeric_or_exact(item, answer)
    checks = res["checks"]
    numeric_ok = checks.get("numeric_ok", checks.get("exact_ok"))
    cov = checks.get("claim_coverage")
    if numeric_ok is not None and cov is not None:
        res["score"] = 0.5 * float(numeric_ok) + 0.5 * float(cov)
        res["correct"] = bool(numeric_ok) and cov == 1.0 and not checks.get("forbidden_claims_hit")
    return res
