"""Muhakeme (neden-sonuç, belirsizlik) değerlendirici — deterministik iddia kapsaması.

Deterministik kısım: ``required_claims`` kapsaması + ``forbidden_claims`` ihlali + (varsa)
kabul edilen cevap exact match. Açıklama kalitesi gibi deterministik ölçülemeyen boyutlar
için opsiyonel LLM judge (``judge.py``) ayrı alan olarak saklanır; bu modül onu çağırmaz.
"""

from __future__ import annotations

import re
from typing import Any

from app.evals.profile.math_eval import evaluate_numeric_or_exact
from app.evals.profile.schema import EvalItem

_UNCERTAINTY = re.compile(
    r"(?i)\b(belirsiz|olası|muhtemel|varsayım|sınırlı|kesin değil"
    r"|uncertain|likely|assum|may|might)\w*"
)


def evaluate_reasoning(item: EvalItem, answer: str) -> dict[str, Any]:
    res = evaluate_numeric_or_exact(item, answer)
    res["checks"]["expresses_uncertainty"] = bool(_UNCERTAINTY.search(answer))
    cov = res["checks"].get("claim_coverage")
    if cov is not None and res["correct"] is not None:
        res["score"] = float(cov) if not res["checks"].get("forbidden_claims_hit") else 0.0
    return res
