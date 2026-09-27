"""Cevap değerlendirme dağıtıcısı — domain'e göre deterministik değerlendiriciyi seçer."""

from __future__ import annotations

from typing import Any

from app.evals.profile.coding_eval import evaluate_coding
from app.evals.profile.math_eval import evaluate_math
from app.evals.profile.reasoning_eval import evaluate_reasoning
from app.evals.profile.schema import EvalItem
from app.evals.profile.statistics_eval import evaluate_statistics
from app.evals.profile.trading_eval import evaluate_trading


def evaluate_answer(
    item: EvalItem, answer: str, *, allow_code_execution: bool = False
) -> dict[str, Any]:
    """{'correct': bool|None, 'score': float|None, 'checks': {...}} döndür.

    ``requires_abstention`` kayıtlarında doğruluk = doğru çekimserlik (içerik uydurmamak).
    """
    if item.requires_abstention:
        from app.evals.profile.text_checks import is_abstention

        ok = is_abstention(answer)
        return {"correct": ok, "score": float(ok), "checks": {"expected_abstention": True}}
    if item.domain == "math":
        return evaluate_math(item, answer)
    if item.domain == "statistics":
        return evaluate_statistics(item, answer)
    if item.domain == "reasoning":
        return evaluate_reasoning(item, answer)
    if item.domain == "trading":
        return evaluate_trading(item, answer)
    return evaluate_coding(item, answer, allow_execution=allow_code_execution)
