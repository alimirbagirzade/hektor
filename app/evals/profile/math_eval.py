"""Matematik değerlendirici — beklenen sayısal değer + tolerans; çoktan seçmelide exact match."""

from __future__ import annotations

from typing import Any

from app.evals.profile.schema import EvalItem
from app.evals.profile.text_checks import (
    claim_coverage,
    exact_match,
    final_number,
    forbidden_hits,
    numeric_match,
)


def evaluate_numeric_or_exact(item: EvalItem, answer: str) -> dict[str, Any]:
    """Sayısal (tolerans) → exact match → iddia kapsama sırasıyla deterministik kontrol."""
    checks: dict[str, Any] = {}
    correct: bool | None = None
    if item.expected_numeric_value is not None:
        got = final_number(answer)
        checks["parsed_value"] = got
        checks["expected_value"] = item.expected_numeric_value
        checks["tolerance"] = item.numeric_tolerance
        correct = numeric_match(got, item.expected_numeric_value, item.numeric_tolerance)
        checks["numeric_ok"] = correct
    elif item.accepted_answers:
        correct = exact_match(answer, item.accepted_answers)
        checks["exact_ok"] = correct
    cov, missing = claim_coverage(item.required_claims, answer)
    if cov is not None:
        checks["claim_coverage"] = cov
        checks["missing_claims"] = missing
        if correct is None:
            correct = cov == 1.0
    bad = forbidden_hits(item.forbidden_claims, answer)
    if bad:
        checks["forbidden_claims_hit"] = bad
        correct = False
    return {
        "correct": correct,
        "score": None if correct is None else float(correct),
        "checks": checks,
    }


def evaluate_math(item: EvalItem, answer: str) -> dict[str, Any]:
    return evaluate_numeric_or_exact(item, answer)
