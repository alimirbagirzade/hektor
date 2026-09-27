"""Tutarlılık — aynı soruya tekrarlanan (farklı seed'li) cevapların final cevap uyumu."""

from __future__ import annotations

from collections import Counter

from app.evals.profile.leakage import normalize_text
from app.evals.profile.text_checks import final_answer_segment, final_number, is_abstention


def answer_key(answer: str) -> str:
    """Karşılaştırma anahtarı: çekimserlik > son sayı (6 anlamlı hane) > normalize final segment."""
    if is_abstention(answer):
        return "<abstain>"
    num = final_number(answer)
    if num is not None:
        return f"num:{num:.6g}"
    return "txt:" + normalize_text(final_answer_segment(answer))[:120]


def consistency_score(answers: list[str]) -> float | None:
    """Mod cevaba uyan cevapların oranı (1.0 = tamamen tutarlı). <2 cevapta None."""
    if len(answers) < 2:
        return None
    keys = [answer_key(a) for a in answers]
    _, top = Counter(keys).most_common(1)[0]
    return top / len(keys)
