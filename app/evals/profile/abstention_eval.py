"""Çekimserlik + halüsinasyon değerlendirici (CLAUDE.md Kural 7: kaynak uydurma).

- ``abstention_correct``: model, çekimser kalması gereken yerde (``requires_abstention``)
  çekimser kaldı mı; kalmaması gereken yerde cevap verdi mi.
- ``hallucinated``: çekimser kalması gerekirken içerik uydurdu YA DA yasak iddia üretti
  ya da alınan bağlamda olmayan bir kaynağa atıf yaptı (grounding sonucu ile birleştirilir).
"""

from __future__ import annotations

from typing import Any

from app.evals.profile.schema import EvalItem
from app.evals.profile.text_checks import forbidden_hits, is_abstention


def evaluate_abstention(
    item: EvalItem, answer: str, *, fabricated_citations: int = 0
) -> dict[str, Any]:
    abstained = is_abstention(answer)
    abstention_correct = abstained == item.requires_abstention
    forbidden = forbidden_hits(item.forbidden_claims, answer)
    hallucinated = (
        (item.requires_abstention and not abstained) or bool(forbidden) or fabricated_citations > 0
    )
    return {
        "abstained": abstained,
        "abstention_correct": abstention_correct,
        "hallucinated": hallucinated,
        "forbidden_claims_hit": forbidden,
        "fabricated_citations": fabricated_citations,
    }
