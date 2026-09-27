"""Hata teşhisi — "skor düşük" değil "skor NEDEN düşük".

Tek-sistem nedenleri (``diagnose_item``) ve sistemler-arası nedenler
(``cross_system_diagnosis``: aynı soruda base doğru ama LoRA yanlış → LoRA_reasoning_failure;
tekil adapter doğru ama birleşik profil yanlış → profile_merge_failure) ayrı hesaplanır.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.evals.profile.schema import EvalItem, ItemScore

FAILURE_CATEGORIES: tuple[str, ...] = (
    "retrieval_failure",
    "reranker_failure",
    "context_missing",
    "context_noise",
    "LoRA_reasoning_failure",
    "math_failure",
    "statistics_failure",
    "unsupported_claim",
    "hallucination",
    "citation_failure",
    "router_failure",
    "profile_merge_failure",
    "base_model_failure",
)


def diagnose_item(
    item: EvalItem,
    score: ItemScore,
    *,
    uses_rag: bool,
    retrieval: Mapping[str, Any] | None = None,
    router_selected_domain: str | None = None,
) -> list[str]:
    """Tek sistemdeki yanlış cevabın gözlenebilir nedenleri (doğruysa boş liste)."""
    if score.correct is not False:
        return []
    cats: list[str] = []
    if uses_rag and item.requires_rag:
        retrieved = list((retrieval or {}).get("retrieved_chunk_ids") or [])
        candidates = list((retrieval or {}).get("candidate_chunk_ids") or [])
        expected = set(item.expected_chunk_ids)
        if not retrieved:
            cats.append("context_missing")
        elif expected and not expected & set(retrieved):
            if expected & set(candidates):
                cats.append("reranker_failure")  # aday havuzunda vardı, sıralama eledi
            else:
                cats.append("retrieval_failure")
        elif score.retrieval_precision is not None and score.retrieval_precision < 0.3:
            cats.append("context_noise")
    if item.domain == "math" and (
        item.requires_calculation or item.expected_numeric_value is not None
    ):
        cats.append("math_failure")
    if item.domain == "statistics":
        cats.append("statistics_failure")
    if score.checks.get("missing_claims") or score.checks.get("forbidden_claims_hit"):
        cats.append("unsupported_claim")
    if score.hallucinated:
        cats.append("hallucination")
    if score.citation_accuracy is not None and score.citation_accuracy < 1.0:
        cats.append("citation_failure")
    if router_selected_domain is not None and router_selected_domain not in (
        item.domain,
        "general",
    ):
        cats.append("router_failure")
    return list(dict.fromkeys(cats))


def cross_system_diagnosis(
    item_id: str,
    by_system: Mapping[str, ItemScore],
    *,
    base_of: Mapping[str, str],
    individual_of: Mapping[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Sistemler arası neden ekle.

    base_of:       sistem → aynı RAG ayarlı LoRA'sız karşılığı (C→A, D→B).
    individual_of: birleşik profil sistemi → aynı RAG ayarlı tekil-adapter sistemleri.
    Hiçbir sistem doğru değilse ve A (base) yanlışsa: base_model_failure.
    """
    extra: dict[str, list[str]] = {}
    for system, sc in by_system.items():
        if sc.correct is not False:
            continue
        base = base_of.get(system)
        if base and by_system.get(base) is not None and by_system[base].correct is True:
            extra.setdefault(system, []).append("LoRA_reasoning_failure")
        for ind in (individual_of or {}).get(system, []):
            if by_system.get(ind) is not None and by_system[ind].correct is True:
                extra.setdefault(system, []).append("profile_merge_failure")
                break
    if by_system and all(sc.correct is False for sc in by_system.values()):
        for system in by_system:
            extra.setdefault(system, []).append("base_model_failure")
    return extra
