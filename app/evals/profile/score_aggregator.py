"""Skor birleştirici — ham metrikler HER ZAMAN ayrı; overall yalnız raporlama kolaylığı.

``eval_weights`` (bu modül) ile LoRA profil ağırlıkları (mix_profiles.yaml) AYNI KAVRAM
DEĞİLDİR: biri rapor özet katsayısı, diğeri adapter ölçek katsayısıdır.
Overall skor terfi kararında KULLANILMAZ (bkz. regression_eval).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from app.evals.profile.schema import EVAL_DOMAINS, ItemScore

# Eval ağırlık anahtarı → metrik adı (kısa takma adlar).
_ALIASES = {"abstention": "abstention_accuracy"}
LOWER_IS_BETTER: frozenset[str] = frozenset({"hallucination_rate", "latency"})


def _avg(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 6) if vals else None


def aggregate(
    scores: list[ItemScore], resources: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Kalem skorlarından metrik sözlüğü (+ domain başına n)."""
    m: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for d in EVAL_DOMAINS:
        ds = [s for s in scores if s.domain == d and s.correct is not None]
        counts[d] = len(ds)
        m[f"{d}_accuracy"] = _avg(float(bool(s.correct)) for s in ds)
    measurable = [s for s in scores if s.correct is not None]
    m["answer_correctness"] = _avg(float(bool(s.correct)) for s in measurable)
    # Kısmi kredi (iddia kapsaması, sayı+iddia yarı-yarıya) — answer_correctness ikili iken
    # bu, "ne kadar doğru" ortalamasıdır.
    m["factual_correctness"] = _avg(s.score for s in measurable)
    m["retrieval_recall_at_k"] = _avg(s.retrieval_recall for s in scores)
    m["retrieval_precision"] = _avg(s.retrieval_precision for s in scores)
    m["context_relevance"] = _avg(s.context_relevance for s in scores)
    m["grounding"] = _avg(s.grounding for s in scores)
    m["citation_accuracy"] = _avg(s.citation_accuracy for s in scores)
    abst = [s for s in scores if "abstention_correct" in s.checks]
    m["abstention_accuracy"] = _avg(float(bool(s.checks["abstention_correct"])) for s in abst)
    m["hallucination_rate"] = _avg(float(s.hallucinated) for s in scores) if scores else None
    m["consistency"] = _avg(s.checks.get("consistency") for s in scores)
    m["latency"] = _avg(s.latency_s for s in scores)
    m["tokens_per_second"] = _avg(s.tokens_per_second for s in scores)
    res = dict(resources or {})
    m["VRAM_usage"] = res.get("vram_mb")
    m["RAM_usage"] = res.get("ram_mb")
    m["n_items"] = len(scores)
    m["n_unmeasured"] = len(scores) - len(measurable)
    m["n_by_domain"] = counts
    return m


def overall_score(metrics: Mapping[str, Any], eval_weights: Mapping[str, float]) -> dict[str, Any]:
    """Ağırlıklı özet; eksik metrikler ağırlıktan düşülür ve AÇIKÇA listelenir."""
    used: dict[str, float] = {}
    missing: list[str] = []
    total = 0.0
    for key, w in eval_weights.items():
        name = _ALIASES.get(key, key)
        v = metrics.get(name)
        if v is None:
            missing.append(name)
            continue
        val = 1.0 - float(v) if name in LOWER_IS_BETTER else float(v)
        used[name] = float(w)
        total += float(w) * val
    wsum = sum(used.values())
    return {
        "overall": round(total / wsum, 6) if wsum > 0 else None,
        "weights_used": used,
        "missing_metrics": missing,
        "note": (
            "overall yalnız raporlama içindir; terfi kararı domain metrikleri + regression gate ile"
        ),
    }
