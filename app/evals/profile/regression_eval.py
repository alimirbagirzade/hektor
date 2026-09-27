"""Regression kapısı — aday profil, production profiliyle (yoksa base+RAG baseline) kıyaslanır.

Overall skor YÜKSELDİ diye terfi OLMAZ. Kurallar (``configs/eval/profile_eval.yaml``):

- ``must_not_decrease``: aday < mevcut ise ENGEL (ör. statistics_accuracy, grounding).
- ``max_drop``: domain başına izin verilen en fazla düşüş (ör. math 0.03); aşılırsa ENGEL.
- ``must_not_increase``: aday > mevcut ise ENGEL (ör. hallucination_rate).
- ``minimums``: mutlak alt eşik (ör. abstention_accuracy ≥ 0.8).
- ``min_items_per_domain``: kıyaslanan domain'de örneklem yetersizse ENGEL (gürültü ≠ kanıt).
- Metrik ölçülemediyse (None) ENGEL — kanıtlanamayan iyileşme terfi gerekçesi değildir.

Yalnız deterministik metrikler kullanılır; LLM judge skoru bu kapıya girmez.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

_EPS = 1e-9


@dataclass
class GateFinding:
    metric: str
    rule: str
    candidate: float | None
    reference: float | None
    detail: str


@dataclass
class GateResult:
    passed: bool
    reference_label: str
    blockers: list[GateFinding] = field(default_factory=list)
    regressions: list[GateFinding] = field(default_factory=list)  # engel olmayan düşüşler
    overall_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["note"] = "overall_delta bilgi amaçlıdır; kapı kararı domain kurallarıyla verilir"
        return d


def _num(v: Any) -> float | None:
    return None if v is None else float(v)


def evaluate_regression_gate(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    reference_label: str = "production",
    candidate_overall: float | None = None,
    reference_overall: float | None = None,
) -> GateResult:
    res = GateResult(passed=True, reference_label=reference_label)
    if candidate_overall is not None and reference_overall is not None:
        res.overall_delta = round(candidate_overall - reference_overall, 6)

    def block(metric: str, rule: str, c: float | None, r: float | None, detail: str) -> None:
        res.blockers.append(GateFinding(metric, rule, c, r, detail))

    for metric in rules.get("must_not_decrease") or []:
        c, r = _num(candidate.get(metric)), _num(reference.get(metric))
        if c is None or r is None:
            block(metric, "must_not_decrease", c, r, "metrik ölçülemedi")
        elif c + _EPS < r:
            block(metric, "must_not_decrease", c, r, f"düştü: {r:.3f} → {c:.3f}")

    for metric, allowed in (rules.get("max_drop") or {}).items():
        c, r = _num(candidate.get(metric)), _num(reference.get(metric))
        if c is None or r is None:
            block(metric, "max_drop", c, r, "metrik ölçülemedi")
            continue
        drop = r - c
        if drop > float(allowed) + _EPS:
            block(metric, "max_drop", c, r, f"anlamlı regression: -{drop:.3f} (izin {allowed})")
        elif drop > _EPS:
            res.regressions.append(
                GateFinding(metric, "max_drop", c, r, f"küçük düşüş -{drop:.3f} (izin içinde)")
            )

    for metric in rules.get("must_not_increase") or []:
        c, r = _num(candidate.get(metric)), _num(reference.get(metric))
        if c is None or r is None:
            block(metric, "must_not_increase", c, r, "metrik ölçülemedi")
        elif c > r + _EPS:
            block(metric, "must_not_increase", c, r, f"arttı: {r:.3f} → {c:.3f}")

    for metric, minimum in (rules.get("minimums") or {}).items():
        c = _num(candidate.get(metric))
        if c is None or c + _EPS < float(minimum):
            block(metric, "minimum", c, float(minimum), f"eşik altı (min {minimum})")

    min_n = int(rules.get("min_items_per_domain") or 0)
    if min_n:
        c_n = candidate.get("n_by_domain") or {}
        r_n = reference.get("n_by_domain") or {}
        watched = set(rules.get("must_not_decrease") or []) | set(rules.get("max_drop") or {})
        for metric in sorted(watched):
            if not metric.endswith("_accuracy"):
                continue
            dom = metric.removesuffix("_accuracy")
            n_c, n_r = int(c_n.get(dom, 0)), int(r_n.get(dom, 0))
            if min(n_c, n_r) < min_n:
                block(metric, "min_items", n_c, n_r, f"yetersiz örneklem (<{min_n}) — kanıt değil")

    res.passed = not res.blockers
    return res
