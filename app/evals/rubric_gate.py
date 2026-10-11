"""Rubrik + kritik hata kapısı (protokol §10; denetim 2026-10-11 E2).

Tek kural: adayın HERHANGİ bir cevabında (her tekrar dahil) kritik hata varsa karar
**NO-GO**'dur — ortalamaya gömülmez, başka boyuttaki kazançla telafi edilmez.

Puan: ``app.evals.llm30.RUBRIC`` boyutları (doğruluk 0–4, alt madde 0–2, tutarlılık 0–2,
doğrulama 0–2, kaynak desteği 0–2 — yalnız kaynaklı görevde) toplanır ve 0–4 ölçeğine
normalize edilir (``candidate_compare`` soru puanıyla aynı ölçek).

Tekrar: Ollama temperature 0'da bile deterministik değildir (aynı tohum farklı çalışma
zamanında birebir aynı çıktıyı garanti etmez). Bu yüzden her karşılaştırma soru başına ≥2
tekrar ister; tekrarlar soru içinde ortalanır (bağımsız örnek SAYILMAZ), fark soru
düzeyinde eşlenmiş bootstrap ile hesaplanır. Rapor her zaman n (soru) ve tekrar sayısını
taşır.

Kapı asla "GO" demez: geçen aday ``insan_onayi_bekliyor`` olur — son adım insandır.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

#: Kapının olası kararları (en kötüden en iyiye). "GO" bilinçli olarak YOK.
NO_GO = "NO-GO"
INSUFFICIENT = "yetersiz_kanit"
AWAITING_HUMAN = "insan_onayi_bekliyor"

MIN_REPEATS = 2
DEFAULT_MARGIN = 0.25  # 0–4 ölçeğinde kötüleşme toleransı (candidate_compare ile aynı)


class RubricError(ValueError):
    """Rubrik puanı şema dışı."""


def rubric_dims(*, requires_source: bool) -> dict[str, int]:
    """Bu görevde puanlanan boyutlar → azami puan (kaynaksız görevde kaynak boyutu yok)."""
    from app.evals.llm30 import RUBRIC

    return {d: m for d, m in RUBRIC.items() if requires_source or d != "kaynak_destegi"}


def critical_codes() -> dict[str, str]:
    from app.evals.final_set import all_critical_codes

    return all_critical_codes()


def score_rubric(rubric: Mapping[str, Any], *, requires_source: bool) -> float:
    """Boyut puanlarını doğrula → 0–4 ölçeğinde soru puanı (eksik/fazla boyut = HATA)."""
    dims = rubric_dims(requires_source=requires_source)
    if set(rubric) != set(dims):
        raise RubricError(f"rubrik boyutları {sorted(dims)} olmalı (gelen {sorted(rubric)})")
    total = 0
    for dim, top in dims.items():
        value = rubric[dim]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= top:
            raise RubricError(f"{dim}: 0–{top} arası tam sayı olmalı (gelen {value!r})")
        total += value
    return 4.0 * total / sum(dims.values())


def check_codes(codes: Iterable[str]) -> list[str]:
    known = critical_codes()
    out = [str(c) for c in codes]
    unknown = [c for c in out if c not in known]
    if unknown:
        raise RubricError(f"bilinmeyen kritik hata kodu: {unknown}")
    return out


def _paired_ci(diffs: list[float], *, seed: int) -> dict[str, float | int]:
    from app.evals.v15_protocol import paired_interval

    return paired_interval(diffs, seed=seed)


def gate(
    records: Iterable[Mapping[str, Any]],
    *,
    candidate: str,
    reference: str | None = None,
    min_repeats: int = MIN_REPEATS,
    margin: float = DEFAULT_MARGIN,
    seed: int = 42,
) -> dict[str, Any]:
    """Puanlanmış cevaplardan kapı kararı.

    ``records``: ``{"model", "question_id", "repeat", "score" (0–4 | None),
    "critical_errors" (liste)}``. ``model`` 2×2'de koşul harfi (A/B/C/D) da olabilir.
    """
    per: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for r in records:
        per.setdefault(str(r["model"]), {}).setdefault(str(r["question_id"]), []).append(r)
    if candidate not in per:
        raise RubricError(f"aday kayıtları yok: {candidate}")
    if reference is not None and reference not in per:
        raise RubricError(f"referans kayıtları yok: {reference}")

    summary: dict[str, Any] = {}
    q_means: dict[str, dict[str, float]] = {}
    for model, by_q in per.items():
        reps = [len(v) for v in by_q.values()]
        unscored = sum(1 for v in by_q.values() for r in v if r.get("score") is None)
        crit = [
            {"question_id": qid, "repeat": r.get("repeat"), "codes": list(r["critical_errors"])}
            for qid, v in sorted(by_q.items())
            for r in v
            if r.get("critical_errors")
        ]
        means: dict[str, float] = {}
        spreads: list[float] = []
        for qid, v in by_q.items():
            vals = [float(r["score"]) for r in v if r.get("score") is not None]
            if vals and len(vals) == len(v):
                means[qid] = sum(vals) / len(vals)
                spreads.append(max(vals) - min(vals))
        q_means[model] = means
        summary[model] = {
            "n_questions": len(by_q),
            "repeats_min": min(reps),
            "repeats_max": max(reps),
            "n_answers": sum(reps),
            "n_unscored": unscored,
            "mean_score": (sum(means.values()) / len(means)) if means else None,
            "critical_answers": len(crit),
            "critical": crit,
            # Tekrarlar arası puan farkı: büyükse tek koşu güvenilmez (determinizmsizlik).
            "repeat_spread_mean": (sum(spreads) / len(spreads)) if spreads else None,
            "questions_with_repeat_disagreement": sum(1 for s in spreads if s > 0),
        }

    cand = summary[candidate]
    reasons: list[str] = []
    decision = AWAITING_HUMAN
    if cand["critical_answers"]:
        decision = NO_GO
        reasons.append(
            f"Adayda {cand['critical_answers']} kritik hatalı cevap (her biri tek başına NO-GO)"
        )
    if cand["n_unscored"]:
        reasons.append(f"Adayda {cand['n_unscored']} puanlanmamış cevap")
        decision = decision if decision == NO_GO else INSUFFICIENT
    if cand["repeats_min"] < min_repeats:
        reasons.append(
            f"Soru başına tekrar {cand['repeats_min']} < {min_repeats} (Ollama temp 0'da "
            "deterministik değil; tek koşu karar kanıtı değildir)"
        )
        decision = decision if decision == NO_GO else INSUFFICIENT
    comparison: dict[str, Any] | None = None
    if reference is not None:
        common = sorted(set(q_means[candidate]) & set(q_means[reference]))
        diffs = [q_means[candidate][q] - q_means[reference][q] for q in common]
        if summary[reference]["repeats_min"] < min_repeats:
            reasons.append(f"Referansta soru başına tekrar < {min_repeats}")
            decision = decision if decision == NO_GO else INSUFFICIENT
        if len(diffs) >= 2:
            comparison = {"reference": reference, **_paired_ci(diffs, seed=seed)}
            if comparison["high"] < -margin:
                decision = NO_GO
                reasons.append(
                    f"Gerileme: soru düzeyi eşlenmiş fark GA üst sınırı "
                    f"{comparison['high']:.3f} < −{margin}"
                )
            elif decision == AWAITING_HUMAN and not (
                comparison["low"] > -margin and comparison["mean"] >= 0
            ):
                decision = INSUFFICIENT
                reasons.append(
                    f"Fark GA [{comparison['low']:.3f}, {comparison['high']:.3f}] kararsız "
                    f"(tolerans {margin})"
                )
        else:
            decision = decision if decision == NO_GO else INSUFFICIENT
            reasons.append("Eşlenmiş karşılaştırma için en az iki ortak puanlı soru gerekli")
    if decision == AWAITING_HUMAN:
        reasons.append("Kapı geçti → son karar İNSAN incelemesinde (otomatik GO yok)")
    return {
        "decision": decision,
        "candidate": candidate,
        "reasons": reasons,
        "models": summary,
        "comparison": comparison,
        "min_repeats": min_repeats,
        "margin": margin,
    }


def records_from_llm30_raw(path: Path) -> list[dict[str, Any]]:
    """``llm30_run`` ``raw.jsonl`` → kapı kayıtları (puanlar insan/hakem tarafından doldurulur).

    ``score_by_dimension`` RUBRIC boyutlarıyla, ``critical_errors`` kod listesiyle doldurulmuş
    olmalı; boş (None) satır puanlanmamış sayılır.
    """
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dims = r.get("score_by_dimension")
        score = (
            score_rubric(dims, requires_source=bool(r.get("rag_enabled")))
            if isinstance(dims, dict)
            else None
        )
        crit = r.get("critical_errors")
        out.append(
            {
                "model": r["condition"],
                "question_id": r["question_id"],
                "repeat": r.get("seed"),
                "score": score,
                "critical_errors": check_codes(crit) if isinstance(crit, list) else [],
            }
        )
    return out


def format_gate(res: Mapping[str, Any]) -> str:
    """Kısa insan-okur özet (CLI)."""
    lines = [f"KAPI: {res['decision']}  (aday {res['candidate']})"]
    for model, s in res["models"].items():
        mean = s["mean_score"]
        lines.append(
            f"  {model}: n={s['n_questions']} soru · tekrar {s['repeats_min']}–{s['repeats_max']}"
            f" · {s['n_answers']} cevap · ort {('—' if mean is None else f'{mean:.2f}')}/4"
            f" · kritik {s['critical_answers']} · puansız {s['n_unscored']}"
        )
    comp = res.get("comparison")
    if comp:
        lines.append(
            f"  fark (aday − {comp['reference']}): {comp['mean']:+.3f} "
            f"GA95 [{comp['low']:+.3f}, {comp['high']:+.3f}] · n={comp['n_questions']} soru"
        )
    lines.extend(f"  - {r}" for r in res["reasons"])
    return "\n".join(lines)
