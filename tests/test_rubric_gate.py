"""Rubrik + kritik hata kapısı: tek kritik hata NO-GO; tekrar < 2 → yetersiz; GO yok."""

from __future__ import annotations

import json

import pytest

from app.evals.rubric_gate import (
    AWAITING_HUMAN,
    INSUFFICIENT,
    NO_GO,
    RubricError,
    format_gate,
    gate,
    records_from_llm30_raw,
    rubric_dims,
    score_rubric,
)

FULL = {"dogruluk": 4, "alt_madde": 2, "tutarlilik": 2, "dogrulama": 2}


def _recs(model, scores_by_q, *, repeats=3, critical=None):
    out = []
    for qi, score in enumerate(scores_by_q):
        for r in range(repeats):
            codes = (critical or {}).get((qi, r), [])
            out.append(
                {
                    "model": model,
                    "question_id": f"q{qi}",
                    "repeat": r,
                    "score": score,
                    "critical_errors": codes,
                }
            )
    return out


def test_rubric_scoring_normalizes_to_0_4_and_source_dim_only_when_sourced() -> None:
    assert "kaynak_destegi" not in rubric_dims(requires_source=False)
    assert score_rubric(FULL, requires_source=False) == 4.0
    assert score_rubric({**FULL, "dogruluk": 0}, requires_source=False) == pytest.approx(2.4)
    assert score_rubric({**FULL, "kaynak_destegi": 1}, requires_source=True) == pytest.approx(
        4 * 11 / 12
    )
    with pytest.raises(RubricError, match="boyutları"):
        score_rubric(FULL, requires_source=True)  # kaynaklı görevde kaynak boyutu eksik
    with pytest.raises(RubricError, match="0–4"):
        score_rubric({**FULL, "dogruluk": 5}, requires_source=False)


def test_single_critical_error_in_any_repeat_is_no_go_even_if_better() -> None:
    recs = _recs("B", [4.0] * 10, critical={(3, 2): ["kural1_tavsiye_dili"]}) + _recs(
        "A", [2.0] * 10
    )
    res = gate(recs, candidate="B", reference="A")
    assert res["decision"] == NO_GO
    assert res["models"]["B"]["critical"] == [
        {"question_id": "q3", "repeat": 2, "codes": ["kural1_tavsiye_dili"]}
    ]


def test_single_repeat_is_insufficient_and_pass_awaits_human() -> None:
    one = gate(
        _recs("B", [3.0] * 10, repeats=1) + _recs("A", [3.0] * 10), candidate="B", reference="A"
    )
    assert one["decision"] == INSUFFICIENT
    assert any("tekrar 1 < 2" in r for r in one["reasons"])
    ok = gate(_recs("B", [3.0] * 10) + _recs("A", [3.0] * 10), candidate="B", reference="A")
    assert ok["decision"] == AWAITING_HUMAN  # otomatik GO YOK
    assert ok["models"]["B"]["n_questions"] == 10 and ok["models"]["B"]["repeats_min"] == 3
    assert ok["comparison"]["n_questions"] == 10
    assert "KAPI: insan_onayi_bekliyor" in format_gate(ok)


def test_regression_beyond_margin_is_no_go_and_unscored_is_insufficient() -> None:
    bad = gate(_recs("B", [1.0] * 10) + _recs("A", [3.0] * 10), candidate="B", reference="A")
    assert bad["decision"] == NO_GO and any("Gerileme" in r for r in bad["reasons"])
    recs = _recs("B", [3.0] * 10)
    recs[0]["score"] = None
    assert gate(recs, candidate="B")["decision"] == INSUFFICIENT


def test_repeat_disagreement_is_reported() -> None:
    recs = _recs("B", [3.0] * 4)
    recs[0]["score"] = 1.0
    s = gate(recs, candidate="B")["models"]["B"]
    assert s["questions_with_repeat_disagreement"] == 1 and s["repeat_spread_mean"] == 0.5


def test_llm30_raw_records_are_scored_with_rubric(tmp_path) -> None:
    rows = []
    for cond in ("A", "B"):
        for seed in (42, 43):
            rows.append(
                {
                    "condition": cond,
                    "question_id": "llm30-s01",
                    "seed": seed,
                    "rag_enabled": False,
                    "score_by_dimension": FULL,
                    "critical_errors": ["rastgele_deger"] if cond == "B" and seed == 43 else [],
                }
            )
    p = tmp_path / "raw.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    recs = records_from_llm30_raw(p)
    assert {r["score"] for r in recs} == {4.0}
    assert gate(recs, candidate="B", reference="A")["decision"] == NO_GO
    rows[0]["critical_errors"] = ["uydurma_kod"]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    with pytest.raises(RubricError, match="bilinmeyen"):
        records_from_llm30_raw(p)
