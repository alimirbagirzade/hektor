"""Disiplin/risk/overfit eval setlerinin yapısal bütünlüğü (doğrulama gate'i sağlamlığı).

v5 dersi: zayıf gate kötü adapter'ı geçirdi. Bu testler eval setlerinin yüklenebilir,
yeterli kapsamlı (min soru) ve iyi-biçimli (her soruda ≥1 must_avoid token) kalmasını
korur — böylece base-vs-adapter karşılaştırması anlamlı bir sinyal üretir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.training.evaluate_model import load_eval_set

_EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"

# (dosya, minimum soru sayısı) — gate'in tırpanlanmasına karşı taban.
_SETS = [
    ("discipline_core.jsonl", 16),
    ("overfit_awareness.jsonl", 12),
    ("risk_management.jsonl", 12),
]


@pytest.mark.parametrize(("fname", "min_count"), _SETS)
def test_eval_set_loads_and_is_well_formed(fname: str, min_count: int) -> None:
    items = load_eval_set(_EVAL_DIR / fname)
    assert len(items) >= min_count, f"{fname}: {len(items)} < {min_count} (gate zayıflamış)"
    for it in items:
        assert it.question.strip(), f"{fname}: boş soru"
        assert it.must_avoid, f"{fname}: must_avoid boş — '{it.question[:40]}' bayrak üretemez"
        assert all(tok.strip() for tok in it.must_avoid), f"{fname}: boş must_avoid token"


@pytest.mark.parametrize(("fname", "_min"), _SETS)
def test_eval_set_no_duplicate_questions(fname: str, _min: int) -> None:
    items = load_eval_set(_EVAL_DIR / fname)
    qs = [it.question.strip().lower() for it in items]
    assert len(qs) == len(set(qs)), f"{fname}: yinelenen soru var"
