"""ReleaseGate fail-closed davranışı (CLAUDE.md Kural 2 — bozuk metrik yayını geçmemeli).

Yayın kapısı asla FAIL-OPEN olmamalı: hesaplanamayan (None/NaN/inf/sayısal-olmayan)
bir metrik eşiği KARŞILAMADI sayılır; eskiden NaN sessizce geçer (nan<eşik=False),
None ise TypeError ile çökerdi.
"""

from __future__ import annotations

import math

from app.reliability.release_gate import ReleaseGate

_OK = {
    "recall_at_10": 0.90,
    "citation_accuracy": 0.90,
    "grounding_score": 0.90,
    "abstention_correct": 0.95,
}


def test_all_thresholds_met_passes() -> None:
    assert ReleaseGate().check(_OK).passed is True


def test_below_threshold_fails() -> None:
    m = {**_OK, "recall_at_10": 0.50}
    res = ReleaseGate().check(m)
    assert res.passed is False
    assert any("recall_at_10" in f for f in res.failures)


def test_missing_metric_fails_closed() -> None:
    m = {k: v for k, v in _OK.items() if k != "recall_at_10"}
    res = ReleaseGate().check(m)
    assert res.passed is False
    assert any("EKSİK" in f for f in res.failures)


def test_nan_metric_fails_closed_not_open() -> None:
    """REGRESYON: NaN metrik (ör. sıfıra-bölme) SESSİZCE GEÇERDİ (fail-open) — artık FAIL."""
    m = {**_OK, "recall_at_10": math.nan}
    res = ReleaseGate().check(m)
    assert res.passed is False
    assert any("recall_at_10" in f for f in res.failures)


def test_inf_metric_fails_closed() -> None:
    """+inf gibi anlamsız metrik de fail-closed (inf<eşik=False ile geçemez)."""
    m = {**_OK, "grounding_score": math.inf}
    res = ReleaseGate().check(m)
    assert res.passed is False


def test_none_metric_fails_closed_not_crash() -> None:
    """REGRESYON: None metrik TypeError ile ÇÖKERDİ — artık temiz FAIL (çökme yok)."""
    m = {**_OK, "citation_accuracy": None}
    res = ReleaseGate().check(m)  # çökmemeli
    assert res.passed is False
    assert any("citation_accuracy" in f for f in res.failures)


def test_bool_metric_fails_closed() -> None:
    """bool metrik (True/False) geçersiz sayılır — metrik gerçek sayı olmalı."""
    m = {**_OK, "abstention_correct": True}
    res = ReleaseGate().check(m)
    assert res.passed is False


def test_non_numeric_metric_fails_closed() -> None:
    """Sayısal-olmayan değer (str) fail-closed, çökme yok."""
    m = {**_OK, "recall_at_10": "0.9"}
    res = ReleaseGate().check(m)
    assert res.passed is False
