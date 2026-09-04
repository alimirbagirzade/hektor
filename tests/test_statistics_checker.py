"""İstatistik denetleyici testleri (permütasyon p-değeri determinizmi + uyarılar)."""

from __future__ import annotations

import pytest

from app.tools.statistics_checker import correlation_report, describe_series


def test_describe_basic() -> None:
    rep = describe_series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert rep.n == 10
    assert rep.mean == pytest.approx(5.5)
    assert rep.median == pytest.approx(5.5)
    assert rep.min == 1 and rep.max == 10


def test_describe_small_sample_warns() -> None:
    rep = describe_series([1, 2, 3])
    assert any("küçük" in w.lower() for w in rep.warnings)


def test_describe_empty_raises() -> None:
    with pytest.raises(ValueError):
        describe_series([])


def test_perfect_correlation() -> None:
    x = list(range(1, 21))
    y = [2 * v for v in x]
    rep = correlation_report(x, y, seed=42, n_permutations=500)
    assert rep.pearson == pytest.approx(1.0, abs=1e-9)
    assert rep.spearman == pytest.approx(1.0, abs=1e-9)
    assert rep.p_value < 0.05
    assert rep.significant is True


def test_permutation_pvalue_deterministic() -> None:
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [2, 1, 4, 3, 6, 5, 8, 7]
    a = correlation_report(x, y, seed=11, n_permutations=300)
    b = correlation_report(x, y, seed=11, n_permutations=300)
    assert a.p_value == b.p_value
    assert a.pearson == b.pearson


def test_high_correlation_triggers_causation_warning() -> None:
    x = list(range(1, 21))
    y = [2 * v + (1 if v % 2 else -1) for v in x]
    rep = correlation_report(x, y, seed=5, n_permutations=200)
    assert any("nedensellik" in w.lower() for w in rep.warnings)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        correlation_report([1, 2, 3], [1, 2], seed=1)


def test_too_few_points_raise() -> None:
    with pytest.raises(ValueError):
        correlation_report([1, 2], [3, 4], seed=1)


def test_pvalue_never_zero() -> None:
    # +1 düzeltmesi sayesinde p hiç tam 0 olmaz (yanlı-0 önlemi).
    x = list(range(1, 31))
    y = [3 * v for v in x]
    rep = correlation_report(x, y, seed=1, n_permutations=100)
    assert rep.p_value > 0.0
