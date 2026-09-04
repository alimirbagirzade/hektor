"""Monte Carlo olasılık simülatörü testleri (determinizm + risk-of-ruin)."""

from __future__ import annotations

import pytest

from app.tools.probability_simulator import monte_carlo_equity


def test_deterministic_same_seed() -> None:
    rets = [0.05, -0.02, 0.03, -0.04, 0.06, -0.01]
    a = monte_carlo_equity(rets, seed=7, n_paths=500)
    b = monte_carlo_equity(rets, seed=7, n_paths=500)
    assert a.to_dict() == b.to_dict()


def test_different_seed_changes_result() -> None:
    rets = [0.05, -0.02, 0.03, -0.04, 0.06, -0.01]
    a = monte_carlo_equity(rets, seed=1, n_paths=500)
    b = monte_carlo_equity(rets, seed=2, n_paths=500)
    # Bootstrap örneklemesi farklı → ruin/dağılım farklı (en azından bir alanda).
    assert a.ruin_probability != b.ruin_probability or a.mean_final_equity != b.mean_final_equity


def test_all_positive_no_ruin_no_loss() -> None:
    res = monte_carlo_equity([0.01, 0.02, 0.03], seed=42, n_paths=300, ruin_fraction=0.5)
    assert res.ruin_probability == 0.0
    assert res.prob_loss == 0.0
    assert res.mean_final_equity > 10_000.0


def test_negative_series_has_ruin() -> None:
    res = monte_carlo_equity([-0.4, -0.3, -0.2], seed=42, n_paths=500, ruin_fraction=0.5)
    assert res.ruin_probability > 0.0
    assert res.prob_loss > 0.0


def test_expected_value_and_variance_reported() -> None:
    rets = [0.10, -0.10, 0.10, -0.10]
    res = monte_carlo_equity(rets, seed=3, n_paths=100)
    assert res.per_trade_mean == pytest.approx(0.0, abs=1e-9)
    assert res.per_trade_std > 0


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        monte_carlo_equity([], seed=1)


def test_bad_ruin_fraction_raises() -> None:
    with pytest.raises(ValueError):
        monte_carlo_equity([0.01], seed=1, ruin_fraction=1.5)


def test_probabilities_in_unit_range() -> None:
    res = monte_carlo_equity([0.05, -0.06, 0.02, -0.03], seed=9, n_paths=400)
    assert 0.0 <= res.ruin_probability <= 1.0
    assert 0.0 <= res.prob_loss <= 1.0


def test_negative_n_trades_raises() -> None:
    with pytest.raises(ValueError, match="n_trades"):
        monte_carlo_equity([0.01, -0.02], seed=1, n_trades=-5)


def test_negative_n_paths_raises() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        monte_carlo_equity([0.01, -0.02], seed=1, n_paths=-1)


def test_non_finite_returns_raise() -> None:
    """NaN/inf getiri cumprod boyunca yayılıp tüm metrikleri sessizce zehirler → net hata."""
    import math

    with pytest.raises(ValueError):
        monte_carlo_equity([0.01, math.nan], seed=1)
    with pytest.raises(ValueError):
        monte_carlo_equity([0.01, math.inf], seed=1)


def test_return_below_minus_one_raises() -> None:
    """getiri < -1.0 → 1+r < 0 → cumprod equity işaretini çevirir (anlamsız); reddedilmeli."""
    with pytest.raises(ValueError):
        monte_carlo_equity([0.01, -1.5], seed=1)


def test_expected_shortfall_count_based_with_ties() -> None:
    # Ayrık getiriler → final equity'lerde eşitlik; ES sayıya göre seçilir (kuyruk şişmez).
    rets = [0.0, 0.01, -0.01]
    res = monte_carlo_equity(rets, seed=3, n_paths=2000)
    # ES en kötü %5'in ortalaması → VaR%95'ten (5. yüzdelik tek nokta) küçük olamaz
    assert res.expected_shortfall_pct >= res.var_95_pct - 1e-9
    # determinizm korunur
    res2 = monte_carlo_equity(rets, seed=3, n_paths=2000)
    assert res.expected_shortfall_pct == res2.expected_shortfall_pct


def test_zero_n_trades_raises() -> None:
    # Açık 0 (falsy) sessizce rets.size'a dönmemeli → doğrulamaya düşüp hata vermeli
    with pytest.raises(ValueError, match="n_trades"):
        monte_carlo_equity([0.01, -0.02], seed=1, n_trades=0)
