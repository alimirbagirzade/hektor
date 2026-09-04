"""Permütasyon entropisi (Bandt-Pompe) testleri — normalize [0,1], look-ahead'siz."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.trading.indicators import compute_indicator, permutation_entropy
from app.verification.exams.registry import get_spec, list_specs


def test_aralik_0_1() -> None:
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.standard_normal(200).cumsum())
    pe = permutation_entropy(close, period=20).dropna()
    assert (pe >= -1e-9).all()
    assert (pe <= 1 + 1e-9).all()


def test_monoton_trend_sifir() -> None:
    # Sürekli artış → her 3'lü pencere artan sıralı → tek ordinal desen → entropi 0
    close = pd.Series([float(x) for x in range(1, 9)])
    pe = permutation_entropy(close, period=4, order=3)
    assert abs(pe.iloc[-1]) < 1e-12


def test_warmup_nan() -> None:
    # order=3, period=4 → ilk geçerli değer index = order + period - 2 = 5
    close = pd.Series([float(x) for x in range(1, 9)])
    pe = permutation_entropy(close, period=4, order=3)
    assert pe.iloc[:5].isna().all()
    assert not pe.iloc[5:].isna().any()


def test_rastgele_pozitif() -> None:
    # Çok desenli (rastgele) seri → entropi kesinlikle > 0
    rng = np.random.default_rng(42)
    close = pd.Series(rng.standard_normal(100).cumsum())
    pe = permutation_entropy(close, period=20).dropna()
    assert (pe > 0).any()
    assert pe.iloc[-1] > 0.0


def test_order_2_kucuk_hata() -> None:
    close = pd.Series([1.0, 2.0, 3.0])
    try:
        permutation_entropy(close, period=2, order=1)
    except ValueError as exc:
        assert "order" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("order<2 ValueError beklenir")


def test_compute_indicator_permentropy() -> None:
    close = pd.Series(100 + np.random.default_rng(7).standard_normal(60).cumsum())
    df = pd.DataFrame({"close": close})
    pd.testing.assert_series_equal(
        compute_indicator("PERMENTROPY", df, 12), permutation_entropy(close, 12)
    )


def test_registry_permentropy_kayitli() -> None:
    spec = get_spec("PERMENTROPY")
    assert spec.name == "PERMENTROPY"
    assert "PERMENTROPY" in {s.name for s in list_specs()}
