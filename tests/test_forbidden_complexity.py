"""forbidden_pattern_rate (0711.0729) + complexity_entropy (1808.01926) testleri.

Bandt-Pompe ordinal-desen tabanlı; saf numpy/pandas, look-ahead yok, warmup NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.trading.indicators import (
    complexity_entropy,
    compute_indicator,
    forbidden_pattern_rate,
)


def _rand(n: int, seed: int = 0) -> pd.Series:
    return pd.Series(100 + np.random.default_rng(seed).standard_normal(n).cumsum())


def test_forbidden_aralik_ve_warmup() -> None:
    fr = forbidden_pattern_rate(_rand(120), period=14, order=3)
    val = fr.dropna()
    assert (val >= -1e-9).all() and (val <= 1 + 1e-9).all()  # [0,1]
    assert fr.iloc[:14].isna().all()  # tam pencere oluşana dek tanımsız


def test_forbidden_monoton_yuksek() -> None:
    # Sürekli artan seri → her pencere ordinal deseni [0,1,2]; 6 desenin 5'i 'yasak'.
    close = pd.Series(np.arange(1, 41, dtype=float))
    fr = forbidden_pattern_rate(close, period=6, order=3).dropna()
    assert np.allclose(fr.to_numpy(), 1.0 - 1.0 / 6.0, atol=1e-9)  # ≈0.8333


def test_complexity_aralik_ve_monoton_sifir() -> None:
    comp = complexity_entropy(_rand(150), period=20, order=3)
    val = comp.dropna()
    assert (val >= -1e-9).all() and (val <= 1 + 1e-9).all()  # [0,1]
    # Monoton seri → tek desen (delta dağılım) → H=0 → karmaşıklık 0.
    mono = complexity_entropy(pd.Series(np.arange(1, 41, dtype=float)), period=6, order=3).dropna()
    assert np.allclose(mono.to_numpy(), 0.0, atol=1e-9)


def test_registry_dispatch() -> None:
    df = pd.DataFrame({"close": _rand(80, seed=3)})
    pd.testing.assert_series_equal(
        compute_indicator("FORBIDDEN", df, 14), forbidden_pattern_rate(df["close"], 14)
    )
    pd.testing.assert_series_equal(
        compute_indicator("COMPLEXITY", df, 20), complexity_entropy(df["close"], 20)
    )


def test_look_ahead_yok_prefix_kararli() -> None:
    # t anındaki değer gelecekteki barlara bağlı olmamalı: tam seri ile kesilmiş
    # önek aynı erken değerleri vermeli (yalnız geçmiş pencere kullanılır).
    full = _rand(100, seed=5)
    for fn in (forbidden_pattern_rate, complexity_entropy):
        a = fn(full, period=10, order=3)
        b = fn(full.iloc[:60], period=10, order=3)
        pd.testing.assert_series_equal(a.iloc[:60], b, check_names=False)
