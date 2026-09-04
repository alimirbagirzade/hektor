"""Deflated Sharpe (overfit haircut, 1905.05023 ruhu) — değişmez özellik testleri.

`_metrics`'in deflated Sharpe'ı Lo (2002) SE-kırpmasıyla hesaplar: daima ≤ ham Sharpe,
parametre sayısıyla düşer, örneklem büyüdükçe ham'a yaklaşır, negatif Sharpe'ı şişirmez.
Tamamen çevrimdışı + deterministik (CLAUDE.md Kural 6).
"""

from __future__ import annotations

import pandas as pd

from app.trading.backtester import _metrics


def _pos(n: int) -> pd.Series:
    return pd.Series(1.0, index=range(n))


def test_deflated_le_raw_and_in_dict() -> None:
    # Pozitif ortalamalı seri → pozitif Sharpe; deflated kırpılmış olmalı.
    base = [0.01, -0.002, 0.008, 0.001, 0.006, -0.003, 0.009, 0.002]
    rets = pd.Series(base * 25)  # 200 gözlem
    m = _metrics(rets, _pos(len(rets)), "1d", n_params=4)
    assert m.sharpe > 0
    assert m.sharpe_deflated <= m.sharpe  # haircut
    assert m.sharpe_deflated >= 0.0
    assert "sharpe_deflated" in m.to_dict()


def test_more_params_lower_deflated() -> None:
    # AYNI seri; yalnız parametre sayısı artar → deflated düşer (overfit cezası).
    rets = pd.Series([0.01, -0.002, 0.008, 0.001, 0.006, -0.003, 0.009, 0.002] * 25)
    pos = _pos(len(rets))
    few = _metrics(rets, pos, "1d", n_params=0)
    many = _metrics(rets, pos, "1d", n_params=12)
    assert many.sharpe_deflated <= few.sharpe_deflated
    assert few.sharpe == many.sharpe  # ham Sharpe parametreden bağımsız


def test_larger_sample_smaller_haircut() -> None:
    # Aynı dağılım, daha uzun seri → SE küçülür → deflated ham'a yaklaşır (gap daralır).
    base = [0.01, -0.002, 0.008, 0.001, 0.006, -0.003, 0.009, 0.002]
    short = pd.Series(base * 5)  # 40
    long = pd.Series(base * 100)  # 800
    ms = _metrics(short, _pos(len(short)), "1d", n_params=3)
    ml = _metrics(long, _pos(len(long)), "1d", n_params=3)
    gap_short = ms.sharpe - ms.sharpe_deflated
    gap_long = ml.sharpe - ml.sharpe_deflated
    assert gap_long <= gap_short


def test_negative_sharpe_not_inflated() -> None:
    # Kaybeden seri (negatif ortalama) → deflated ham'ı ŞİŞİRMEMELİ (eşit kalır).
    rets = pd.Series([-0.01, 0.002, -0.008, -0.001, -0.006, 0.003, -0.009] * 20)
    m = _metrics(rets, _pos(len(rets)), "1d", n_params=5)
    assert m.sharpe < 0
    assert m.sharpe_deflated == m.sharpe  # negatif kırpılmaz/şişirilmez
