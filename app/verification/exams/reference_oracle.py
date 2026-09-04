"""ReferenceOracle — sınavların TEK güvenli "gerçek" kaynağı.

L3/L4 sınavlarında modelin çıktısı bununla karşılaştırılır. Referans daima:
  - registry indikatörleri için ``compute_indicator`` (saf vektörize pandas/numpy),
  - registry-dışı formüller için ``safe_eval`` (whitelist'li AST, eval/exec yok)
ile üretilir. Girdi vektörleri seed'li ve deterministiktir (CLAUDE.md Kural 6).
Tek-bar/tek-değer fonksiyonları kullanılır; gelecek bar okunmaz (look-ahead yok).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from app.trading.indicators import compute_indicator
from app.verification.exams.safe_eval import safe_eval

__all__ = ["ReferenceOracle"]


class ReferenceOracle:
    """Sınavlar için güvenli referans hesaplayıcı."""

    @staticmethod
    def synthetic_closes(
        n: int, seed: int, *, low: float = 100.0, high: float = 110.0, decimals: int = 2
    ) -> pd.Series:
        """Seed'li, temiz (yuvarlanmış) kapanış fiyatı serisi üretir.

        Kısa ve yuvarlak sayılar bilinçlidir: modelin formülü ELLE uygulayabilmesi
        (aritmetik dayanıklılık değil, ANLAMA test edilsin) için.
        """
        rng = np.random.default_rng(seed)
        vals = rng.uniform(low, high, size=n).round(decimals)
        return pd.Series(vals, dtype=float)

    @staticmethod
    def indicator_series(name: str, df: pd.DataFrame, period: int) -> pd.Series:
        """Registry indikatörünün referans serisi (compute_indicator sarmalayıcı)."""
        return compute_indicator(name, df, period)

    @staticmethod
    def formula_value(expr: str, variables: Mapping[str, float]) -> float:
        """Registry-dışı bir formülün skaler referans değeri (güvenli AST)."""
        return safe_eval(expr, variables)
