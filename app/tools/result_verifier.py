"""Araç sonucu doğrulayıcı — hesap çıktısı "final" olmadan önce mantık sınırları.

LLM bir aracın çıktısını cevaba koymadan önce buradan geçirmeli. Gerçekçi-olmayan
değerler (Sharpe > 5, Kelly > 1, olasılık [0,1] dışında, inf/nan) uyarı üretir.
Saf-Python; hiçbir şey çalıştırmaz (Kural 5 — eval/exec yok).
"""

from __future__ import annotations

import math
from typing import Any


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def verify_backtest_metrics(metrics: dict[str, Any]) -> list[str]:
    """Backtest metriklerinde gerçekçi-olmayan değerleri işaretle (look-ahead şüphesi vb.)."""
    w: list[str] = []
    sharpe = metrics.get("sharpe")
    if sharpe is not None and _is_finite(sharpe) and float(sharpe) > 5:
        w.append(f"Gerçekçi olmayan Sharpe ({sharpe}) — look-ahead/leak şüphesi.")
    pf = metrics.get("profit_factor")
    if pf is not None and _is_finite(pf) and float(pf) < 0:
        w.append(f"Negatif profit factor ({pf}) — hesap hatası.")
    wr = metrics.get("win_rate_pct")
    if wr is not None and _is_finite(wr) and not (0 <= float(wr) <= 100):
        w.append(f"win_rate_pct [0,100] dışında ({wr}).")
    dd = metrics.get("max_drawdown_pct")
    if dd is not None and _is_finite(dd) and float(dd) < -100:
        w.append(f"max_drawdown_pct < -100 imkânsız ({dd}).")
    for key in ("sharpe", "sortino", "total_return_pct", "profit_factor"):
        v = metrics.get(key)
        if v is not None and not _is_finite(v):
            w.append(f"{key} sonlu değil (inf/nan): {v}.")
    return w


def verify_kelly(fraction: float) -> list[str]:
    """Kelly fraksiyonu mantık sınırları (0 ≤ f ≤ 1; > 0.5 yüksek risk)."""
    w: list[str] = []
    if not _is_finite(fraction):
        return [f"Kelly fraksiyonu sonlu değil: {fraction}."]
    f = float(fraction)
    if f < 0:
        w.append(f"Kelly fraksiyonu negatif ({f:.3f}) — negatif beklenti.")
    if f > 1:
        w.append(f"Kelly fraksiyonu > 1 ({f:.3f}) — geçersiz (kaldıraçsız imkânsız).")
    elif f > 0.5:
        w.append(f"Kelly fraksiyonu yüksek ({f:.0%}) — yarı/çeyrek Kelly önerilir.")
    return w


def verify_probability(value: float, name: str = "olasılık") -> list[str]:
    """Bir olasılık değerinin [0, 1] aralığında olduğunu doğrula."""
    if not _is_finite(value):
        return [f"{name} sonlu değil: {value}."]
    v = float(value)
    if not (0.0 <= v <= 1.0):
        return [f"{name} [0,1] dışında ({v})."]
    return []
