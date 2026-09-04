"""İstatistik denetleyici — saf numpy (scipy/statsmodels YOK).

LLM'in istatistik iddialarını (korelasyon, p-değeri, örneklem yeterliliği) kafadan
"anlamlı" demesini engeller. p-değeri t-testi yerine **permütasyon testi** ile
hesaplanır (seed zorunlu → determinizm, Kural 6). Korelasyon ≠ nedensellik uyarısı
ve örneklem-büyüklüğü uyarıları üretir.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class StatsReport:
    """Tek serinin betimsel istatistikleri + uyarılar."""

    n: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "warnings": self.warnings,
        }


@dataclass
class CorrelationReport:
    """İki seri arası korelasyon + permütasyon p-değeri + uyarılar."""

    n: int
    pearson: float
    spearman: float
    p_value: float  # permütasyon testi (iki yönlü), seed'e bağlı deterministik
    n_permutations: int
    seed: int
    significant: bool  # p_value < 0.05 (yalnız istatistiksel; nedensellik DEĞİL)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "p_value": self.p_value,
            "n_permutations": self.n_permutations,
            "seed": self.seed,
            "significant": self.significant,
            "warnings": self.warnings,
        }


def _sample_size_warnings(n: int) -> list[str]:
    w: list[str] = []
    if n < 8:
        w.append(f"Çok küçük örneklem (n={n} < 8) — sonuç istatistiksel olarak anlamsız.")
    elif n < 30:
        w.append(f"Küçük örneklem (n={n} < 30) — istatistiksel güç zayıf, dikkatli yorumla.")
    return w


def describe_series(values: Sequence[float]) -> StatsReport:
    """Tek bir sayısal serinin betimsel istatistikleri (+ örneklem uyarısı)."""
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        raise ValueError("values boş.")
    return StatsReport(
        n=int(arr.size),
        mean=round(float(arr.mean()), 6),
        median=round(float(np.median(arr)), 6),
        std=round(float(arr.std(ddof=0)), 6),
        min=round(float(arr.min()), 6),
        max=round(float(arr.max()), 6),
        warnings=_sample_size_warnings(int(arr.size)),
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ortalama-rank (bağları ortalama rank ile) — saf numpy, scipy.stats.rankdata yerine."""
    order = a.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # bağları (eşit değerleri) ortalama rank ile düzelt
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def correlation_report(
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int,
    n_permutations: int = 1000,
) -> CorrelationReport:
    """İki seri arası Pearson + Spearman korelasyonu ve permütasyon p-değeri.

    p-değeri: y rastgele n_permutations kez karıştırılır; |permüte korelasyon| ≥
    |gözlenen| oranı (iki yönlü). Seed zorunlu → deterministik (Kural 6).
    Uyarılar: örneklem büyüklüğü + korelasyon≠nedensellik (yüksek |r| durumunda).
    """
    xa = np.asarray([float(v) for v in x], dtype=float)
    ya = np.asarray([float(v) for v in y], dtype=float)
    if xa.size != ya.size:
        raise ValueError("x ve y aynı uzunlukta olmalı.")
    if xa.size < 3:
        raise ValueError("Korelasyon için en az 3 gözlem gerekir.")

    obs = _pearson(xa, ya)
    spear = _spearman(xa, ya)

    rng = np.random.default_rng(seed)
    abs_obs = abs(obs)
    count = 0
    for _ in range(int(n_permutations)):
        perm = rng.permutation(ya)
        if abs(_pearson(xa, perm)) >= abs_obs:
            count += 1
    # +1 düzeltmesi (gözlenenin kendisi) → p hiç 0 olmaz (yanlı-0 önlemi).
    p_value = (count + 1) / (int(n_permutations) + 1)

    warnings = _sample_size_warnings(int(xa.size))
    if abs_obs >= 0.5:
        warnings.append(
            "Yüksek korelasyon nedensellik DEĞİLDİR — gizli değişken/ortak sebep olabilir; "
            "yön ve mekanizma ayrıca test edilmeli."
        )

    return CorrelationReport(
        n=int(xa.size),
        pearson=round(obs, 6),
        spearman=round(spear, 6),
        p_value=round(p_value, 6),
        n_permutations=int(n_permutations),
        seed=int(seed),
        significant=bool(p_value < 0.05),
        warnings=warnings,
    )
