"""Monte Carlo olasılık simülatörü — saf numpy, seed zorunlu (Kural 6: determinizm).

LLM'in olasılık/beklenen-değer/risk-of-ruin hesabını "kafadan" yapmasını engeller;
işlem getirisi serisinden bootstrap ile equity yolları simüle eder. Çıktı bir
*hipotez/test noktası*dır, yatırım tavsiyesi DEĞİLDİR (Kural 1).

Bağımlılık yok: scipy/statsmodels gerekmez (yüzdelikler + RNG saf numpy).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MonteCarloResult:
    """Bootstrap Monte Carlo equity simülasyonu özeti (tümü deterministik, seed'e bağlı)."""

    n_paths: int
    n_trades: int
    seed: int
    per_trade_mean: float  # beklenen değer (işlem başı ortalama getiri)
    per_trade_std: float  # oynaklık (işlem başı std)
    mean_final_equity: float
    median_final_equity: float
    p05_final_equity: float
    p95_final_equity: float
    var_95_pct: float  # 5. yüzdelikte başlangıca göre kayıp % (Value-at-Risk)
    expected_shortfall_pct: float  # en kötü %5 yolun ortalama kayıp %'si
    prob_loss: float  # final < başlangıç olasılığı
    ruin_fraction: float  # ruin seviyesi (başlangıcın kesri, ör. 0.5 = %50 kayıp)
    ruin_probability: float  # yol boyunca herhangi bir anda ruin seviyesine düşme olasılığı
    note: str = (
        "Geçmiş işlem getirilerinden bootstrap simülasyonudur — gelecek garantisi "
        "değil, hipotez/test noktasıdır (yatırım tavsiyesi değildir)."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_paths": self.n_paths,
            "n_trades": self.n_trades,
            "seed": self.seed,
            "per_trade_mean": self.per_trade_mean,
            "per_trade_std": self.per_trade_std,
            "mean_final_equity": self.mean_final_equity,
            "median_final_equity": self.median_final_equity,
            "p05_final_equity": self.p05_final_equity,
            "p95_final_equity": self.p95_final_equity,
            "var_95_pct": self.var_95_pct,
            "expected_shortfall_pct": self.expected_shortfall_pct,
            "prob_loss": self.prob_loss,
            "ruin_fraction": self.ruin_fraction,
            "ruin_probability": self.ruin_probability,
            "note": self.note,
        }


def monte_carlo_equity(
    trade_returns: Sequence[float],
    *,
    seed: int,
    n_paths: int = 1000,
    n_trades: int | None = None,
    initial_equity: float = 10_000.0,
    ruin_fraction: float = 0.5,
) -> MonteCarloResult:
    """İşlem getirisi serisinden bootstrap Monte Carlo equity simülasyonu.

    Parametreler
    ------------
    trade_returns : işlem-bazlı getiri serisi (ondalık; +0.05 = %5 kazanç).
    seed          : ZORUNLU — aynı seed aynı sonucu verir (Kural 6).
    n_paths       : simüle edilecek yol sayısı.
    n_trades      : her yoldaki işlem sayısı (None → girdi uzunluğu).
    initial_equity: başlangıç sermayesi.
    ruin_fraction : ruin seviyesi başlangıcın kesri olarak (0.5 = sermayenin %50'sine
                    düşmek "ruin" sayılır). 0 < ruin_fraction < 1.

    ``trade_returns`` boşsa ValueError.
    """
    rets = np.asarray([float(r) for r in trade_returns], dtype=float)
    if rets.size == 0:
        raise ValueError("trade_returns boş — simülasyon için en az bir işlem getirisi gerekir.")
    if not np.all(np.isfinite(rets)):
        # NaN/inf getiri cumprod boyunca yayılıp tüm metrikleri sessizce zehirlerdi.
        raise ValueError("trade_returns sonlu olmalı (NaN/inf yok) — bozuk veri girişi.")
    if np.any(rets < -1.0):
        # 1+r < 0 → cumprod equity işaretini çevirir (anlamsız negatif sermaye). Tek
        # işlemde >%100 kayıp imkânsız (kaldıraçsız long); bozuk girdiyi açıkça reddet.
        raise ValueError("trade_returns < -1.0 olamaz (tek işlemde >%100 kayıp imkânsız).")
    if not (0.0 < ruin_fraction < 1.0):
        raise ValueError("ruin_fraction (0, 1) aralığında olmalı.")
    # `is not None` (truthy değil) → açık n_trades=0 doğrulamaya düşer, sessizce rets.size'a
    # dönmez (0 falsy olduğu için eski kontrol onu atlıyordu).
    n_trades = int(n_trades) if n_trades is not None else int(rets.size)
    n_paths = int(n_paths)
    # Negatif/sıfır değerler numpy'da kriptik hata verir → açık doğrulama (Kural: net hata).
    if n_trades <= 0:
        raise ValueError(f"n_trades pozitif olmalı, verilen: {n_trades}")
    if n_paths <= 0:
        raise ValueError(f"n_paths pozitif olmalı, verilen: {n_paths}")

    rng = np.random.default_rng(seed)
    # Bootstrap: her yol için n_trades işlem yerine koyarak örneklenir.
    sampled = rng.choice(rets, size=(n_paths, n_trades), replace=True)
    growth = 1.0 + sampled
    # Yol boyunca kümülatif equity (ruin için min izlenir).
    equity_paths = initial_equity * np.cumprod(growth, axis=1)
    final_equity = equity_paths[:, -1]
    path_min = equity_paths.min(axis=1)

    ruin_level = initial_equity * ruin_fraction
    ruin_probability = float(np.mean(path_min <= ruin_level))

    p05 = float(np.percentile(final_equity, 5))
    p95 = float(np.percentile(final_equity, 95))
    var_95_pct = float((initial_equity - p05) / initial_equity * 100.0)
    # Expected Shortfall: en kötü %5'i SAYIYA göre seç (eşik-bağlı `<= p05` filtresi,
    # final equity'lerde eşitlik olunca >%5 örnekler → kuyruğu şişirir). Sıralayıp
    # ilk k'yı al → tam %5 (deterministik).
    k = max(1, round(n_paths * 0.05))
    worst = np.sort(final_equity)[:k]
    es_pct = float((initial_equity - worst.mean()) / initial_equity * 100.0)

    return MonteCarloResult(
        n_paths=n_paths,
        n_trades=n_trades,
        seed=int(seed),
        per_trade_mean=round(float(rets.mean()), 6),
        per_trade_std=round(float(rets.std(ddof=0)), 6),
        mean_final_equity=round(float(final_equity.mean()), 2),
        median_final_equity=round(float(np.median(final_equity)), 2),
        p05_final_equity=round(p05, 2),
        p95_final_equity=round(p95, 2),
        var_95_pct=round(var_95_pct, 2),
        expected_shortfall_pct=round(es_pct, 2),
        prob_loss=round(float(np.mean(final_equity < initial_equity)), 4),
        ruin_fraction=ruin_fraction,
        ruin_probability=round(ruin_probability, 4),
    )
