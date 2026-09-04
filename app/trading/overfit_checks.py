"""Overfit / robustness checks for backtest results.

These are guardrails, not guarantees. They encode the project's skepticism:
too few trades, implausible profit factor, or catastrophic drawdown all reduce
confidence regardless of headline return.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.trading.backtester import BacktestMetrics, run_backtest
from app.trading.strategy_ir import StrategyIR


@dataclass
class OverfitReport:
    warnings: list[str]
    in_sample: BacktestMetrics | None = None
    out_sample: BacktestMetrics | None = None

    @property
    def degradation_pct(self) -> float | None:
        if not (self.in_sample and self.out_sample):
            return None
        a = self.in_sample.total_return_pct
        b = self.out_sample.total_return_pct
        if a == 0:
            return None
        return round((a - b) / abs(a) * 100, 2)


def static_checks(metrics: BacktestMetrics, min_trades: int = 30) -> list[str]:
    warnings: list[str] = []
    if metrics.n_trades < min_trades:
        warnings.append(
            f"Çok az işlem ({metrics.n_trades} < {min_trades}); istatistiksel anlam zayıf."
        )
    if metrics.profit_factor > 5 and metrics.n_trades < 100:
        warnings.append(
            f"Şüpheli yüksek profit factor ({metrics.profit_factor}); overfit ihtimali."
        )
    if metrics.max_drawdown_pct < -50:
        warnings.append(f"Aşırı drawdown ({metrics.max_drawdown_pct}%); risk kabul edilemez.")
    if metrics.sharpe > 4:
        warnings.append(f"Gerçekçi olmayan Sharpe ({metrics.sharpe}); look-ahead/leak kontrol et.")
    return warnings


def in_out_of_sample(
    df: pd.DataFrame, ir: StrategyIR, split: float = 0.7, min_trades: int = 30
) -> OverfitReport:
    n = len(df)
    cut = int(n * split)
    if cut <= 0 or cut >= n:
        # Boş IS veya OOS dilimi: run_backtest boş df'te çökmez ama TÜM-SIFIR metrik üretir →
        # geçerli bir "0% OOS" gibi görünüp sahte-OOS sonucuna yol açardı. Açıkça yetersiz-veri
        # uyarısı ver, yanıltıcı backtest koşma (in/out_sample=None → degradation_pct=None).
        return OverfitReport(
            warnings=[
                f"Veri IS/OOS bölünmesi için yetersiz (n={n}, split={split} → "
                f"IS={cut} / OOS={n - cut} bar); overfit denetimi atlandı."
            ],
            in_sample=None,
            out_sample=None,
        )
    in_df, out_df = df.iloc[:cut], df.iloc[cut:]

    in_res = run_backtest(in_df, ir)
    out_res = run_backtest(out_df, ir)

    # BUG-M7 fix: IS metrikleri de kontrol edilmeli — IS'in kendisi şüpheliyse
    # OOS'a hiç geçilmeden uyarı verilmeli.
    is_warnings = [f"[IS] {w}" for w in static_checks(in_res.metrics, min_trades=min_trades)]
    oos_warnings = static_checks(out_res.metrics, min_trades=min_trades)
    warnings = is_warnings + oos_warnings

    deg_in = in_res.metrics.total_return_pct
    deg_out = out_res.metrics.total_return_pct
    if deg_in > 0 and deg_out < 0:
        warnings.append("Örneklem-içi pozitif, örneklem-dışı negatif: klasik overfit işareti.")

    return OverfitReport(warnings=warnings, in_sample=in_res.metrics, out_sample=out_res.metrics)
