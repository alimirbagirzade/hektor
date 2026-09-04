"""Pozisyon büyüklüğü ve risk yönetimi hesaplamaları.

Üç yöntem:
  1. Kelly kriteri  — optimal büyüklük (tam + yarı)
  2. Drawdown ölçekleme — kötü dönemde küçül
  3. Sabit risk       — işlem başına hedef kayıp yüzdesi

Tüm hesaplamalar *öneri* niteliğindedir; gerçek işlem kararı
kullanıcıya aittir. Bu modul canlı emir üretmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ─── Sonuç yapıları ───────────────────────────────────────────────────────────


@dataclass
class KellyResult:
    win_rate: float  # kazanma oranı (0-1)
    avg_win: float  # ortalama kazançlı işlem getirisi
    avg_loss: float  # ortalama zararlı işlem getirisi (mutlak)
    odds: float  # b = avg_win / avg_loss
    full_kelly: float  # tam Kelly fraksiyonu (0-1)
    half_kelly: float  # yarı Kelly (daha güvenli)
    quarter_kelly: float  # çeyrek Kelly (çok muhafazakâr)
    capped_kelly: float  # max %25 ile sınırlanmış önerilen fraksiyon


@dataclass
class DrawdownScaleResult:
    current_drawdown_pct: float  # anlık dd (negatif)
    max_allowed_pct: float  # eşik (örn. -20)
    scale_factor: float  # 0-1: pozisyona çarpılan katsayı
    in_drawdown_zone: bool  # eşiği aştı mı


@dataclass
class FixedRiskResult:
    equity: float  # toplam sermaye ($)
    risk_per_trade_pct: float  # işlem başına riske atılacak % (örn. 1.0)
    stop_distance_pct: float  # stop noktasına mesafe % (örn. 2.0 → %2 zarar)
    position_size_pct: float  # sermayenin yüzdesi olarak pozisyon büyüklüğü
    position_size_usd: float  # mutlak $ miktarı


@dataclass
class RiskReport:
    """Bir backtest sonucundan üretilen tam risk raporu."""

    strategy_name: str
    n_trades: int
    kelly: KellyResult
    drawdown_scale: DrawdownScaleResult
    fixed_risk: FixedRiskResult
    warnings: list[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "n_trades": self.n_trades,
            "kelly": {
                "win_rate": self.kelly.win_rate,
                "avg_win": self.kelly.avg_win,
                "avg_loss": self.kelly.avg_loss,
                "odds": self.kelly.odds,
                "full_kelly": self.kelly.full_kelly,
                "half_kelly": self.kelly.half_kelly,
                "quarter_kelly": self.kelly.quarter_kelly,
                "capped_kelly": self.kelly.capped_kelly,
            },
            "drawdown_scale": {
                "current_drawdown_pct": self.drawdown_scale.current_drawdown_pct,
                "max_allowed_pct": self.drawdown_scale.max_allowed_pct,
                "scale_factor": self.drawdown_scale.scale_factor,
                "in_drawdown_zone": self.drawdown_scale.in_drawdown_zone,
            },
            "fixed_risk": {
                "equity": self.fixed_risk.equity,
                "risk_per_trade_pct": self.fixed_risk.risk_per_trade_pct,
                "stop_distance_pct": self.fixed_risk.stop_distance_pct,
                "position_size_pct": self.fixed_risk.position_size_pct,
                "position_size_usd": self.fixed_risk.position_size_usd,
            },
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


# ─── Hesaplama fonksiyonları ──────────────────────────────────────────────────


def compute_kelly(
    trade_returns: pd.Series,
    max_fraction: float = 0.25,
) -> KellyResult:
    """Kelly fraksiyonu — girdi: işlem bazlı getiri serisi (decimals).

    Örnekler: +0.05 → %5 kazanç, -0.02 → %2 kayıp.
    """
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    p = float(len(wins)) / len(trade_returns) if len(trade_returns) > 0 else 0.0
    q = 1.0 - p
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.abs().mean()) if len(losses) > 0 else 1e-9

    b = avg_win / avg_loss if avg_loss > 0 else 0.0

    # f* = (b*p - q) / b
    full_kelly = max(0.0, (b * p - q) / b) if b > 0 else 0.0

    full_kelly = round(full_kelly, 4)
    half_kelly = round(full_kelly / 2, 4)
    quarter_kelly = round(full_kelly / 4, 4)
    capped = round(min(full_kelly, max_fraction), 4)

    return KellyResult(
        win_rate=round(p, 4),
        avg_win=round(avg_win, 4),
        avg_loss=round(avg_loss, 4),
        odds=round(b, 4),
        full_kelly=full_kelly,
        half_kelly=half_kelly,
        quarter_kelly=quarter_kelly,
        capped_kelly=capped,
    )


def compute_drawdown_scale(
    equity_curve: pd.Series,
    max_allowed_pct: float = -20.0,
) -> DrawdownScaleResult:
    """Anlık drawdown'a göre pozisyon ölçekleme faktörü.

    Mantık:
      - dd = 0           → scale = 1.0 (tam pozisyon)
      - dd = max_allowed → scale = 0.0 (pozisyon sıfır)
      - Arada lineer interpolasyon
    """
    if len(equity_curve) == 0:
        return DrawdownScaleResult(0.0, max_allowed_pct, 1.0, False)

    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    current_dd = float(dd.iloc[-1]) * 100  # yüzde

    max_allowed = min(max_allowed_pct, 0.0)  # negatif olmalı
    in_zone = current_dd < max_allowed

    if current_dd >= 0:
        scale = 1.0
    elif current_dd <= max_allowed:
        scale = 0.0
    else:
        scale = round(1.0 - (current_dd / max_allowed), 4)

    return DrawdownScaleResult(
        current_drawdown_pct=round(current_dd, 2),
        max_allowed_pct=max_allowed,
        scale_factor=scale,
        in_drawdown_zone=in_zone,
    )


def compute_fixed_risk(
    equity: float = 10_000.0,
    risk_per_trade_pct: float = 1.0,
    stop_distance_pct: float = 2.0,
) -> FixedRiskResult:
    """Sabit risk hesabı.

    Mantık: pozisyon_büyüklüğü = (equity * risk%) / stop%
    Örnek:
      equity=10000, risk=1%, stop=2%
      → 10000 * 0.01 / 0.02 = 5000 $  (sermayenin %50'si)
    """
    risk_frac = risk_per_trade_pct / 100.0
    stop_frac = stop_distance_pct / 100.0

    pos_pct = 0.0 if stop_frac <= 0 else round(min(risk_frac / stop_frac, 1.0) * 100, 2)

    pos_usd = round(equity * pos_pct / 100.0, 2)

    return FixedRiskResult(
        equity=equity,
        risk_per_trade_pct=risk_per_trade_pct,
        stop_distance_pct=stop_distance_pct,
        position_size_pct=pos_pct,
        position_size_usd=pos_usd,
    )


# ─── Ana API ─────────────────────────────────────────────────────────────────


def analyze_risk(
    strategy_name: str,
    equity_curve: pd.Series,
    position: pd.Series,
    returns: pd.Series,
    *,
    equity_usd: float = 10_000.0,
    max_dd_threshold_pct: float = -20.0,
    risk_per_trade_pct: float = 1.0,
    atr_stop_pct: float = 2.0,
) -> RiskReport:
    """Backtest sonucundan tam risk raporu üret.

    Parametreler
    ------------
    equity_curve      : kümülatif getiri serisi (run_backtest'ten)
    position          : 0/1 pozisyon serisi
    returns           : net bar başı getiri serisi
    equity_usd        : başlangıç sermayesi ($ cinsinden)
    max_dd_threshold  : bu eşiği geçince pozisyon küçültülür (örn. -20)
    risk_per_trade_pct: işlem başına riske atılacak % (sabit risk yöntemi)
    atr_stop_pct      : varsayılan stop mesafesi % (ATR/fiyat)
    """
    warnings: list[str] = []

    # İşlem başı getiri: entry bar'larının sonraki getirileri
    entry_mask = (position.shift(1).fillna(0) == 0) & (position == 1)
    trade_returns = _extract_trade_returns(position, returns)

    n_trades = int(entry_mask.sum())
    if n_trades < 30:
        warnings.append(
            f"Az işlem ({n_trades} < 30) — Kelly tahmini istatistiksel olarak güvenilmez."
        )

    kelly = compute_kelly(trade_returns)
    dd_scale = compute_drawdown_scale(equity_curve, max_dd_threshold_pct)
    fixed = compute_fixed_risk(equity_usd, risk_per_trade_pct, atr_stop_pct)

    if kelly.full_kelly <= 0:
        warnings.append("Kelly fraksiyonu ≤ 0 — strateji bu veride negatif beklenti gösteriyor.")
    if kelly.full_kelly > 0.5:
        warnings.append(
            f"Tam Kelly çok yüksek ({kelly.full_kelly:.0%}) — yarı veya çeyrek Kelly önerilir."
        )
    if dd_scale.in_drawdown_zone:
        warnings.append(
            f"Anlık drawdown ({dd_scale.current_drawdown_pct:.1f}%) eşiği aştı "
            f"({dd_scale.max_allowed_pct:.1f}%) — pozisyon küçültülmeli."
        )

    recommendation = _build_recommendation(kelly, dd_scale, fixed, warnings)

    return RiskReport(
        strategy_name=strategy_name,
        n_trades=n_trades,
        kelly=kelly,
        drawdown_scale=dd_scale,
        fixed_risk=fixed,
        warnings=warnings,
        recommendation=recommendation,
    )


def _extract_trade_returns(position: pd.Series, returns: pd.Series) -> pd.Series:
    """Her trade için bileşik NET getiriyi çıkar (giriş + ÇIKIŞ maliyeti dahil).

    Bloklar EFEKTİF pozisyon (``position.shift(1)``) üzerinden sınırlanır — ``returns``
    de bu gecikmeli pozisyondan türetilir (backtester ``_net_returns``: giriş maliyeti
    eff_pos 0→1 barında, ÇIKIŞ maliyeti eff_pos 1→0 barında yüklenir). Çıkış-maliyeti
    barı (eff_pos'un 0'a döndüğü bar) bloğa DAHİL edilir.

    Eskiden bloklar HAM ``position`` ile sınırlanıyordu; ``returns`` gecikmeli pozisyondan
    geldiği için blok bir bar kayıyordu → hem son tutuş-barı getirisi hem çıkış maliyeti
    bloğun DIŞINDA kalıyordu. Sonuç: per-trade getiri (ve dolayısıyla Kelly tahmini) çıkış
    maliyetini yok sayıp ŞİŞİYORDU (CLAUDE.md kural 3 — maliyeti yok sayma). Headline
    metrikler (Sharpe/equity) etkilenmiyordu; yalnız bu Kelly dekompozisyonu hatalıydı.

    Seri sonunda hâlâ açık pozisyon (çıkış barı yok) → maliyetsiz kapatılır; gerçekleşmemiş
    çıkışın bilinen köşe durumu (backtester ile tutarlı).
    """
    eff_pos = position.shift(1).fillna(0.0)
    if not (eff_pos == 1).any():
        return pd.Series(dtype=float)

    trade_blocks: list[float] = []
    block_rets: list[float] = []
    in_block = False
    for i in range(len(eff_pos)):
        active = eff_pos.iloc[i] == 1
        if active:
            if not in_block:
                in_block = True
                block_rets = []
            block_rets.append(returns.iloc[i])
        elif in_block:
            # eff_pos 1→0: bu bar çıkış maliyetini taşır → bloğa dahil et, sonra kapat.
            block_rets.append(returns.iloc[i])
            trade_blocks.append(float(np.prod([1.0 + r for r in block_rets]) - 1.0))
            in_block = False
            block_rets = []
    if in_block and block_rets:
        # Veri sonunda açık pozisyon: çıkış maliyeti henüz gerçekleşmedi → maliyetsiz kapat.
        trade_blocks.append(float(np.prod([1.0 + r for r in block_rets]) - 1.0))

    return pd.Series(trade_blocks, dtype=float) if trade_blocks else pd.Series(dtype=float)


def _build_recommendation(
    kelly: KellyResult,
    dd_scale: DrawdownScaleResult,
    fixed: FixedRiskResult,
    warnings: list[str],
) -> str:
    if kelly.full_kelly <= 0:
        return "Strateji negatif beklentili — pozisyon önerilmez."

    effective = kelly.half_kelly * dd_scale.scale_factor
    effective = round(effective, 4)

    lines = [
        f"Önerilen pozisyon fraksiyonu (yarı Kelly × drawdown skalası): "
        f"{effective:.1%} (sermayenin {effective * 100:.1f}%'i)",
        f"Sabit risk yöntemiyle: {fixed.position_size_pct:.1f}% sermaye "
        f"(= {fixed.position_size_usd:,.0f} $ varsayılan sermayeye göre)",
    ]
    if dd_scale.in_drawdown_zone:
        lines.append(f"⚠ Drawdown bölgesi — ölçek faktörü {dd_scale.scale_factor:.2f} uygulandı.")
    if warnings:
        lines.append("Uyarılar: " + "; ".join(warnings))
    return "  ".join(lines)
