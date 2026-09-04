"""reward_signal.py — Backtest sonuçlarından doğrulanabilir ödül sinyali hesaplar.

Her kriter otomatik doğrulanabilir (LLM değerlendirmesi gerekmez):
  - execution_ok   : backtest hatasız çalıştı mı?
  - trade_count_ok : yeterli işlem sayısı (≥ MIN_TRADES)?
  - sharpe_ok      : Sharpe oranı kabul edilebilir (≥ MIN_SHARPE)?
  - drawdown_ok    : maksimum düşüş sınır içinde (≤ MAX_DD_PCT)?
  - return_ok      : pozitif toplam getiri?
  - win_rate_ok    : kazanma oranı minimum eşiğin üstünde (≥ MIN_WIN_RATE)?

Bileşik skor = ağırlıklı ortalama → [0.0, 1.0]

DPO için:
  composite ≥ REWARD_PASS   → "chosen"
  composite ≤ REWARD_REJECT → "rejected"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Eşikler
MIN_TRADES = 10
MIN_SHARPE = 0.8
MAX_DD_PCT = 35.0
MIN_WIN_RATE = 0.38
REWARD_PASS = 0.65
REWARD_REJECT = 0.35

# Kriter ağırlıkları (toplam 1.0)
_WEIGHTS: dict[str, float] = {
    "execution_ok": 0.20,
    "trade_count_ok": 0.20,
    "sharpe_ok": 0.25,
    "drawdown_ok": 0.15,
    "return_ok": 0.10,
    "win_rate_ok": 0.10,
}


@dataclass
class RewardCriteria:
    execution_ok: float = 0.0
    trade_count_ok: float = 0.0
    sharpe_ok: float = 0.0
    drawdown_ok: float = 0.0
    return_ok: float = 0.0
    win_rate_ok: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def composite(self) -> float:
        return round(sum(getattr(self, k) * w for k, w in _WEIGHTS.items()), 4)

    @property
    def label(self) -> str:
        if self.composite >= REWARD_PASS:
            return "chosen"
        if self.composite <= REWARD_REJECT:
            return "rejected"
        return "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite": self.composite,
            "label": self.label,
            "execution_ok": self.execution_ok,
            "trade_count_ok": self.trade_count_ok,
            "sharpe_ok": self.sharpe_ok,
            "drawdown_ok": self.drawdown_ok,
            "return_ok": self.return_ok,
            "win_rate_ok": self.win_rate_ok,
            "notes": self.notes,
        }


def compute_reward(
    metrics: dict[str, Any],
    verdict: str = "inconclusive",
    had_error: bool = False,
) -> RewardCriteria:
    """Backtest metriklerinden RewardCriteria hesapla."""
    c = RewardCriteria()

    if had_error or verdict == "error":
        c.notes.append("Backtest hatası — execution_ok=0")
        return c

    c.execution_ok = 1.0

    n_trades = int(metrics.get("n_trades", 0))
    if n_trades >= MIN_TRADES:
        c.trade_count_ok = 1.0
    else:
        c.trade_count_ok = round(n_trades / MIN_TRADES, 2)
        c.notes.append(f"Az işlem: {n_trades} < {MIN_TRADES}")

    # Backtester to_dict() "sharpe" anahtarı üretir; eski "sharpe_ratio" geriye-uyum için.
    sharpe = float(metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0)))
    if sharpe >= MIN_SHARPE:
        c.sharpe_ok = min(1.0, round(sharpe / (MIN_SHARPE * 2), 2))
    elif sharpe > 0:
        c.sharpe_ok = round(sharpe / MIN_SHARPE * 0.5, 2)
    else:
        c.notes.append(f"Negatif/sıfır Sharpe: {sharpe:.2f}")

    dd = abs(float(metrics.get("max_drawdown_pct", 0.0)))
    if dd <= MAX_DD_PCT:
        c.drawdown_ok = round(1.0 - dd / MAX_DD_PCT * 0.5, 2)
    else:
        c.drawdown_ok = 0.0
        c.notes.append(f"Yüksek drawdown: {dd:.1f}% > {MAX_DD_PCT}%")

    ret = float(metrics.get("total_return_pct", 0.0))
    c.return_ok = 1.0 if ret > 0 else 0.0
    if ret <= 0:
        c.notes.append(f"Negatif/sıfır getiri: {ret:.1f}%")

    # Backtester "win_rate_pct" (0-100) üretir; eşik/normalizasyon kesir (0-1) bekler.
    wr = float(metrics.get("win_rate_pct", metrics.get("win_rate", 0.0)))
    if wr > 1.0:
        wr /= 100.0
    if wr >= MIN_WIN_RATE:
        c.win_rate_ok = min(1.0, round(wr / 0.6, 2))
    else:
        c.win_rate_ok = round(wr / MIN_WIN_RATE * 0.5, 2)
        c.notes.append(f"Düşük win_rate: {wr:.1%}")

    return c


def build_preference_pairs(
    scored: list[tuple[str, RewardCriteria]],
    min_gap: float = 0.25,
) -> list[dict[str, str]]:
    """(session_id, RewardCriteria) listesinden DPO çifti üret.

    Her chosen–rejected çifti, composite farkı ≥ min_gap ise eşleştirilir.
    """
    chosen = [(sid, rc) for sid, rc in scored if rc.label == "chosen"]
    rejected = [(sid, rc) for sid, rc in scored if rc.label == "rejected"]
    pairs = []
    for c_id, c_rc in chosen:
        for r_id, r_rc in rejected:
            if c_rc.composite - r_rc.composite >= min_gap:
                pairs.append({"chosen_id": c_id, "rejected_id": r_id})
    return pairs
