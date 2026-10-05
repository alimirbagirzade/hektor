"""event_engine.py — stop, hedef, pozisyon büyüklüğü ve maliyetleri UYGULAYAN backtest (Faz 2B).

Eski ``backtester`` (vektörize, long/flat, kapanıştan kapanışa) stop'u ve boyutlandırmayı hiç
uygulamaz; sohbetten gelen strateji BU motorla test edilir. Yürütme bar bar (yol bağımlı stop
gereği döngü; göstergeler vektörizedir).

Zamanlama (sabit, rapora yazılır):
1. Sinyaller bar t KAPANIŞINDA, yalnız t ve öncesi verilerle hesaplanır.
2. Emir bar t+1 AÇILIŞINDA dolar. Aynı barda sinyal + dolum yoktur.
3. Stop/hedef seviyesi giriş dolum fiyatından ve SİNYAL BARININ ATR'sinden kurulur (karar anında
   bilinen değer).

Muhafazakâr bar içi varsayımlar:
- Giriş barında da stop ve hedef geçerlidir (giriş açılışta; barın kalan aralığı sonradadır).
- Aynı barda hem stop hem hedef dokunursa STOP önce sayılır.
- Açılış stop'un ötesindeyse (gap) dolum AÇILIŞ fiyatından (stop'tan kötü) olur.
- Açılış hedefin ötesindeyse dolum HEDEF fiyatından sayılır (daha iyi açılış kâr sayılmaz).
- Her dolumda (stop/hedef dahil): fiyat aleyhe ``kayma + makas/2`` (bp) kaydırılır ve nominal
  üzerinden komisyon kesilir. Açık pozisyonun nominali üzerinden her bar KAPANIŞINDA bar
  süresi kadar fonlama kesilir (giriş barı dahil).
- Dönem içinde açık kalan pozisyon dönemin son kapanışında maliyetle kapatılır ("dönem sonu").
- Dönem başında pozisyon yoktur; dönem öncesindeki sinyaller işlem açmaz (ısınma verisi yalnız
  gösterge hesabında kullanılır). Dönemin son barındaki giriş sinyali yok sayılır (sayılır).

``shift(1)`` tek başına sızıntı kanıtı sayılmaz: her koşuda ``prefix_invariance_check`` verinin
rastgele (seed'li) noktalardan kesilmiş öneklerinde sinyal ve ATR'lerin tam veriyle AYNI çıktığını
doğrular; uyuşmazlık = gelecekten okuma → koşu reddedilir.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.trading.indicators import compute_indicator
from app.trading.strategy_ir import is_number_literal, parse_rule
from app.trading.strategy_spec import TestableStrategy

ENGINE_VERSION = "hektor-event-1.0.0"
METRICS_VERSION = "m1"

ASSUMPTIONS = [
    "Sinyal bar kapanışında, yalnız o bar ve öncesiyle hesaplanır; dolum sonraki bar açılışında.",
    "Stop/hedef: giriş dolum fiyatı + sinyal barının ATR'si (karar anında bilinen).",
    "Giriş barında da stop/hedef geçerli; aynı barda ikisi birden → stop önce.",
    "Gap: açılış stop'un ötesindeyse dolum açılıştan (daha kötü); hedefin ötesindeyse hedeften.",
    "Her dolumda fiyat aleyhe (kayma + makas/2) kaydırılır ve nominal üzerinden komisyon kesilir.",
    "Açık nominal üzerinden her bar kapanışında bar süresi kadar fonlama kesilir.",
    "Dönem başında pozisyon yok; ısınma verisi yalnız göstergelerde; dönem sonunda açık "
    "pozisyon son kapanışta maliyetle kapanır.",
]

METRIC_DEFINITIONS = {
    "total_return_pct": "(dönem sonu özsermaye / başlangıç − 1) × 100; tüm maliyetler dahil",
    "max_drawdown_pct": "bar kapanışı özsermaye eğrisinde zirveden en derin düşüş (%)",
    "sharpe": "bar getirilerinin ortalaması / std × √(yıllık bar sayısı); risksiz oran 0",
    "n_trades": "kapanan işlem sayısı (dönem sonu kapanışı dahil)",
    "win_rate_pct": "net kârı > 0 olan işlem oranı (%)",
    "profit_factor": "kazanan işlemlerin net toplamı / kaybedenlerin net toplamı (kayıp yoksa "
    "tanımsız)",
    "exposure_pct": "kapanışta pozisyon taşınan bar oranı (%)",
    "costs_pct": "komisyon + kayma/makas + fonlama, başlangıç özsermayesinin yüzdesi",
}


@dataclass
class Trade:
    direction: str
    entry_time: str
    entry_fill: float
    exit_time: str = ""
    exit_fill: float = 0.0
    exit_reason: str = ""
    qty: float = 0.0
    notional: float = 0.0
    stop_at_entry: float | None = None
    tp_at_entry: float | None = None
    commission: float = 0.0
    slip_spread_cost: float = 0.0
    funding: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    bars_held: int = 0


@dataclass
class RunResult:
    metrics: dict[str, Any]
    trades: list[Trade]
    equity: pd.Series
    window: dict[str, Any]
    counters: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "window": self.window,
            "counters": self.counters,
            "trades": [asdict(t) for t in self.trades],
        }


# ── göstergeler + sinyaller ──────────────────────────────────────────────────


def _atr_cols(spec: TestableStrategy) -> set[str]:
    cols = set()
    if spec.stop.type in ("atr_initial", "atr_trailing"):
        cols.add(f"atr_{spec.stop.atr_period}")
    if spec.take_profit.type == "atr_multiple":
        cols.add(f"atr_{spec.take_profit.atr_period}")
    return cols


def compute_columns(df: pd.DataFrame, spec: TestableStrategy) -> pd.DataFrame:
    out = df.copy()
    for ind in spec.indicators:
        out[ind.column] = compute_indicator(ind.name, df, ind.period)
    for col in (spec.required_columns() | _atr_cols(spec)) - set(out.columns):
        name, period = col.rsplit("_", 1)
        out[col] = compute_indicator(name.upper(), df, int(period))
    return out


_OPS = {
    "<": np.less,
    "<=": np.less_equal,
    ">": np.greater,
    ">=": np.greater_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def _rules_mask(cols: pd.DataFrame, rules: list[str]) -> np.ndarray:
    if not rules:
        return np.zeros(len(cols), dtype=bool)
    mask = np.ones(len(cols), dtype=bool)
    for rule in rules:
        lhs, op, rhs = parse_rule(rule)
        left = cols[lhs].to_numpy(dtype=float)
        right = float(rhs) if is_number_literal(rhs) else cols[rhs].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            res = _OPS[op](left, right)
        # NaN (ısınma) karşılaştırması sinyal ÜRETMEZ.
        valid = ~np.isnan(left) & (True if isinstance(right, float) else ~np.isnan(right))
        mask &= res & valid
    return mask


def signals(cols: pd.DataFrame, spec: TestableStrategy) -> tuple[np.ndarray, np.ndarray]:
    return _rules_mask(cols, spec.entry_rules), _rules_mask(cols, spec.exit_rules)


def prefix_invariance_check(
    df: pd.DataFrame, spec: TestableStrategy, *, seed: int = 42, n_cuts: int = 4
) -> dict[str, Any]:
    """Önek kesildiğinde sinyal/ATR değişiyorsa hesap gelecekten okuyor demektir."""
    n = len(df)
    full = compute_columns(df, spec)
    f_entry, f_exit = signals(full, spec)
    rng = np.random.default_rng(seed)
    lo = max(2, int(n * 0.2))
    cuts = sorted({int(x) for x in rng.integers(lo, n, size=n_cuts)} | {n - 1})
    atr_cols = sorted(_atr_cols(spec))
    for k in cuts:
        part = compute_columns(df.iloc[:k], spec)
        p_entry, p_exit = signals(part, spec)
        if not (np.array_equal(p_entry, f_entry[:k]) and np.array_equal(p_exit, f_exit[:k])):
            return {"ok": False, "seed": seed, "cuts": cuts, "mismatch_cut": k, "what": "sinyal"}
        for col in atr_cols:
            a = part[col].to_numpy(dtype=float)
            b = full[col].to_numpy(dtype=float)[:k]
            if not np.allclose(a, b, equal_nan=True, rtol=1e-12, atol=1e-12):
                return {"ok": False, "seed": seed, "cuts": cuts, "mismatch_cut": k, "what": col}
    return {"ok": True, "seed": seed, "cuts": cuts}


# ── yürütme ──────────────────────────────────────────────────────────────────


def _bars_per_year(tf: str) -> int:
    from app.trading.backtester import _BARS_PER_YEAR

    return _BARS_PER_YEAR[tf]


def _bar_days(tf: str) -> float:
    from app.trading.data_quality import _TF_SECONDS

    return _TF_SECONDS[tf] / 86400.0


def run(
    df: pd.DataFrame,
    spec: TestableStrategy,
    *,
    start: Any = None,
    end: Any = None,
    initial_equity: float = 1.0,
) -> RunResult:
    """``[start, end]`` döneminde stratejiyi yürüt (göstergeler tüm önceki veriyle ısınır)."""
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
        raise ValueError("Veri artan zaman indeksli olmalı (data_quality.load_checked_csv).")
    cols = compute_columns(df, spec)
    entry_sig, exit_sig = signals(cols, spec)
    idx = df.index
    p0 = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start), side="left"))
    p1 = len(df) - 1 if end is None else int(idx.searchsorted(pd.Timestamp(end), side="right")) - 1
    if p1 - p0 < 2:
        raise ValueError("Dönem çok kısa (en az 3 bar).")

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr_stop = (
        cols[f"atr_{spec.stop.atr_period}"].to_numpy(dtype=float)
        if spec.stop.type in ("atr_initial", "atr_trailing")
        else None
    )
    atr_tp = (
        cols[f"atr_{spec.take_profit.atr_period}"].to_numpy(dtype=float)
        if spec.take_profit.type == "atr_multiple"
        else None
    )

    d = 1.0 if spec.direction == "long" else -1.0
    costs = spec.costs
    # CostModel doğrulayıcısı None'u reddeder; ``or 0.0`` yalnız tip denetleyicisi içindir.
    adv = ((costs.slippage_bps_per_side or 0.0) + (costs.spread_bps or 0.0) / 2.0) / 1e4
    comm = (costs.commission_bps_per_side or 0.0) / 1e4
    fund_per_bar = (costs.funding_bps_per_day or 0.0) / 1e4 * _bar_days(spec.timeframe)

    def fill(px: float, side: float) -> float:
        return px * (1.0 + side * adv)

    realized = float(initial_equity)
    equity = np.empty(p1 - p0 + 1)
    trades: list[Trade] = []
    counters = {
        "ignored_last_bar_entry": 0,
        "skipped_entry_no_atr": 0,
        "stop_exits": 0,
        "gap_stop_exits": 0,
        "same_bar_stop_first": 0,
        "tp_exits": 0,
        "rule_exits": 0,
        "end_exits": 0,
    }
    pos: Trade | None = None
    qty = 0.0
    stop: float | None = None
    tp: float | None = None
    pending_entry = -1  # sinyal barı
    pending_exit = False
    held_bars = 0

    def close_pos(px: float, j: int, reason: str) -> None:
        nonlocal pos, realized, qty, stop, tp, held_bars
        assert pos is not None
        side = -d
        f = fill(px, side)
        commission = qty * f * comm
        gross = d * qty * (f - pos.entry_fill)
        realized += gross - commission
        pos.exit_time = idx[j].isoformat()
        pos.exit_fill = f
        pos.exit_reason = reason
        pos.commission += commission
        pos.slip_spread_cost += qty * abs(f - px)
        pos.gross_pnl = d * qty * (f - pos.entry_fill)
        pos.net_pnl = pos.gross_pnl - pos.commission - pos.funding
        pos.bars_held = held_bars
        trades.append(pos)
        pos, qty, stop, tp, held_bars = None, 0.0, None, None, 0

    for j in range(p0, p1 + 1):
        entered_now = False
        # 1) açılış emirleri
        if j > p0 and pos is not None and pending_exit:
            close_pos(o[j], j, "kural")
            counters["rule_exits"] += 1
        pending_exit = False
        if j > p0 and pos is None and pending_entry >= 0:
            sb = pending_entry
            f = fill(o[j], d)
            stop_lvl: float | None = None
            dist = 0.0
            ok = True
            if spec.stop.type == "fixed_pct":
                dist = f * float(spec.stop.value or 0)
            elif spec.stop.type in ("atr_initial", "atr_trailing"):
                a = float(atr_stop[sb]) if atr_stop is not None else math.nan
                if not math.isfinite(a) or a <= 0:
                    ok = False
                else:
                    dist = float(spec.stop.value or 0) * a
            if spec.stop.type != "none":
                stop_lvl = f - d * dist
            tp_lvl: float | None = None
            if ok and spec.take_profit.type == "fixed_pct":
                tp_lvl = f * (1.0 + d * float(spec.take_profit.value or 0))
            elif ok and spec.take_profit.type == "atr_multiple":
                a2 = float(atr_tp[sb]) if atr_tp is not None else math.nan
                if math.isfinite(a2) and a2 > 0:
                    tp_lvl = f + d * float(spec.take_profit.value or 0) * a2
                else:
                    ok = False
            elif ok and spec.take_profit.type == "r_multiple":
                tp_lvl = f + d * float(spec.take_profit.value or 0) * dist
            if not ok:
                counters["skipped_entry_no_atr"] += 1
            else:
                eq = realized
                if spec.sizing.type == "fixed_fraction":
                    notional = float(spec.sizing.fraction or 0) * eq
                else:
                    risk = float(spec.sizing.risk_pct or 0) * eq
                    notional = min(risk / (dist / f), spec.sizing.max_leverage * eq)
                qty = notional / f
                commission = notional * comm
                realized -= commission
                pos = Trade(
                    direction=spec.direction,
                    entry_time=idx[j].isoformat(),
                    entry_fill=f,
                    qty=qty,
                    notional=notional,
                    stop_at_entry=stop_lvl,
                    tp_at_entry=tp_lvl,
                    commission=commission,
                    slip_spread_cost=qty * abs(f - o[j]),
                )
                stop, tp = stop_lvl, tp_lvl
                entered_now = True
        pending_entry = -1

        # 2) bar içi stop / hedef
        if pos is not None:
            if d > 0:
                gap_stop = stop is not None and not entered_now and o[j] <= stop
                hit_stop = stop is not None and lo[j] <= stop
                gap_tp = tp is not None and not entered_now and o[j] >= tp
                hit_tp = tp is not None and h[j] >= tp
            else:
                gap_stop = stop is not None and not entered_now and o[j] >= stop
                hit_stop = stop is not None and h[j] >= stop
                gap_tp = tp is not None and not entered_now and o[j] <= tp
                hit_tp = tp is not None and lo[j] <= tp
            if gap_stop:
                close_pos(o[j], j, "gap_stop")
                counters["gap_stop_exits"] += 1
            elif hit_stop:
                if hit_tp:
                    counters["same_bar_stop_first"] += 1
                assert stop is not None
                close_pos(stop, j, "stop")
                counters["stop_exits"] += 1
            elif gap_tp or hit_tp:
                assert tp is not None
                close_pos(tp, j, "hedef")
                counters["tp_exits"] += 1

        # 3) kapanış: fonlama, takip eden stop, son bar, sinyaller
        if pos is not None:
            held_bars += 1
            f_cost = qty * c[j] * fund_per_bar
            realized -= f_cost
            pos.funding += f_cost
            if spec.stop.type == "atr_trailing" and atr_stop is not None and stop is not None:
                a = float(atr_stop[j])
                if math.isfinite(a) and a > 0:
                    cand = c[j] - d * float(spec.stop.value or 0) * a
                    stop = max(stop, cand) if d > 0 else min(stop, cand)
            if j == p1:
                close_pos(c[j], j, "donem_sonu")
                counters["end_exits"] += 1
            elif exit_sig[j]:
                pending_exit = True
        if pos is None and entry_sig[j] and j >= p0:
            if j < p1:
                pending_entry = j
            else:
                counters["ignored_last_bar_entry"] += 1
        open_pnl = d * qty * (c[j] - pos.entry_fill) if pos is not None else 0.0
        equity[j - p0] = realized + open_pnl

    eq = pd.Series(equity, index=idx[p0 : p1 + 1])
    metrics = _metrics(eq, trades, spec, initial_equity, counters)
    window = {
        "start": idx[p0].isoformat(),
        "end": idx[p1].isoformat(),
        "n_bars": p1 - p0 + 1,
        "warmup_bars_before": p0,
    }
    return RunResult(metrics=metrics, trades=trades, equity=eq, window=window, counters=counters)


def _metrics(
    eq: pd.Series,
    trades: list[Trade],
    spec: TestableStrategy,
    initial: float,
    counters: dict[str, int],
) -> dict[str, Any]:
    rets = eq.pct_change().dropna()
    std = float(rets.std()) if len(rets) > 1 else 0.0
    sharpe = (
        float(rets.mean()) / std * math.sqrt(_bars_per_year(spec.timeframe))
        if std > 0 and math.isfinite(std)
        else 0.0
    )
    dd = float((eq / eq.cummax() - 1.0).min()) * 100 if len(eq) else 0.0
    pnl = [t.net_pnl for t in trades]
    wins = sum(p for p in pnl if p > 0)
    losses = -sum(p for p in pnl if p < 0)
    pf: float | None = round(wins / losses, 4) if losses > 0 else None
    costs = sum(t.commission + t.slip_spread_cost + t.funding for t in trades)
    exposure = float(sum(t.bars_held for t in trades)) / max(1, len(eq)) * 100
    return {
        "total_return_pct": round((float(eq.iloc[-1]) / initial - 1.0) * 100, 4),
        "max_drawdown_pct": round(dd, 4),
        "sharpe": round(sharpe, 4),
        "n_trades": len(trades),
        "win_rate_pct": round(sum(1 for p in pnl if p > 0) / len(pnl) * 100, 4) if pnl else 0.0,
        "profit_factor": pf,
        "exposure_pct": round(exposure, 4),
        "costs_pct": round(costs / initial * 100, 4),
        "commission_pct": round(sum(t.commission for t in trades) / initial * 100, 4),
        "slip_spread_pct": round(sum(t.slip_spread_cost for t in trades) / initial * 100, 4),
        "funding_pct": round(sum(t.funding for t in trades) / initial * 100, 4),
    }
