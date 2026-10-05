"""independent_backtest.py — üretim motorundan BAĞIMSIZ yeniden hesap (denetim aracı).

Amaç: ``app/trading/event_engine.py`` sonuçlarını, onun hiçbir hesap fonksiyonunu (gösterge,
kural, dolum, maliyet) içe aktarmadan, YALNIZ belgelenmiş varsayımlardan
(``docs/PROTOKOL_STRATEJI_TESTI.md`` · Motor bölümü) yeniden kurarak karşılaştırmak.

Bilinçli kısıtlar:
- Saf Python: ``pandas`` / ``numpy`` / ``app.*`` İÇE AKTARILMAZ.
- Desteklenen alt küme küçük ve açık: göstergeler EMA (ewm span, adjust=False) ve ATR (Wilder:
  ewm alpha=1/n, adjust=False; ilk TR = high−low); kurallar ``<kolon> <op> <kolon|sayı>``;
  stop ``atr_initial`` / ``fixed_pct``; hedef ``none`` / ``r_multiple`` / ``fixed_pct``;
  boyut ``fixed_fraction`` / ``risk_per_trade``.
- Takip eden stop, ATR katı hedef bu denetimin kapsamı DIŞINDADIR (kapsam sınırı raporda yazılır).

Varsayımlar (motor belgesinden, kod değil):
1. Sinyal bar t kapanışında; dolum t+1 açılışında; fiyat aleyhe (kayma + makas/2) bp.
2. Stop/hedef: giriş dolumu ± kat × SİNYAL BARININ ATR'si; R hedefi stop mesafesinin katı.
3. Giriş barında stop/hedef geçerli (gap kontrolü yok); aynı barda ikisi → stop önce.
4. Sonraki barlarda açılış stop'un ötesindeyse dolum açılıştan; hedefin ötesindeyse hedeften.
5. Komisyon her dolumda nominal × bp; fonlama her bar kapanışında qty × kapanış × bp/gün × gün.
6. Kural çıkışı: kapanışta sinyal → sonraki açılışta çıkış. Dönem sonu: son kapanışta çıkış.
7. Dönem başında pozisyon yok; dönem son barındaki giriş sinyali yok sayılır.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_RULE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*([a-z][a-z0-9_]*|-?\d+(?:\.\d+)?)\s*$"
)


@dataclass
class Bars:
    time: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]


def read_csv(path: str | Path) -> Bars:
    t, o, h, lo, c = [], [], [], [], []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            t.append(row["time"])
            o.append(float(row["open"]))
            h.append(float(row["high"]))
            lo.append(float(row["low"]))
            c.append(float(row["close"]))
    return Bars(t, o, h, lo, c)


def ema(x: list[float], n: int) -> list[float]:
    a = 2.0 / (n + 1.0)
    out = [x[0]]
    for v in x[1:]:
        out.append(out[-1] + a * (v - out[-1]))
    return out


def atr(b: Bars, n: int) -> list[float]:
    tr = [b.high[0] - b.low[0]]
    for i in range(1, len(b.close)):
        pc = b.close[i - 1]
        tr.append(max(b.high[i] - b.low[i], abs(b.high[i] - pc), abs(b.low[i] - pc)))
    out = [tr[0]]
    for v in tr[1:]:
        out.append(out[-1] + (v - out[-1]) / n)
    return out


def column(b: Bars, name: str, cache: dict[str, list[float]]) -> list[float]:
    if name in cache:
        return cache[name]
    base = {"open": b.open, "high": b.high, "low": b.low, "close": b.close}
    if name in base:
        cache[name] = base[name]
    else:
        kind, per = name.rsplit("_", 1)
        if kind == "ema":
            cache[name] = ema(b.close, int(per))
        elif kind == "atr":
            cache[name] = atr(b, int(per))
        else:
            raise ValueError(f"Bağımsız denetim kapsamı dışında gösterge: {name}")
    return cache[name]


def rule_mask(b: Bars, rules: list[str], cache: dict[str, list[float]]) -> list[bool]:
    n = len(b.close)
    if not rules:
        return [False] * n
    mask = [True] * n
    ops = {
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
    }
    for r in rules:
        m = _RULE.match(r)
        if not m:
            raise ValueError(f"Kural okunamadı: {r}")
        lhs, op, rhs = m.groups()
        left = column(b, lhs, cache)
        right: list[float] | float
        try:
            right = float(rhs)
        except ValueError:
            right = column(b, rhs, cache)
        for i in range(n):
            y = right if isinstance(right, float) else right[i]
            mask[i] = mask[i] and ops[op](left[i], y)
    return mask


@dataclass
class ITrade:
    entry_time: str
    entry_fill: float
    qty: float
    notional: float
    stop: float | None
    target: float | None
    commission: float = 0.0
    slip_spread: float = 0.0
    funding: float = 0.0
    exit_time: str = ""
    exit_fill: float = 0.0
    exit_reason: str = ""
    gross: float = 0.0
    net: float = 0.0


@dataclass
class IResult:
    trades: list[ITrade] = field(default_factory=list)
    final_equity: float = 1.0


def run(b: Bars, spec: dict[str, Any], start: str, end: str, *, bar_days: float) -> IResult:
    """Spec (TestableStrategy JSON'u) + dönem [start, end] (ISO, dahil) → işlemler."""
    times = b.time
    p0 = next(i for i, t in enumerate(times) if _iso(t) >= _iso(start))
    p1 = max(i for i, t in enumerate(times) if _iso(t) <= _iso(end))
    cache: dict[str, list[float]] = {}
    entry = rule_mask(b, spec["entry_rules"], cache)
    exit_ = rule_mask(b, spec.get("exit_rules") or [], cache)
    d = 1.0 if spec["direction"] == "long" else -1.0
    c = spec["costs"]
    adv = (float(c["slippage_bps_per_side"]) + float(c["spread_bps"]) / 2.0) / 1e4
    comm = float(c["commission_bps_per_side"]) / 1e4
    fund = float(c["funding_bps_per_day"]) / 1e4 * bar_days
    st, tp, sz = spec["stop"], spec.get("take_profit") or {"type": "none"}, spec["sizing"]
    if st["type"] not in ("none", "fixed_pct", "atr_initial"):
        raise ValueError(f"Bağımsız denetim kapsamı dışında stop: {st['type']}")
    if tp["type"] not in ("none", "fixed_pct", "r_multiple"):
        raise ValueError(f"Bağımsız denetim kapsamı dışında hedef: {tp['type']}")
    atr_s = (
        column(b, f"atr_{st.get('atr_period', 14)}", cache) if st["type"] == "atr_initial" else []
    )

    res = IResult()
    eq = 1.0
    pos: ITrade | None = None
    sig = -1
    want_exit = False

    def px_fill(px: float, side: float) -> float:
        return px * (1.0 + side * adv)

    def close(px: float, j: int, why: str) -> None:
        nonlocal pos, eq
        assert pos is not None
        f = px_fill(px, -d)
        cm = pos.qty * f * comm
        pos.commission += cm
        pos.slip_spread += pos.qty * abs(f - px)
        pos.gross = d * pos.qty * (f - pos.entry_fill)
        pos.net = pos.gross - pos.commission - pos.funding
        eq += pos.gross - cm
        pos.exit_time, pos.exit_fill, pos.exit_reason = times[j], f, why
        res.trades.append(pos)
        pos = None

    for j in range(p0, p1 + 1):
        fresh = False
        if j > p0 and pos is not None and want_exit:
            close(b.open[j], j, "kural")
        want_exit = False
        if j > p0 and pos is None and sig >= 0:
            f = px_fill(b.open[j], d)
            dist = 0.0
            if st["type"] == "fixed_pct":
                dist = f * float(st["value"])
            elif st["type"] == "atr_initial":
                dist = float(st["value"]) * atr_s[sig]
            stop = f - d * dist if st["type"] != "none" else None
            target = None
            if tp["type"] == "fixed_pct":
                target = f * (1.0 + d * float(tp["value"]))
            elif tp["type"] == "r_multiple":
                target = f + d * float(tp["value"]) * dist
            if sz["type"] == "fixed_fraction":
                notional = float(sz["fraction"]) * eq
            else:
                notional = min(
                    float(sz["risk_pct"]) * eq / (dist / f), float(sz.get("max_leverage", 1)) * eq
                )
            qty = notional / f
            cm = notional * comm
            eq -= cm
            pos = ITrade(times[j], f, qty, notional, stop, target, commission=cm)
            pos.slip_spread = qty * abs(f - b.open[j])
            fresh = True
        sig = -1
        if pos is not None:
            o, h, lo = b.open[j], b.high[j], b.low[j]
            s, t = pos.stop, pos.target
            if d > 0:
                g_s = s is not None and not fresh and o <= s
                h_s = s is not None and lo <= s
                g_t = t is not None and not fresh and o >= t
                h_t = t is not None and h >= t
            else:
                g_s = s is not None and not fresh and o >= s
                h_s = s is not None and h >= s
                g_t = t is not None and not fresh and o <= t
                h_t = t is not None and lo <= t
            if g_s:
                close(o, j, "gap_stop")
            elif h_s:
                assert s is not None
                close(s, j, "stop")
            elif g_t or h_t:
                assert t is not None
                close(t, j, "hedef")
        if pos is not None:
            fc = pos.qty * b.close[j] * fund
            pos.funding += fc
            eq -= fc
            if j == p1:
                close(b.close[j], j, "donem_sonu")
            elif exit_[j]:
                want_exit = True
        if pos is None and entry[j] and j < p1:
            sig = j
    res.final_equity = eq
    return res


def _iso(t: str) -> str:
    """Karşılaştırılabilir UTC damga (ör. '2024-01-01T00:00:00+00:00' → '2024-01-01T00:00:00')."""
    t = t.replace("Z", "+00:00")
    if t.endswith("+00:00"):
        t = t[:-6]
    return t


def compare_trades(prod: list[dict[str, Any]], ind: list[ITrade], *, n: int = 10) -> dict:
    """İşlem bazında fark tablosu (en az ``n`` ya da hepsi)."""
    rows = []
    k = min(len(prod), len(ind))
    pick = list(range(k)) if k <= n else _spread(k, n)
    worst = 0.0
    for i in pick:
        p, q = prod[i], ind[i]
        diff = {
            "entry_time_eq": _iso(p["entry_time"]) == _iso(q.entry_time),
            "exit_time_eq": _iso(p["exit_time"]) == _iso(q.exit_time),
            "reason_eq": p["exit_reason"] == q.exit_reason,
            "entry_fill": p["entry_fill"] - q.entry_fill,
            "exit_fill": p["exit_fill"] - q.exit_fill,
            "qty": p["qty"] - q.qty,
            "cost": (p["commission"] + p["slip_spread_cost"] + p["funding"])
            - (q.commission + q.slip_spread + q.funding),
            "net_pnl": p["net_pnl"] - q.net,
        }
        rel = max(
            abs(diff["entry_fill"]) / max(1e-12, abs(q.entry_fill)),
            abs(diff["exit_fill"]) / max(1e-12, abs(q.exit_fill)),
            abs(diff["qty"]) / max(1e-12, abs(q.qty)),
            abs(diff["net_pnl"]) / max(1e-12, abs(q.notional)),
        )
        worst = max(worst, rel)
        rows.append(
            {
                "index": i,
                "entry_time": q.entry_time,
                "exit_time": q.exit_time,
                "exit_reason": q.exit_reason,
                "entry_fill": q.entry_fill,
                "exit_fill": q.exit_fill,
                "qty": q.qty,
                "cost": q.commission + q.slip_spread + q.funding,
                "net_pnl": q.net,
                "diff": diff,
            }
        )
    same_struct = all(
        r["diff"]["entry_time_eq"] and r["diff"]["exit_time_eq"] and r["diff"]["reason_eq"]
        for r in rows
    )
    # Raporlanan seçki dışında TÜM işlemler de taranır (yapı + net PnL göreli farkı).
    all_struct = len(prod) == len(ind)
    all_worst = 0.0
    for p, q in zip(prod, ind, strict=False):
        all_struct = all_struct and (
            _iso(p["entry_time"]) == _iso(q.entry_time)
            and _iso(p["exit_time"]) == _iso(q.exit_time)
            and p["exit_reason"] == q.exit_reason
        )
        all_worst = max(
            all_worst,
            abs(p["net_pnl"] - q.net) / max(1e-12, abs(q.notional)),
            abs(p["entry_fill"] - q.entry_fill) / max(1e-12, abs(q.entry_fill)),
            abs(p["exit_fill"] - q.exit_fill) / max(1e-12, abs(q.exit_fill)),
        )
    return {
        "n_prod": len(prod),
        "n_independent": len(ind),
        "compared": len(rows),
        "rows": rows,
        "same_structure": same_struct and all_struct,
        "max_rel_diff": max(worst, all_worst),
        "all_trades_checked": min(len(prod), len(ind)),
    }


def _spread(k: int, n: int) -> list[int]:
    """İlk, son ve aradan eşit aralıklı ``n`` indeks (kapsam tüm döneme yayılsın)."""
    if n <= 1:
        return [0]
    return sorted({round(i * (k - 1) / (n - 1)) for i in range(n)})


def finite(x: float) -> bool:
    return math.isfinite(x)
