"""Faz 2B — stop/hedef/boyut/maliyet uygulayan motor + veri denetimi.

Elle kurulmuş barlarla: sinyal kapanışta → dolum sonraki açılışta; giriş barında stop; aynı barda
stop+hedef → stop önce; gap'te açılıştan dolum; hedef gap'inde hedef fiyatı; maliyet birimleri
(bp) ve fonlama; short simetrisi; takip eden stop yalnız lehe; ısınma verisi işlem açmaz ve OOS
getirisine karışmaz; önek değişmezliği gelecekten okumayı yakalar.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.trading import event_engine as ee
from app.trading.data_quality import DataQualityError, load_checked_csv
from app.trading.strategy_spec import TestableStrategy

ZERO = "test: maliyetsiz mekanik doğrulama"


def _spec(**kw) -> TestableStrategy:
    base = {
        "name": "t",
        "market": "TEST",
        "timeframe": "1h",
        "direction": "long",
        "entry_rules": ["close > 100"],
        "exit_rules": [],
        "stop": {"type": "fixed_pct", "value": 0.05},
        "sizing": {"type": "fixed_fraction", "fraction": 1.0},
        "costs": {
            "commission_bps_per_side": 0,
            "slippage_bps_per_side": 0,
            "spread_bps": 0,
            "funding_bps_per_day": 0,
            "zero_reasons": {
                "commission_bps_per_side": ZERO,
                "slippage_bps_per_side": ZERO,
                "spread_bps": ZERO,
                "funding_bps_per_day": ZERO,
            },
        },
    }
    base.update(kw)
    return TestableStrategy.model_validate(base)


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h", tz="UTC")
    o, h, lo, c = zip(*rows, strict=True)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 1.0}, index=idx)


FLAT = (90.0, 91.0, 89.0, 90.0)  # sinyal yok (close ≤ 100)
SIG = (99.0, 101.0, 98.0, 101.0)  # kapanış 101 → giriş sinyali


def test_signal_at_close_fills_next_open() -> None:
    df = _bars([FLAT, SIG, (102, 103, 101, 102), (102, 103, 101, 102)])
    r = ee.run(df, _spec(entry_rules=["close > 100"]))
    t = r.trades[0]
    assert t.entry_time == df.index[2].isoformat() and t.entry_fill == 102.0
    assert t.exit_reason == "donem_sonu" and t.exit_fill == 102.0


def test_stop_in_entry_bar_and_same_bar_stop_first() -> None:
    # Giriş barı: açılış 100, düşük 94 → %5 stop (95) AYNI barda tetiklenir.
    df = _bars([FLAT, SIG, (100, 112, 94, 99), FLAT])
    spec = _spec(take_profit={"type": "fixed_pct", "value": 0.1})  # hedef 110 da dokundu
    r = ee.run(df, spec)
    t = r.trades[0]
    assert t.exit_reason == "stop" and t.exit_fill == pytest.approx(95.0)
    assert r.counters["same_bar_stop_first"] == 1


def test_gap_through_stop_fills_at_open() -> None:
    df = _bars([FLAT, SIG, (100, 101, 99, 100), (80, 82, 79, 81), FLAT])
    r = ee.run(df, _spec(entry_rules=["close > 100"]))
    t = r.trades[0]
    assert t.exit_reason == "gap_stop" and t.exit_fill == 80.0  # stop 95'ten KÖTÜ


def test_tp_gap_counts_at_target_not_better_open() -> None:
    df = _bars([FLAT, SIG, (100, 101, 99, 100), (130, 131, 129, 130), FLAT])
    r = ee.run(df, _spec(take_profit={"type": "fixed_pct", "value": 0.1}))
    t = r.trades[0]
    assert t.exit_reason == "hedef" and t.exit_fill == pytest.approx(110.0)


def test_costs_units_and_funding() -> None:
    costs = {
        "commission_bps_per_side": 10,  # %0.1
        "slippage_bps_per_side": 5,
        "spread_bps": 10,  # dolum başına 5 bp
        "funding_bps_per_day": 24,  # 1h bar → 1 bp / bar
    }
    df = _bars([FLAT, SIG, (100, 100, 100, 100), (100, 100, 100, 100)])
    r = ee.run(df, _spec(costs=costs, stop={"type": "none"}))
    t = r.trades[0]
    assert t.entry_fill == pytest.approx(100 * (1 + 0.001))  # 5+5 bp aleyhe
    assert t.exit_fill == pytest.approx(100 * (1 - 0.001))
    qty = 1.0 / t.entry_fill
    assert t.commission == pytest.approx(0.001 + qty * t.exit_fill * 0.001)
    assert t.funding == pytest.approx(2 * qty * 100 * 0.0001)  # giriş barı + son bar
    assert r.metrics["total_return_pct"] < 0  # düz fiyatta maliyet kaybı
    assert r.metrics["costs_pct"] == pytest.approx(
        (t.commission + t.slip_spread_cost + t.funding) * 100, abs=1e-4
    )  # metrik 4 haneye yuvarlanır


def test_missing_cost_rejected_zero_needs_reason() -> None:
    with pytest.raises(ValueError, match="Maliyet eksik"):
        _spec(costs={"commission_bps_per_side": 1, "slippage_bps_per_side": 1})
    with pytest.raises(ValueError, match="gerekçe"):
        _spec(
            costs={
                "commission_bps_per_side": 0,
                "slippage_bps_per_side": 1,
                "spread_bps": 1,
                "funding_bps_per_day": 1,
            }
        )
    with pytest.raises(ValueError, match="negatif"):
        _spec(
            costs={
                "commission_bps_per_side": -1,
                "slippage_bps_per_side": 1,
                "spread_bps": 1,
                "funding_bps_per_day": 1,
            }
        )


def test_short_mirror_stop() -> None:
    sig = (101, 102, 99, 99)  # close < 100 → short sinyali
    df = _bars([(110, 111, 109, 110), sig, (100, 106, 99, 104), (104, 105, 103, 104)])
    r = ee.run(df, _spec(direction="short", entry_rules=["close < 100"]))
    t = r.trades[0]
    assert t.exit_reason == "stop" and t.exit_fill == pytest.approx(105.0)
    assert t.net_pnl == pytest.approx(-(1.0 / 100) * 5, rel=1e-9)


def test_trailing_stop_only_moves_favourably() -> None:
    spec = _spec(stop={"type": "atr_trailing", "value": 1.0, "atr_period": 1})
    rows = [FLAT, SIG] + [(100 + i, 101 + i, 99.5 + i, 100.5 + i) for i in range(6)]
    rows += [(105, 105.2, 90, 91)]  # sert düşüş → takip eden stop'ta çıkış
    df = _bars(rows)
    r = ee.run(df, spec)
    t = r.trades[0]
    assert t.exit_reason == "stop" and t.exit_fill > t.stop_at_entry  # stop yukarı taşındı


def test_risk_sizing_requires_stop_and_caps_leverage() -> None:
    with pytest.raises(ValueError, match="stop olmadan"):
        _spec(stop={"type": "none"}, sizing={"type": "risk_per_trade", "risk_pct": 0.01})
    df = _bars([FLAT, SIG, (100, 101, 99, 100), FLAT])
    spec = _spec(sizing={"type": "risk_per_trade", "risk_pct": 0.01, "max_leverage": 2})
    t = ee.run(df, spec).trades[0]
    # %1 risk / %5 stop = 0.2 nominal (sınırın altında)
    assert t.notional == pytest.approx(0.2)
    spec2 = _spec(
        stop={"type": "fixed_pct", "value": 0.001},
        sizing={"type": "risk_per_trade", "risk_pct": 0.01, "max_leverage": 2},
    )
    assert ee.run(df, spec2).trades[0].notional == pytest.approx(2.0)  # kaldıraç tavanı


def test_warmup_only_and_oos_window_isolated() -> None:
    rng = np.random.default_rng(1)
    n = 400
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    o = np.concatenate([[close[0]], close[:-1]])
    df = _bars(
        list(zip(o, np.maximum(o, close) + 0.5, np.minimum(o, close) - 0.5, close, strict=True))
    )
    spec = _spec(
        indicators=[{"name": "SMA", "period": 50}],
        entry_rules=["close > sma_50"],
        exit_rules=["close < sma_50"],
    )
    start = df.index[300]
    r = ee.run(df, spec, start=start)
    assert r.window["warmup_bars_before"] == 300 and r.window["n_bars"] == 100
    assert all(pd.Timestamp(t.entry_time) > start for t in r.trades)  # ısınmada işlem yok
    assert r.equity.iloc[0] == 1.0  # OOS eğrisi geçmiş getiriyi taşımaz
    # Isınma OOS sinyallerini değiştirmez: tam veride hesaplanan sma, OOS'ta kullanılır.
    assert not math.isnan(ee.compute_columns(df, spec)["sma_50"].iloc[300])


def test_prefix_invariance_detects_lookahead(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    df = _bars([(x, x + 1, x - 1, x) for x in close])
    spec = _spec(indicators=[{"name": "EMA", "period": 20}], entry_rules=["close > ema_20"])
    assert ee.prefix_invariance_check(df, spec)["ok"]

    real = ee.compute_columns

    def leaky(d, s):  # gelecekten okuyan (merkezli) gösterge
        out = real(d, s)
        out["ema_20"] = d["close"].rolling(21, center=True, min_periods=1).mean()
        return out

    monkeypatch.setattr(ee, "compute_columns", leaky)
    res = ee.prefix_invariance_check(df, spec)
    assert not res["ok"] and res["what"] == "sinyal"


def test_unsupported_rule_language_rejected() -> None:
    with pytest.raises(ValueError, match=r"kural dili|kolon değil"):
        _spec(entry_rules=["fiyat yeni zirve yaparsa"])
    with pytest.raises(ValueError, match="kolon değil"):
        _spec(entry_rules=["vwap_bands_9 > close"])


# ── veri denetimi ────────────────────────────────────────────────────────────


def _csv(tmp_path, text: str):
    p = tmp_path / "d.csv"
    p.write_text(text, encoding="utf-8")
    return p


def _rows(n: int, start: str = "2024-01-01 00:00", tz: str = "") -> str:
    idx = pd.date_range(start, periods=n, freq="1h")
    lines = ["time,open,high,low,close,volume"]
    for i, t in enumerate(idx):
        p = 100 + i * 0.1
        lines.append(f"{t.strftime('%Y-%m-%d %H:%M:%S')}{tz},{p},{p + 1},{p - 1},{p + 0.05},1")
    return "\n".join(lines) + "\n"


def test_naive_times_require_timezone(tmp_path) -> None:
    p = _csv(tmp_path, _rows(60))
    with pytest.raises(DataQualityError, match="saat dilimi"):
        load_checked_csv(p, timeframe="1h")
    df, rep = load_checked_csv(p, timeframe="1h", tz="Europe/Istanbul")
    assert str(df.index.tz) == "UTC" and df.index[0].hour == 21  # +03 → UTC
    assert rep.tz_source.startswith("kullanıcı")
    _df2, rep2 = load_checked_csv(_csv(tmp_path, _rows(60, tz="+00:00")), timeframe="1h")
    assert rep2.tz_source.startswith("damgadaki ofset")


def test_sorting_duplicates_gaps_recorded(tmp_path) -> None:
    lines = _rows(80).splitlines()
    header, body = lines[0], lines[1:]
    body = body[:40] + body[45:]  # 5 bar eksik
    body = [*body[::-1], body[0]]  # ters sıra + birebir yineleme
    df, rep = load_checked_csv(_csv(tmp_path, "\n".join([header, *body])), timeframe="1h", tz="UTC")
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    joined = " | ".join(rep.actions)
    assert "artan sıraya" in joined and "yinelenen 1" in joined
    assert rep.gaps["estimated_missing_bars"] == 5


def test_conflicting_duplicate_and_bad_ohlc_rejected(tmp_path) -> None:
    lines = _rows(60).splitlines()
    t = lines[5].split(",")[0]
    conflict = [*lines, f"{t},1,2,0.5,1.5,1"]
    with pytest.raises(DataQualityError, match="ÇELİŞEN"):
        load_checked_csv(_csv(tmp_path, "\n".join(conflict)), timeframe="1h", tz="UTC")
    bad = [*lines[:3], lines[3].replace(lines[3].split(",")[2], "1", 1), *lines[4:]]
    with pytest.raises(DataQualityError, match="OHLC"):
        load_checked_csv(_csv(tmp_path, "\n".join(bad)), timeframe="1h", tz="UTC")


def test_timeframe_mismatch_and_missing_time_column(tmp_path) -> None:
    with pytest.raises(DataQualityError, match="uyuşmuyor"):
        load_checked_csv(_csv(tmp_path, _rows(60)), timeframe="15m", tz="UTC")
    with pytest.raises(DataQualityError, match="Zaman kolonu yok"):
        load_checked_csv(_csv(tmp_path, "open,high,low,close\n1,2,0.5,1\n"), timeframe="1h")
