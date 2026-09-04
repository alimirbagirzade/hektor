"""Load OHLCV market data from CSV (and generate synthetic data for testing).

Expected CSV columns (case-insensitive): time/date, open, high, low, close, volume.
The synthetic generator lets you run the full backtest pipeline without sourcing
real data first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REQUIRED = ["open", "high", "low", "close"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # Binance CSV'si "open time" kullanır; "close time" gerekmez
    if "open time" in df.columns:
        df = df.rename(columns={"open time": "time"})
        if "close time" in df.columns:
            df = df.drop(columns=["close time"])
    time_col = next((c for c in ("time", "date", "datetime", "timestamp") if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df = df.set_index(time_col)
        df.index.name = "time"
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV eksik kolonlar: {missing}")
    df = df[[*_REQUIRED, "volume"]].dropna(subset=_REQUIRED)
    df = df.astype(dict.fromkeys([*_REQUIRED, "volume"], "float64"))
    # OHLC bütünlüğü: high en yüksek, low en düşük olmalı. Bozuk barlar (high<low,
    # high<open/close, low>open/close) backtest/indikatörü sessizce bozar → reddet.
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if bad.any():
        raise ValueError(
            f"Geçersiz OHLC barı ({int(bad.sum())} satır): high/low, open/close ile "
            "tutarsız (high<low gibi). Veri bozuk — düzelt veya kaynağı değiştir."
        )
    return df


def generate_synthetic_ohlcv(
    n: int = 2000,
    start_price: float = 2000.0,
    seed: int = 42,
    freq: str = "15min",
) -> pd.DataFrame:
    """Geometric-Brownian-ish synthetic series with a mild regime shift."""
    rng = np.random.default_rng(seed)
    drift = np.where(np.arange(n) < n // 2, 0.0001, -0.00005)
    rets = drift + rng.normal(0, 0.004, n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    # high/low open VE close'u kapsamalı; aksi halde open aralık dışına taşar ve
    # load_ohlcv OHLC bütünlük kontrolü barı "bozuk veri" diye reddeder (round-trip kırılır).
    base_hi = np.maximum(open_, close)
    base_lo = np.minimum(open_, close)
    high = base_hi * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = base_lo * (1 - np.abs(rng.normal(0, 0.002, n)))
    vol = rng.uniform(100, 1000, n)
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx
    )


def write_synthetic_csv(path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_ohlcv(**kwargs)
    df.to_csv(path, index_label="time")
    return path
