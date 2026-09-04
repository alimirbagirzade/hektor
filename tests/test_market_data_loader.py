"""Sentetik OHLCV üreteci + yükleyici testleri (round-trip bütünlük regresyonu)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.trading.market_data_loader import (
    generate_synthetic_ohlcv,
    load_ohlcv,
    write_synthetic_csv,
)


def test_synthetic_ohlc_integrity() -> None:
    # high tüm O/C/L'yi kapsamalı, low hepsinin altında olmalı.
    df = generate_synthetic_ohlcv(n=2000, seed=7)
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["high"] >= df["low"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()


@pytest.mark.parametrize("seed", [1, 7, 42, 99, 123])
def test_round_trip_not_rejected(tmp_path: Path, seed: int) -> None:
    # write_synthetic_csv → load_ohlcv ValueError fırlatmamalı (eskiden ~%70 bar bozuktu).
    csv = write_synthetic_csv(tmp_path / f"s{seed}.csv", n=1000, seed=seed)
    df = load_ohlcv(csv)
    assert len(df) == 1000
