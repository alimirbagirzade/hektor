"""Teknik indikatörler birim testleri — tamamen çevrimdışı."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.trading.indicators import atr, bollinger, compute_indicator, ema, macd, rsi, sma


@pytest.fixture
def close_series() -> pd.Series:
    rng = np.random.default_rng(0)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlc_df(close_series: pd.Series) -> pd.DataFrame:
    close = close_series
    rng = np.random.default_rng(1)
    high = close * (1 + np.abs(rng.normal(0, 0.005, len(close))))
    low = close * (1 - np.abs(rng.normal(0, 0.005, len(close))))
    open_ = pd.Series(np.concatenate([[close.iloc[0]], close.values[:-1]]))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_ema_length(close_series: pd.Series) -> None:
    result = ema(close_series, 20)
    assert len(result) == len(close_series)


def test_ema_approaches_price(close_series: pd.Series) -> None:
    # EMA yakınsak olmalı: sabit seriye uygulanırsa seriye yaklaşmalı
    flat = pd.Series([100.0] * 100)
    result = ema(flat, 10)
    assert abs(result.iloc[-1] - 100.0) < 0.01


def test_sma_window_nan(close_series: pd.Series) -> None:
    result = sma(close_series, 20)
    # İlk 19 değer NaN olmalı
    assert result.iloc[:19].isna().all()
    assert result.iloc[19:].notna().all()


def test_rsi_bounds(close_series: pd.Series) -> None:
    result = rsi(close_series, 14)
    assert (result >= 0).all() and (result <= 100).all()


def test_rsi_flat_series_midpoint() -> None:
    flat = pd.Series([100.0] * 50)
    result = rsi(flat, 14)
    # Değişim yok → RSI 50'ye yakınsamalı
    assert abs(result.iloc[-1] - 50.0) < 1.0


def test_rsi_lossless_uptrend_is_100() -> None:
    # Kesintisiz yükselen seri (hiç kayıp yok) → RSI üst sınıra (100) gitmeli, 50 DEĞİL.
    up = pd.Series([float(i) for i in range(1, 40)], name="close")
    result = rsi(up, 14)
    assert result.iloc[-1] == pytest.approx(100.0)
    assert result.iloc[20] == pytest.approx(100.0)


def test_atr_positive(ohlc_df: pd.DataFrame) -> None:
    result = atr(ohlc_df["high"], ohlc_df["low"], ohlc_df["close"], 14)
    assert (result.dropna() > 0).all()


def test_macd_three_series(close_series: pd.Series) -> None:
    line, signal, hist = macd(close_series)
    assert len(line) == len(close_series)
    assert len(signal) == len(close_series)
    # histogram = macd - signal
    diff = (line - signal - hist).abs()
    assert diff.max() < 1e-10


def test_bollinger_bands_order(close_series: pd.Series) -> None:
    upper, mid, lower = bollinger(close_series, 20)
    valid = upper.dropna().index
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_compute_indicator_ema(ohlc_df: pd.DataFrame) -> None:
    result = compute_indicator("EMA", ohlc_df, 20)
    expected = ema(ohlc_df["close"], 20)
    pd.testing.assert_series_equal(result, expected)


def test_compute_indicator_rsi(ohlc_df: pd.DataFrame) -> None:
    result = compute_indicator("RSI", ohlc_df, 14)
    assert (result >= 0).all() and (result <= 100).all()


def test_compute_indicator_atr(ohlc_df: pd.DataFrame) -> None:
    result = compute_indicator("ATR", ohlc_df, 14)
    assert (result.dropna() > 0).all()


def test_compute_indicator_macd(ohlc_df: pd.DataFrame) -> None:
    result = compute_indicator("MACD", ohlc_df, 12)
    assert len(result) == len(ohlc_df)


def test_compute_indicator_unknown_raises(ohlc_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Bilinmeyen"):
        compute_indicator("UNKNOWN", ohlc_df, 14)
