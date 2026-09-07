"""load_ohlcv zaman sırası güvencesi (Kademe-2 av bulgusu, 2026-09-07 — BLOCKER).

Bulgu: loader zaman indeksini sıralamıyordu. Satırları yeniden-eskiye sıralı bir CSV
(Investing.com / Yahoo dışa aktarımı) ters yönde backtest ediliyor; göstergeler
sonraki tarihlerden hesaplanıyor (fiili look-ahead, Kural 4), getiriler ters dönüyor
ve IS/OOS bölmesi "OOS" diye en eski veriyi ölçüyordu. Verdict sessizce yanlış çıkıyor
ve aynı dosya eğitim verisine kanıt olarak sızabiliyordu.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.trading.market_data_loader import generate_synthetic_ohlcv, load_ohlcv


def _yaz(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index_label="time")
    return path


def test_tersten_sirali_csv_artan_yuklenir(tmp_path: Path) -> None:
    """Yeniden-eskiye sıralı dosya, artan sıralı dosyayla AYNI seriyi vermeli."""
    df = generate_synthetic_ohlcv(n=120, seed=7)
    artan = load_ohlcv(_yaz(df, tmp_path / "artan.csv"))
    tersten = load_ohlcv(_yaz(df.iloc[::-1], tmp_path / "ters.csv"))

    assert tersten.index.is_monotonic_increasing
    pd.testing.assert_frame_equal(artan, tersten)


def test_tersten_sirali_backtest_ayni_sonucu_verir(tmp_path: Path) -> None:
    """Asıl zarar burada: sıralanmamış dosya farklı (yanlış) bir verdict üretiyordu."""
    from app.trading.backtester import run_backtest
    from app.trading.strategy_ir import example_ir

    df = generate_synthetic_ohlcv(n=600, seed=11)
    ir = example_ir()
    artan = run_backtest(load_ohlcv(_yaz(df, tmp_path / "a.csv")), ir)
    tersten = run_backtest(load_ohlcv(_yaz(df.iloc[::-1], tmp_path / "t.csv")), ir)

    assert artan.metrics.total_return_pct == tersten.metrics.total_return_pct
    assert artan.metrics.n_trades == tersten.metrics.n_trades


def test_okunamayan_zaman_damgasi_dusurulur(tmp_path: Path) -> None:
    """errors='coerce' NaT üretir; konumu olmayan satır seriye giremez."""
    df = generate_synthetic_ohlcv(n=30, seed=3)
    path = tmp_path / "nat.csv"
    ham = df.reset_index()
    ham.columns = ["time", *ham.columns[1:]]
    # Zaman kolonunu METİN yap: bozuk damga ancak ham CSV'de böyle görünür.
    ham["time"] = ham["time"].astype(str)
    ham.loc[5, "time"] = "bozuk-tarih"
    ham.to_csv(path, index=False)

    out = load_ohlcv(path)
    assert len(out) == 29
    assert out.index.notna().all()
    assert out.index.is_monotonic_increasing


def test_tekrarli_zaman_damgasi_tekillestirilir(tmp_path: Path) -> None:
    """Aynı damganın iki kaydı kalırsa konumsal hesap bir barı iki kez sayar."""
    df = generate_synthetic_ohlcv(n=20, seed=5)
    cift = pd.concat([df, df.iloc[[10]]]).sort_index(kind="stable")
    out = load_ohlcv(_yaz(cift, tmp_path / "dup.csv"))

    assert len(out) == 20
    assert not out.index.has_duplicates


def test_zaten_sirali_dosya_degismez(tmp_path: Path) -> None:
    """Düzeltme mevcut (doğru) dosyaların davranışını DEĞİŞTİRMEMELİ."""
    df = generate_synthetic_ohlcv(n=50, seed=9)
    out = load_ohlcv(_yaz(df, tmp_path / "ok.csv"))
    assert len(out) == 50
    assert out.index.is_monotonic_increasing
    assert out["close"].iloc[0] == df["close"].iloc[0]
