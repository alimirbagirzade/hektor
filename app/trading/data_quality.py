"""data_quality.py — gerçek OHLCV CSV'sinin denetimi + kayıtlı temizlik özeti (Faz 2B).

``market_data_loader.load_ohlcv``'dan daha katıdır; sohbetten gelen stratejinin testi YALNIZ bu
yoldan veri alır:

- Zaman kolonu ZORUNLU (sırası doğrulanamayan veri reddedilir).
- Saat dilimi: zaman damgaları ofsetli ise UTC'ye çevrilir; ofsetsiz (naive) ise kullanıcının
  verdiği ``tz`` ile yerelleştirilir. ``tz`` yoksa REDDEDİLİR (sessizce UTC varsayılmaz).
- Okunamayan zaman damgası (NaT) satırları düşürülür ve sayılır; %1'i aşarsa reddedilir.
- Sıra bozuksa artan sıraya getirilir (kaydedilir). Yinelenen zaman: birebir aynı satır
  düşürülür (kaydedilir); farklı değerli yineleme REDDEDİLİR.
- Geçersiz OHLC (high<low, high<open/close, low>open/close), pozitif olmayan fiyat, eksik OHLC
  değeri REDDEDİLİR (ilk örnek damgalarla).
- Bildirilen zaman dilimi ile verinin ortanca adımı uyuşmazsa REDDEDİLİR.
- Eksik barlar DOLDURULMAZ; boşluk sayısı, tahmini eksik bar ve en büyük boşluk kaydedilir
  (hafta sonu / tatil kapanışı olabilir — yorumu kullanıcıya bırakılır).

Çıktı: temiz DataFrame (UTC indeks) + ``DataReport`` (ham dosya özeti, temiz veri özeti, dönem,
yapılan her işlem). Rapor test sonucuna eklenir ve sonuç iddiası doğrulamasında kullanılır.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

_REQUIRED = ("open", "high", "low", "close")
_TIME_COLS = ("time", "date", "datetime", "timestamp", "open time")
_TF_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}
MAX_NAT_SHARE = 0.01


class DataQualityError(ValueError):
    """Veri testi engelleyecek kadar sorunlu (kullanıcıya gösterilir)."""


@dataclass
class DataReport:
    file_name: str
    file_sha256: str
    clean_sha256: str
    timeframe: str
    tz_source: str
    rows_raw: int
    rows_clean: int
    start: str
    end: str
    actions: list[str] = field(default_factory=list)
    gaps: dict[str, Any] = field(default_factory=dict)
    median_step_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_sha256(df: pd.DataFrame) -> str:
    """Temiz verinin kanonik özeti (UTC ISO zaman + OHLCV, sabit biçim)."""
    buf = io.StringIO()
    out = df[[*_REQUIRED, "volume"]].copy()
    out.index = out.index.strftime("%Y-%m-%dT%H:%M:%SZ")
    out.to_csv(buf, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()


def _parse_times(raw: pd.Series, tz: str | None) -> tuple[pd.DatetimeIndex, str]:
    sample = raw.dropna().astype(str).head(50)
    has_offset = sample.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})\s*$", regex=True).any()
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all() and len(raw):
        # Epoch damgası (ms ya da s) — tanımı gereği UTC.
        unit = "ms" if numeric.abs().median() > 1e11 else "s"
        idx = pd.to_datetime(numeric, unit=unit, utc=True)
        return pd.DatetimeIndex(idx), f"epoch ({unit}), UTC"
    if has_offset:
        idx = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed")
        return pd.DatetimeIndex(idx), "damgadaki ofset → UTC"
    if not tz:
        raise DataQualityError(
            "Zaman damgalarında saat dilimi yok ve saat dilimi belirtilmedi. Verinin saat "
            "dilimini seçin (ör. UTC, Europe/Istanbul); sessizce UTC varsayılmaz."
        )
    naive = pd.to_datetime(raw, errors="coerce", format="mixed")
    try:
        idx = pd.DatetimeIndex(naive).tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    except Exception as exc:
        raise DataQualityError(f"Saat dilimi uygulanamadı ({tz}): {exc}") from exc
    return idx.tz_convert("UTC"), f"kullanıcı: {tz} → UTC"


def load_checked_csv(
    path: str | Path, *, timeframe: str, tz: str | None = None
) -> tuple[pd.DataFrame, DataReport]:
    path = Path(path)
    if not path.is_file():
        raise DataQualityError(f"Dosya yok: {path.name}")
    if timeframe not in _TF_SECONDS:
        raise DataQualityError(f"Zaman dilimi tanınmıyor: {timeframe}")
    raw_bytes = path.read_bytes()
    file_sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise DataQualityError(f"CSV okunamadı: {exc}") from exc
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    time_col = next((c for c in _TIME_COLS if c in df.columns), None)
    if time_col is None:
        raise DataQualityError(
            "Zaman kolonu yok (time/date/datetime/timestamp). Satır sırası doğrulanamaz; "
            "test yapılmadı."
        )
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise DataQualityError(f"Eksik kolon(lar): {missing}")
    actions: list[str] = []
    rows_raw = len(df)
    idx, tz_source = _parse_times(df[time_col], tz)
    df = df.drop(columns=[time_col])
    df.index = idx
    df.index.name = "time"
    if "volume" not in df.columns:
        df["volume"] = 0.0
        actions.append("hacim kolonu yok → 0 ile dolduruldu (stratejide hacim kullanılmamalı)")
    df = df[[*_REQUIRED, "volume"]]

    nat = int(df.index.isna().sum())
    if nat:
        if nat / max(1, rows_raw) > MAX_NAT_SHARE:
            raise DataQualityError(
                f"{nat}/{rows_raw} satırın zamanı okunamadı (>%{MAX_NAT_SHARE * 100:g}) — "
                "biçimi düzeltin."
            )
        df = df[df.index.notna()]
        actions.append(f"okunamayan zamanlı {nat} satır düşürüldü")

    try:
        df = df.astype("float64")
    except (TypeError, ValueError) as exc:
        raise DataQualityError(f"Sayısal olmayan fiyat değeri: {exc}") from exc
    nan_rows = df[list(_REQUIRED)].isna().any(axis=1)
    if nan_rows.any():
        ex = ", ".join(str(t) for t in df.index[nan_rows][:3])
        raise DataQualityError(f"Eksik OHLC değeri ({int(nan_rows.sum())} satır): {ex}")

    if not df.index.is_monotonic_increasing:
        inversions = int((pd.Series(df.index).diff().dt.total_seconds() < 0).sum())
        df = df.sort_index(kind="stable")
        actions.append(f"zaman sırası bozuktu ({inversions} ters adım) → artan sıraya getirildi")

    if df.index.has_duplicates:
        dup = df[df.index.duplicated(keep=False)]
        conflicting = dup.groupby(level=0).nunique().gt(1).any(axis=1)
        bad = [str(t) for t, v in conflicting.items() if bool(v)]
        if bad:
            raise DataQualityError(
                f"Aynı zaman damgasında ÇELİŞEN satırlar ({len(bad)} damga): {', '.join(bad[:3])}"
            )
        n0 = len(df)
        df = df[~df.index.duplicated(keep="first")]
        actions.append(f"birebir yinelenen {n0 - len(df)} satır düşürüldü")

    nonpos = (df[list(_REQUIRED)] <= 0).any(axis=1)
    if nonpos.any():
        ex = ", ".join(str(t) for t in df.index[nonpos][:3])
        raise DataQualityError(f"Pozitif olmayan fiyat ({int(nonpos.sum())} satır): {ex}")
    bad_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    if bad_ohlc.any():
        ex = ", ".join(str(t) for t in df.index[bad_ohlc][:3])
        raise DataQualityError(f"Geçersiz OHLC barı ({int(bad_ohlc.sum())} satır): {ex}")
    if len(df) < 50:
        raise DataQualityError(f"Yetersiz veri: {len(df)} bar (en az 50).")

    step = pd.Series(df.index).diff().dt.total_seconds().dropna()
    expected = _TF_SECONDS[timeframe]
    median = float(step.median())
    if abs(median - expected) > 0.01 * expected:
        raise DataQualityError(
            f"Veri adımı ({median:g} sn) bildirilen zaman dilimiyle ({timeframe} = {expected} sn) "
            "uyuşmuyor."
        )
    gap_mask = step > 1.5 * expected
    gaps = step[gap_mask]
    missing_bars = int(((gaps / expected).round() - 1).sum()) if len(gaps) else 0
    largest = float(gaps.max()) if len(gaps) else 0.0
    gap_info = {
        "n_gaps": int(gap_mask.sum()),
        "estimated_missing_bars": missing_bars,
        "largest_gap_s": largest,
        "largest_gap_at": str(df.index[1:][gap_mask.to_numpy()][gaps.argmax()])
        if len(gaps)
        else "",
        "note": "Eksik barlar doldurulmadı; piyasa kapanışı (hafta sonu/tatil) olabilir.",
    }
    if missing_bars:
        actions.append(f"{gap_info['n_gaps']} boşluk (~{missing_bars} eksik bar) — doldurulmadı")

    report = DataReport(
        file_name=path.name,
        file_sha256=file_sha,
        clean_sha256=clean_sha256(df),
        timeframe=timeframe,
        tz_source=tz_source,
        rows_raw=rows_raw,
        rows_clean=len(df),
        start=df.index[0].isoformat(),
        end=df.index[-1].isoformat(),
        actions=actions,
        gaps=gap_info,
        median_step_s=median,
    )
    return df, report
