"""fetch_public_ohlcv.py — hesap/anahtar/ödeme gerektirmeyen birincil kaynaktan küçük OHLCV seti.

Kaynak: Binance kamu veri arşivi (https://data.binance.vision) — borsanın kendi yayımladığı
aylık spot kline dosyaları + her dosyanın ``.CHECKSUM`` (SHA256) eşi. Hesap ya da API anahtarı
gerekmez; indirilen her zip SHA256 ile doğrulanır, doğrulanmayan dosya KULLANILMAZ.

Bu veri yalnız **teknik doğrulama piyasası** içindir (hesapların ve veri akışının doğruluğu);
kullanıcının nihai trading tercihi DEĞİLDİR. Sentetik veri değildir; köken dosyası bunu ve
kaynağı açıkça yazar.

Çıktı (``--out`` altında):
- ``<sembol>_<aralık>_<başlangıç>_<bitiş>.csv`` : ``time`` (ISO-8601, +00:00 ofsetli, mum AÇILIŞ
  zamanı), ``open, high, low, close, volume`` (taban varlık hacmi).
- aynı adla ``.provenance.json`` : kaynak URL'leri, doğrulanan sağlamalar, sembol, piyasa türü,
  saat dilimi, dönem, indirme zamanı, satır sayısı, CSV SHA256.

Kullanım::

    uv run python scripts/fetch_public_ohlcv.py --symbol BTCUSDT --interval 1h \
        --start 2023-01 --end 2024-12 --out data/market/raw
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.binance.vision/data/spot/monthly/klines"
USER_AGENT = "hektor-research-fetch/1.0 (tek seferlik kucuk tarihsel veri)"


def _months(start: str, end: str) -> list[str]:
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:  # sabit https kaynak
        return resp.read()


def _ts(raw: str) -> dt.datetime:
    """Binance açılış zamanı: 2025'ten itibaren mikrosaniye, öncesi milisaniye."""
    v = int(raw)
    seconds = v / 1e6 if v > 10**14 else v / 1e3
    return dt.datetime.fromtimestamp(seconds, tz=dt.UTC)


def fetch(symbol: str, interval: str, start: str, end: str, out_dir: Path) -> dict:
    months = _months(start, end)
    files = []
    rows: list[list[str]] = []
    for ym in months:
        name = f"{symbol}-{interval}-{ym}.zip"
        url = f"{BASE}/{symbol}/{interval}/{name}"
        blob = _get(url)
        want = _get(url + ".CHECKSUM").decode("utf-8").split()[0].strip().lower()
        got = hashlib.sha256(blob).hexdigest()
        if got != want:
            raise SystemExit(f"SAĞLAMA TUTMADI: {name} ({got} != {want}) — veri kullanılmadı.")
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            inner = zf.namelist()
            if len(inner) != 1:
                raise SystemExit(f"Beklenmeyen zip içeriği: {inner}")
            text = zf.read(inner[0]).decode("utf-8")
        n0 = len(rows)
        for rec in csv.reader(io.StringIO(text)):
            if not rec or not rec[0].strip().isdigit():
                continue  # başlık satırı (varsa)
            t = _ts(rec[0])
            rows.append([t.isoformat(), rec[1], rec[2], rec[3], rec[4], rec[5]])
        files.append(
            {"url": url, "sha256": got, "checksum_url": url + ".CHECKSUM", "rows": len(rows) - n0}
        )
    rows.sort(key=lambda r: r[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol}_{interval}_{start}_{end}"
    csv_path = out_dir / f"{stem}.csv"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["time", "open", "high", "low", "close", "volume"])
    w.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    csv_path.write_bytes(data)
    prov = {
        "label": "GERÇEK piyasa verisi — teknik doğrulama piyasası (kullanıcının nihai trading "
        "tercihi DEĞİLDİR). Sentetik değildir.",
        "source": "Binance kamu veri arşivi (data.binance.vision) — borsanın yayımladığı spot "
        "kline dosyaları; hesap/anahtar/ödeme gerekmez",
        "source_base_url": BASE,
        "symbol": symbol,
        "market_type": "spot",
        "interval": interval,
        "timezone": "UTC (mum açılış zamanı, ISO-8601 +00:00)",
        "period_requested": {"start_month": start, "end_month": end},
        "period_actual": {"first_open": rows[0][0], "last_open": rows[-1][0]},
        "n_rows": len(rows),
        "columns": "time, open, high, low, close, volume (taban varlık hacmi)",
        "files": files,
        "checksums_verified": True,
        "downloaded_at": dt.datetime.now(dt.UTC).isoformat(),
        "csv_file": csv_path.name,
        "csv_sha256": hashlib.sha256(data).hexdigest(),
        "limits": "Kline verisinde alış/satış kotasyonu yok → makas (spread) ölçülmedi, "
        "varsayımdır. Spot piyasa → fonlama yok.",
        "tool": "scripts/fetch_public_ohlcv.py",
    }
    (out_dir / f"{stem}.provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return prov


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--start", default="2023-01", help="YYYY-AA")
    ap.add_argument("--end", default="2024-12", help="YYYY-AA")
    ap.add_argument("--out", default="data/market/raw")
    a = ap.parse_args(argv)
    prov = fetch(a.symbol, a.interval, a.start, a.end, Path(a.out))
    print(json.dumps({k: prov[k] for k in ("csv_file", "csv_sha256", "n_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
