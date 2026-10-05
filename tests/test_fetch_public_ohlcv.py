"""fetch_public_ohlcv: ay aralığı, zaman birimi (ms/µs) ve sağlama kapısı — ağ YOK."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "fetch_public_ohlcv.py"
spec = importlib.util.spec_from_file_location("fetch_public_ohlcv", _P)
assert spec and spec.loader
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


def _zip(rows: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("X-1h-2024-01.csv", "\n".join(rows) + "\n")
    return buf.getvalue()


def test_months_and_timestamp_units() -> None:
    assert fp._months("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]
    ms = fp._ts("1704067200000")  # 2024-01-01 milisaniye
    us = fp._ts("1735689600000000")  # 2025-01-01 mikrosaniye (2025+ arşiv biçimi)
    assert ms.isoformat() == "2024-01-01T00:00:00+00:00"
    assert us.isoformat() == "2025-01-01T00:00:00+00:00"


def test_checksum_mismatch_refuses_data(monkeypatch, tmp_path) -> None:
    blob = _zip(["1704067200000,1,2,0.5,1.5,10,0,0,0,0,0,0"])
    good = hashlib.sha256(blob).hexdigest()
    calls = {"sha": good}

    def fake_get(url: str) -> bytes:
        return f"{calls['sha']}  X.zip".encode() if url.endswith(".CHECKSUM") else blob

    monkeypatch.setattr(fp, "_get", fake_get)
    prov = fp.fetch("X", "1h", "2024-01", "2024-01", tmp_path)
    assert prov["n_rows"] == 1 and prov["checksums_verified"] and "GERÇEK" in prov["label"]
    text = (tmp_path / prov["csv_file"]).read_text(encoding="utf-8")
    assert text.splitlines()[1].startswith("2024-01-01T00:00:00+00:00,1,2,0.5,1.5,10")
    calls["sha"] = "0" * 64
    with pytest.raises(SystemExit, match="SAĞLAMA TUTMADI"):
        fp.fetch("X", "1h", "2024-01", "2024-01", tmp_path / "b")
