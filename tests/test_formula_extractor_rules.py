"""Kural tabanlı formül tespiti — kelime-sınırı regresyon testi (kaynak uydurma önleme)."""

from __future__ import annotations

from pathlib import Path

from app.memory.sqlite_store import SqliteStore
from app.research.formula_extractor import FormulaExtractor


def _fx(tmp_path: Path) -> FormulaExtractor:
    return FormulaExtractor(store=SqliteStore(db_path=tmp_path / "t.db"))


def test_no_substring_false_positive(tmp_path: Path) -> None:
    # "schema"→EMA, "plasma"→SMA, "theatre"→ATR, "obvious"→OBV alt-dize eşleşmeleri
    # kelime-sınırıyla artık YANLIŞ formül üretmemeli (sahte latex=None kayıt yok).
    text = "The schema and plasma in this theatre are obvious to small teams."
    found = {f["name"] for f in _fx(tmp_path)._rule_based_extract(text, "p")}
    assert not ({"EMA", "SMA", "ATR", "OBV"} & found)


def test_real_indicator_still_matched(tmp_path: Path) -> None:
    text = "The RSI and MACD indicators are momentum oscillators."
    found = {f["name"] for f in _fx(tmp_path)._rule_based_extract(text, "p")}
    assert "RSI" in found and "MACD" in found
