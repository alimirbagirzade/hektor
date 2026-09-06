"""Yüklemede otomatik kart + anlama skoru (`auto_card_on_upload`) — çevrimdışı.

Ingest / kart üretici / skorlayıcı stub'lanır; LLM ve Ollama yok. Doğrulanan sözleşme:
- Ayar KAPALI (varsayılan) → ingest sonrası kart üretilmez.
- Ayar AÇIK → ingest → kart → (kart içerikliyse) anlama skoru.
- Kart boş dönerse skor HESAPLANMAZ (boş kart zaten kaydedilmez).
- Makale atlandıysa (zaten indeksli) kart üretilmez.
- Kart/skor hatası yüklemeyi patlatmaz (yalnız log).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.web import server


class _Cagri:
    def __init__(self) -> None:
        self.kart: list[str] = []
        self.skor: list[str] = []
        self.kaydet: list[object] = []


def _kur(monkeypatch: pytest.MonkeyPatch, *, skipped: bool, dolu_kart: bool, kart_hata=False):
    c = _Cagri()

    class _Indexer:
        def ingest_one(self, disc):
            return SimpleNamespace(paper_id="paper_x", skipped=skipped)

    class _Builder:
        def build(self, pid: str):
            c.kart.append(pid)
            if kart_hata:
                raise RuntimeError("LLM yok")
            return SimpleNamespace(has_content=dolu_kart)

    class _Scorer:
        def score(self, pid: str):
            c.skor.append(pid)
            return {"paper_id": pid}

    class _Store:
        def save_comprehension_score(self, r) -> None:
            c.kaydet.append(r)

    monkeypatch.setattr("app.ingestion.paper_loader.compute_file_hash", lambda p: "h" * 20)
    monkeypatch.setattr("app.memory.paper_indexer.PaperIndexer", _Indexer)
    monkeypatch.setattr("app.brain.knowledge_card_builder.KnowledgeCardBuilder", _Builder)
    monkeypatch.setattr("app.verification.comprehension_scorer.ComprehensionScorer", _Scorer)
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", _Store)
    return c


def test_ayar_varsayilan_kapali() -> None:
    assert get_settings().auto_card_on_upload is False


def test_kapaliyken_kart_uretilmez(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(get_settings(), "auto_card_on_upload", False, raising=False)
    c = _kur(monkeypatch, skipped=False, dolu_kart=True)
    server._ingest_one(tmp_path / "a.pdf")
    assert c.kart == [] and c.skor == []


def test_acikken_kart_ve_skor(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(get_settings(), "auto_card_on_upload", True, raising=False)
    c = _kur(monkeypatch, skipped=False, dolu_kart=True)
    server._ingest_one(tmp_path / "a.pdf")
    assert c.kart == ["paper_x"] and c.skor == ["paper_x"] and len(c.kaydet) == 1


def test_bos_kartta_skor_atlanir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(get_settings(), "auto_card_on_upload", True, raising=False)
    c = _kur(monkeypatch, skipped=False, dolu_kart=False)
    server._ingest_one(tmp_path / "a.pdf")
    assert c.kart == ["paper_x"] and c.skor == []  # boş kart → boşuna LLM yakma


def test_atlanan_makalede_kart_yok(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(get_settings(), "auto_card_on_upload", True, raising=False)
    c = _kur(monkeypatch, skipped=True, dolu_kart=True)
    server._ingest_one(tmp_path / "a.pdf")
    assert c.kart == []


def test_kart_hatasi_yuklemeyi_patlatmaz(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(get_settings(), "auto_card_on_upload", True, raising=False)
    c = _kur(monkeypatch, skipped=False, dolu_kart=True, kart_hata=True)
    with caplog.at_level("ERROR"):
        server._ingest_one(tmp_path / "a.pdf")  # raise ETMEMELİ
    assert c.kart == ["paper_x"] and c.skor == []
    assert any("Otomatik kart üretilemedi" in r.getMessage() for r in caplog.records)
