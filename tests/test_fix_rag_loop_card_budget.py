"""Kart bütçesi/açlık regresyon testleri — tamamen çevrimdışı (LLM/ağ yok).

Kapsanan kök sorun: başarısız kart denemesi "üretildi" sayılıyordu.
  (A) Kartlanamayan makale her turda aynı bütçeyi yiyip alttakileri sıraya getirmiyordu.
  (B) `builder.build()` BOŞ (kaydedilmemiş) kart dönse de sayaç artıyordu.
  (C) Ajan "Kart üretildi: N" derken N deneme sayısıydı (CLAUDE.md kural 2 ihlali).

Store/builder monkeypatch ile sahtelenir; `RagLearningLoop` gerçek koddur.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.research.rag_learning_loop import (
    _MAX_CARD_ATTEMPTS,
    RagLearningLoop,
    RagLoopState,
    _as_attempt_ledger,
)


class _Card:
    """`KnowledgeCard` yerine geçen asgari sahte (yalnız `has_content` okunur)."""

    def __init__(self, has_content: bool) -> None:
        self.has_content = has_content


class _Store:
    """SqliteStore stub: kart üretimi başarılıysa makale 'kartlı' olur."""

    def __init__(self, paper_ids: list[str]) -> None:
        self._ids = list(paper_ids)
        self.carded: set[str] = set()
        self.approved: list[str] = []

    def list_papers(self) -> list[Any]:
        return [SimpleNamespace(paper_id=pid) for pid in self._ids]

    def has_knowledge_card(self, paper_id: str) -> bool:
        return paper_id in self.carded

    def list_pending_cards(self) -> list[dict[str, Any]]:
        return []

    def approve_card(self, card_id: str) -> bool:  # pragma: no cover - çağrılmaz
        return False


def _install(monkeypatch: pytest.MonkeyPatch, store: _Store, builder_cls: type) -> None:
    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", lambda: store)
    monkeypatch.setattr("app.brain.knowledge_card_builder.KnowledgeCardBuilder", builder_cls)


def _make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RagLearningLoop:
    """Durum dosyasını tmp veri köküne bağla (bkz. tests/test_rag_learning_loop.py)."""
    from app.config import settings as settings_mod

    (tmp_path / "storage").mkdir(exist_ok=True)
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    settings_mod.get_settings.cache_clear()
    return RagLearningLoop()


# --------------------------------------------------------------- (B) boş kart sayılmaz


def test_bos_kart_sayaci_artirmaz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`build()` içeriksiz kart dönerse "üretildi" SAYILMAZ (kaydedilmemiştir)."""
    store = _Store(["p1", "p2"])
    calls: list[str] = []

    class BosBuilder:
        def build(self, paper_id: str) -> _Card:
            calls.append(paper_id)
            return _Card(has_content=False)

    _install(monkeypatch, store, BosBuilder)
    loop = _make(tmp_path, monkeypatch)

    assert loop._build_missing_cards(limit=2) == 0  # deneme var, kart yok
    assert calls == ["p1", "p2"]  # bütçe kadar DENENDİ
    assert store.approved == []  # boş kart onaya gitmez


def test_icerikli_kart_sayilir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """İçerikli kart hem sayılır hem defterden düşer (defter şişmesin)."""
    store = _Store(["p1", "p2"])

    class DoluBuilder:
        def build(self, paper_id: str) -> _Card:
            store.carded.add(paper_id)  # gerçek builder kaydeder
            return _Card(has_content=True)

    _install(monkeypatch, store, DoluBuilder)
    loop = _make(tmp_path, monkeypatch)

    assert loop._build_missing_cards(limit=5) == 2
    assert loop._state.card_attempts == {}  # başarılı makaleler defterde kalmaz


def test_karisik_sonuc_yalniz_basariyi_sayar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bir başarı + bir boş + bir istisna → sayaç 1 (üçü de denenmiş olsa bile)."""
    store = _Store(["p1", "p2", "p3"])

    class KarisikBuilder:
        def build(self, paper_id: str) -> _Card:
            if paper_id == "p1":
                store.carded.add(paper_id)
                return _Card(has_content=True)
            if paper_id == "p2":
                return _Card(has_content=False)
            raise RuntimeError("builder patladı")

    _install(monkeypatch, store, KarisikBuilder)
    loop = _make(tmp_path, monkeypatch)

    assert loop._build_missing_cards(limit=3) == 1
    assert loop._state.card_attempts == {"p2": 1, "p3": 1}  # yalnız başarısızlar defterde


# --------------------------------------------------------------- (A) açlık biter


def test_kartlanamayan_makale_tavandan_sonra_atlanir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kalıcı kartlanamayan makale tavan kadar denenir, sonra sıra DİĞERLERİNE geçer.

    Bütçe 1 (turda tek deneme) ve p1 hep boş dönüyor. Eski davranışta p1 her turda
    bütçeyi yerdi (built += 1) ve p2'ye sıra HİÇ gelmezdi (açlık).
    """
    store = _Store(["p1", "p2"])  # list_papers en yeniden eskiye: p1 önce
    calls: list[str] = []

    class Builder:
        def build(self, paper_id: str) -> _Card:
            calls.append(paper_id)
            if paper_id == "p1":
                return _Card(has_content=False)  # kaynak bozuk → hiç kartlanamaz
            store.carded.add(paper_id)
            return _Card(has_content=True)

    _install(monkeypatch, store, Builder)
    loop = _make(tmp_path, monkeypatch)

    sonuclar = [loop._build_missing_cards(limit=1) for _ in range(_MAX_CARD_ATTEMPTS + 1)]

    assert calls.count("p1") == _MAX_CARD_ATTEMPTS  # tavandan fazla denenmez
    assert "p2" in calls  # sıra alttaki makaleye GELDİ (açlık bitti)
    assert sum(sonuclar) == 1  # yalnız p2 gerçekten kartlandı
    assert loop._state.card_attempts["p1"] == _MAX_CARD_ATTEMPTS


def test_defter_diske_yazilir_ve_yeni_ornekte_okunur(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defter kalıcı: ajan her koşuda yeni örnek kurar, tavan yine de işler."""
    store = _Store(["p1"])
    calls: list[str] = []

    class BosBuilder:
        def build(self, paper_id: str) -> _Card:
            calls.append(paper_id)
            return _Card(has_content=False)

    _install(monkeypatch, store, BosBuilder)
    _make(tmp_path, monkeypatch)  # state dosyasını tmp veri köküne bağla

    for _ in range(_MAX_CARD_ATTEMPTS + 2):
        RagLearningLoop()._build_missing_cards(limit=1)  # her turda TAZE örnek

    assert calls.count("p1") == _MAX_CARD_ATTEMPTS
    kayit = json.loads(
        (tmp_path / "storage" / "rag_learning_state.json").read_text(encoding="utf-8")
    )
    assert kayit["card_attempts"] == {"p1": _MAX_CARD_ATTEMPTS}  # defter DİSKTE


# --------------------------------------------------------------- (C) ajan raporu


def test_run_reader_bos_kartta_sifir_raporlar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_reader` deneme değil GERÇEK kart sayar → boş yanıtta "kart 0"."""
    from app.research.paper_reader import run_reader

    store = _Store(["p1", "p2"])

    class BosBuilder:
        def build(self, paper_id: str) -> _Card:
            return _Card(has_content=False)

    _install(monkeypatch, store, BosBuilder)
    loop = _make(tmp_path, monkeypatch)

    class _ReaderStore(_Store):
        def get_comprehension_score(self, paper_id: str) -> None:
            return None

    reader_store = _ReaderStore(["p1", "p2"])
    reader_store.carded = store.carded  # aynı gerçekliği paylaş

    r = run_reader(cards=2, scores=0, loop=loop, store=reader_store, use_llm_score=False)
    assert r["ok"] is True
    assert r["kart"] == 0  # deneme yapıldı ama içerikli kart yok
    assert len(r["kalan_kartsiz"]) == 2  # dürüst rapor: iş bitmedi


# --------------------------------------------------------------- (4) state uyumluluğu


def test_eski_state_dosyasi_eksik_anahtarla_okunur(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`card_attempts` içermeyen ESKİ state dosyası hatasız okunur (varsayılan {})."""
    (tmp_path / "storage").mkdir(exist_ok=True)
    eski = {
        "enabled": True,
        "interval_min": 45,
        "cards_per_cycle": 3,
        "total_cards": 7,
        "rebuilt_paper_ids": ["p9"],
        "rebuild_attempts": {"p9": 2},
    }
    (tmp_path / "storage" / "rag_learning_state.json").write_text(
        json.dumps(eski), encoding="utf-8"
    )
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    from app.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    st = RagLearningLoop().get_status()
    assert st["card_attempts"] == {}  # eksik anahtar → varsayılan
    assert st["enabled"] is True and st["interval_min"] == 45  # eski alanlar korunur
    assert st["total_cards"] == 7 and st["rebuild_attempts"] == {"p9": 2}


def test_bozuk_defter_cokertmez() -> None:
    """Elle bozulmuş defter (yanlış tip) okunabilir hâle getirilir, çökertmez."""
    assert _as_attempt_ledger(None) == {}
    assert _as_attempt_ledger(["p1"]) == {}
    assert _as_attempt_ledger({"p1": "2", "p2": "abc", "p3": 1}) == {"p1": 2, "p3": 1}


def test_state_from_dict_bilinmeyen_alani_yok_sayar() -> None:
    """İleri/geri uyum: bilinmeyen anahtar state'i bozmaz."""
    s = RagLoopState.from_dict({"enabled": True, "gelecekteki_alan": 1, "card_attempts": None})
    assert s.enabled is True
    assert s.card_attempts == {}  # None → varsayılan (from_dict None'ı atlar)
