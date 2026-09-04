"""RAG öğrenme döngüsü testi — tamamen çevrimdışı (ağır adımlar monkeypatch'lenir).

Ağ (arXiv) ve LLM (kart/skor) çağrıları yer almaz; yalnız durum makinesi,
ayar kelepçeleme, sayaç güncellemesi ve eğitim-sırasında-duraklat davranışı test edilir.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.research.rag_learning_loop import RagLearningLoop, is_substantive_card


def test_is_substantive_card_rejects_degenerate() -> None:
    # Dejenere "..." kartı (LLM ön-madde/çekişmede üretir) ONAYLANMAMALI.
    assert is_substantive_card({"title": "...", "main_claim": "..."}) is False
    assert is_substantive_card({"title": "", "main_claim": ""}) is False
    assert is_substantive_card({"title": "Vol", "main_claim": "kısa"}) is False  # main_claim<40
    # Gerçek içerik KABUL edilmeli (anlamlı title≥8 + main_claim≥40).
    assert (
        is_substantive_card(
            {
                "title": "Volatilite Kümelenmesi",
                "main_claim": "Volatilite kümelenmesi momentum sinyallerinin kalıcılığını artırır.",
            }
        )
        is True
    )


def _make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RagLearningLoop:
    # Durum dosyası CWD'ye DEĞİL `settings.state_dir`'e bağlıdır → veri kökünü yönlendir.
    # (conftest `_settings_cache_reset` fixture'ı test sonunda cache'i temizler.)
    from app.config import settings as settings_mod

    (tmp_path / "storage").mkdir(exist_ok=True)
    monkeypatch.setenv("ACHILLES_ROOT_PATH", str(tmp_path))
    settings_mod.get_settings.cache_clear()
    return RagLearningLoop()


def test_default_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = _make(tmp_path, monkeypatch)
    st = loop.get_status()
    assert st["enabled"] is False
    assert st["stage"] == "idle"
    assert st["running"] is False


def test_enable_persists_across_instances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = _make(tmp_path, monkeypatch)
    loop.set_enabled(True)
    # Yeni örnek aynı state dosyasından okur → kalıcılık.
    loop2 = RagLearningLoop()
    assert loop2.get_status()["enabled"] is True


def test_config_clamped_and_persisted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = _make(tmp_path, monkeypatch)
    loop.set_config(interval_min=99999, cards_per_cycle=3, fetch_enabled=False)
    st = loop.get_status()
    assert st["interval_min"] == 1440  # üst sınıra kelepçelendi
    assert st["cards_per_cycle"] == 3
    assert st["fetch_enabled"] is False


def test_cycle_runs_steps_and_updates_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = _make(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "_training_running", lambda: False)
    monkeypatch.setattr(loop, "_fetch_new_papers", lambda: (2, 2))
    monkeypatch.setattr(loop, "_build_missing_cards", lambda limit: 1)
    monkeypatch.setattr(loop, "_score_missing", lambda limit: 3)
    monkeypatch.setattr(loop, "_refresh_mastery", lambda: 42)

    res = asyncio.run(loop.run_one_cycle())

    assert res["ok"] is True
    st = loop.get_status()
    assert st["cycles_completed"] == 1
    assert st["total_fetched"] == 2
    assert st["total_cards"] == 1
    assert st["total_scored"] == 3
    assert st["mastery_percent"] == 42
    assert st["running"] is False
    assert st["stage"] == "idle"
    assert len(st["history"]) == 1


def test_cycle_skipped_during_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = _make(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "_training_running", lambda: True)
    called = {"cards": False}

    def _cards(limit: int) -> int:
        called["cards"] = True
        return 0

    monkeypatch.setattr(loop, "_build_missing_cards", _cards)
    monkeypatch.setattr(loop, "_refresh_mastery", lambda: 10)

    res = asyncio.run(loop.run_one_cycle())

    assert res.get("skipped") == "training_running"
    assert called["cards"] is False  # ağır iş atlandı
    st = loop.get_status()
    assert st["stage"] == "paused_training"
    assert st["running"] is False


def test_fetch_skipped_when_not_due(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Çekim aralığı henüz dolmadıysa makale çekilmez ama kart/skor yine yapılır."""
    loop = _make(tmp_path, monkeypatch)
    loop.set_config(fetch_interval_hours=24)
    # Az önce çekim yapılmış gibi işaretle → bu turda çekim 'due' değil.
    import datetime as dt

    loop._state.last_fetch_at = dt.datetime.now(dt.UTC).isoformat()

    fetch_called = {"n": 0}

    def _fetch() -> tuple[int, int]:
        fetch_called["n"] += 1
        return (5, 5)

    monkeypatch.setattr(loop, "_training_running", lambda: False)
    monkeypatch.setattr(loop, "_fetch_new_papers", _fetch)
    monkeypatch.setattr(loop, "_build_missing_cards", lambda limit: 2)
    monkeypatch.setattr(loop, "_score_missing", lambda limit: 2)
    monkeypatch.setattr(loop, "_refresh_mastery", lambda: 50)

    res = asyncio.run(loop.run_one_cycle())

    assert res["ok"] is True
    assert fetch_called["n"] == 0  # çekim atlandı
    assert res["cards"] == 2
    assert res["scored"] == 2


def test_rebuild_runs_only_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loop = _make(tmp_path, monkeypatch)
    monkeypatch.setattr(loop, "_training_running", lambda: False)
    monkeypatch.setattr(loop, "_fetch_new_papers", lambda: (0, 0))
    monkeypatch.setattr(loop, "_build_missing_cards", lambda limit: 0)
    monkeypatch.setattr(loop, "_score_missing", lambda limit: 0)
    monkeypatch.setattr(loop, "_refresh_mastery", lambda: 60)
    calls = {"n": 0}

    def _rebuild(limit: int) -> int:
        calls["n"] += 1
        return 4

    monkeypatch.setattr(loop, "_rebuild_empty_cards", _rebuild)

    # Kapalıyken (varsayılan) rebuild çağrılmaz.
    res_off = asyncio.run(loop.run_one_cycle())
    assert calls["n"] == 0
    assert res_off["rebuilt"] == 0

    # Açıkken rebuild çağrılır ve sayaçlar güncellenir.
    loop.set_config(rebuild_empty=True)
    res_on = asyncio.run(loop.run_one_cycle())
    assert calls["n"] == 1
    assert res_on["rebuilt"] == 4
    st = loop.get_status()
    assert st["last_rebuilt"] == 4
    assert st["total_rebuilt"] == 4


def test_rebuild_caps_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """İçerik üretemeyen makale (ör. bozuk kaynak) en çok _MAX_REBUILD_ATTEMPTS kez denenir."""
    from app.research.rag_learning_loop import _MAX_REBUILD_ATTEMPTS

    loop = _make(tmp_path, monkeypatch)
    loop.set_config(rebuild_empty=True)
    calls = {"n": 0}

    class FakeStore:
        def list_approved_cards(self) -> list[dict]:
            return [{"paper_id": "p1", "card_json": {"title": "", "main_claim": ""}}]

        def list_pending_cards(self) -> list[dict]:
            return []

        def approve_card(self, card_id: str) -> bool:
            return False

    class FakeBuilder:
        def build(self, pid: str) -> None:
            calls["n"] += 1  # hep boş üretir (içerik gelmez)

    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", FakeStore)
    monkeypatch.setattr("app.brain.knowledge_card_builder.KnowledgeCardBuilder", FakeBuilder)
    monkeypatch.setattr("app.lora.dataset_builder.build_dataset", lambda cards: [])

    for _ in range(6):  # 6 tur dene; tavan 3'te durmalı
        loop._rebuild_empty_cards(limit=5)

    assert calls["n"] == _MAX_REBUILD_ATTEMPTS  # tavandan fazla denenmez
    assert loop._state.rebuild_attempts["p1"] == _MAX_REBUILD_ATTEMPTS


def test_rebuild_skips_papers_with_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zaten içerik üreten (with_real) makale rebuild edilmez."""
    loop = _make(tmp_path, monkeypatch)
    loop.set_config(rebuild_empty=True)
    calls = {"n": 0}

    class _Ex:
        def __init__(self, pid: str) -> None:
            self.metadata = {"paper_id": pid}

    class FakeStore:
        def list_approved_cards(self) -> list[dict]:
            return [{"paper_id": "p1", "card_json": {"title": "x", "main_claim": "y"}}]

        def list_pending_cards(self) -> list[dict]:
            return []

        def approve_card(self, card_id: str) -> bool:
            return False

    class FakeBuilder:
        def build(self, pid: str) -> None:
            calls["n"] += 1

    monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", FakeStore)
    monkeypatch.setattr("app.brain.knowledge_card_builder.KnowledgeCardBuilder", FakeBuilder)
    monkeypatch.setattr("app.lora.dataset_builder.build_dataset", lambda cards: [_Ex("p1")])

    n = loop._rebuild_empty_cards(limit=5)
    assert n == 0
    assert calls["n"] == 0  # p1 zaten içerikli → denenmez
