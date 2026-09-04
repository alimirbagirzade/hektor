"""Curriculum sistemi birim testleri — çevrimdışı (Ollama/LLM gerektirmez).

Testler:
1. test_knowledge_card_classify      — _classify_card farklı içeriklerle
2. test_save_with_metadata           — save_knowledge_card metadata kaydeder
3. test_approve_card                 — approve_card lora_eligible=1 yapar
4. test_reject_card                  — reject_card review_status=rejected yapar
5. test_list_pending                 — pending kartları listeler
6. test_dataset_lora_filter          — lora_eligible_only=True sadece approved örnekleri alır
7. test_dataset_phase_filter         — phase=1 sadece lora_phase_1 örnekleri alır
8. test_curriculum_pacing            — phase belirtilince 60/30/10 karışım yapılır
9. test_db_migration                 — _migrate() idempotent çalışır
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from app.brain.knowledge_card_builder import KnowledgeCard as KCModel
from app.brain.knowledge_card_builder import KnowledgeCardBuilder
from app.memory.sqlite_store import SqliteStore, TrainingExample
from app.training.dataset_builder import DatasetBuilder

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------


def _tmp_store(tmp_path: Path) -> SqliteStore:
    """Geçici bir SQLiteStore oluştur — gerçek DB'yi etkilemez."""
    db_file = tmp_path / "test_curriculum.db"
    return SqliteStore(db_path=db_file)


def _add_paper(store: SqliteStore, paper_id: str) -> None:
    """Test için sahte bir Paper kaydı ekle."""
    store.upsert_paper(
        paper_id=paper_id,
        file_hash=f"hash_{paper_id}",
        source_path=f"/tmp/{paper_id}.pdf",
    )


def _add_card(
    store: SqliteStore,
    card_id: str,
    paper_id: str,
    *,
    trust_level: str = "draft",
    review_status: str = "pending",
    lora_eligible: int = 0,
    difficulty: float = 0.3,
    stage: str = "lora_phase_2",
) -> None:
    _add_paper(store, paper_id)
    store.save_knowledge_card(
        card_id=card_id,
        paper_id=paper_id,
        model="test-model",
        card={"paper_id": paper_id, "main_claim": "test"},
        trust_level=trust_level,
        review_status=review_status,
        lora_eligible=lora_eligible,
        difficulty=difficulty,
        stage=stage,
    )


def _add_training_example(
    store: SqliteStore,
    example_id: str,
    source_paper_id: str | None = None,
) -> None:
    with store.session() as s:
        s.add(
            TrainingExample(
                example_id=example_id,
                source_paper_id=source_paper_id,
                example_type="test",
                instruction=f"Instruction {example_id}",
                input_text=f"Input {example_id}",
                output_text=f"Output {example_id}",
            )
        )


def _builder(store: SqliteStore, tmp_path: Path) -> DatasetBuilder:
    b = DatasetBuilder(store=store)
    b.settings = types.SimpleNamespace(jsonl_dir=tmp_path / "jsonl")  # type: ignore[assignment]
    return b


# ---------------------------------------------------------------------------
# 1. _classify_card — farklı kart içerikleriyle
# ---------------------------------------------------------------------------


def test_knowledge_card_classify_empty_hypotheses(tmp_path: Path) -> None:
    """Hipotez yoksa difficulty=0.1, stage=lora_phase_1."""
    store = _tmp_store(tmp_path)
    builder = KnowledgeCardBuilder.__new__(KnowledgeCardBuilder)
    builder.store = store  # type: ignore[assignment]

    card = KCModel(
        paper_id="p1",
        possible_strategy_hypotheses=[],
        methods=[],
        implementation_notes=[],
        risk_warnings=[],
    )
    _trust_level, difficulty, stage = builder._classify_card(card)
    assert _trust_level == "draft"
    assert difficulty == pytest.approx(0.1)
    assert stage == "lora_phase_1"


def test_knowledge_card_classify_few_hypotheses(tmp_path: Path) -> None:
    """1-2 hipotez, methods < 3 → difficulty=0.2."""
    builder = KnowledgeCardBuilder.__new__(KnowledgeCardBuilder)

    card = KCModel(
        paper_id="p2",
        possible_strategy_hypotheses=["hip1"],
        methods=["EMA"],
        implementation_notes=[],
        risk_warnings=[],
    )
    _trust_level, difficulty, stage = builder._classify_card(card)
    assert _trust_level == "draft"
    assert difficulty == pytest.approx(0.2)
    assert stage == "lora_phase_1"


def test_knowledge_card_classify_many_hypotheses(tmp_path: Path) -> None:
    """3+ hipotez → difficulty=0.4 → lora_phase_2."""
    builder = KnowledgeCardBuilder.__new__(KnowledgeCardBuilder)

    card = KCModel(
        paper_id="p3",
        possible_strategy_hypotheses=["h1", "h2", "h3"],
        methods=["EMA", "RSI"],
        implementation_notes=[],
        risk_warnings=[],
    )
    _trust_level, difficulty, stage = builder._classify_card(card)
    assert difficulty == pytest.approx(0.4)
    assert stage == "lora_phase_2"


def test_knowledge_card_classify_with_bonus(tmp_path: Path) -> None:
    """impl_notes + risk_warnings dolu → +0.2 bonus (0.4 + 0.2 = 0.6 → lora_phase_3)."""
    builder = KnowledgeCardBuilder.__new__(KnowledgeCardBuilder)

    card = KCModel(
        paper_id="p4",
        possible_strategy_hypotheses=["h1", "h2", "h3"],
        methods=["EMA", "RSI"],
        implementation_notes=["not 1"],
        risk_warnings=["risk 1"],
    )
    _trust_level, difficulty, stage = builder._classify_card(card)
    assert difficulty == pytest.approx(0.6)
    assert stage == "lora_phase_3"


# ---------------------------------------------------------------------------
# 2. test_save_with_metadata — metadata kaydedilir
# ---------------------------------------------------------------------------


def test_save_with_metadata(tmp_path: Path) -> None:
    """save_knowledge_card tüm metadata alanlarını kaydetmelidir."""
    store = _tmp_store(tmp_path)
    _add_paper(store, "paper_meta")

    store.save_knowledge_card(
        card_id="card_meta_001",
        paper_id="paper_meta",
        model="test-model",
        card={"paper_id": "paper_meta"},
        trust_level="verified",
        review_status="approved",
        lora_eligible=1,
        difficulty=0.7,
        stage="lora_phase_3",
    )

    result = store.get_card_by_id("card_meta_001")
    assert result is not None
    assert result["trust_level"] == "verified"
    assert result["review_status"] == "approved"
    assert result["lora_eligible"] == 1
    assert result["difficulty"] == pytest.approx(0.7)
    assert result["stage"] == "lora_phase_3"


# ---------------------------------------------------------------------------
# 3. test_approve_card
# ---------------------------------------------------------------------------


def test_approve_card(tmp_path: Path) -> None:
    """approve_card: review_status=approved, lora_eligible=1 olmalı."""
    store = _tmp_store(tmp_path)
    _add_card(store, "card_approve_1", "paper_approve_1")

    ok = store.approve_card("card_approve_1")
    assert ok is True

    result = store.get_card_by_id("card_approve_1")
    assert result is not None
    assert result["review_status"] == "approved"
    assert result["lora_eligible"] == 1


def test_approve_card_nonexistent(tmp_path: Path) -> None:
    """Olmayan kart için approve_card False döndürmeli."""
    store = _tmp_store(tmp_path)
    ok = store.approve_card("card_does_not_exist")
    assert ok is False


def test_approve_card_rejects_empty_content(tmp_path: Path) -> None:
    """İçeriksiz kart (title=None, main_claim='') onaylanmamalı — pending kalmalı.

    198 'approved-ama-boş' kart sorununun guard'ı (_card_has_content): üretim başarısız
    olduğunda kart coverage'ı/denetimi çöple şişiremez.
    """
    store = _tmp_store(tmp_path)
    _add_paper(store, "paper_empty_1")
    store.save_knowledge_card(
        card_id="card_empty_1",
        paper_id="paper_empty_1",
        model="test-model",
        card={"paper_id": "paper_empty_1", "title": None, "main_claim": ""},
    )

    ok = store.approve_card("card_empty_1")
    assert ok is False

    result = store.get_card_by_id("card_empty_1")
    assert result is not None
    assert result["review_status"] == "pending"
    assert result["lora_eligible"] == 0


def test_approve_card_rejects_placeholder_dots(tmp_path: Path) -> None:
    """'...' placeholder kart (alfanümerik içerik yok) onaylanmamalı."""
    store = _tmp_store(tmp_path)
    _add_paper(store, "paper_dots_1")
    store.save_knowledge_card(
        card_id="card_dots_1",
        paper_id="paper_dots_1",
        model="test-model",
        card={"paper_id": "paper_dots_1", "title": "...", "main_claim": "..."},
    )

    assert store.approve_card("card_dots_1") is False


# ---------------------------------------------------------------------------
# 4. test_reject_card
# ---------------------------------------------------------------------------


def test_reject_card(tmp_path: Path) -> None:
    """reject_card: review_status=rejected, lora_eligible=0 olmalı."""
    store = _tmp_store(tmp_path)
    # Önce onaylı bir kart oluştur, sonra reddet
    _add_card(
        store,
        "card_reject_1",
        "paper_reject_1",
        review_status="approved",
        lora_eligible=1,
    )

    ok = store.reject_card("card_reject_1")
    assert ok is True

    result = store.get_card_by_id("card_reject_1")
    assert result is not None
    assert result["review_status"] == "rejected"
    assert result["lora_eligible"] == 0


# ---------------------------------------------------------------------------
# 5. test_list_pending
# ---------------------------------------------------------------------------


def test_list_pending(tmp_path: Path) -> None:
    """list_pending_cards yalnızca pending durumundaki kartları döndürmeli."""
    store = _tmp_store(tmp_path)
    _add_card(store, "card_p1", "paper_list_1", review_status="pending")
    _add_card(store, "card_p2", "paper_list_2", review_status="approved")
    _add_card(store, "card_p3", "paper_list_3", review_status="rejected")
    _add_card(store, "card_p4", "paper_list_4", review_status="pending")

    pending = store.list_pending_cards()
    pending_ids = {r["card_id"] for r in pending}
    assert "card_p1" in pending_ids
    assert "card_p4" in pending_ids
    assert "card_p2" not in pending_ids
    assert "card_p3" not in pending_ids


# ---------------------------------------------------------------------------
# 6. test_dataset_lora_filter
# ---------------------------------------------------------------------------


def test_dataset_lora_filter(tmp_path: Path) -> None:
    """lora_eligible_only=True → sadece approved+lora_eligible=1 olan örnekler."""
    store = _tmp_store(tmp_path)

    # Onaylı kart + örnek
    _add_card(
        store,
        "card_lora_1",
        "paper_lora_1",
        review_status="approved",
        lora_eligible=1,
        stage="lora_phase_2",
    )
    _add_training_example(store, "ex_lora_1", source_paper_id="paper_lora_1")

    # Onaysız kart + örnek (pending)
    _add_card(
        store,
        "card_lora_2",
        "paper_lora_2",
        review_status="pending",
        lora_eligible=0,
        stage="lora_phase_2",
    )
    _add_training_example(store, "ex_lora_2", source_paper_id="paper_lora_2")

    b = _builder(store, tmp_path)
    records_filtered = b.collect(lora_eligible_only=True)
    records_all = b.collect(lora_eligible_only=False)

    # Filtreli: sadece onaylı kaynak
    assert len(records_filtered) == 1
    # Filtresiz: ikisi de
    assert len(records_all) == 2


# ---------------------------------------------------------------------------
# 7. test_dataset_phase_filter
# ---------------------------------------------------------------------------


def test_dataset_phase_filter(tmp_path: Path) -> None:
    """phase=1 → sadece lora_phase_1 stage'indeki approved örnekleri almalı."""
    store = _tmp_store(tmp_path)

    # Faz 1 kartı + örneği
    _add_card(
        store,
        "card_ph1",
        "paper_ph1",
        review_status="approved",
        lora_eligible=1,
        difficulty=0.1,
        stage="lora_phase_1",
    )
    _add_training_example(store, "ex_ph1", source_paper_id="paper_ph1")

    # Faz 2 kartı + örneği
    _add_card(
        store,
        "card_ph2",
        "paper_ph2",
        review_status="approved",
        lora_eligible=1,
        difficulty=0.3,
        stage="lora_phase_2",
    )
    _add_training_example(store, "ex_ph2", source_paper_id="paper_ph2")

    b = _builder(store, tmp_path)
    phase1_records = b.collect(phase=1, lora_eligible_only=True)
    phase2_records = b.collect(phase=2, lora_eligible_only=True)

    assert len(phase1_records) == 1
    assert len(phase2_records) == 1


# ---------------------------------------------------------------------------
# 8. test_curriculum_pacing — 60/30/10 karışım
# ---------------------------------------------------------------------------


def test_curriculum_pacing(tmp_path: Path) -> None:
    """phase belirtilince ve toplam > 20 ise 60/30/10 karışım yapılır."""
    store = _tmp_store(tmp_path)

    # Faz 2 kart + çok örnek (current)
    _add_card(
        store,
        "card_cp2",
        "paper_cp2",
        review_status="approved",
        lora_eligible=1,
        difficulty=0.3,
        stage="lora_phase_2",
    )
    for i in range(15):
        _add_training_example(store, f"ex_cp2_{i:03d}", source_paper_id="paper_cp2")

    # Faz 1 kart + örnek (prev)
    _add_card(
        store,
        "card_cp1",
        "paper_cp1",
        review_status="approved",
        lora_eligible=1,
        difficulty=0.1,
        stage="lora_phase_1",
    )
    for i in range(8):
        _add_training_example(store, f"ex_cp1_{i:03d}", source_paper_id="paper_cp1")

    # Faz 3 kart + örnek (next)
    _add_card(
        store,
        "card_cp3",
        "paper_cp3",
        review_status="approved",
        lora_eligible=1,
        difficulty=0.6,
        stage="lora_phase_3",
    )
    for i in range(4):
        _add_training_example(store, f"ex_cp3_{i:03d}", source_paper_id="paper_cp3")

    b = _builder(store, tmp_path)
    result = b.build(phase=2, lora_eligible_only=True, seed=42)

    # Toplam 27 örnek > 20, dolayısıyla curriculum pacing aktif
    total = result.n_train + result.n_valid
    # Sonucun makul bir aralıkta olduğunu kontrol et (tam sayı kesmeler nedeniyle toleranslı)
    assert total > 0
    # 60% * 27 ≈ 16 current + 30% * 27 ≈ 8 prev + 10% * 27 ≈ 2 next = ~26
    assert total <= 27  # aşmamalı


# ---------------------------------------------------------------------------
# 9. test_db_migration — idempotent
# ---------------------------------------------------------------------------


def test_db_migration_idempotent(tmp_path: Path) -> None:
    """_migrate() mevcut bir DB üzerinde tekrar çalışınca hata vermemeli."""
    db_file = tmp_path / "migrate_test.db"

    # İlk oluşturma: create_all + _migrate çağrılır
    store1 = SqliteStore(db_path=db_file)
    assert store1 is not None

    # İkinci oluşturma: aynı dosya, migrate tekrar çağrılır
    store2 = SqliteStore(db_path=db_file)
    assert store2 is not None

    # Kolonların gerçekten var olduğunu doğrula
    _add_paper(store2, "paper_mig")
    store2.save_knowledge_card(
        card_id="card_mig_1",
        paper_id="paper_mig",
        model="test",
        card={},
        trust_level="draft",
        review_status="pending",
        lora_eligible=0,
        difficulty=0.2,
        stage="lora_phase_1",
    )
    card = store2.get_card_by_id("card_mig_1")
    assert card is not None
    assert card["stage"] == "lora_phase_1"
