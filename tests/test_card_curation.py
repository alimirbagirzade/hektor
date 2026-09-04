"""Kart kürasyonu testleri — orphan karantina + version-collapse (offline, hermetik)."""

from __future__ import annotations

from pathlib import Path

from app.lora.card_curation import (
    apply_curation,
    card_richness,
    pick_canonical_card_id,
    plan_curation,
)
from app.memory.sqlite_store import SqliteStore


def _card(
    card_id: str,
    paper_id: str,
    *,
    summary: str = "RSI momentum osilatörü ve formülü.",
    created_at: str = "2026-01-01T00:00:00Z",
    difficulty: float = 0.4,
    eligible: int = 1,
    status: str = "approved",
) -> dict:
    return {
        "card_id": card_id,
        "paper_id": paper_id,
        "review_status": status,
        "lora_eligible": eligible,
        "difficulty": difficulty,
        "created_at": created_at,
        "card_json": {"title": "RSI", "summary": summary, "formulas": []},
    }


# --- saf fonksiyonlar -------------------------------------------------------


def test_card_richness_prefers_longer_content() -> None:
    poor = _card("a", "p1", summary="kısa")
    rich = _card("b", "p1", summary="çok daha uzun ve içerik bakımından zengin bir özet metni")
    assert card_richness(rich) > card_richness(poor)


def test_pick_canonical_selects_richest_then_newest() -> None:
    """En zengin kart kanonik; eşit zenginlikte en yeni created_at kazanır (deterministik)."""
    poor = _card("poor", "p1", summary="kısa özet")
    rich_old = _card("rich_old", "p1", summary="uzun zengin özet " * 4, created_at="2026-01-01")
    rich_new = _card("rich_new", "p1", summary="uzun zengin özet " * 4, created_at="2026-06-01")
    # Sıralamadan bağımsız aynı sonuç
    assert pick_canonical_card_id([poor, rich_old, rich_new]) == "rich_new"
    assert pick_canonical_card_id([rich_new, poor, rich_old]) == "rich_new"


def test_plan_curation_flags_orphans() -> None:
    cards = [_card("c1", "paper_real"), _card("c2", "paper_ghost")]
    plan = plan_curation(cards, valid_paper_ids={"paper_real"})
    assert plan.orphan_card_ids == {"c2"}
    assert plan.orphan_paper_ids == {"paper_ghost"}
    assert "c1" in plan.keep_card_ids


def test_plan_curation_version_collapse_keeps_one_per_paper() -> None:
    """Aynı paper'dan 3 kart → 1 tutulur, 2 düşürülür."""
    cards = [
        _card("c1", "paper_x", summary="kısa"),
        _card("c2", "paper_x", summary="orta uzunlukta bir özet metni burada"),
        _card("c3", "paper_x", summary="en uzun ve en zengin özet metni " * 3),
    ]
    plan = plan_curation(cards, valid_paper_ids={"paper_x"})
    assert plan.keep_card_ids == {"c3"}  # en zengin
    assert plan.redundant_card_ids == {"c1", "c2"}
    assert plan.collapsed_paper_ids == {"paper_x"}
    assert plan.demote_card_ids == {"c1", "c2"}


def test_plan_curation_ignores_ineligible_cards() -> None:
    """Zaten lora_eligible=0 / rejected kart plana girmez (no-op)."""
    cards = [
        _card("c1", "paper_ghost", eligible=0),
        _card("c2", "paper_ghost", status="rejected"),
    ]
    plan = plan_curation(cards, valid_paper_ids={"paper_real"})
    assert plan.demote_card_ids == set()


# --- DB uygula (hermetik) ---------------------------------------------------


def _seed(store: SqliteStore) -> None:
    store.upsert_paper(paper_id="paper_real", file_hash="h_real", source_path="x.pdf")
    # paper_real: 2 kart (collapse) — biri zengin tutulur
    store.save_knowledge_card(
        card_id="keep",
        paper_id="paper_real",
        model="t",
        card={"title": "RSI", "summary": "uzun zengin özet metni " * 4},
        review_status="approved",
        lora_eligible=1,
        difficulty=0.4,
    )
    store.save_knowledge_card(
        card_id="dup",
        paper_id="paper_real",
        model="t",
        card={"title": "RSI", "summary": "kısa"},
        review_status="approved",
        lora_eligible=1,
        difficulty=0.4,
    )
    # paper_ghost: papers tablosunda YOK → orphan
    store.save_knowledge_card(
        card_id="orphan1",
        paper_id="paper_ghost",
        model="t",
        card={"title": "DL", "summary": "%92 doğruluk iddiası içeren orphan kart metni"},
        review_status="approved",
        lora_eligible=1,
        difficulty=0.4,
    )


def test_apply_curation_dry_run_does_not_mutate(tmp_path: Path) -> None:
    store = SqliteStore(db_path=tmp_path / "c.db")
    _seed(store)
    report = apply_curation(store, dry_run=True)
    assert report.dry_run is True
    assert report.orphan_demoted == 1
    assert report.redundant_demoted == 1
    # DB değişmedi — hepsi hâlâ eligible
    assert store.get_card_by_id("orphan1")["lora_eligible"] == 1
    assert store.get_card_by_id("dup")["lora_eligible"] == 1
    assert store.get_card_by_id("keep")["lora_eligible"] == 1


def test_apply_curation_run_demotes_orphan_and_redundant(tmp_path: Path) -> None:
    store = SqliteStore(db_path=tmp_path / "c.db")
    _seed(store)
    report = apply_curation(store, dry_run=False)
    assert report.dry_run is False
    assert report.total_demoted == 2
    # orphan + dup düşürüldü; kanonik korundu
    assert store.get_card_by_id("orphan1")["lora_eligible"] == 0
    assert store.get_card_by_id("dup")["lora_eligible"] == 0
    assert store.get_card_by_id("keep")["lora_eligible"] == 1
    # review_status KORUNDU (silinmedi/reddedilmedi) → geri alınabilir
    assert store.get_card_by_id("orphan1")["review_status"] == "approved"


def test_apply_curation_is_idempotent(tmp_path: Path) -> None:
    """İkinci --run hiçbir yeni kart düşürmemeli (zaten eligible olanlar kaldı)."""
    store = SqliteStore(db_path=tmp_path / "c.db")
    _seed(store)
    apply_curation(store, dry_run=False)
    second = apply_curation(store, dry_run=False)
    assert second.total_demoted == 0
