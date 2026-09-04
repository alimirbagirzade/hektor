"""LoRA kontrol düzlemi (orkestratör) testleri."""

from __future__ import annotations

from pathlib import Path

from app.lora.control_plane import LoRAControlPlane
from app.memory.sqlite_store import SqliteStore


def _save_approved_card(store: SqliteStore, card_id: str, difficulty: float) -> None:
    store.save_knowledge_card(
        card_id=card_id,
        paper_id=f"paper_{card_id}",
        model="test",
        card={
            "title": "RSI",
            "summary": (
                "RSI, fiyatın aşırı alım ve aşırı satım bölgelerini ölçen bir "
                "momentum osilatörüdür ve 0-100 arasında değer alır."
            ),
            "formulas": [],
        },
        review_status="approved",
        lora_eligible=1,
        difficulty=difficulty,
    )


def test_run_audit_produces_report() -> None:
    """run_audit Gate kümesini çalıştırıp rapor üretmeli."""
    store = SqliteStore()
    _save_approved_card(store, "c1", 0.3)
    plane = LoRAControlPlane(store=store)

    report = plane.run_audit()
    assert report.total_input >= 1
    assert len(report.stages) >= 8


def test_run_full_includes_split() -> None:
    """run_full Gate 8'i ekleyip dataset_split üretmeli."""
    store = SqliteStore()
    for i in range(6):
        _save_approved_card(store, f"f{i}", 0.4)
    plane = LoRAControlPlane(store=store)

    report = plane.run_full()
    assert report.dataset_split is not None
    gate_ids = {s.gate_id for s in report.stages}
    assert 8 in gate_ids


def test_gate_7_safety_scans_quality_rejected_cards(tmp_path: Path) -> None:
    """B1: Gate 4'te (kısa cevap) elenen ama sır taşıyan kart Gate 7'de YAKALANMALI.

    Safety BLOCKER, Gate 4 elemesinden BAĞIMSIZ tüm içerikli kartları tarar. Eski kodda
    gate_7 yalnız clean_cards'ı (Gate 4 geçenler) tarıyordu → kısa+sırlı kart kaçıyordu.
    """
    store = SqliteStore(db_path=tmp_path / "b1.db")
    store.save_knowledge_card(
        card_id="leak1",
        paper_id="paper_leak1",
        model="test",
        # summary <50 karakter → Gate 4 'cevap çok kısa' diye eler; ama sır içerir.
        card={"title": "Not", "summary": "password=hunter2xy"},
        review_status="approved",
        lora_eligible=1,
        difficulty=0.3,
    )
    plane = LoRAControlPlane(store=store)
    report = plane.run_audit()

    gate7 = next(s for s in report.stages if s.gate_id == 7)
    assert gate7.passed is False  # sır yakalandı (eski kodda kaçardı)


def test_generate_report_writes_markdown(tmp_path: Path) -> None:
    """generate_report Markdown üretip dosyaya yazmalı."""
    store = SqliteStore()
    _save_approved_card(store, "c1", 0.5)
    plane = LoRAControlPlane(store=store)
    report = plane.run_audit()

    out = tmp_path / "report.md"
    markdown = plane.generate_report(report, output_path=out)
    assert "# LoRA Denetim Raporu" in markdown
    assert out.exists()
