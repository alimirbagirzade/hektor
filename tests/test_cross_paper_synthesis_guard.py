"""cross_paper_synthesizer üçlü 'cross-paper' sözleşmesi (CLAUDE.md kural 7).

İkili yol `_select_cross_paper` ile ≥2 FARKLI makaleyi zorlar; üçlü yol da aynı sözleşmeye
uymalı. Tek makale 3 kategoride formüle sahipse üçlü tek makaleye çöker → 'farklı akademik
makalelerden' iddiası YANLIŞ olur (uydurma provenans) ve LoRA eğitim verisine sızar. Bu
testler guard'ı (Kademe-2 bulgusu) ve meşru çok-makale sentezini bozmadığını kilitler.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.research.cross_paper_synthesizer import CrossPaperSynthesizer


def _stub_synth(
    formulas: list[dict[str, Any]], saved: list[str], monkeypatch: pytest.MonkeyPatch
) -> CrossPaperSynthesizer:
    """Gerçek store/LLM bağlamadan synthesize_all'ı sınanabilir kıl (yan etkiler stub)."""
    synth = CrossPaperSynthesizer.__new__(CrossPaperSynthesizer)
    synth.store = type("S", (), {"list_formulas": staticmethod(lambda: formulas)})()  # type: ignore[assignment]
    monkeypatch.setattr(synth, "_exists", lambda ex_id: False, raising=False)
    monkeypatch.setattr(synth, "_build", lambda sel, cats: {"ok": True}, raising=False)
    monkeypatch.setattr(synth, "_save", lambda ex_id, ex: saved.append(ex_id), raising=False)
    return synth


def test_triple_synthesis_skips_single_paper(monkeypatch: pytest.MonkeyPatch) -> None:
    """3 kategori de AYNI paper_id ise hiçbir sentez örneği kaydedilmemeli (Kural 7)."""
    formulas = [
        {"formula_id": "f1", "paper_id": "p1", "category": "momentum", "name": "A"},
        {"formula_id": "f2", "paper_id": "p1", "category": "volatility", "name": "B"},
        {"formula_id": "f3", "paper_id": "p1", "category": "trend", "name": "C"},
    ]
    saved: list[str] = []
    synth = _stub_synth(formulas, saved, monkeypatch)
    total = synth.synthesize_all(force=True)
    assert total == 0
    assert saved == []


def test_triple_synthesis_allows_distinct_papers(monkeypatch: pytest.MonkeyPatch) -> None:
    """≥2 farklı makale içeren üçlü meşrudur — guard sentezi körlememeli."""
    formulas = [
        {"formula_id": "f1", "paper_id": "p1", "category": "momentum", "name": "A"},
        {"formula_id": "f2", "paper_id": "p2", "category": "volatility", "name": "B"},
        {"formula_id": "f3", "paper_id": "p3", "category": "trend", "name": "C"},
    ]
    saved: list[str] = []
    synth = _stub_synth(formulas, saved, monkeypatch)
    total = synth.synthesize_all(force=True)
    assert total >= 1
    assert saved
