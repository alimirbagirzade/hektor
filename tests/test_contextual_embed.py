"""build_embed_text testleri (Faz P2 — Contextual Retrieval ön-eki)."""

from __future__ import annotations

from app.memory.paper_indexer import build_embed_text

_TEXT = "ATR volatiliteyi ölçer ve momentum filtresinde kullanılır."


def test_contextual_off_returns_raw() -> None:
    assert build_embed_text(_TEXT, "Başlık", "Methods", contextual=False) == _TEXT


def test_contextual_on_adds_title_and_section() -> None:
    out = build_embed_text(_TEXT, "Momentum Paper", "Methods", contextual=True)
    assert out == f"Momentum Paper / Methods: {_TEXT}"


def test_contextual_on_title_only() -> None:
    out = build_embed_text(_TEXT, "Momentum Paper", None, contextual=True)
    assert out == f"Momentum Paper: {_TEXT}"


def test_contextual_on_no_meta_falls_back_to_text() -> None:
    assert build_embed_text(_TEXT, "", "", contextual=True) == _TEXT
    assert build_embed_text(_TEXT, None, None, contextual=True) == _TEXT


def test_original_text_unchanged_in_prefix() -> None:
    # Ön-ek eklenir ama orijinal metin korunur (Chroma document'ı için kritik).
    out = build_embed_text(_TEXT, "T", "S", contextual=True)
    assert out.endswith(_TEXT)
