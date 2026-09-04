"""cross_paper_synthesizer._synthesis_seed — determinizm (CLAUDE.md kural 6).

Sentez LLM çağrısının seed'i girdiden (formül bloğu) deterministik türetilir:
aynı girdi → aynı seed (reprodüklenebilir), farklı girdi → farklı seed (çeşitlilik).
"""

from __future__ import annotations

from app.research.cross_paper_synthesizer import _synthesis_seed


def test_seed_deterministic_for_same_block() -> None:
    """Aynı blok her çağrıda AYNI seed'i vermeli (süreç-içi + süreçler-arası kararlı)."""
    block = "• [momentum] RSI (makale: p1)\n  Formül: 100-100/(1+RS)"
    assert _synthesis_seed(block) == _synthesis_seed(block)


def test_seed_varies_by_block() -> None:
    """Farklı blok → farklı seed (sabit-seed 'hep aynı çıktı' tuzağına düşmez)."""
    a = _synthesis_seed("• [momentum] RSI")
    b = _synthesis_seed("• [volatilite] ATR")
    assert a != b


def test_seed_is_nonneg_int_in_range() -> None:
    """Seed negatif olmayan 32-bit aralıkta int (Ollama/OpenAI seed sözleşmesi)."""
    s = _synthesis_seed("herhangi bir blok")
    assert isinstance(s, int)
    assert 0 <= s < 2**32


def test_seed_stable_known_value() -> None:
    """Regresyon kilidi: bilinen girdi için seed sabit kalmalı (sha256[:8] kararlı).

    Değer değişirse (ör. hash tabanı/dilim değişti) reprodüklenebilirlik bozulur.
    """
    # sha256("abc")[:8] = "ba7816bf" → int = 0xba7816bf
    assert _synthesis_seed("abc") == 0xBA7816BF
