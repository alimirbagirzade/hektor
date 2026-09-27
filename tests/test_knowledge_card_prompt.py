"""Kart talimatı (app/prompts/knowledge_card.md) sözleşme testi.

Regresyon (2026-09-27 lora-audit incelemesi): talimat hipotezlerin "somut ve ölçülebilir"
olmasını istiyordu; model bunu makalede OLMAYAN rakam uydurarak karşıladı (292 onaylı
kartın ~%5'i: "outperform ... by 15%", "90% accuracy") ve yönlendirici dil üretti
("investors should consider ..."). Bu kuralların talimattan sessizce düşmesini engeller.
"""

from __future__ import annotations

import re

from app.brain.prompt_loader import load_prompt


def _prompt() -> str:
    return load_prompt("knowledge_card")


def test_prompt_no_longer_demands_measurable_hypotheses() -> None:
    """Rakam uydurtan "somut ve ölçülebilir" talimatı geri gelmemeli."""
    assert "ölçülebilir" not in _prompt()


def test_prompt_forbids_numbers_not_in_source() -> None:
    p = _prompt()
    assert "Rakam uydurma yasak" in p
    assert "AYNEN geçmeyen" in p
    assert "The paper reports" in p  # doğrulanan sonuç kaynağa atfedilir


def test_prompt_requires_test_if_hypothesis_with_oos_and_costs() -> None:
    p = _prompt()
    assert "Test if" in p
    assert "out-of-sample" in p
    assert "net of transaction costs" in p
    # Makaleden hipotez çıkmıyorsa boş bırak — uydurma hipotez yok.
    assert re.search(r"listeyi BOŞ bırak", p)


def test_prompt_bans_advice_language() -> None:
    p = _prompt()
    assert "Tavsiye dili yasak" in p
    for phrase in ("traders/investors should", "directly applicable", "guaranteed"):
        assert phrase in p, phrase


def test_prompt_requires_main_claim_summary() -> None:
    """Regresyon (A/B, 2026-09-27): "hiçbir şeyi uydurma" tek başına kalınca model ders
    kitabında (Hogg) konu özetini de "uydurma" sayıp main_claim'i boş bıraktı → kart hiç
    üretilmedi. Konu özetinin uydurma OLMADIĞI açıkça yazılı kalmalı."""
    p = _prompt()
    assert "main_claim her zaman doldurulur" in p
    assert "uydurma DEĞİLDİR" in p


def test_prompt_keeps_json_schema_fields() -> None:
    """Şema alanları builder'ın beklediğiyle aynı kalmalı."""
    p = _prompt()
    for field in (
        "title",
        "main_claim",
        "methods",
        "trading_relevance",
        "limitations",
        "possible_strategy_hypotheses",
        "risk_warnings",
        "implementation_notes",
    ):
        assert f'"{field}"' in p, field
