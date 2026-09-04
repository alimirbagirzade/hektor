"""Doğrulama kapıları (Gate 0-8) entegrasyon testleri."""

from __future__ import annotations

from app.lora.dataset_builder import SYSTEM_PROMPT
from app.lora.gates import (
    gate_0_source,
    gate_1_schema,
    gate_2_curriculum,
    gate_3_domain,
    gate_4_quality,
    gate_5_math,
    gate_7_safety,
    gate_8_split,
)


def _approved_card(card_id: str, summary: str, difficulty: float = 0.5) -> dict:
    return {
        "card_id": card_id,
        "paper_id": f"paper_{card_id}",
        "review_status": "approved",
        "lora_eligible": 1,
        "difficulty": difficulty,
        "created_at": "2026-01-01T00:00:00Z",
        "card_json": {
            "title": "RSI",
            "summary": summary,
            "formulas": [],
        },
    }


def _example(source_id: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "RSI nedir?"},
            {"role": "assistant", "content": "RSI bir momentum osilatörüdür."},
        ],
        "metadata": {"source_id": source_id},
    }


def test_gate_0_passes_for_valid_card() -> None:
    """Geçerli onaylı kart (domain + created_at) Gate 0'ı geçmeli."""
    card = _approved_card("c1", "RSI momentum göstergesi ve formül içerir.")
    result = gate_0_source([card])
    assert result.passed is True


def test_gate_0_rejects_unapproved() -> None:
    """Onaysız kart Gate 0'da reddedilmeli."""
    card = _approved_card("c1", "momentum formül")
    card["review_status"] = "pending"
    result = gate_0_source([card])
    assert result.passed is False
    assert result.rejected_count == 1


def test_gate_0_rejects_orphan_paper_id() -> None:
    """paper_id papers tablosunda (valid set) YOKSA kart orphan → reddedilmeli (kural 7)."""
    card = _approved_card("c1", "RSI momentum göstergesi ve formül içerir.")
    # card.paper_id = 'paper_c1'; geçerli sette yok → orphan.
    result = gate_0_source([card], valid_paper_ids={"paper_other"})
    assert result.passed is False
    assert result.rejected_count == 1
    assert any("orphan" in d for d in result.details)


def test_gate_0_passes_when_paper_id_present() -> None:
    """paper_id geçerli sette varsa (kaynak gerçek) Gate 0 geçmeli."""
    card = _approved_card("c1", "RSI momentum göstergesi ve formül içerir.")
    result = gate_0_source([card], valid_paper_ids={"paper_c1"})
    assert result.passed is True


def test_gate_0_existence_check_is_opt_in() -> None:
    """valid_paper_ids verilmezse varlık denetimi atlanır (geriye-dönük uyum)."""
    card = _approved_card("c1", "RSI momentum göstergesi ve formül içerir.")
    # Aynı kart, set verilmeden → eski davranış (orphan denetimi yok) → geçer.
    assert gate_0_source([card]).passed is True


def test_gate_6_uppercase_turkish_markers_flagged() -> None:
    """BÜYÜK harf 'ÇELİŞKİ'/'TUTARSIZ' işaretleri tr_fold ile insan incelemesine düşmeli."""
    from app.lora.gates import gate_6_philosophy

    card = _approved_card("c6", "Bu kartta açık bir ÇELİŞKİ ve TUTARSIZ ifade var.")
    result = gate_6_philosophy([card])
    assert result.review_count >= 1


# --------------------------------------------------------------------------- #
# Gate 6 disiplin dili (tavsiye/yönlendirme/üstünlük) — adversarial
# --------------------------------------------------------------------------- #
# Gerçek-pozitif: tavsiye/yönlendirme dili işaretlenmeli. Yanlış-pozitif:
# olumsuzlanmış (alçakgönüllü) ifade işaretlenMEMELİ.


def test_gate_6_directly_applied_advice_flagged() -> None:
    """'can be directly applied' tavsiye dili inceleme işaretlemeli."""
    from app.lora.gates import gate_6_philosophy

    card = _approved_card("c1", "This model can be directly applied to short-term trading.")
    result = gate_6_philosophy([card])
    assert result.review_count >= 1


def test_gate_6_traders_should_directive_flagged() -> None:
    """'Traders should…' yönlendirme dili inceleme işaretlemeli (kural 1 ihlali)."""
    from app.lora.gates import gate_6_philosophy

    card = _approved_card("c1", "Traders should combine technical and fundamental analysis.")
    result = gate_6_philosophy([card])
    assert result.review_count >= 1


def test_gate_6_superior_performance_flagged() -> None:
    """Falsifiye-edilemez 'superior performance' iddiası inceleme işaretlemeli."""
    from app.lora.gates import gate_6_philosophy

    card = _approved_card(
        "c1", "Transformers have demonstrated superior performance in forecasting."
    )
    result = gate_6_philosophy([card])
    assert result.review_count >= 1


def test_gate_6_negated_directly_applicable_not_flagged_en() -> None:
    """'not directly applicable' (olumsuzlanmış, alçakgönüllü) işaretlenMEMELİ (FP)."""
    from app.lora.gates import gate_6_philosophy

    card = _approved_card(
        "c1", "The method is not directly applicable to continuous action spaces."
    )
    result = gate_6_philosophy([card])
    assert result.review_count == 0


def test_gate_6_negated_directly_applicable_not_flagged_tr() -> None:
    """'doğrudan uygulanabilir değildir' Türkçe olumsuzlama işaretlenMEMELİ (FP)."""
    from app.lora.gates import gate_6_philosophy

    card = _approved_card(
        "c1", "Bu yöntem sürekli aksiyon uzayına doğrudan uygulanabilir değildir."
    )
    result = gate_6_philosophy([card])
    assert result.review_count == 0


def test_gate_6_soft_block_fails_when_discipline_pervasive() -> None:
    """İncelemeli oran eşiği aşarsa (yeterli kartla) Gate 6 yumuşak-blok BAŞARISIZ olmalı."""
    from app.lora.gates import gate_6_philosophy

    advice = [
        _approved_card(f"a{i}", "This approach can be directly applied to live trading.")
        for i in range(12)
    ]
    clean = [
        _approved_card(f"c{i}", "RSI fiyatın momentumunu ölçen bir osilatördür ve sınırlıdır.")
        for i in range(13)
    ]
    result = gate_6_philosophy(advice + clean)  # 12/25 = 0.48 > 0.25
    assert result.passed is False
    assert result.review_count == 12


def test_gate_6_soft_block_passes_when_discipline_rare() -> None:
    """Disiplin dili seyrekse (oran eşik altı) Gate 6 geçmeli."""
    from app.lora.gates import gate_6_philosophy

    advice = [_approved_card("a0", "This can be directly applied to trading.")]
    clean = [
        _approved_card(f"c{i}", "RSI fiyatın momentumunu ölçen bir osilatördür ve sınırlıdır.")
        for i in range(24)
    ]
    result = gate_6_philosophy(advice + clean)  # 1/25 = 0.04 < 0.25
    assert result.passed is True
    assert result.review_count == 1


def test_gate_6_small_batch_never_soft_blocks() -> None:
    """min_cards altı küçük batch'te oran yüksek olsa da yumuşak-blok uygulanmaz."""
    from app.lora.gates import gate_6_philosophy

    cards = [_approved_card("a0", "This can be directly applied to trading.")]
    result = gate_6_philosophy(cards)  # 1 kart < 20 → passed True
    assert result.passed is True
    assert result.review_count == 1


def test_gate_5_flags_bare_performance_claim_without_blocking() -> None:
    """Gate 5 çıplak '92% accuracy' iddiasını inceleme işaretlemeli, reddetmemeli."""
    card = _approved_card("c1", "This model predicts price movements with 92% accuracy.")
    result = gate_5_math([card])
    assert result.passed is True  # blok değil
    assert result.review_count >= 1


def test_gate_1_schema_validates_message_order() -> None:
    """Doğru rol sıralı örnek Gate 1'i geçmeli."""
    result = gate_1_schema([_example("s1")])
    assert result.passed is True


def test_gate_2_rejects_invalid_difficulty() -> None:
    """0-1 dışı difficulty Gate 2'de reddedilmeli."""
    card = _approved_card("c1", "momentum", difficulty=1.5)
    result = gate_2_curriculum([card])
    assert result.passed is False


def test_gate_3_requires_domain() -> None:
    """Domain bulunamayan kart Gate 3'te reddedilmeli."""
    card = _approved_card("c1", "xyzzy plugh")
    card["card_json"]["title"] = "qqq"
    result = gate_3_domain([card])
    assert result.passed is False


def test_gate_4_quality_returns_clean_list() -> None:
    """Gate 4 temiz kart listesini de döndürmeli."""
    card = _approved_card(
        "c1",
        "RSI fiyatın aşırı alım ve aşırı satım bölgelerini ölçen momentum osilatörüdür.",
    )
    result, clean = gate_4_quality([card])
    assert result.passed is True
    assert len(clean) == 1


def test_gate_5_math_flags_overconfident() -> None:
    """Aşırı emin ifade Gate 5'te reddedilmeli."""
    card = _approved_card("c1", "Bu strateji kesinlikle garanti kazandırır her zaman.")
    result = gate_5_math([card])
    assert result.passed is False


def test_gate_7_safety_blocks_secret() -> None:
    """Sır içeren kart Gate 7'de (BLOCKER) reddedilmeli."""
    card = _approved_card("c1", "password=hunter2 ile giriş yapılır gerçekten uzun metin.")
    result = gate_7_safety([card])
    assert result.passed is False


def test_gate_7_safety_scans_risk_warnings_field() -> None:
    """Sır YALNIZ risk_warnings alanındaysa bile Gate 7 (BLOCKER) yakalamalı.

    Kör nokta: ``_card_text`` eskiden limitations/datasets/risk_warnings'i
    toplamıyordu → bu serbest-metin alanlarındaki sır/PII güvenlik kapısını
    (BLOCKER) ATLIYORDU. Temiz özet + sır-içeren risk_warnings → REDDEDİLMELİ.
    """
    card = _approved_card("c1", "RSI momentum göstergesi, tamamen zararsız özet metni.")
    card["card_json"]["risk_warnings"] = [
        "Üretim erişimi için password=hunter2 kullanın (kaza sonucu sızdı)."
    ]
    result = gate_7_safety([card])
    assert result.passed is False
    assert result.rejected_count == 1


def test_gate_7_safety_clean_risk_warnings_pass() -> None:
    """Sırsız risk_warnings (meşru uyarı) Gate 7'yi geçmeli (alan dahil ama temiz)."""
    card = _approved_card("c1", "RSI momentum göstergesi, zararsız özet.")
    card["card_json"]["risk_warnings"] = ["Aşırı uydurma riski; örneklem dışı test şart."]
    card["card_json"]["limitations"] = ["Yalnız günlük zaman diliminde test edildi."]
    result = gate_7_safety([card])
    assert result.passed is True


def test_gate_8_split_no_leakage_for_distinct_sources() -> None:
    """Farklı kaynaklı örnekler Gate 8'de sızıntısız bölünmeli."""
    examples = [_example(f"src_{i}") for i in range(12)]
    result, split = gate_8_split(examples)
    assert result.passed is True
    assert len(split.train) + len(split.valid) + len(split.test) == 12


def test_gate_8_fails_on_empty_valid_or_test_small_n() -> None:
    """Az benzersiz kaynakta (n_groups≤5) valid/test boş kalır → Gate 8 BLOKLAMALI.

    Eski hâlde boş valid/test 'sızıntı yok' diye PASS verip sahte OOS garantisi üretiyordu
    (READY_TO_TRAIN). Artık boş valid/test FAIL (CLAUDE.md kural 2).
    """
    # 2 benzersiz kaynak → split: train=2, valid=0, test=0
    examples = [_example("src_a"), _example("src_b")]
    result, split = gate_8_split(examples)
    assert result.passed is False
    assert not split.valid and not split.test
    assert result.rejected_count >= 1
