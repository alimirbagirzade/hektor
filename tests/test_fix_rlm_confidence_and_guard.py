"""RLM: atıfsız cevabın bedava güven puanı + eksik trading uyarısı.

İki Kademe-2 av bulgusu (2026-09-07):

1. ``_citation_score([]) == 1.0`` — atıf HİÇ yokken tam puan dönüyordu ve bu değer
   ağırlıklı ortalamaya 0.30 ağırlıkla giriyordu. Satır-içi atıf taşımayan bir cevap
   SIRF atıfsız olduğu için güven skorunu şişiriyordu (Kural 7'nin tersi).
2. ``apply_trading_guard`` idempotens kontrolü serbest bir cümleye bakıyordu; o cümle
   modelin metninde ya da alıntılanan makale parçasında geçtiğinde zorunlu uyarının
   kalan üç satırı hiç eklenmiyordu (Kural 1 sessizce eksik uygulanıyordu).
"""

from __future__ import annotations

from app.rlm.rlm_controller import (
    _TRADING_DISCLAIMER,
    _TRADING_GUARD_HEADER,
    apply_trading_guard,
)
from app.verification.citation_verifier import CitationCheck
from app.verification.confidence_scorer import ConfidenceScorer, _citation_score
from app.verification.context_sufficiency import SufficiencyLevel, SufficiencyResult
from app.verification.grounding_verifier import GroundingLevel, GroundingResult


def _sufficiency() -> SufficiencyResult:
    return SufficiencyResult(level=SufficiencyLevel.SUFFICIENT, missing_items=[], can_answer=True)


def _grounding(level: GroundingLevel) -> GroundingResult:
    return GroundingResult(claim="iddia", level=level, evidence_chunk_id=None)


def _cit(exists: bool) -> CitationCheck:
    return CitationCheck(paper_id="a", chunk_id="b", exists=exists, claim="iddia", supported=exists)


# ---------------------------------------------------------------- citation score


def test_atifsiz_cevap_tam_puan_ALMAZ() -> None:
    """Asıl bulgu: boş atıf kümesi artık 1.0 değil, 'ölçülemedi' (None)."""
    assert _citation_score([]) is None


def test_atif_orani_hala_dogru_hesaplanir() -> None:
    """Atıf varken davranış DEĞİŞMEZ (regresyon yok)."""
    cits = [_cit(True), _cit(False)]
    assert _citation_score(cits) == 0.5


def test_atifsiz_cevap_skoru_sismez() -> None:
    """Atıfsız cevap, aynı içerikte atıflı-mükemmel cevaptan YÜKSEK skor alamaz."""
    scorer = ConfidenceScorer()
    zayif_dayanak = [_grounding(GroundingLevel.UNSUPPORTED)]

    atifsiz = scorer.score(_sufficiency(), [], zayif_dayanak, [])
    tam_atifli = scorer.score(_sufficiency(), [_cit(True)], zayif_dayanak, [])

    assert atifsiz.details["citation_absent"] is True
    assert tam_atifli.details["citation_absent"] is False
    # Eski kodda ikisi de 0.30 * 1.0 alıyordu → eşitlerdi. Artık atıflı olan üstte.
    assert atifsiz.score < tam_atifli.score
    # Atıfsızda rapor edilen alan 0.0 olmalı → §16 LoRA aday kapısı (>=0.90) eler.
    assert atifsiz.citation_score == 0.0


def test_atifsizda_kalan_agirliklar_normalize_edilir() -> None:
    """Bileşen çıkarıldı diye skor haksızca çökmemeli: her şey mükemmelse skor 1.0."""
    scorer = ConfidenceScorer()
    mukemmel = scorer.score(_sufficiency(), [], [_grounding(GroundingLevel.SUPPORTED)], [])
    assert mukemmel.score == 1.0


# ---------------------------------------------------------------- trading guard


def test_serbest_cumle_zorunlu_uyariyi_ATLATMAZ() -> None:
    """Asıl bulgu: alıntılanan metinde geçen cümle uyarıyı iptal ediyordu."""
    cevap = (
        "Makale, bu sonucun yatırım tavsiyesi değildir biçiminde okunmasını öneriyor. "
        "Alım sinyali üreten strateji backtest edilmelidir."
    )
    out = apply_trading_guard(cevap, "bu strateji için alım sinyali ver")
    assert _TRADING_GUARD_HEADER in out
    assert _TRADING_DISCLAIMER in out
    # Uyarının DÖRT satırı da eklenmeli (eskiden hiçbiri eklenmiyordu).
    assert "canlı sinyal değildir" in out
    assert "backtest gerektirir" in out.replace("\n", " ")


def test_idempotent_kalir() -> None:
    """İki kez uygulamak uyarıyı ÇİFTLEMEZ (mevcut sözleşme korunur)."""
    bir = apply_trading_guard("alım sinyali öner", "strateji")
    iki = apply_trading_guard(bir, "strateji")
    assert bir == iki
    assert iki.count(_TRADING_GUARD_HEADER) == 1


def test_trading_disi_cevaba_uyari_eklenmez() -> None:
    """Yanlış-pozitif koruması: sürekli uyarı = göz ardı edilen uyarı."""
    cevap = "Konformal tahmin, dağılımdan bağımsız tahmin aralıkları üretir."
    assert apply_trading_guard(cevap, "konformal tahmin nedir") == cevap
