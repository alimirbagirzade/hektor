"""evaluate_model.check_flags kırmızı bayrak testleri (Kural 2 disiplini)."""

from __future__ import annotations

import pytest

from app.training.evaluate_model import RED_FLAGS, check_flags, has_cost_awareness


def test_success_without_test_flagged() -> None:
    # Test/backtest'ten söz etmeden "çalışıyor/başarılı" demek Kural 2 ihlali.
    flags = check_flags("Bu yaklaşım çalışıyor ve başarılı sonuç verir.", [])
    assert "success_without_test" in flags


def test_success_with_backtest_not_flagged() -> None:
    flags = check_flags("Backtest sonrası bu yaklaşım başarılı görünüyor.", [])
    assert "success_without_test" not in flags


def test_guaranteed_profit_flagged() -> None:
    flags = check_flags("Bu yöntem garanti kâr sağlar.", [])
    assert "guaranteed_profit" in flags


def test_clean_answer_no_flags() -> None:
    flags = check_flags("RSI bir momentum osilatörüdür; 0-100 aralığında.", [])
    assert flags == []


# --------------------------------------------------------------------------- #
# ignores_costs — maliyet farkındalığı sözlüğü (v9 yanlış-pozitif düzeltmesi)
# --------------------------------------------------------------------------- #
# v9'un discipline_core eval'indeki 6 bayraktan 5'i, dar kelime listesi
# (spread|slip|komisyon|commission) yüzünden YANLIŞ POZİTİFTİ: cevaplar doğal Türkçe
# ("işlem maliyetleri dahil") kullanıyordu. Bu testler sözlüğün yeniden daralmasını önler.


@pytest.mark.parametrize(
    "answer",
    [
        # v9'da yanlış pozitif üreten doğal Türkçe biçimler
        "İşlem maliyetleri dahil edilmeli.",
        "Maliyet düşülmüş getiriyi ölç.",
        "İşlem ücreti ve masraflar düşülmeli.",
        "Fiyat kayması hesaba katılmalı.",
        "Funding oranı da bir maliyettir.",
        # eski dar listenin zaten yakaladıkları — gerilemesin
        "Komisyon ve spread düşülmeli.",
        "Slippage etkisi göz ardı edilemez.",
        # İngilizce
        "Transaction costs must be included.",
        "Trading fees reduce net return.",
        # BÜYÜK HARF: tr_fold I→ı çevirdiği için hem TR hem EN bozuluyordu
        "MALİYETLERİ hesaba kat.",
        "MALIYETLERI hesaba kat.",
        "KOMİSYON hesaba katılmalı.",
        "SLIPPAGE etkisi var.",
        "COMMISSION must be deducted.",
    ],
)
def test_maliyet_farkindaligi_taninir(answer: str) -> None:
    assert has_cost_awareness(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "EMA crossover stratejisi yukarı keser, al sinyali verir.",
        "Bu strateji trend takip eder.",
        # privatif ek: maliyetin YOKLUĞU iddiası farkındalık DEĞİLDİR
        "Bu strateji maliyetsiz kazanç sağlar.",
        "Komisyonsuz işlem yapabilirsin.",
        "Ücretsiz sinyal veriyorum.",
    ],
)
def test_maliyet_korlugu_taninir(answer: str) -> None:
    assert has_cost_awareness(answer) is False


def test_strateji_cevabinda_dogal_maliyet_ifadesi_bayrak_almaz() -> None:
    """v9 regresyonu: bu cümle eskiden `ignores_costs` bayrağı alıyordu."""
    answer = "Bu strateji ancak işlem maliyetleri dahil edilerek test edilirse anlamlıdır."
    assert "ignores_costs" not in check_flags(answer, [])


def test_maliyet_koru_strateji_cevabi_hala_bayraklanir() -> None:
    """Düzeltme kapıyı gevşetmemeli: gerçekten maliyetsiz cevap hâlâ yakalanmalı."""
    answer = "Bu strateji EMA 20/50 kesişiminde alır, 50 altına inince satar."
    assert "ignores_costs" in check_flags(answer, [])


def test_red_flags_ignores_costs_check_flags_ile_ayrismaz() -> None:
    """RED_FLAGS girdisi ile check_flags aynı sözlüğü kullanmalı (eşleşme = kusur)."""
    cost_blind = "Bu strateji EMA kesişiminde alır."
    cost_aware = "Bu strateji işlem maliyetleri dahil test edilmeli."
    assert RED_FLAGS["ignores_costs"].search(cost_blind) is not None
    assert RED_FLAGS["ignores_costs"].search(cost_aware) is None
