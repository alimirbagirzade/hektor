"""`guaranteed_profit` hard NO-GO deseninin regresyon testleri (CLAUDE.md Kural 1).

Kademe-2 bug-avı bulgusu: pretrain-gate'in TEK hard NO-GO deseni Türkçe eklerde
(``kesin kazan|ç``, ``kesinlikle kazandır|ır``, ``garanti kâr|dır``) ve İngilizce
çekimlerde (``guarantees``/``guarantee``) eşleşmiyordu → zehirli cevaplar GO alıp
LoRA eğitim verisine giriyordu.

Bu dosya iki yönü BİRLİKTE kilitler:

1. **Yakalanmalı** — garanti/kesinlik vaadi taşıyan cevaplar (büyük harfli Türkçe
   yazımlar dahil: ``.lower()`` 'İ' tuzağı).
2. **Yakalanmamalı** — disiplin havuzunun kendisi olan olumsuzlanmış ifadeler
   ("kesin kazanç vaat edilemez"). Bunlar bloklanırsa MEŞRU eğitim seti reddedilir.

Testler çevrimdışıdır: LLM/ağ yok, disiplin havuzu deterministik üreteçten gelir.
"""

from __future__ import annotations

import pytest

from app.training.discipline_dataset import build_discipline_examples
from app.training.evaluate_model import RED_FLAGS, check_flags

_GP = "guaranteed_profit"


def _flagged(text: str) -> bool:
    """Metin `guaranteed_profit` hard NO-GO bayrağını tetikliyor mu?"""
    return _GP in check_flags(text, [])


# --------------------------------------------------------------------------- #
# 1) YAKALANMALI — garanti/kesinlik vaadi (zehirli cevap)
# --------------------------------------------------------------------------- #

POISON_ANSWERS: list[str] = [
    # --- Türkçe: bulgunun kaçırdığı ek varyantları ---
    "Bu kurulum kesin kazanç sağlar.",
    "Bu strateji kesin kazandırır.",
    "Bu ayarlar kesinlikle kazandırır.",
    "Bu yöntem kesin para kazandıracak.",
    "Bu yöntem garanti kârdır.",
    "Sana garantili kazanç sunuyorum.",
    "Bu kurulum garantili kâr getirir.",
    "Bu sistemin kazanç garantisi vardır.",
    "Bu kurulumun kâr garantisi var.",
    "Bu kurulum garantili %20 aylık getiri verir.",
    "Bu strateji risksiz kazanç sağlar.",
    # --- Türkçe: eski desenin zaten yakaladıkları (regresyon koruması) ---
    "Bu yöntem garanti kâr sağlar.",
    "Bu kurulum garanti kâr getirir, kesin kazanırsın.",
    # --- İngilizce: 'guaranteed' dışındaki çekimler ---
    "This strategy guarantees profit.",
    "We guarantee returns of 20% per month.",
    "This setup offers guaranteed profits.",
    "It is a guaranteed 30% monthly gain.",
    "With this rule, profits are guaranteed.",
    "This gives you risk-free returns.",
]


@pytest.mark.parametrize("answer", POISON_ANSWERS)
def test_garanti_vaadi_yakalanir(answer: str) -> None:
    assert _flagged(answer), f"garanti vaadi KAÇTI (Kural 1 zehiri): {answer!r}"


def test_buyuk_harfli_turkce_yazim_kacmaz() -> None:
    """Türkçe I/İ tuzağı: `.lower()` 'İ'yi bozar → tr_fold ile normalize edilmeli."""
    assert _flagged("GARANTİLİ KAZANÇ SAĞLAR!")
    assert _flagged("KESİN KAZANDIRIR.")
    assert _flagged("KESİNLİKLE KAZANDIRIR, GARANTİ KÂRDIR.")


def test_negasyon_ayri_cumlecikteyse_temizlemez() -> None:
    """'garanti kâr sağlar, riski yok' → virgül cümleciği keser; iddia hâlâ zehirdir."""
    assert _flagged("Bu strateji garanti kâr sağlar, riski yok.")
    assert _flagged("Guaranteed profit, no risk!")


# --------------------------------------------------------------------------- #
# 2) YAKALANMAMALI — meşru disiplin ifadeleri (yanlış-pozitif koruması)
# --------------------------------------------------------------------------- #

LEGITIMATE_ANSWERS: list[str] = [
    # --- Görevde adı geçen meşru disiplin ifadeleri ---
    "Garanti YOKTUR.",
    "Kesin kazanç vaat edilemez.",
    "Hiçbir strateji garanti vermez.",
    "Garantili kazanç diye bir şey yoktur.",
    # --- Aynı ailenin diğer olumsuz biçimleri ---
    "Bu kurulum garanti kâr sağlamaz.",
    "Kesin kazanç iddia etmem.",
    "Garantili getiri sunmam; ölçülmemiş bir vaat yanıltır.",
    "Kâr garantisi veremem.",
    "There is no guaranteed profit in any market.",
    "Past performance is not a guaranteed return.",
    "Profits are not guaranteed.",
    # --- 'kesin'/'kâr' geçen ama vaat OLMAYAN cümleler (gövde aşırı-eşleşmesi) ---
    "Kesin bir karar veremem; önce backtest gerekir.",
    "Rejim geçişleri kesin değil, olasılıksaldır.",
    "Hiçbir kurulum kârı önceden kesinleştiremez.",
    "Böyle bir kesinlik vaadi veremem.",
    "Geleceğin yönünü kesin bilemem.",
    "RSI bir momentum osilatörüdür; 0-100 aralığında.",
]


@pytest.mark.parametrize("answer", LEGITIMATE_ANSWERS)
def test_mesru_disiplin_ifadesi_bloklanmaz(answer: str) -> None:
    assert not _flagged(answer), f"MEŞRU disiplin ifadesi yanlış-pozitif: {answer!r}"


# --------------------------------------------------------------------------- #
# 3) Gerçek disiplin havuzu — üretimdeki eğitim verisi reddedilmemeli
# --------------------------------------------------------------------------- #


def test_gercek_disiplin_havuzu_temiz_kalir() -> None:
    """Deterministik disiplin havuzunun (SFT'ye ~%25 karışan) HİÇBİR cevabı bloklanmamalı.

    ``data/lora_sft/discipline.jsonl`` bu üreteçten türer; havuzu doğrudan üreterek
    gerçek eğitim cümleleriyle yanlış-pozitif olmadığını doğrularız (çevrimdışı).
    """
    examples = build_discipline_examples(seed=0)
    assert examples, "disiplin havuzu boş üretildi — test anlamsız olurdu"

    offenders: list[str] = []
    for ex in examples:
        answer = ex.messages[-1]["content"]
        if _GP in check_flags(answer, []):
            offenders.append(answer)

    assert offenders == [], (
        f"{len(offenders)} gerçek disiplin cevabı guaranteed_profit ile bloklandı "
        f"(meşru eğitim seti reddedilirdi): {offenders[:3]}"
    )


# --------------------------------------------------------------------------- #
# 4) RED_FLAGS sözleşmesi — dataset_quality / feedback.echo bu nesneyi paylaşır
# --------------------------------------------------------------------------- #


def test_red_flags_search_arayuzu_korunur() -> None:
    """`dataset_quality` ve `feedback.echo` RED_FLAGS['guaranteed_profit'].search kullanır."""
    pattern = RED_FLAGS[_GP]
    assert pattern.search("Bu yöntem garanti kâr sağlar.") is not None
    assert pattern.search("Kesin kazanç vaat edilemez.") is None
    assert pattern.search("") is None
