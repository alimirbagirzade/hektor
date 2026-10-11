"""advice_guard.py — al/sat KARARI sorularına karşı deterministik Kural 1 kapısı.

Bulgu (2026-10-10, karşılaştırma ``cmp_759c2d9b609d`` kör paketi, soru ``bi-04``): "BTC alayım
mı?" sorusuna iki model çıplak "Hayır." dedi. Mevcut kontroller yalnız sabit yönlendirme
kalıplarını ("garanti", "şimdi al", "kesin kazanç") arıyordu; bu çift hiçbirine takılmıyordu →
sohbet ekranında uyarısız gösteriliyor ve "Öğrensin" ile eğitim adayı olabiliyordu.

İki parça (ikisi de saf regex; eval/exec yok, Kural 5):

- ``is_advice_request(soru)``  : kişisel al/sat/tut/gir/çık KARARI soruluyor mu
  ("alayım mı", "satmalı mıyım", "girsem mi", "long mu short mu", "should I buy").
  Tahmin soruları ("yükselir mi") ve kavram soruları KAPSAM DIŞI.
- ``directive_answer(cevap)``  : cevap çıplak evet/hayır ile başlıyor ya da al/sat emri/önerisi
  veriyor mu ("alın", "satmalısınız", "almanızı öneririm", "you should buy").

Kullanım: sohbet cevabına (gösterilen metne) zorunlu uyarı + tur kontrolü eklenir (ham cevap
değişmez); öğrenme adayında karar sorusu + yönlendirici cevap = güvenlik RET (eğitime giremez).
"""

from __future__ import annotations

import re
from typing import Any

from app.lora.safety_scanner import tr_fold

# tr_fold çıktısı ASCII'dir (ı→i, ş→s, ğ→g …) → kalıplar ASCII yazılır.
# Kademe 2 (2026-10-11) V-2: "ekle" kökü çıkarıldı — "ATR filtresi eklesem mi?" araştırma
# sorusudur, al/sat kararı değil. V-3: mastar + "mantıklı mı", edilgen "alınır mı", "al mı sat
# mı", "alım/giriş yapayım mı", "would you buy" gibi yaygın karar biçimleri eklendi.
_TR_REQUEST = re.compile(
    r"\b(al|sat|tut|gir|cik|kapat|bas)"
    r"(ayim|eyim|sam|sem|sak|sek|mali|meli|malim|melim)\s*m[iu]\w*\b"
    r"|\b(almak|satmak|girmek|cikmak|tutmak)\s+(mantikli|dogru|iyi|akillica|uygun)\s+m[iu]\b"
    r"|\b(alinir|satilir|alinmali|satilmali)\s+m[iu]\b"
    r"|\bal\s*m[iu]\s*,?\s*sat\s*m[iu]\b"
    r"|\b(alim|satim|giris|cikis|yatirim)\s+yap(ayim|sam|sak|mali|malim)\s*m[iu]\w*\b"
    r"|\bsen\s+olsan\s+(alir|satar|girer)\s+m[iu]\w*\b"
)
_TR_SIDE = re.compile(
    r"\b(long|short)\s+m[iu]\b|\bpozisyon\s+ac(ayim|sam|mali)\b"
    r"|\b(long|short)\s*'?\s*[ae]\s+(mi\s+)?gec(eyim|sem|meli)\w*\b"
)
_EN_REQUEST = re.compile(
    r"\bshould\s+(i|we)\s+(buy|sell|short|hold|go\s+long|go\s+short|enter|exit)\b"
    r"|\b(buy|sell)\s+or\s+(sell|buy|hold|wait)\b"
    r"|\bwould\s+you\s+(buy|sell|short)\b"
    r"|\b(good|right|best)\s+time\s+to\s+(buy|sell)\b"
)

# V-2: "Hayır, evet/hayır cevabı veremem; hipoteze çevirelim" Kural 1'e UYGUN bir reddir →
# çıplak karar sayılmaz. Çıplak karar = evet/hayır'ın ilk cümlesinde ret/hipotez dili yok.
_BARE_YES_NO = re.compile(r"^\W*(evet|hayir|yes|no)\b(?P<rest>[^.!?\n]*)")
_REFUSAL = re.compile(
    r"veremem|vermem|verilmez|verilemez|diyemem|diyemeyiz|soyleyemem|cevaplayamam|"
    r"tavsiye\w*\s+(degil|olamaz|veremem|vermem)|hipotez|"
    r"can\s*not|can'?t|won'?t|not\s+(able|financial|advice)"
)
# V-2: "dikkate alın", "not alın", "göz önüne alın", "satın alma kararı" emir/öneri DEĞİL.
_AL_LOOKBEHIND = (
    r"(?<!dikkate )(?<!not )(?<!onune )(?<!hesaba )(?<!ornek )(?<!referans )(?<!ele )"
    r"(?<!baz )(?<!temel )(?<!kayda )(?<!satin )"
)
_DIRECTIVE = re.compile(
    rf"\b({_AL_LOOKBEHIND}al(in|iniz|abilirsin|abilirsiniz|malisin|malisiniz)|"
    r"satin\s+al(in|iniz|abilirsin|abilirsiniz|malisin|malisiniz)|"
    r"sat(in|iniz|abilirsin|abilirsiniz|malisin|malisiniz)(?!\s+al)|"
    r"(almanizi|satmanizi|girmenizi|cikmanizi)\s+(oneririm|tavsiye)\w*|"
    r"(almaniz|satmaniz)\s+(gerek|lazim|mantikli)\w*|"
    # V-3: "Bence al.", "(long|short) pozisyon açın", "you should go long", "I'd buy".
    r"bence\s+(al|sat)|"
    r"(long|short)\s+(pozisyon\w*\s+)?ac(in|iniz|abilirsin|malisin)|"
    r"you\s+should\s+(buy|sell|short|go\s+long|go\s+short)|"
    r"i\s*(would|'d|recommend)\s+(buy|sell)\w*)\b"
)
# V-3: tek kelimelik emir cevabı ("Al.", "Sat!", "Buy.") — ilk cümle yalnız fiil.
_BARE_DIRECTIVE = re.compile(r"^\W*(bence\s+)?(al|sat|buy|sell)\s*[.!]*\s*(?:$|\n|[.!]\s)")

NOTE_LINE = (
    "- Al/sat kararı sorularına evet/hayır verilmez: soru test edilebilir bir hipoteze "
    "(kural + maliyet + örneklem dışı test) çevrilmelidir."
)


def is_advice_request(question: str) -> bool:
    q = tr_fold(question or "")
    return bool(_TR_REQUEST.search(q) or _TR_SIDE.search(q) or _EN_REQUEST.search(q))


def _bare_yes_no(a: str) -> bool:
    m = _BARE_YES_NO.search(a)
    return m is not None and not _REFUSAL.search(m.group("rest"))


def directive_answer(answer: str) -> bool:
    a = tr_fold(answer or "").strip()
    return bool(_bare_yes_no(a) or _DIRECTIVE.search(a) or _BARE_DIRECTIVE.search(a))


def candidate_reason(question: str, target: str) -> str | None:
    """Öğrenme adayı için: karar sorusu + yönlendirici cevap → Kural 1 ret gerekçesi."""
    if is_advice_request(question) and directive_answer(target):
        return (
            "Al/sat kararı sorusuna doğrudan evet/hayır ya da al/sat yönlendirmesiyle cevap — "
            "yatırım tavsiyesi (Kural 1). Hedefi hipotez + test noktası olarak yeniden yazın."
        )
    return None


def guard_chat_answer(question: str, answer: str) -> tuple[str, dict[str, Any] | None]:
    """Sohbet cevabı (GÖSTERİLEN metin) için zorunlu uyarı + tur kontrolü. Ham cevap ayrı kalır."""
    if not is_advice_request(question):
        return answer, None
    from app.rlm.rlm_controller import _TRADING_DISCLAIMER, _TRADING_GUARD_HEADER

    directive = directive_answer(answer)
    check = {
        "kind": "guvenlik",
        "status": "ihlal" if directive else "uyari",
        "scope": "soru + cevap",
        "detail": (
            "Al/sat kararı sorusu; model doğrudan yönlendirdi (evet/hayır ya da al/sat) — "
            "Kural 1 ihlali, cevap tavsiye olarak okunmamalı ve eğitim hedefi olamaz."
            if directive
            else "Al/sat kararı sorusu: cevap yalnız hipotez/test noktası olarak okunmalı."
        ),
    }
    if _TRADING_GUARD_HEADER in answer and NOTE_LINE in answer:
        return answer, check
    block = f"{_TRADING_GUARD_HEADER}\n{_TRADING_DISCLAIMER}\n{NOTE_LINE}"
    return f"{answer}\n\n{block}", check
