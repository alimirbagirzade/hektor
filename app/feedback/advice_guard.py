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
_TR_REQUEST = re.compile(
    r"\b(al|sat|tut|gir|cik|kapat|ekle|bas)"
    r"(ayim|eyim|sam|sem|sak|sek|mali|meli|malim|melim)\s*m[iu]\w*\b"
)
_TR_SIDE = re.compile(r"\b(long|short)\s+m[iu]\b|\bpozisyon\s+ac(ayim|sam|mali)\b")
_EN_REQUEST = re.compile(
    r"\bshould\s+(i|we)\s+(buy|sell|short|hold|go\s+long|go\s+short|enter|exit)\b"
    r"|\b(buy|sell)\s+or\s+(sell|buy|hold)\b"
)

_BARE_YES_NO = re.compile(r"^\W*(evet|hayir|yes|no)\b")
_DIRECTIVE = re.compile(
    r"\b(al(in|iniz|abilirsin|abilirsiniz|malisin|malisiniz)|"
    r"sat(in|iniz|abilirsin|abilirsiniz|malisin|malisiniz)|"
    r"(almanizi|satmanizi|girmenizi|cikmanizi)\s+(oneririm|tavsiye)\w*|"
    r"(almaniz|satmaniz)\s+(gerek|lazim|mantikli)\w*|"
    r"you\s+should\s+(buy|sell|short)|i\s+(would|recommend)\s+(buy|sell)\w*)\b"
)

NOTE_LINE = (
    "- Al/sat kararı sorularına evet/hayır verilmez: soru test edilebilir bir hipoteze "
    "(kural + maliyet + örneklem dışı test) çevrilmelidir."
)


def is_advice_request(question: str) -> bool:
    q = tr_fold(question or "")
    return bool(_TR_REQUEST.search(q) or _TR_SIDE.search(q) or _EN_REQUEST.search(q))


def directive_answer(answer: str) -> bool:
    a = tr_fold(answer or "").strip()
    return bool(_BARE_YES_NO.search(a[:40]) or _DIRECTIVE.search(a))


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
