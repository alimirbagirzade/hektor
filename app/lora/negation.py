"""Olumsuzlama tespiti — Gate 5 (aşırı-emin ifade), Gate 6 (tavsiye dili) ve Gate 7
(finansal yönlendirme) ORTAK kullanır.

Olumsuzlanmış ifade ihtiyatlı dildir: "…without guaranteed performance gains" (2026-09-27
lora-audit, card_d489175087d1) Gate 5'te, "may not be directly applicable" (card_8aad7e24a61e)
Gate 6'da, "There is no guaranteed profit in trading." Gate 7'de yanlış işaretleniyordu.

Desenler ``tr_fold``'lanmış metne karşı çalışır. ÖNCE: olumsuzlayıcı + en fazla bir ara kelime
("is not necessarily …"); ara kelime "only" OLAMAZ — "not only superior performance but…"
olumsuzlama değildir. SONRA: Türkçe yüklem olumsuzlaması ("garantisi yoktur", "uygulanabilir
değildir").

Olumlu deyimler (Kademe 2 B4, 2026-09-28): "hiç şüphesiz", "no doubt", "without (a) doubt",
"can't lose", "never fails", "nothing but" gibi kalıplar olumsuzlayıcı kelime TAŞIR ama anlamca
PEKİŞTİRİR — "hiç şüphesiz garanti kâr" olumsuzlama sanılıp BLOCKER kapıdan kaçıyordu (FN).
Bu deyimlerle başlayan olumsuzlayıcı eşleşmesi olumsuzlama SAYILMAZ.

Modül bağımsızdır (``safety_scanner`` ↔ ``math_verifier`` döngüsel içe-aktarımı olmasın).
"""

from __future__ import annotations

import re

# "n't" kısaltmaları (don't/doesn't/won't/can't/isn't …) düz ve kıvrık kesme işaretiyle.
NEG_BEFORE_RE = re.compile(
    r"\b(?:not|no|non|without|never|cannot|\w*n['’]t|hardly|neither|nor|hic|hicbir)"
    r"(?![\w'’])(?:\s+(?!only\b)\w+)?[\s\-]*$"
)
# Türkçe yüklem olumsuzlaması: eşleşmenin ekinden sonra en fazla BİR ara kelime ("garanti
# kâr yoktur" → 'garanti' için ara kelime 'kâr'; "kâr garantisi yoktur" → ara kelime yok).
NEG_AFTER_RE = re.compile(r"^\w*\s*(?:\w+\s+)?(?:degil|yok|edilmez|olmaz|olmad)")
NEG_WINDOW: int = 30

# Olumsuzlayıcı kelimeyle BAŞLAYAN ama pekiştiren deyimler (tr_fold'lanmış). Olumsuzlayıcı
# eşleşmesi bunlardan biriyle başlıyorsa olumsuzlama değildir. ("şüphesiz"/"kuşkusuz" tek
# başına zaten olumsuzlayıcı içermez; yalnız "hiç" ile birleşince yakalanıyordu.)
AFFIRMATIVE_IDIOM_RE = re.compile(
    r"^(?:hic\s+(?:suphesiz|kuskusuz|suphe\s+yok)"
    r"|no\s+(?:doubt|question)"
    r"|without\s+(?:a\s+|any\s+)?(?:doubt|question)"
    r"|can['’]?t\s+lose|cannot\s+lose|can\s+not\s+lose|won['’]t\s+lose"
    r"|never\s+(?:fails?|loses?)"
    r"|not\s+only"
    r"|nothing\s+but)\b"
)


def is_negated(folded: str, start: int, end: int) -> bool:
    """`folded[start:end]` eşleşmesi olumsuzlanmış mı (öncesinde ya da sonrasında)?

    Olumsuzlayıcı olumlu bir deyimin parçasıysa ("hiç şüphesiz", "no doubt", "can't lose")
    olumsuzlama sayılmaz.
    """
    before = folded[max(0, start - NEG_WINDOW) : start]
    after = folded[end : end + NEG_WINDOW]
    match = NEG_BEFORE_RE.search(before)
    if match and not AFFIRMATIVE_IDIOM_RE.match(before[match.start() :]):
        return True
    return bool(NEG_AFTER_RE.search(after))
