"""Matematik / istatistik doğrulayıcı — deterministik regex tabanlı.

Gate 5 için kullanılır. Hesap tutarlılığı, istatistiksel kırmızı bayraklar
ve aşırı emin yatırım dili gibi sorunları işaretler. LLM gerektirmez.
Şüpheli durumları otomatik reddetmek yerine `requires_review` ile
insan incelemesine yönlendirir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.lora.safety_scanner import tr_fold

# İstatistiksel kırmızı bayraklar — varlığı uyarı doğurur.
STATISTICAL_FLAGS: list[str] = [
    "lookahead",
    "look-ahead",
    "look ahead",
    "survivorship bias",
    "survivorship",
    "data snooping",
    "data-snooping",
    "p-hacking",
    "p hacking",
]

# Aşırı emin / yanıltıcı yatırım ifadeleri — varlığı uyarı doğurur.
OVERCONFIDENT_PHRASES: list[str] = [
    "kesinlikle",
    "garanti",
    "her zaman kazanır",
    "risk yok",
    "asla kaybetmez",
    "guaranteed",
    "always wins",
    "no risk",
    "%100 kazanç",
    "100% profit",
]

# Olumsuzlama tespiti — Gate 5 (aşırı-emin ifade) ve Gate 6 (tavsiye dili) ORTAK kullanır.
# Olumsuzlanmış ifade ihtiyatlı dildir: "…without guaranteed performance gains" (2026-09-27
# lora-audit, card_d489175087d1) Gate 5'te, "may not be directly applicable"
# (card_8aad7e24a61e) Gate 6'da yanlış işaretleniyordu. Desenler tr_fold'lanmış metne karşı
# çalışır. ÖNCE: olumsuzlayıcı + en fazla bir ara kelime ("is not necessarily …"); ara kelime
# "only" OLAMAZ — "not only superior performance but…" olumsuzlama değildir. SONRA: Türkçe
# yüklem olumsuzlaması ("garantisi yoktur", "uygulanabilir değildir").
NEG_BEFORE_RE = re.compile(
    r"\b(?:not|no|non|without|never|cannot|can't|isn't|aren't|hardly|neither|nor|hic|hicbir)"
    r"\b(?:\s+(?!only\b)\w+)?[\s\-]*$"
)
NEG_AFTER_RE = re.compile(r"^\w*\s*(?:degil|yok|edilmez|olmaz|olmad)")
NEG_WINDOW: int = 30


def is_negated(folded: str, start: int, end: int) -> bool:
    """`folded[start:end]` eşleşmesi olumsuzlanmış mı (öncesinde ya da sonrasında)?"""
    before = folded[max(0, start - NEG_WINDOW) : start]
    after = folded[end : end + NEG_WINDOW]
    return bool(NEG_BEFORE_RE.search(before) or NEG_AFTER_RE.search(after))


# "%<sayı>" desenini yakalar.
_PERCENT_RE = re.compile(r"%\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*%")

# Risk-yüzdesi kuralı için YEREL bağlam (tr_fold'lanmış metin, her iki yana karakter).
_RISK_PCT_WINDOW: int = 40
_RISK_WORD_RE = re.compile(r"\brisk")
_RETURN_WORD_RE = re.compile(
    r"return|getiri|profit|\bkar\b|kazanc|gain|cumulative|kumulatif|growth|buyume|"
    r"increase|artis|outperform|performance|performans"
)

# --------------------------------------------------------------------------- #
# Doğrulanmamış / aşırı-kesin performans iddiaları (Gate 5 — requires_review)
# --------------------------------------------------------------------------- #
# v5 disiplin-gerilemesini besleyen kart sınıfı: "92% accuracy",
# "outperforming ... by 15%", "Sharpe ratio of at least 1.5" gibi sayısal
# performans iddiaları — hangi veri seti/dönem/OOS belirtmeden. Eski Gate 5
# bunları geçiriyordu ('accuracy' OVERCONFIDENT_PHRASES'te değil + sayı %1000
# eşiğinin altında). BLOK DEĞİL; insan incelemesine yönlendirilir (bağlam önemli).
#
# Desenler tr_fold'lanmış metne (küçük harf + aksansız) karşı çalışır.
_PERFORMANCE_CLAIM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "doğruluk/isabet yüzdesi iddiası",
        re.compile(r"\d+(?:[.,]\d+)?\s*%\s*(?:accuracy|dogruluk|isabet|basari)"),
    ),
    (
        "doğruluk/isabet yüzdesi iddiası",
        re.compile(r"(?:accuracy|dogruluk|isabet|basari)\w*[^.\n]{0,20}?\d+(?:[.,]\d+)?\s*%"),
    ),
    (
        "'%X üstünlük' (outperform by) iddiası",
        re.compile(r"outperform\w*[^.\n]{0,40}?\bby\s*\d+(?:[.,]\d+)?\s*%"),
    ),
    (
        "'en az X Sharpe' iddiası",
        re.compile(r"sharpe[^.\n]{0,30}?(?:of at least|at least|en az)\s*\d"),
    ),
]

# Kanıt bağlamı işaretleri — iddianın YANINDA (pencere içinde) varsa iddia
# doğrulanmış/falsifiye-edilebilir sayılır ve işaretlenmez. Bu, meşru
# "backtest sonucu %60 isabet (2010-2020, OOS)" gibi bağlamlı ölçümleri
# yanlış-pozitiften korur (yalnız ÇIPLAK pazarlama iddiası işaretlenir).
_EVIDENCE_MARKER_RE = re.compile(
    r"backtest|geriye don[uü]k test|out[- ]of[- ]sample|\boos\b|in[- ]sample|"
    r"walk[- ]forward|holdout|hold[- ]out|dogrulama set|validation set|test set|"
    r"test seti|cross[- ]valid|capraz dogrulama|k-fold|"
    r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:19|20)?\d{2}\b"
)
# İddianın çevresinde kanıt aradığımız karakter penceresi (her iki yana).
_EVIDENCE_WINDOW: int = 70


# --------------------------------------------------------------------------- #
# Kaynak sayı kontrolü (Gate 5 — requires_review, BLOK DEĞİL)
# --------------------------------------------------------------------------- #
# 2026-09-27 lora-audit incelemesi: kart üreticisi makalede OLMAYAN yüzdeler uyduruyordu
# ("could reduce VaR by 10-15%", "will reduce forgetting by at least 30%"); yukarıdaki
# performans-iddiası kalıpları yalnız bir kısmını yakalıyordu. Karttaki her yüzde, kartın
# dayandığı makalenin chunk metninde aranır; yoksa inceleme işareti (Kural 7). Büyük
# kitaplarda "var" zayıf kanıt olduğundan BLOK değil, insan incelemesi.
_SRC_PCT_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s*(?:%|percent\b|per cent\b)")
# Ondalıklı sayı yeterince ayırt edicidir → kaynakta % işaretsiz de kabul ("accuracies of
# 63.707" tablosu). Tam sayı ise % bağlamı ister ("15" her metinde sayfa no. olarak geçer).
_SRC_DECIMAL_RE = re.compile(r"(?<![\w.,])(\d+[.,]\d+)(?!\d)")
# Cümle sınırı — sayı İÇİNDEKİ nokta bölmez ("63.707%" tek parça kalır).
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)[.;](?!\d)|\n")
# Bu cümlelerdeki yüzde iddia DEĞİL: önerilen test eşiği ("Test if … above 70%") ya da
# açıkça örnek parametre ("e.g., 2% below current price").
_NOT_A_CLAIM_RE = re.compile(
    r"\b(?:test|evaluate|measure|assess|examine)\s+(?:if|whether)\b|\be\.?g\b"
)


def _norm_number(raw: str) -> str:
    return f"{float(raw.replace(',', '.')):g}"


def source_numbers(source_text: str) -> frozenset[str]:
    """Kaynak metindeki yüzdeler + ondalıklı sayılar (normalize: "15.0" → "15")."""
    folded = tr_fold(source_text)
    nums = {_norm_number(m.group(1)) for m in _SRC_PCT_RE.finditer(folded)}
    nums |= {_norm_number(m.group(1)) for m in _SRC_DECIMAL_RE.finditer(folded)}
    return frozenset(nums)


def _numbers_missing_from_source(folded: str, source_nums: frozenset[str]) -> list[str]:
    """Karttaki (iddia cümlelerindeki) yüzdelerden kaynakta OLMAYANLAR, artan sırada."""
    claims: set[str] = set()
    # "e.g." kısaltmasının noktaları cümleyi bölmesin (örnek parametre muafiyeti kaybolurdu).
    folded = re.sub(r"\be\.g\.", "eg", folded)
    for sentence in _SENTENCE_SPLIT_RE.split(folded):
        if _NOT_A_CLAIM_RE.search(sentence):
            continue
        claims |= {_norm_number(m.group(1)) for m in _SRC_PCT_RE.finditer(sentence)}
    return sorted(claims - source_nums, key=float)


@dataclass
class MathVerifyResult:
    """Bir metnin matematik/istatistik doğrulama sonucu."""

    passed: bool
    issues: list[str] = field(default_factory=list)
    requires_review: bool = False


def _extract_percentages(text: str) -> list[float]:
    """Metindeki yüzde değerlerini float listesi olarak çıkar."""
    values: list[float] = []
    for match in _PERCENT_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        if raw is None:
            continue
        try:
            values.append(float(raw.replace(",", ".")))
        except ValueError:
            continue
    return values


def _check_percentage_sanity(text: str) -> list[str]:
    """Yüzde değerlerinin makul aralıkta olup olmadığını kontrol et.

    Risk yüzdesi bağlamında %100'ün çok üstündeki getiri iddiaları
    (örn. >%1000 yıllık) şüpheli olarak işaretlenir.
    """
    issues: list[str] = []
    # tr_fold: 'YILLIK GETİRİ'/'RİSK' büyük harf bağlamı str.lower() ile (İ→i+nokta)
    # kaçıyordu → modülün diğer taramalarıyla tutarlı olsun (Gate 5 bağlam tespiti).
    folded = tr_fold(text)
    percentages = _extract_percentages(text)

    has_return_context = any(
        tr_fold(kw) in folded
        for kw in ("getiri", "return", "kâr", "kar", "profit", "yıllık", "annual")
    )
    if has_return_context:
        for pct in percentages:
            if pct > 1000:
                issues.append(f"şüpheli yüksek getiri iddiası (%{pct:g})")

    # Risk bağlamı YÜZDENİN YANINDA aranır. Eskiden metnin herhangi bir yerinde "risk"
    # geçmesi yetiyordu → "cumulative returns of 340%, 185%, 371%" (card_89d30aac91b1,
    # 2026-09-27) risk yüzdesi sanılıyordu. Getiri bağlamı bitişikse yüzde risk değildir.
    for match in _PERCENT_RE.finditer(folded):
        raw = match.group(1) or match.group(2)
        try:
            pct = float(raw.replace(",", "."))
        except ValueError:
            continue
        if pct <= 100:
            continue
        window = folded[max(0, match.start() - _RISK_PCT_WINDOW) : match.end() + _RISK_PCT_WINDOW]
        if _RISK_WORD_RE.search(window) and not _RETURN_WORD_RE.search(window):
            issues.append(f"risk yüzdesi %100'ü aşıyor (%{pct:g}) — tutarsız olabilir")
    return issues


def _check_performance_claims(folded: str) -> list[str]:
    """Doğrulanmamış sayısal performans iddialarını kanıt-kapısıyla işaretle.

    `folded`: tr_fold'lanmış metin. Her iddia için, çevresindeki
    `_EVIDENCE_WINDOW` karakter içinde kanıt işareti (backtest/OOS/dönem…)
    yoksa uyarı eklenir. Aynı örüntüden tek uyarı yeter.
    """
    issues: list[str] = []
    for label, pattern in _PERFORMANCE_CLAIM_PATTERNS:
        for match in pattern.finditer(folded):
            window_start = max(0, match.start() - _EVIDENCE_WINDOW)
            window_end = min(len(folded), match.end() + _EVIDENCE_WINDOW)
            if _EVIDENCE_MARKER_RE.search(folded[window_start:window_end]):
                continue  # kanıt bağlamı bitişik → meşru ölçüm, işaretleme
            snippet = match.group(0).strip()[:50]
            issues.append(f"doğrulanmamış performans iddiası: {label} ('{snippet}')")
            break  # aynı örüntü için bir uyarı yeterli
    return issues


def _has_unnegated(folded: str, phrase: str) -> bool:
    """`phrase`'in en az bir OLUMSUZLANMAMIŞ geçişi var mı (`folded`: tr_fold'lanmış)?

    Tek bir olumsuzlanmamış geçiş yeterlidir → "not guaranteed … but guaranteed profit"
    yine yakalanır; olumsuzlama yalnız kendi geçişini muaf tutar.
    """
    start = 0
    while (i := folded.find(phrase, start)) >= 0:
        end = i + len(phrase)
        if not is_negated(folded, i, end):
            return True
        start = end
    return False


def verify_math_content(text: str, source_nums: frozenset[str] | None = None) -> MathVerifyResult:
    """Bir metni matematik/istatistik açısından doğrula.

    Geriye dönen sonuç:
      - `issues`: bulunan tüm uyarılar
      - `requires_review`: insan incelemesi gerekiyor mu
      - `passed`: aşırı emin yatırım ifadesi yoksa True (bunlar blocker)

    `source_nums` (``source_numbers(kaynak_metin)``) verilirse, karttaki yüzdelerden kaynakta
    OLMAYANLAR inceleme işareti alır. ``None`` ya da boş küme (kaynak metni yok) → kontrol
    atlanır; doğrulanamayan kart yanlışlıkla işaretlenmez.
    """
    if not text:
        return MathVerifyResult(passed=True)

    # Türkçe-bilinçli normalize: "KESİNLİKLE"/"GARANTİ" gibi büyük harf yazımlar
    # str.lower() ile (İ→i + birleşik nokta) taramayı atlatabiliyordu — tr_fold engeller.
    folded = tr_fold(text)
    issues: list[str] = []

    for flag in STATISTICAL_FLAGS:
        if tr_fold(flag) in folded:
            issues.append(f"istatistiksel risk işareti: '{flag}'")

    overconfident_found = False
    for phrase in OVERCONFIDENT_PHRASES:
        if _has_unnegated(folded, tr_fold(phrase)):
            issues.append(f"aşırı emin yatırım ifadesi: '{phrase}'")
            overconfident_found = True

    issues.extend(_check_percentage_sanity(text))
    # Doğrulanmamış sayısal performans iddiaları (requires_review, BLOK DEĞİL).
    issues.extend(_check_performance_claims(folded))
    if source_nums:
        missing = _numbers_missing_from_source(folded, source_nums)
        if missing:
            listed = ", ".join(f"%{n}" for n in missing)
            issues.append(f"kaynakta olmayan sayı: {listed}")

    requires_review = bool(issues)

    return MathVerifyResult(
        passed=not overconfident_found,
        issues=issues,
        requires_review=requires_review,
    )
