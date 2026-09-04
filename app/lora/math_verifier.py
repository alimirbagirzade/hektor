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

# "%<sayı>" desenini yakalar.
_PERCENT_RE = re.compile(r"%\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*%")

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

    has_risk_context = tr_fold("risk") in folded
    if has_risk_context:
        for pct in percentages:
            if pct > 100:
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


def verify_math_content(text: str) -> MathVerifyResult:
    """Bir metni matematik/istatistik açısından doğrula.

    Geriye dönen sonuç:
      - `issues`: bulunan tüm uyarılar
      - `requires_review`: insan incelemesi gerekiyor mu
      - `passed`: aşırı emin yatırım ifadesi yoksa True (bunlar blocker)
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
        if tr_fold(phrase) in folded:
            issues.append(f"aşırı emin yatırım ifadesi: '{phrase}'")
            overconfident_found = True

    issues.extend(_check_percentage_sanity(text))
    # Doğrulanmamış sayısal performans iddiaları (requires_review, BLOK DEĞİL).
    issues.extend(_check_performance_claims(folded))

    requires_review = bool(issues)

    return MathVerifyResult(
        passed=not overconfident_found,
        issues=issues,
        requires_review=requires_review,
    )
