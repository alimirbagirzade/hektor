"""Matematik / istatistik doğrulayıcı — deterministik regex tabanlı.

Gate 5 için kullanılır. Hesap tutarlılığı, istatistiksel kırmızı bayraklar
ve aşırı emin yatırım dili gibi sorunları işaretler. LLM gerektirmez.
Şüpheli durumları otomatik reddetmek yerine `requires_review` ile
insan incelemesine yönlendirir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.lora.negation import NEG_AFTER_RE, NEG_BEFORE_RE, NEG_WINDOW, is_negated
from app.lora.safety_scanner import NO_RISK_RE, RISK_YOK_RE, tr_fold

__all__ = [
    "NEG_AFTER_RE",
    "NEG_BEFORE_RE",
    "NEG_WINDOW",
    "MathVerifyResult",
    "is_negated",
    "overconfident_hits",
    "source_numbers",
    "verify_math_content",
]

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

# Kesinlik (certainty) dili — garanti vaadi DEĞİL. pretrain-gate satır taramasında bunlar
# uyarıdır; geri kalan OVERCONFIDENT_PHRASES (garanti/risksizlik vaadi) blocker'dır.
CERTAINTY_ONLY_PHRASES: frozenset[str] = frozenset({"kesinlikle"})

# Her ifadenin tr_fold'lanmış metindeki kelime-sınırlı deseni. Eskiden çıplak alt-dize
# aranıyordu (Kademe 2 B6, 2026-09-28): "no risk" → "no risk-free arbitrage"/"no risk
# premium", "guaranteed" → "The estimator is guaranteed to converge." BLOCKER'a düşüyordu.
# (desen, olumsuzlama-öncesi-bakılır-mı, finansal-nesne-gerekir-mi)
_OVERCONFIDENT_RULES: dict[str, tuple[re.Pattern[str], bool, bool]] = {
    "kesinlikle": (re.compile(r"\bkesinlikle\b"), True, False),
    # Çıplak "garanti"/"guaranteed" yalnız YAKININDA finansal nesne (kâr/getiri/kazanç/
    # profit/return/gain/win…) varsa vaattir; "yakınsama garantisi", "guaranteed to
    # converge" matematik dilidir. "garanti kâr"/"guaranteed profit" nesneyi taşıdığı için
    # yakalanmaya devam eder.
    "garanti": (re.compile(r"\bgaranti\w*"), True, True),
    "her zaman kazanır": (re.compile(r"\bher zaman kazanir\w*"), True, False),
    # "risk yok"/"no risk" zaten olumsuz yapıdır: öncesindeki olumsuzlayıcı ("hiçbir risk
    # yok") iddiayı PEKİŞTİRİR, olumsuzlamaz → yalnız sonrası (Türkçe "… değil") bakılır.
    "risk yok": (RISK_YOK_RE, False, False),
    "asla kaybetmez": (re.compile(r"\basla kaybetmez\w*"), True, False),
    "guaranteed": (re.compile(r"\bguaranteed\b"), True, True),
    "always wins": (re.compile(r"\balways wins?\b"), True, False),
    "no risk": (NO_RISK_RE, False, False),
    "%100 kazanç": (re.compile(r"%\s*100\s*kazanc\w*"), True, False),
    "100% profit": (re.compile(r"\b100\s*%\s*profit\w*"), True, False),
}

# Çıplak garanti kelimesinin vaat sayılması için gereken finansal nesne (tr_fold'lanmış).
# "loss" BİLEREK yok (ML'de "loss function" her yerde); "lose/losing" var ("never lose,
# guaranteed"). "risk-free/risksiz" de nesne sayılır ("guaranteed … risk-free").
_FINANCIAL_OBJECT_RE = re.compile(
    r"\b(?:profit\w*|returns?|gains?|win|wins|winning|income|earnings?|payoffs?|payouts?"
    r"|lose|loses|losing|risk[\s\-]?free|risksiz\w*"
    r"|kar|kari|karin|karini|karli\w*|karlar\w*|kazan\w*|getiri\w*)\b"
)
# "guaranteed to return the optimum" — fiil 'return' getiri değildir.
_VERB_RETURN_RE = re.compile(r"\bto\s+$")
_FIN_OBJECT_WINDOW: int = 40
_CLAUSE_BREAK_RE = re.compile(r"[.!?;\n]")


def _has_financial_object(folded: str, start: int, end: int) -> bool:
    """Eşleşmenin yakınında (aynı cümle, ±40 karakter) finansal nesne var mı?"""
    left = folded[max(0, start - _FIN_OBJECT_WINDOW) : start]
    right = folded[end : end + _FIN_OBJECT_WINDOW]
    # Cümle sınırının ötesine taşma (ondalık "2.5" nokta sayılmasın diye rakam arası hariç).
    left_breaks = [m.end() for m in _CLAUSE_BREAK_RE.finditer(left) if not _is_decimal(left, m)]
    if left_breaks:
        left = left[left_breaks[-1] :]
    right_break = next(
        (m.start() for m in _CLAUSE_BREAK_RE.finditer(right) if not _is_decimal(right, m)), None
    )
    if right_break is not None:
        right = right[:right_break]
    for segment in (left, right):
        for obj in _FINANCIAL_OBJECT_RE.finditer(segment):
            if obj.group(0) == "return" and _VERB_RETURN_RE.search(segment[: obj.start()]):
                continue  # "guaranteed to return …" fiildir
            return True
    return False


def _is_decimal(text: str, match: re.Match[str]) -> bool:
    i = match.start()
    return (
        match.group(0) == "."
        and 0 < i < len(text) - 1
        and text[i - 1].isdigit()
        and text[i + 1].isdigit()
    )


def overconfident_hits(folded: str) -> list[str]:
    """`folded` (tr_fold'lanmış) metindeki OLUMSUZLANMAMIŞ aşırı-emin ifade etiketleri.

    Tek bir olumsuzlanmamış geçiş yeterlidir → "not guaranteed … but guaranteed profit"
    yine yakalanır; olumsuzlama yalnız kendi geçişini muaf tutar.
    """
    hits: list[str] = []
    for label, (pattern, check_before, needs_object) in _OVERCONFIDENT_RULES.items():
        for match in pattern.finditer(folded):
            start, end = match.start(), match.end()
            if check_before:
                if is_negated(folded, start, end):
                    continue
            elif NEG_AFTER_RE.search(folded[end : end + NEG_WINDOW]):
                continue
            if needs_object and not _has_financial_object(folded, start, end):
                continue
            hits.append(label)
            break
    return hits


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
#
# Aralıklar İKİ ucuyla okunur ("10-15%", "10–15%", "10 to 15%", "10%-15%"): eskiden yalnız
# üst uç (15) alınıyordu → karttaki uydurma alt uç (10) hiç denetlenmiyordu (Kademe 2 B2).
# Türkçe ön-ek biçimi ("%15", "%10-15", "%10 ile %15") da yüzdedir.
_NUM = r"\d+(?:[.,]\d+)?"
_RANGE_SEP = r"\s*(?:-|–|—|\bto\b|\bile\b|\bila\b)\s*"
_SRC_PCT_RE = re.compile(
    rf"(?<![\w.,])({_NUM})(?:\s*%?{_RANGE_SEP}({_NUM}))?\s*(?:%|percent\b|per cent\b)"
)
_SRC_PCT_PREFIX_RE = re.compile(rf"(?<![\w%])%\s*({_NUM})(?:{_RANGE_SEP}%?\s*({_NUM}))?")
# Ondalıklı sayı yeterince ayırt edicidir → kaynakta % işaretsiz de kabul ("accuracies of
# 63.707" tablosu). Tam sayı ise % bağlamı ister ("15" her metinde sayfa no. olarak geçer).
_SRC_DECIMAL_RE = re.compile(r"(?<![\w.,])(\d+[.,]\d+)(?!\d)")
# Cümle sınırı — sayı İÇİNDEKİ nokta bölmez ("63.707%" tek parça kalır).
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)[.;](?!\d)|\n")
# Açıkça örnek parametre ("e.g., 2% below current price") → cümledeki yüzdeler iddia DEĞİL.
_EXAMPLE_RE = re.compile(r"\be\.?g\b")
# Önerilen test cümlesi ("Test if … above 70%"). Muafiyet YALNIZ eşik ifadesine uygulanır;
# aynı cümledeki etki büyüklüğü ("by 15%", "%15 daha") iddia olarak kalır (inceleme).
_TEST_IF_RE = re.compile(r"\b(?:test|evaluate|measure|assess|examine)\s+(?:if|whether)\b")
_THRESHOLD_BEFORE_RE = re.compile(
    r"(?:\babove|\bbelow|\bexceeds?|\bexceeding|\bover|\bunder|\bbeyond|\bgreater than"
    r"|\bless than|\bmore than|\bhigher than|\blower than|\bat least|\bat most"
    r"|\bthreshold(?: of)?|[<>≤≥]=?)\s*$"
)
_THRESHOLD_AFTER_RE = re.compile(
    r"^['’]?\w*\s+(?:uzerin\w*|ustun\w*|altin\w*|asar\w*|asiyor\w*|gecer\w*|esig\w*)"
)
_THRESHOLD_WINDOW: int = 30


def _norm_number(raw: str) -> str:
    return f"{float(raw.replace(',', '.')):g}"


def _iter_percentages(folded: str) -> list[tuple[str, int, int]]:
    """Yüzde değerleri (normalize) + eşleşme aralığı; aralıklarda her iki uç ayrı değer."""
    out: list[tuple[str, int, int]] = []
    for pattern in (_SRC_PCT_RE, _SRC_PCT_PREFIX_RE):
        for m in pattern.finditer(folded):
            for raw in (m.group(1), m.group(2)):
                if raw:
                    out.append((_norm_number(raw), m.start(), m.end()))
    return out


def source_numbers(source_text: str) -> frozenset[str]:
    """Kaynak metindeki yüzdeler + ondalıklı sayılar (normalize: "15.0" → "15")."""
    folded = tr_fold(source_text)
    nums = {value for value, _, _ in _iter_percentages(folded)}
    nums |= {_norm_number(m.group(1)) for m in _SRC_DECIMAL_RE.finditer(folded)}
    return frozenset(nums)


def _is_threshold(sentence: str, start: int, end: int) -> bool:
    """Yüzde eşik ifadesi mi ("above 70%", "%70'in üzerinde")?"""
    before = sentence[max(0, start - _THRESHOLD_WINDOW) : start]
    after = sentence[end : end + _THRESHOLD_WINDOW]
    return bool(_THRESHOLD_BEFORE_RE.search(before) or _THRESHOLD_AFTER_RE.search(after))


def _numbers_missing_from_source(folded: str, source_nums: frozenset[str]) -> list[str]:
    """Karttaki (iddia cümlelerindeki) yüzdelerden kaynakta OLMAYANLAR, artan sırada."""
    claims: set[str] = set()
    # "e.g." kısaltmasının noktaları cümleyi bölmesin (örnek parametre muafiyeti kaybolurdu).
    folded = re.sub(r"\be\.g\.", "eg", folded)
    for sentence in _SENTENCE_SPLIT_RE.split(folded):
        if _EXAMPLE_RE.search(sentence):
            continue
        is_test = bool(_TEST_IF_RE.search(sentence))
        for value, start, end in _iter_percentages(sentence):
            if is_test and _is_threshold(sentence, start, end):
                continue  # önerilen test eşiği — iddia değil
            claims.add(value)
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


def verify_math_content(text: str, source_nums: frozenset[str] | None = None) -> MathVerifyResult:
    """Bir metni matematik/istatistik açısından doğrula.

    Geriye dönen sonuç:
      - `issues`: bulunan tüm uyarılar
      - `requires_review`: insan incelemesi gerekiyor mu
      - `passed`: aşırı emin yatırım ifadesi yoksa True (bunlar blocker)

    `source_nums` (``source_numbers(kaynak_metin)``) verilirse, karttaki yüzdelerden kaynakta
    OLMAYANLAR inceleme işareti alır. ``None`` (kaynak metni yok) → kontrol atlanır;
    doğrulanamayan kart yanlışlıkla işaretlenmez. BOŞ küme ise kaynağın metni var ama hiç
    sayı içermiyor demektir → karttaki her yüzde kaynakta YOK (inceleme; Kademe 2 B2 — eskiden
    boş küme de "kontrol yok" sayılıp uydurma yüzde sessizce geçiyordu).
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

    hits = overconfident_hits(folded)
    issues.extend(f"aşırı emin yatırım ifadesi: '{phrase}'" for phrase in hits)
    overconfident_found = bool(hits)

    issues.extend(_check_percentage_sanity(text))
    # Doğrulanmamış sayısal performans iddiaları (requires_review, BLOK DEĞİL).
    issues.extend(_check_performance_claims(folded))
    if source_nums is not None:
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
