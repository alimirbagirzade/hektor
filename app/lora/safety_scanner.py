"""Güvenlik / sır tarayıcısı — eğitim verisinde gizli ve tehlikeli içerik.

Gate 7 (BLOCKER) için kullanılır. Tek ihlal bile batch'i reddeder.
Regex tabanlı; LLM gerektirmez. API anahtarı, özel anahtar, parola,
kişisel veri ve kesin finansal yönlendirme arar.

İki desen daraltıldı (Kademe-2 bug-avı bulgusu B5, 2026-06-22) — Gate 7 artık
TÜM nonempty kartları tarıdığından kör desenler yanlış-pozitifle GO'yu kilitliyordu:

* ``national_id``: çıplak ``\\d{11}`` her 11 haneli sayıyı (hacim, zaman damgası,
  veri-kümesi boyutu) TC kimlik sanıyordu. Artık ya **TC checksum** doğrulanır
  (gerçek kimlik, etiketsiz bile yakalanır) ya da yakın **bağlam anahtarı**
  ('TC'/'kimlik'/'TCKN'/'vatandaş') aranır.
* ``api_key``: çıplak 32+ alfanümerik token uzun hash / LaTeX / değişken adı /
  base64 parçasını sır sanıyordu. Artık ya bilinen **sır ön-eki** (sk-/ghp_/AKIA…)
  ya da **çok-sınıflı + yüksek-entropi** token gerekir; saf hex hash ve düz
  tanımlayıcılar elenir, gerçek anahtarlar yakalanır.

Daraltma yalnızca yanlış-pozitifi düşürür; gerçek sır/PII yakalama korunur
(testlerle adversarial doğrulandı).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field


def tr_fold(text: str) -> str:
    """Türkçe-bilinçli normalize: büyük/küçük harf + aksan farklarını eşitle.

    Python ``str.lower()`` Türkçe büyük 'İ'yi 'i' + birleşik nokta (U+0307)
    dizisine çevirir; bu yüzden ``"ŞİMDİ AL".lower()`` içinde "şimdi al" GEÇMEZ.
    Bu fonksiyon İ/I'yı doğru eşler, casefold uygular ve birleşik aksanları
    temizler — böylece büyük harf / aksanlı yazımlar finansal-yönlendirme
    taramasını atlatamaz (BLOCKER gate güvenliği).
    """
    folded = text.replace("İ", "i").replace("I", "ı").casefold()
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# --------------------------------------------------------------------------- #
# api_key — daraltılmış sezgi (entropi + karakter-sınıfı + bilinen ön-ek)
# --------------------------------------------------------------------------- #

# 32+ karakter aday token. base64/credential karakterlerini ('+','/','=') de İÇERİR:
# kanonik AWS secret access key gibi '/'+'+' taşıyan sırlar aksi halde 32-altı parçalara
# bölünüp entropi/sınıf sezgisini hiç tetiklemeden Gate 7'yi (BLOCKER) ATLIYORDU (FN). \b
# yok çünkü '+/=' kelime-sınırı değil; çok-sınıf+yüksek-entropi kapısı hex hash/path/
# tanımlayıcıyı yine eler (yalnız gerçek base64 sır geçer).
_API_KEY_CANDIDATE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_\-+/]{32,}={0,2}")
# Gerçek rastgele anahtar bu eşiklerin üstünde; saf hex hash (≤2 sınıf) ve düz
# tanımlayıcı/LaTeX (düşük entropi) altında kalır.
_API_KEY_MIN_ENTROPY: float = 3.5
_API_KEY_MIN_CHAR_CLASSES: int = 3

# Tartışmasız sır biçimleri — sınıf/entropi sezgisinden bağımsız yakalanır.
# (Yüksek-kesinlik: bu ön-ekler düz metinde yüksek-entropili gövdeden önce
# kazara neredeyse hiç gelmez.)
_SECRET_PREFIX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub PAT / OAuth / refresh
    re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),  # Stripe
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),  # OpenAI
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),  # AWS access key id
    re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b"),  # Google API key
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),  # GitLab PAT
]


def _shannon_entropy(token: str) -> float:
    """Token'ın karakter başına Shannon entropisini (bit) döndür."""
    if not token:
        return 0.0
    length = len(token)
    return -sum((n / length) * math.log2(n / length) for n in Counter(token).values())


def _char_class_count(token: str) -> int:
    """Token'da kaç ayrı karakter-sınıfı (küçük/büyük/rakam) var?"""
    has_lower = any(c.islower() for c in token)
    has_upper = any(c.isupper() for c in token)
    has_digit = any(c.isdigit() for c in token)
    return sum((has_lower, has_upper, has_digit))


def _detect_api_key(text: str) -> bool:
    """Gerçek API anahtarı / sır benzeri token ara (daraltılmış).

    Önce bilinen sır ön-eklerine, sonra çok-sınıflı + yüksek-entropili 32+ token'a
    bakar. Saf hex hash (yalnız küçük+rakam = 2 sınıf), düz değişken adı (1 sınıf)
    ve LaTeX/yapılandırılmış dizgeler (düşük entropi) elenir.
    """
    for prefix in _SECRET_PREFIX_PATTERNS:
        if prefix.search(text):
            return True
    for match in _API_KEY_CANDIDATE.finditer(text):
        token = match.group(0)
        if (
            _char_class_count(token) >= _API_KEY_MIN_CHAR_CLASSES
            and _shannon_entropy(token) >= _API_KEY_MIN_ENTROPY
        ):
            return True
    return False


# --------------------------------------------------------------------------- #
# national_id — TC checksum + bağlam anahtarı (çıplak 11-hane yerine)
# --------------------------------------------------------------------------- #

_NATIONAL_ID_CANDIDATE: re.Pattern[str] = re.compile(r"\b\d{11}\b")
# Etiketli (gerçek/sahte) kimlik bağlamı. 'tc' kelime sınırına bağlı + ardından
# harf gelmez (tcp/batch/match yanlış-pozitifi olmaz); 'kimlik'/'tckn'/'vatandaş'
# zaten ayırt edici.
_NATIONAL_ID_CONTEXT: re.Pattern[str] = re.compile(
    r"(?i)(?:kimlik|tckn|vatanda[şs]|\bt\.?\s?c\.?(?![a-z]))"
)
_NATIONAL_ID_WINDOW: int = 30


def _valid_tc_checksum(digits: str) -> bool:
    """TC kimlik numarası checksum'ı (10. ve 11. hane) doğru mu?

    Rastgele 11 haneli bir sayının her iki kontrol hanesini de tutturma olasılığı
    ~%1; bu yüzden checksum, etiketsiz gerçek kimliği yakalarken hacim/zaman-damgası
    gibi sayıları elemenin güçlü bir yoludur.
    """
    if len(digits) != 11 or digits[0] == "0":
        return False
    nums = [int(c) for c in digits]
    odd_sum = nums[0] + nums[2] + nums[4] + nums[6] + nums[8]
    even_sum = nums[1] + nums[3] + nums[5] + nums[7]
    check10 = (odd_sum * 7 - even_sum) % 10
    check11 = sum(nums[:10]) % 10
    return nums[9] == check10 and nums[10] == check11


def _detect_national_id(text: str) -> bool:
    """TC kimlik numarası ara (daraltılmış).

    11 haneli her sayı için: checksum geçerliyse (gerçek kimlik) ya da hemen
    öncesinde 'TC'/'kimlik' türü bağlam anahtarı varsa (etiketli sahte/placeholder
    kimlik) ihlal sayılır. Bağlamsız + checksum'sız 11 haneli sayılar geçer.
    """
    for match in _NATIONAL_ID_CANDIDATE.finditer(text):
        digits = match.group(0)
        if _valid_tc_checksum(digits):
            return True
        window_start = max(0, match.start() - _NATIONAL_ID_WINDOW)
        if _NATIONAL_ID_CONTEXT.search(text[window_start : match.start()]):
            return True
    return False


# --------------------------------------------------------------------------- #
# phone — gerçek biçimlendirme iste (çıplak rakam dizisi telefon değil)
# --------------------------------------------------------------------------- #

# Eski desen tüm ön-ek/ayraçları opsiyonel yaptığından pratikte `\d{10}`'a çöküp
# ayraçsız HERHANGİ 10/11-haneli sayıyı (Unix epoch, işlem hacmi, veri-kümesi satır
# sayısı) telefon sanıyordu → national_id'nin (B5) geçirdiği sayıları Gate 7'de yeniden
# FP'yle blokluyordu. Artık GERÇEK telefon biçimi zorunlu: +90 ön-eki, parantezli alan
# kodu, ya da rakam grupları arasında AYRAÇ. Ayraçsız çıplak sayı telefon SAYILMAZ.
_PHONE_FORMATTED: re.Pattern[str] = re.compile(
    r"\+90[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"  # +90 5xx xxx xx xx
    r"|\(0?\d{3}\)[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}"  # (0212) 555 12 34
    r"|\b0\d{2,3}[\s\-]\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b"  # 0212 555 12 34 (ayraç zorunlu)
    r"|\b\d{3}[\s\-]\d{3}[\s\-]\d{2}[\s\-]\d{2}\b"  # 555 123 45 67 (gruplar ayraçlı)
)


# --------------------------------------------------------------------------- #
# credential_assignment — sonek-toleranslı anahtar + sır-benzeri değer kapısı
# --------------------------------------------------------------------------- #

# Anahtar kelime ön-ek alır (aws_secret…, client_secret) ve yalnız credential-y sonek
# bileşenleriyle uzar (secret_access_key) → 'tokenizer'/'token_count'/'max_tokens'/
# 'password_strength' gibi tanımlayıcılar kelime-sınırı nedeniyle EŞLEŞMEZ. Değer ayrıca
# sır-benzeri olmalı → proza ':' ardılı ('token: kelime') ve saf-sayı hiperparametre
# ('token_count=5', 'secret_santa_budget=100000') FP'leri elenir.
_CREDENTIAL_ASSIGN: re.Pattern[str] = re.compile(
    r"(?i)\b[\w-]*?(?:password|passwd|secret|token|api[_-]?key|access[_-]?key)"
    r"(?:[_-](?:key|token|id|secret|value|pwd|pass|access))*\b\s*[=:]\s*(\S+)"
)


def _detect_credential_assignment(text: str) -> bool:
    """password=/secret:/aws_secret_access_key= gibi sır atamalarını ara (daraltılmış).

    Değer tarafı sır-benzeri olmalı: bilinen sır ön-eki, YA DA ≥6 uzunluk + ≥2 karakter
    sınıfı + makul entropi. Böylece 'token: kelime' (proza) ve 'token_count=5' / 'max_
    tokens=512' (saf-sayı hiperparametre) Gate 7'yi kilitlemez; 'password=hunter2' ve
    'aws_secret_access_key=<base64>' gerçek sırları yakalanır.
    """
    for match in _CREDENTIAL_ASSIGN.finditer(text):
        value = match.group(1).strip("'\"`").rstrip(".,;:)")
        if not value:
            continue
        if any(p.search(value) for p in _SECRET_PREFIX_PATTERNS):
            return True
        if len(value) >= 6 and _char_class_count(value) >= 2 and _shannon_entropy(value) >= 2.0:
            return True
    return False


def _regex_detector(pattern: re.Pattern[str]) -> Callable[[str], bool]:
    """Bir derlenmiş regex'i `text -> bool` dedektörüne sar."""

    def _detect(text: str) -> bool:
        return pattern.search(text) is not None

    return _detect


# (etiket, dedektör) çiftleri. Dedektör `text -> bool` döndürür.
FORBIDDEN_DETECTORS: list[tuple[str, Callable[[str], bool]]] = [
    # 32+ karakter API key benzeri token (entropi + sınıf + ön-ek ile daraltıldı).
    ("api_key", _detect_api_key),
    # PEM özel anahtar başlığı.
    (
        "private_key",
        _regex_detector(re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ),
    # Bitcoin/Ethereum benzeri cüzdan adresi.
    (
        "wallet_address",
        _regex_detector(re.compile(r"\b(?:0x[a-fA-F0-9]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b")),
    ),
    # password= / token= / secret= / aws_secret_access_key= atamaları (sonek-toleranslı
    # anahtar + sır-benzeri değer kapısı; proza ':' ve saf-sayı hiperparametre FP'siz).
    ("credential_assignment", _detect_credential_assignment),
    # E-posta adresi.
    (
        "email",
        _regex_detector(re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ),
    # Telefon numarası — GERÇEK biçim (ön-ek/parantez/ayraç) zorunlu; çıplak 10/11-haneli
    # sayı (hacim/epoch/ID) telefon SAYILMAZ (eski desen \d{10}'a çöküp FP'yle GO kilitliyordu).
    ("phone", _regex_detector(_PHONE_FORMATTED)),
    # TC kimlik numarası — checksum veya bağlam anahtarı ile daraltıldı.
    ("national_id", _detect_national_id),
]

# Kesin al/sat / garanti kâr yönlendirmesi (küçük harf eşleşme).
FINANCIAL_DIRECTIVES: list[str] = [
    "buy now",
    "sell now",
    "şimdi al",
    "şimdi sat",
    "garanti kar",
    "garanti kâr",
    "guaranteed profit",
    "risk yok",
    "no risk",
]


@dataclass
class SafetyResult:
    """Güvenlik taraması sonucu. `passed=False` ise batch reddedilir."""

    passed: bool
    violations: list[str] = field(default_factory=list)


def scan_for_secrets(text: str) -> SafetyResult:
    """Metni sır, kişisel veri ve finansal yönlendirme için tara.

    Tek bir ihlal bulunsa bile `passed=False` döner (kısmi geçiş yok).
    """
    if not text:
        return SafetyResult(passed=True)

    violations: list[str] = []

    for label, detect in FORBIDDEN_DETECTORS:
        if detect(text):
            violations.append(f"yasak desen: {label}")

    folded = tr_fold(text)
    seen_directives: set[str] = set()
    for directive in FINANCIAL_DIRECTIVES:
        folded_directive = tr_fold(directive)
        if folded_directive in folded and folded_directive not in seen_directives:
            seen_directives.add(folded_directive)
            violations.append(f"finansal yönlendirme: '{directive}'")

    return SafetyResult(passed=not violations, violations=violations)
