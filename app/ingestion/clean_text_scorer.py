"""Metin temizliği skoru (0-10) — salt-regex, deterministik.

PDF parse kalitesinin bir göstergesi: kontrol karakterleri, encoding-hatası işareti
(U+FFFD) ve aşırı satır-sonu tireli kırılma cezalandırılır. ``eval``/``exec`` yok (Kural 5).
"""

from __future__ import annotations

import re

# tab(\x09), newline(\x0a), CR(\x0d) hariç kontrol karakterleri (parse artığı işareti)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_REPLACEMENT_RE = re.compile("�")  # encoding bozulması
_HYPHEN_BREAK_RE = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ]-\s*\n[a-zçğıöşü]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def score_clean_text(text: str) -> float:
    """Metin temizliğini 0-10 arası puanla (10 = temiz). Boş metin → 0."""
    if not text or not text.strip():
        return 0.0
    n = len(text)
    control = len(_CONTROL_RE.findall(text))
    repl = len(_REPLACEMENT_RE.findall(text))
    n_words = max(1, len(_WORD_RE.findall(text)))
    hyphen_breaks = len(_HYPHEN_BREAK_RE.findall(text))

    score = 10.0
    score -= min(4.0, (control / n) * 4000)  # ~%0.1 kontrol karakteri → -4
    score -= min(3.0, (repl / n) * 3000)  # ~%0.1 replacement → -3
    score -= min(2.0, (hyphen_breaks / n_words) * 40)  # kelime başına aşırı kırılma
    if n < 200:
        score -= 1.0  # çok kısa içerik (muhtemel parse kaybı)
    return round(max(0.0, score), 2)
