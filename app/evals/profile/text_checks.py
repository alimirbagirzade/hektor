"""Deterministik metin kontrolleri — sayı çıkarma, exact match, iddia kapsama, çekimserlik."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.evals.profile.leakage import normalize_text

# 1.234,5 (TR) · 1,234.5 (EN) · 0.05 · -3 · 1e-4 · %12
_NUM = re.compile(r"[-−]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
_FINAL_MARKERS = re.compile(
    r"(?:cevap|sonuç|yanıt|answer|result|final)\s*[:=]\s*(.+)", re.IGNORECASE
)
_CHOICE = re.compile(r"(?:^|[\s(\[:])([A-E])(?:[)\].:\s]|$)")

_ABSTAIN = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"bilmiyorum",
        r"emin değilim",
        r"yeterli (bilgi|veri|kanıt|kaynak)",
        r"(kaynak|bağlam|belge|makale)\w*\s+(bu|bununla|buna)?\s*(ilgili)?\s*"
        r"(bilgi|kanıt)?\s*(yok|bulunmuyor|bulunamadı|içermiyor)",
        r"cevaplayamam",
        r"yanıtlayamam",
        r"belirlenemez",
        r"retrieval boş",
        r"i (do not|don't) know",
        r"not enough (information|evidence|context)",
        r"insufficient (information|evidence|context)",
        r"cannot (be )?(determine|answer)",
        r"no (relevant )?(source|evidence|information)",
    )
)


def parse_number(token: str) -> float | None:
    t = token.replace("−", "-").strip()
    if "," in t and "." in t:
        # Son ayırıcı ondalık kabul edilir.
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        parts = t.split(",")
        # 1,234 → binlik (3 hane) ; 0,05 → ondalık
        t = t.replace(",", "") if len(parts[-1]) == 3 and len(parts) > 1 else t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM.finditer(text):
        v = parse_number(m.group(0))
        if v is not None:
            out.append(v)
    return out


def final_answer_segment(text: str) -> str:
    """'Cevap: ...' işaretli son satır varsa onu, yoksa son boş-olmayan satırı döndür."""
    matches = list(_FINAL_MARKERS.finditer(text))
    if matches:
        return matches[-1].group(1).strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def final_number(text: str) -> float | None:
    seg_nums = extract_numbers(final_answer_segment(text))
    if seg_nums:
        return seg_nums[-1]
    nums = extract_numbers(text)
    return nums[-1] if nums else None


def numeric_match(value: float | None, expected: float, tolerance: float | None) -> bool:
    if value is None:
        return False
    # Yüzde↔oran (12 vs 0.12) esnekliği BİLİNÇLİ YOK: 2 beklenirken 200'ü kabul eden bir
    # sahte PASS'e yol açar. Birim eval kaydında (soru metninde) açıkça belirtilmelidir.
    tol = tolerance if tolerance is not None else max(1e-6, abs(expected) * 1e-4)
    return abs(value - expected) <= tol


def exact_match(answer: str, accepted: Iterable[str]) -> bool:
    """Çoktan seçmeli / kısa cevap: normalize edilmiş final segment tam eşleşmesi."""
    acc = [a for a in accepted if a.strip()]
    if not acc:
        return False
    seg = final_answer_segment(answer)
    norm_seg = normalize_text(seg)
    for a in acc:
        na = normalize_text(a)
        if len(a.strip()) == 1 and a.strip().upper() in "ABCDE":
            m = _CHOICE.findall(seg.upper())
            if m and m[-1] == a.strip().upper():
                return True
            continue
        if na and (norm_seg == na or normalize_text(answer) == na):
            return True
    return False


def claim_present(claim: str, answer: str) -> bool:
    """İddia cevapta var mı: normalize alt-dize ya da tüm içerik kelimeleri mevcut.

    ``a|b|c`` biçimi eşanlamlı alternatiflerdir (biri yeterli): ör.
    ``"örneklem dışı|out-of-sample|oos"``.
    """
    na = normalize_text(answer)
    for alt in claim.split("|"):
        nc = normalize_text(alt)
        if not nc:
            continue
        if re.search(rf"(?<!\w){re.escape(nc)}", na):
            return True
        words = [w for w in nc.split() if len(w) > 3]
        if len(words) > 1 and all(w in na for w in words):
            return True
    return False


def claim_coverage(claims: Iterable[str], answer: str) -> tuple[float | None, list[str]]:
    cl = [c for c in claims if c.strip()]
    if not cl:
        return None, []
    missing = [c for c in cl if not claim_present(c, answer)]
    return (len(cl) - len(missing)) / len(cl), missing


def forbidden_hits(claims: Iterable[str], answer: str) -> list[str]:
    return [c for c in claims if c.strip() and claim_present(c, answer)]


def is_abstention(answer: str) -> bool:
    return any(rx.search(answer) for rx in _ABSTAIN)
