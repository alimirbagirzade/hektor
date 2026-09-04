"""Trading hipotez değerlendiricisi — bir fikrin "sinyal" değil "test edilebilir hipotez"
olduğunu denetler (CLAUDE.md Mutlak Kurallar 1-4).

Çıktı asla yatırım tavsiyesi DEĞİLDİR; bir hipotezin test edilmeye HAZIR olup
olmadığını puanlar. Salt-regex (eval/exec yok, Kural 5), deterministik.

Denetlenen maddeler:
- testable      : ölçülebilir/test edilebilir çerçeve var mı (Kural 2)
- costs         : komisyon/slippage/maliyet farkındalığı (Kural 3)
- out_of_sample : örneklem-dışı / look-ahead / walk-forward farkındalığı (Kural 2,4)
- no_advice     : "kesin/garanti/%100/risksiz" gibi tavsiye-kesinlik dili YOK (Kural 1)
- risk_noted    : risk notu var mı
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Kesinlik / tavsiye dili — VARSA hipotez reddedilir (Kural 1).
_ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"%\s*100|100\s*%|yüzde\s*yüz"),
    re.compile(r"(?i)\b(garanti|risksiz|kesin(likle)?|mutlaka)\b"),
    # fiil + tireli/boşluklu varyantlar: guarantee(s/d), sure[-/ ]fire, can('|'? )t lose
    re.compile(
        r"(?i)\b(guarantee[ds]?|risk[\s-]?free|always wins?|sure[\s-]?fire|can'?\s?t lose)\b"
    ),
    re.compile(r"(?i)\bkesin sinyal\b|\bal ve unut\b|\bhemen al\b|\bşimdi al\b"),
)
_TESTABLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(test|backtest|geri test|ölç|sına|doğrula|deneme|hipotez)\w*"),
    re.compile(r"(?i)\b(eğer|when|if)\b.*\b(ise|then|olur|artar|azalır)\b"),
)
_COST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(komisyon|slippage|kayma|işlem maliyet|maliyet|spread|ücret|fee)\w*"),
)
_OOS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(örneklem[\s-]?dışı|out[\s-]?of[\s-]?sample|oos|walk[\s-]?forward)\b"),
    re.compile(r"(?i)\b(look[\s-]?ahead|ileriye[\s-]?bakış|shift|gecikme|geciktir)\w*"),
)
_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(risk|drawdown|stop[\s-]?loss|zarar|kayıp|pozisyon boyut)\w*"),
)

_TEXT_FIELDS = (
    "hypothesis_text",
    "hypothesis",
    "text",
    "title",
    "risk_notes",
    "assumptions",
    "required_data",
    "description",
)


@dataclass
class HypothesisEvalResult:
    """Tek bir trading hipotezinin değerlendirme sonucu (hipotez; tavsiye değil)."""

    hypothesis_id: str
    score: float  # 0.0–1.0
    verdict: str  # candidate / needs_revision / rejected
    checklist: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    note: str = "Bu bir test edilebilirlik puanıdır, yatırım tavsiyesi değildir."

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "score": self.score,
            "verdict": self.verdict,
            "checklist": self.checklist,
            "warnings": self.warnings,
            "note": self.note,
        }


def _collect_text(hypothesis: dict[str, Any] | str) -> str:
    if isinstance(hypothesis, str):
        return hypothesis
    parts: list[str] = []
    for key in _TEXT_FIELDS:
        val = hypothesis.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list | tuple):
            parts.extend(str(v) for v in val)
    return "  ".join(parts)


def _any(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


# Olumsuzlama-bilinçli advice taraması (gates.gate_6 ile aynı ilke). Bir kesinlik kelimesi
# olumsuzlanmışsa eşleşme ALÇAKGÖNÜLLÜ ifadedir, tavsiye DEĞİL: "risksiz değildir" / "garanti
# yok" / "not guaranteed" / "no strategy is risk-free" reddi TETİKLEMEZ. ANCAK olumsuzlama
# yalnız eşleşmeyle AYNI CÜMLECİKTE bastırır — aksi hâlde ayrı cümle/komuttaki olumsuzlama
# ("Garanti kâr. Şüphe yok." / "Kesinlikle al, asla satma.") gerçek tavsiyeyi kör edip kapıdan
# geçiriyordu (Kural 1 sahte-PASS). Sonra-olumsuzlama yalnız Türkçe son-ektir: İngilizce sonrası
# 'never/no' ("risk-free and never fails") olumsuzlama değil üstünlük iddiasıdır.
_CLAUSE_BOUND_RE: re.Pattern[str] = re.compile(
    r"[.;:!?,]|\b(?:ama|fakat|ancak|but|however|yet)\b", re.IGNORECASE
)
_ADVICE_NEG_BEFORE: re.Pattern[str] = re.compile(
    r"(?i)\b(?:de[ğg]il\w*|yok\w*|olma[zd]\w*|hi[çc]bir|hi[çc]\w*|asla"
    r"|not|no|never|non|without|cannot|can'?t|hardly|isn'?t|aren'?t)\b"
)
_ADVICE_NEG_AFTER: re.Pattern[str] = re.compile(r"(?i)\b(?:de[ğg]il\w*|yok\w*|olma[zd]\w*)\b")
_ADVICE_NEG_WINDOW: int = 24


def _clause_before(text: str, start: int) -> str:
    """``start`` öncesi, pencere içindeki SON cümlecik sınırından sonrası (sınırı aşmaz)."""
    seg = text[max(0, start - _ADVICE_NEG_WINDOW) : start]
    last_end = 0
    for m in _CLAUSE_BOUND_RE.finditer(seg):
        last_end = m.end()
    return seg[last_end:]


def _clause_after(text: str, end: int) -> str:
    """``end`` sonrası, pencere içindeki İLK cümlecik sınırına kadar (sınırı aşmaz)."""
    seg = text[end : end + _ADVICE_NEG_WINDOW]
    m = _CLAUSE_BOUND_RE.search(seg)
    return seg[: m.start()] if m else seg


def _advice_present(text: str) -> bool:
    """Kesinlik/tavsiye dili VAR mı — yalnız AYNI CÜMLECİKTEKİ olumsuzlamayı alçakgönüllü
    sayıp atlayarak.

    Kural 1: çıktı alçakgönüllü _hipotez_ olmalı. Naif kelime-eşleşmesi "risksiz değildir"
    gibi olumsuz ifadeleri tavsiye sanıp iyi hipotezleri reddediyordu; eski ±pencere düzeltmesi
    ise ANCHOR'suz olduğu için ayrı cümledeki/komuttaki olumsuzlamayı gerçek tavsiyeyi BASTIRMAK
    için kullanıp tavsiyeyi kapıdan geçiriyordu. Artık olumsuzlama yalnız eşleşmeyle aynı
    cümlecikteyse (sınır: ``.;:!?,`` + bağlaç) bastırır; sonra-olumsuzlama yalnız Türkçe son-ek.
    """
    for pat in _ADVICE_PATTERNS:
        for m in pat.finditer(text):
            before = _clause_before(text, m.start())
            after = _clause_after(text, m.end())
            if _ADVICE_NEG_BEFORE.search(before) or _ADVICE_NEG_AFTER.search(after):
                continue  # aynı cümlecikte olumsuzlanmış → alçakgönüllü, tavsiye değil
            return True
    return False


def evaluate_hypothesis(
    hypothesis: dict[str, Any] | str, *, hypothesis_id: str = ""
) -> HypothesisEvalResult:
    """Bir trading hipotezini test-edilebilirlik kuralları için değerlendir."""
    text = _collect_text(hypothesis)
    hid = hypothesis_id
    if not hid and isinstance(hypothesis, dict):
        hid = str(hypothesis.get("hypothesis_id") or hypothesis.get("id") or "")

    has_advice = _advice_present(text)
    checklist = {
        "testable": _any(_TESTABLE_PATTERNS, text),
        "costs": _any(_COST_PATTERNS, text),
        "out_of_sample": _any(_OOS_PATTERNS, text),
        "no_advice": not has_advice,
        "risk_noted": _any(_RISK_PATTERNS, text),
    }
    warnings: list[str] = []
    if has_advice:
        warnings.append("Kesinlik/tavsiye dili tespit edildi (Kural 1) — hipotez reddedildi.")
    if not checklist["testable"]:
        warnings.append("Test edilebilir çerçeve yok — ölçülebilir koşul/backtest planı ekle.")
    if not checklist["costs"]:
        warnings.append("Maliyet (komisyon/slippage) farkındalığı yok (Kural 3).")
    if not checklist["out_of_sample"]:
        warnings.append("Örneklem-dışı / look-ahead farkındalığı yok (Kural 2,4).")
    if not checklist["risk_noted"]:
        warnings.append("Risk notu yok.")

    score = round(sum(1 for v in checklist.values() if v) / len(checklist), 4)

    # Verdict: tavsiye dili veya test-edilemezlik → HARD reddet (Kural 1,2).
    if has_advice or not checklist["testable"]:
        verdict = "rejected"
    elif not (checklist["costs"] and checklist["out_of_sample"]):
        verdict = "needs_revision"
    else:
        verdict = "candidate"

    return HypothesisEvalResult(
        hypothesis_id=hid or "hyp",
        score=score,
        verdict=verdict,
        checklist=checklist,
        warnings=warnings,
    )


def evaluate_many(
    hypotheses: list[dict[str, Any] | str],
) -> list[HypothesisEvalResult]:
    """Birden çok hipotezi sırayla değerlendir (deterministik)."""
    return [evaluate_hypothesis(h, hypothesis_id=f"hyp_{i}") for i, h in enumerate(hypotheses)]
