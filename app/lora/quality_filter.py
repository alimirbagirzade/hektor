"""Kalite filtresi — kısa, tekrarlı ve duplicate eğitim örneklerini ele.

Gate 4 için kullanılır. Deterministik; LLM gerektirmez. Bir kartın
soru/cevap içeriğini çıkarır ve temel kalite kontrollerinden geçirir.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

MIN_ANSWER_CHARS = 50
# Cevap soruyu büyük oranda tekrar ediyorsa ve kendi içeriği yoksa fail.
OVERLAP_REJECT_RATIO = 0.9


@dataclass
class QualityResult:
    """Tek bir kartın kalite kontrolü sonucu."""

    passed: bool
    reason: str
    score: float


def _extract_qa(card: dict) -> tuple[str, str]:
    """Karttan (soru, cevap) metnini çıkar.

    Esnek: doğrudan 'question'/'answer', ya da messages dizisinden
    user/assistant rollerinden okur.
    """
    question = str(card.get("question") or card.get("instruction") or "").strip()
    answer = str(card.get("answer") or card.get("output") or "").strip()

    if (not question or not answer) and isinstance(card.get("messages"), list):
        for msg in card["messages"]:
            role = msg.get("role")
            content = str(msg.get("content") or "").strip()
            if role == "user" and not question:
                question = content
            elif role == "assistant" and not answer:
                answer = content
    return question, answer


def _content_hash(question: str, answer: str) -> str:
    """Soru+cevap için kararlı SHA256 hash üret."""
    payload = f"{question.strip().lower()}\n{answer.strip().lower()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _token_overlap_ratio(question: str, answer: str) -> float:
    """Cevabın kelimelerinin ne kadarının soruda da geçtiğini ölç (0-1)."""
    q_tokens = set(question.lower().split())
    a_tokens = answer.lower().split()
    if not a_tokens:
        return 1.0
    shared = sum(1 for tok in a_tokens if tok in q_tokens)
    return shared / len(a_tokens)


def check_quality(card: dict) -> QualityResult:
    """Tek bir kartı kalite açısından değerlendir.

    Kontroller:
      - Çok kısa cevap (< 50 karakter) → fail
      - Cevap soruyu tekrar ediyor, kendi içeriği yok → fail
    Duplicate kontrolü `QualityFilter` toplu işleminde yapılır.
    """
    question, answer = _extract_qa(card)

    if len(answer) < MIN_ANSWER_CHARS:
        return QualityResult(
            passed=False,
            reason=f"cevap çok kısa ({len(answer)} < {MIN_ANSWER_CHARS} karakter)",
            score=0.0,
        )

    overlap = _token_overlap_ratio(question, answer)
    if overlap >= OVERLAP_REJECT_RATIO:
        return QualityResult(
            passed=False,
            reason=f"cevap soruyu tekrar ediyor (örtüşme {overlap:.0%})",
            score=round(1.0 - overlap, 3),
        )

    return QualityResult(passed=True, reason="ok", score=round(1.0 - overlap, 3))


@dataclass
class QualityFilter:
    """Toplu kalite filtresi — görülen hash'leri takip ederek duplicate eler."""

    seen_hashes: set[str] = field(default_factory=set)

    def filter_batch(self, cards: list[dict]) -> tuple[list[dict], list[dict]]:
        """Kart listesini (passed, rejected) olarak ikiye ayır.

        Duplicate (daha önce görülen içerik hash'i) reddedilir ve karta
        '_quality_reason' notu eklenir.
        """
        passed: list[dict] = []
        rejected: list[dict] = []

        for card in cards:
            question, answer = _extract_qa(card)
            result = check_quality(card)

            if not result.passed:
                rejected.append({**card, "_quality_reason": result.reason})
                continue

            content_hash = _content_hash(question, answer)
            if content_hash in self.seen_hashes:
                rejected.append({**card, "_quality_reason": "duplicate içerik"})
                continue

            self.seen_hashes.add(content_hash)
            passed.append(card)

        return passed, rejected
