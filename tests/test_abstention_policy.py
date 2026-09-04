"""Çekimser kalma politikası testleri."""

from __future__ import annotations

from app.verification.abstention_policy import AbstentionPolicy
from app.verification.confidence_scorer import ConfidenceReport
from app.verification.context_sufficiency import SufficiencyLevel, SufficiencyResult


def _make_confidence(score: float) -> ConfidenceReport:
    decision = "abstain" if score < 0.4 else "warn" if score < 0.7 else "answer"
    return ConfidenceReport(
        score=score,
        context_score=score,
        citation_score=score,
        grounding_score=score,
        has_contradictions=False,
        formula_integrity=1.0,
        decision=decision,
    )


def _make_sufficiency(
    level: SufficiencyLevel = SufficiencyLevel.SUFFICIENT,
    can_answer: bool = True,
) -> SufficiencyResult:
    return SufficiencyResult(level=level, missing_items=[], can_answer=can_answer)


def test_abstain_on_low_confidence() -> None:
    """Düşük güven skoru (0.3) → should_abstain=True olmalı."""
    policy = AbstentionPolicy()
    confidence = _make_confidence(0.3)
    sufficiency = _make_sufficiency(SufficiencyLevel.SUFFICIENT, can_answer=True)

    decision = policy.decide(confidence, sufficiency)
    assert decision.should_abstain is True
    assert len(decision.reason) > 0


def test_answer_on_high_confidence() -> None:
    """Yüksek güven skoru (0.9) → should_abstain=False olmalı."""
    policy = AbstentionPolicy()
    confidence = _make_confidence(0.9)
    sufficiency = _make_sufficiency(SufficiencyLevel.SUFFICIENT, can_answer=True)

    decision = policy.decide(confidence, sufficiency)
    assert decision.should_abstain is False


def test_abstain_on_insufficient_context() -> None:
    """Yetersiz bağlam → should_abstain=True olmalı (güven yüksek olsa bile)."""
    policy = AbstentionPolicy()
    confidence = _make_confidence(0.9)  # Yüksek güven
    sufficiency = _make_sufficiency(SufficiencyLevel.INSUFFICIENT, can_answer=False)

    decision = policy.decide(confidence, sufficiency)
    assert decision.should_abstain is True


def test_abstain_when_cannot_answer_non_insufficient() -> None:
    """can_answer=False (MISSING_FORMULA_CONTINUATION), INSUFFICIENT olmasa da çekimser."""
    policy = AbstentionPolicy()
    confidence = _make_confidence(0.9)  # yüksek güven olsa bile
    sufficiency = _make_sufficiency(SufficiencyLevel.MISSING_FORMULA_CONTINUATION, can_answer=False)
    decision = policy.decide(confidence, sufficiency)
    assert decision.should_abstain is True


def test_warn_threshold_still_answers() -> None:
    """Orta güven (0.6, 'warn') → should_abstain=False olmalı."""
    policy = AbstentionPolicy()
    confidence = _make_confidence(0.6)
    sufficiency = _make_sufficiency(SufficiencyLevel.PARTIALLY_SUFFICIENT, can_answer=True)

    decision = policy.decide(confidence, sufficiency)
    assert decision.should_abstain is False


def test_abstention_reason_turkish() -> None:
    """Çekimser kalındığında Türkçe mesaj döndürülmeli."""
    policy = AbstentionPolicy()
    confidence = _make_confidence(0.2)
    sufficiency = _make_sufficiency()

    decision = policy.decide(confidence, sufficiency)
    assert decision.should_abstain is True
    # Türkçe mesaj içermeli
    assert any(
        word in decision.reason.lower() for word in ["güvenilir", "bağlam", "yeterli", "kaynak"]
    )
