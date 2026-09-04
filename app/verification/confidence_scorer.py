"""Confidence scorer — aggregates all verification components.

Produces a 0–1 confidence score via weighted average;
decides "answer" / "warn" / "abstain".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.memory.contextual_chunker import ChunkQualityFlags
from app.verification.citation_verifier import CitationCheck
from app.verification.context_sufficiency import SufficiencyLevel, SufficiencyResult
from app.verification.contradiction_detector import Contradiction
from app.verification.grounding_verifier import GroundingLevel, GroundingResult

# Karar eşikleri
_ABSTAIN_THRESHOLD = 0.40
_WARN_THRESHOLD = 0.70

# Bileşen ağırlıkları
_WEIGHTS: dict[str, float] = {
    "context": 0.25,
    "citation": 0.30,
    "grounding": 0.30,
    "formula": 0.15,
}


@dataclass
class ConfidenceReport:
    """Güven skoru raporu."""

    score: float  # 0.0–1.0
    context_score: float
    citation_score: float
    grounding_score: float
    has_contradictions: bool
    formula_integrity: float
    decision: str  # "answer" | "abstain" | "warn"
    details: dict = field(default_factory=dict)


def _context_score(sufficiency: SufficiencyResult) -> float:
    level_scores = {
        SufficiencyLevel.SUFFICIENT: 1.0,
        SufficiencyLevel.PARTIALLY_SUFFICIENT: 0.6,
        SufficiencyLevel.MISSING_ARGUMENT_CONCLUSION: 0.4,
        SufficiencyLevel.MISSING_FORMULA_CONTINUATION: 0.3,
        SufficiencyLevel.CONTRADICTORY: 0.2,
        SufficiencyLevel.INSUFFICIENT: 0.0,
    }
    return level_scores.get(sufficiency.level, 0.5)


def _citation_score(citations: list[CitationCheck]) -> float:
    if not citations:
        return 1.0  # Atıf yoksa nötr (ceza verme)
    valid = sum(1 for c in citations if c.exists)
    return valid / len(citations)


def _grounding_score(groundings: list[GroundingResult]) -> float:
    if not groundings:
        return 0.5  # Cümle yoksa nötr
    level_weights = {
        GroundingLevel.SUPPORTED: 1.0,
        GroundingLevel.PARTIALLY_SUPPORTED: 0.5,
        GroundingLevel.SPECULATIVE: 0.3,
        GroundingLevel.UNSUPPORTED: 0.0,
    }
    total = sum(level_weights.get(g.level, 0.0) for g in groundings)
    return total / len(groundings)


def _formula_integrity_score(quality_flags: list[ChunkQualityFlags] | None) -> float:
    if not quality_flags:
        return 1.0  # Bayrak yoksa nötr
    formula_chunks = [f for f in quality_flags if f.has_formula]
    if not formula_chunks:
        return 1.0
    complete = sum(1 for f in formula_chunks if not f.has_incomplete_formula)
    return complete / len(formula_chunks)


class ConfidenceScorer:
    """Tüm doğrulama bileşenlerini birleştiren güven skoru hesaplayıcı."""

    def score(
        self,
        sufficiency: SufficiencyResult,
        citations: list[CitationCheck],
        groundings: list[GroundingResult],
        contradictions: list[Contradiction],
        quality_flags: list[ChunkQualityFlags] | None = None,
    ) -> ConfidenceReport:
        """Ağırlıklı güven skoru hesapla.

        Args:
            sufficiency: Bağlam yeterliliği sonucu.
            citations: Atıf doğrulama sonuçları.
            groundings: Dayanak doğrulama sonuçları.
            contradictions: Tespit edilen çelişkiler.
            quality_flags: Chunk kalite bayrakları (opsiyonel).

        Returns:
            ConfidenceReport nesnesi.
        """
        ctx = _context_score(sufficiency)
        cit = _citation_score(citations)
        gnd = _grounding_score(groundings)
        formula = _formula_integrity_score(quality_flags)

        raw_score = (
            _WEIGHTS["context"] * ctx
            + _WEIGHTS["citation"] * cit
            + _WEIGHTS["grounding"] * gnd
            + _WEIGHTS["formula"] * formula
        )

        # Çelişki cezası
        contradiction_penalty = min(0.20, len(contradictions) * 0.05)
        final_score = max(0.0, min(1.0, raw_score - contradiction_penalty))

        if final_score < _ABSTAIN_THRESHOLD:
            decision = "abstain"
        elif final_score < _WARN_THRESHOLD:
            decision = "warn"
        else:
            decision = "answer"

        return ConfidenceReport(
            score=round(final_score, 4),
            context_score=round(ctx, 4),
            citation_score=round(cit, 4),
            grounding_score=round(gnd, 4),
            has_contradictions=len(contradictions) > 0,
            formula_integrity=round(formula, 4),
            decision=decision,
            details={
                "contradiction_count": len(contradictions),
                "contradiction_penalty": round(contradiction_penalty, 4),
            },
        )
