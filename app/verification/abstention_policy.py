"""Abstention policy — refuse to answer on low confidence or insufficient context.

Protects the system from producing unreliable answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.verification.confidence_scorer import ConfidenceReport
from app.verification.context_sufficiency import SufficiencyLevel, SufficiencyResult

_INSUFFICIENT_MESSAGE = "Bu soruya güvenilir cevap vermek için getirilen bağlam yeterli değil."
_LOW_CONFIDENCE_MESSAGE = (
    "Mevcut kaynaklar bu soruyu yeterli güvenle yanıtlamak için yetersiz. "
    "Lütfen daha fazla makale ingest edin veya soruyu yeniden formüle edin."
)


@dataclass
class AbstentionDecision:
    """Çekimser kalma kararı."""

    should_abstain: bool
    reason: str
    missing_context: list[str] = field(default_factory=list)


class AbstentionPolicy:
    """Cevap verme / çekimser kalma politikası.

    Çekimser kal:
    - Güven skoru "abstain" kararı veriyorsa.
    - Bağlam yeterliliği INSUFFICIENT ise.
    """

    def decide(
        self,
        confidence: ConfidenceReport,
        sufficiency: SufficiencyResult,
    ) -> AbstentionDecision:
        """Çekimser kalınıp kalınmayacağına karar ver.

        Args:
            confidence: Güven skoru raporu.
            sufficiency: Bağlam yeterliliği sonucu.

        Returns:
            AbstentionDecision nesnesi.
        """
        # INSUFFICIENT veya can_answer=False (ör. MISSING_FORMULA_CONTINUATION) → çekimser kal.
        # Eskiden yalnız INSUFFICIENT yakalanıyordu; can_answer=False sızıp cevap üretiliyordu.
        if sufficiency.level == SufficiencyLevel.INSUFFICIENT or not sufficiency.can_answer:
            return AbstentionDecision(
                should_abstain=True,
                reason=_INSUFFICIENT_MESSAGE,
                missing_context=sufficiency.missing_items,
            )

        # Güven skoru çok düşükse
        if confidence.decision == "abstain":
            return AbstentionDecision(
                should_abstain=True,
                reason=_LOW_CONFIDENCE_MESSAGE,
                missing_context=sufficiency.missing_items,
            )

        # Cevap verilebilir
        return AbstentionDecision(
            should_abstain=False,
            reason="",
            missing_context=[],
        )
