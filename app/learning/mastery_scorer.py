"""mastery_scorer.py — Paper Mastery Score hesaplar (0–100).

Spec §7 ağırlıklandırmasını kullanır. LLM gerektirmez.
Statik kısım (parse/metadata/chunk/index) PaperInspector'dan gelir.
Dinamik kısım (retrieval/citation/grounding/abstention) ExamAnswer listesinden hesaplanır.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.learning.paper_inspector import InspectionResult
from app.learning.rag_exam_runner import ExamAnswer

# Spec §7 eşikleri
_THRESHOLD_LEARNED = 90.0
_THRESHOLD_USABLE = 75.0
_THRESHOLD_PARTIAL = 60.0
_THRESHOLD_RECHUNK = 40.0

# Spec §7 maksimum bileşen skorları
_MAX_RETRIEVAL = 15.0
_MAX_CITATION = 15.0
_MAX_GROUNDING = 15.0
_MAX_ABSTENTION = 10.0
_MAX_FORMULA = 5.0


@dataclass
class MasteryScore:
    """100 puanlık Paper Mastery Score."""

    paper_id: str
    test_id: str
    parse_score: float = 0.0  # 0–10
    metadata_score: float = 0.0  # 0–5
    chunk_quality_score: float = 0.0  # 0–15
    index_score: float = 0.0  # 0–10
    retrieval_score: float = 0.0  # 0–15
    citation_score: float = 0.0  # 0–15
    grounding_score: float = 0.0  # 0–15
    abstention_score: float = 0.0  # 0–10
    formula_argument_score: float = 0.0  # 0–5

    @property
    def total_score(self) -> float:
        return round(
            self.parse_score
            + self.metadata_score
            + self.chunk_quality_score
            + self.index_score
            + self.retrieval_score
            + self.citation_score
            + self.grounding_score
            + self.abstention_score
            + self.formula_argument_score,
            2,
        )

    @property
    def final_status(self) -> str:
        t = self.total_score
        if t >= _THRESHOLD_LEARNED:
            return "learned"
        if t >= _THRESHOLD_USABLE:
            return "usable_needs_review"
        if t >= _THRESHOLD_PARTIAL:
            return "partially_learned"
        if t >= _THRESHOLD_RECHUNK:
            return "needs_rechunking"
        return "failed"

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "test_id": self.test_id,
            "total_score": self.total_score,
            "parse_score": self.parse_score,
            "metadata_score": self.metadata_score,
            "chunk_quality_score": self.chunk_quality_score,
            "index_score": self.index_score,
            "retrieval_score": self.retrieval_score,
            "citation_score": self.citation_score,
            "grounding_score": self.grounding_score,
            "abstention_score": self.abstention_score,
            "formula_argument_score": self.formula_argument_score,
            "final_status": self.final_status,
        }


class MasteryScorer:
    """InspectionResult + ExamAnswer listesinden MasteryScore hesaplar."""

    def compute(
        self,
        inspection: InspectionResult,
        exam_answers: list[ExamAnswer],
        test_id: str,
    ) -> MasteryScore:
        score = MasteryScore(paper_id=inspection.paper_id, test_id=test_id)

        # Statik kısım (LLM'siz)
        score.parse_score = round(inspection.parse_score, 2)
        score.metadata_score = round(inspection.metadata_score, 2)
        score.chunk_quality_score = round(inspection.chunk_quality, 2)
        score.index_score = round(inspection.index_score, 2)

        if not exam_answers:
            return score

        regular = [a for a in exam_answers if not a.requires_abstention]
        abstention = [a for a in exam_answers if a.requires_abstention]

        # Retrieval: paper'ın chunk'larına erişilebildi mi?
        if regular:
            has_paper_ctx = sum(1 for a in regular if inspection.paper_id in a.cited_paper_ids)
            retrieval_ratio = has_paper_ctx / len(regular)
            score.retrieval_score = round(retrieval_ratio * _MAX_RETRIEVAL, 2)

        # Citation
        if regular:
            avg_cit = sum(a.citation_score for a in regular) / len(regular)
            score.citation_score = round(avg_cit * _MAX_CITATION, 2)

        # Grounding
        if regular:
            avg_gnd = sum(a.grounding_score for a in regular) / len(regular)
            hallucination_penalty = sum(1 for a in regular if a.hallucination_detected)
            penalty = hallucination_penalty / len(regular) * 5.0
            score.grounding_score = round(max(0.0, avg_gnd * _MAX_GROUNDING - penalty), 2)

        # Abstention
        if abstention:
            correct = sum(1 for a in abstention if a.abstention_correct)
            score.abstention_score = round((correct / len(abstention)) * _MAX_ABSTENTION, 2)
        else:
            score.abstention_score = _MAX_ABSTENTION * 0.5  # neutral if no abstention qs

        # Formula/argument (bonus: knowledge card var ve chunk'lar section'lı ise)
        if inspection.has_knowledge_card and inspection.chunk_count > 0:
            score.formula_argument_score = round(_MAX_FORMULA * 0.6, 2)
            if any(a.passed for a in regular if a.question_type in ("formula", "main_claim")):
                score.formula_argument_score = _MAX_FORMULA

        return score
