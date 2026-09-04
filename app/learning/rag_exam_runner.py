"""rag_exam_runner.py — Soruları mevcut RagAnswerer'a sorar ve cevapları doğrular.

Mevcut verification altyapısını (citation_verifier, grounding_verifier, context_sufficiency)
yeniden kullanır. LLM olmadan çalışır; LLM varsa zenginleştirilmiş cevap üretir.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from app.brain.rag_answerer import RagAnswerer
from app.evals.rag_ragas_offline import context_precision
from app.learning.question_generator import MasteryQuestion
from app.memory.sqlite_store import SqliteStore
from app.verification.citation_verifier import CitationVerifier
from app.verification.context_sufficiency import ContextSufficiencyClassifier
from app.verification.grounding_verifier import GroundingLevel, GroundingVerifier


@dataclass
class ExamAnswer:
    answer_id: str
    question_id: str
    test_id: str
    paper_id: str
    question_text: str
    question_type: str
    requires_abstention: bool
    answer_text: str
    cited_paper_ids: list[str] = field(default_factory=list)
    citation_score: float = 0.0
    grounding_score: float = 0.0
    context_precision: float = 0.0  # RAGAS gözlemsel sinyal — passed'i ETKİLEMEZ
    context_sufficient: bool = False
    abstention_correct: bool = False
    hallucination_detected: bool = False
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "question_id": self.question_id,
            "test_id": self.test_id,
            "paper_id": self.paper_id,
            "answer_text": self.answer_text,
            "cited_paper_ids": self.cited_paper_ids,
            "citation_score": self.citation_score,
            "grounding_score": self.grounding_score,
            "context_precision": self.context_precision,
            "context_sufficient": self.context_sufficient,
            "abstention_correct": self.abstention_correct,
            "hallucination_detected": self.hallucination_detected,
            "passed": self.passed,
        }


class RagExamRunner:
    """Soruları RAG sistemine sorar ve cevapları deterministik olarak doğrular."""

    def __init__(self, store: SqliteStore | None = None) -> None:
        self._store = store or SqliteStore()
        self._answerer = RagAnswerer()
        self._citation_v = CitationVerifier()
        self._grounding_v = GroundingVerifier()
        self._sufficiency = ContextSufficiencyClassifier()

    def run(self, questions: list[MasteryQuestion], paper_id: str) -> list[ExamAnswer]:
        """Her soru için RAG cevabı al ve doğrula."""
        return [self._run_one(q, paper_id) for q in questions]

    def _run_one(self, q: MasteryQuestion, paper_id: str) -> ExamAnswer:
        rag = self._answerer.answer(q.question_text)
        answer_text = rag.answer
        chunks = rag.sources

        # RAGAS context_precision (gözlemsel): getirilen parçaların cevaba alaka oranı;
        # düşük → retrieval gürültülü. SADECE gözlem/log — passed kararına KATILMAZ. Korele
        # token-proxy'yi geçme-kapısına eklemek sınır cevapları haksız eler (Kural 2; v5-sınıfı).
        try:
            ctx_precision = round(context_precision(answer_text, [c.text for c in chunks]), 4)
        except Exception:
            ctx_precision = 0.0

        cited_ids = list({c.paper_id for c in chunks})
        # LLM çevrimdışıyken answerer placeholder döndürür (llm_used=False); bu GERÇEK
        # akıl yürütme değil → no_answer say ki regular soru sahte "geçti" almasın (Kural 2).
        no_answer = (
            not answer_text.strip()
            or "No sources found" in answer_text
            or not getattr(rag, "llm_used", True)
        )

        # Abstention soruları: cevap VERMEMEK doğru davranıştır
        if q.requires_abstention:
            abstention_correct = no_answer or paper_id not in cited_ids
            return ExamAnswer(
                answer_id="ans_" + uuid.uuid4().hex[:12],
                question_id=q.question_id,
                test_id=q.test_id,
                paper_id=paper_id,
                question_text=q.question_text,
                question_type=q.question_type,
                requires_abstention=True,
                answer_text=answer_text,
                cited_paper_ids=cited_ids,
                context_precision=ctx_precision,
                abstention_correct=abstention_correct,
                passed=abstention_correct,
            )

        # Bağlam yeterliliği
        sufficiency = self._sufficiency.classify(q.question_text, chunks)
        context_ok = sufficiency.can_answer

        # Citation skoru: modelin cevap metnindeki GERÇEK atıflarından ([paper:chunk])
        # hesaplanır — getirilen chunk örtüşmesinden DEĞİL. Aksi halde "atıf disiplini"
        # aslında retrieval-kapsamını ölçer (retrieval_score ile çift-sayım) ve model hiç
        # atıf yapmadan puan alır → eval/SFT dürüstlüğü ihlali (CLAUDE.md Kural 2).
        # CitationVerifier cevaptaki [paper:chunk] atıflarını parse edip getirilen
        # chunk'larda var mı diye doğrular; oran = doğrulanan / toplam atıf.
        citation_checks = self._citation_v.verify(answer_text, chunks)
        if citation_checks:
            cit_score = sum(1 for c in citation_checks if c.exists) / len(citation_checks)
        else:
            cit_score = 0.0  # cevapta hiç atıf yok → atıf disiplini puanı yok

        # Grounding skoru
        grounding_results = self._grounding_v.verify(answer_text, chunks)
        if grounding_results:
            supported = sum(
                1
                for g in grounding_results
                if g.level in (GroundingLevel.SUPPORTED, GroundingLevel.PARTIALLY_SUPPORTED)
            )
            gnd_score = supported / len(grounding_results)
            hallucination = any(g.level == GroundingLevel.UNSUPPORTED for g in grounding_results)
        else:
            # Boş grounding → kanıtsız geçme yasak; 0.0 ata (fake-pass kapısını kapat).
            gnd_score = 0.0
            hallucination = False

        # Geçti mi?
        passed = (
            not no_answer
            and context_ok
            and cit_score >= 0.3
            and gnd_score >= 0.4
            and not hallucination
        )

        return ExamAnswer(
            answer_id="ans_" + uuid.uuid4().hex[:12],
            question_id=q.question_id,
            test_id=q.test_id,
            paper_id=paper_id,
            question_text=q.question_text,
            question_type=q.question_type,
            requires_abstention=False,
            answer_text=answer_text,
            cited_paper_ids=cited_ids,
            citation_score=round(cit_score, 4),
            grounding_score=round(gnd_score, 4),
            context_precision=ctx_precision,
            context_sufficient=context_ok,
            hallucination_detected=hallucination,
            passed=passed,
        )
