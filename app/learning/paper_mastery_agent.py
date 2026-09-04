"""paper_mastery_agent.py — Paper Mastery pipeline'ını orkestre eder.

Akış:
  1. PaperInspector → statik skor
  2. QuestionGenerator → sorular
  3. RagExamRunner → RAG cevaplar + doğrulama
  4. MasteryScorer → 100 puan
  5. StatusManager → status güncelle
  6. ReportGenerator → JSON + MD rapor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.learning.mastery_scorer import MasteryScore, MasteryScorer
from app.learning.paper_inspector import PaperInspector
from app.learning.question_generator import QuestionGenerator
from app.learning.rag_exam_runner import RagExamRunner
from app.learning.report_generator import ReportGenerator
from app.learning.status_manager import StatusManager
from app.memory.mastery_store import MasteryStore
from app.memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


@dataclass
class MasteryRunResult:
    paper_id: str
    test_id: str
    score: MasteryScore
    n_questions: int
    n_passed: int
    n_failed: int
    report_json: str
    report_md: str
    error: str | None = None

    def summary(self) -> str:
        return (
            f"{self.paper_id}: {self.score.total_score:.1f}/100 "
            f"({self.score.final_status}) | "
            f"{self.n_passed}/{self.n_questions} soru geçti"
        )


class PaperMasteryAgent:
    """Tek makale için tam mastery pipeline'ını çalıştırır."""

    def __init__(
        self,
        store: SqliteStore | None = None,
        mastery_store: MasteryStore | None = None,
    ) -> None:
        self._store = store or SqliteStore()
        self._ms = mastery_store or MasteryStore()
        self._inspector = PaperInspector(store=self._store)
        self._q_gen = QuestionGenerator(store=self._store)
        self._runner = RagExamRunner(store=self._store)
        self._scorer = MasteryScorer()
        self._status_mgr = StatusManager(store=self._ms)
        self._reporter = ReportGenerator(store=self._ms)

    def run(self, paper_id: str, question_count: int = 20) -> MasteryRunResult:
        """paper_id için tam mastery testini çalıştır."""
        test_id = self._ms.create_test(paper_id)
        logger.info("Mastery testi başladı: %s (test=%s)", paper_id, test_id)

        try:
            inspection = self._inspector.inspect(paper_id)
            if "paper_not_found" in inspection.missing_steps:
                self._ms.finish_test(test_id, 0, 0)
                self._status_mgr.update(paper_id, "failed", "Makale bulunamadı")
                return self._error_result(paper_id, test_id, "Makale bulunamadı")

            questions = self._q_gen.generate(paper_id, test_id, count=question_count)
            self._ms.save_questions([q.to_dict() for q in questions])
            logger.info("%d soru üretildi", len(questions))

            answers = self._runner.run(questions, paper_id)
            for a in answers:
                self._ms.save_answer(a.to_dict())

            score = self._scorer.compute(inspection, answers, test_id)
            self._ms.save_score({**score.to_dict(), "test_id": test_id})

            passed = sum(1 for a in answers if a.passed)
            failed = len(answers) - passed
            self._ms.finish_test(test_id, passed, failed)

            new_status = self._status_mgr.status_from_score(score.total_score)
            self._status_mgr.update(paper_id, new_status, f"Mastery skor: {score.total_score:.1f}")

            # Rapor üretimi best-effort: dosya hatası (Windows kilidi 'os error 32' vb.)
            # BAŞARILI test/skor/durumu BOZMASIN — aksi halde except bloğu finish_test'i
            # (0,0) + status'u 'failed' ile ezip skorla tutarsız kalıcı durum bırakıyordu.
            report_json = ""
            report_md = ""
            try:
                _jp, _mp = self._reporter.generate(paper_id, test_id, score)
                report_json, report_md = str(_jp), str(_mp)
                logger.info("Rapor: %s", report_json)
            except Exception as exc:
                logger.warning("Mastery raporu üretilemedi (test/skor korundu): %s", exc)

            return MasteryRunResult(
                paper_id=paper_id,
                test_id=test_id,
                score=score,
                n_questions=len(questions),
                n_passed=passed,
                n_failed=failed,
                report_json=report_json,
                report_md=report_md,
            )

        except Exception as exc:
            logger.exception("Mastery testi başarısız: %s", exc)
            self._ms.finish_test(test_id, 0, 0)
            self._status_mgr.update(paper_id, "failed", str(exc))
            return self._error_result(paper_id, test_id, str(exc))

    @staticmethod
    def _error_result(paper_id: str, test_id: str, error: str) -> MasteryRunResult:
        from app.learning.mastery_scorer import MasteryScore

        return MasteryRunResult(
            paper_id=paper_id,
            test_id=test_id,
            score=MasteryScore(paper_id=paper_id, test_id=test_id),
            n_questions=0,
            n_passed=0,
            n_failed=0,
            report_json="",
            report_md="",
            error=error,
        )


class LearningQueue:
    """Sıradaki makaleleri yöneten kuyruk yöneticisi."""

    def __init__(
        self,
        store: SqliteStore | None = None,
        mastery_store: MasteryStore | None = None,
    ) -> None:
        self._store = store or SqliteStore()
        self._ms = mastery_store or MasteryStore()
        self._agent = PaperMasteryAgent(store=self._store, mastery_store=self._ms)

    def list_all(self) -> list[dict]:
        return self._ms.list_queue()

    def enqueue_paper(self, paper_id: str, priority: int = 5) -> str:
        return self._ms.enqueue(paper_id, priority)

    def enqueue_all_papers(self) -> int:
        papers = self._store.list_papers()
        for p in papers:
            self._ms.enqueue(p.paper_id)
        return len(papers)

    def run_next(self, question_count: int = 20) -> MasteryRunResult | None:
        item = self._ms.get_next_queued()
        if not item:
            return None
        self._ms.update_queue_status(item["queue_id"], "running")
        result = self._agent.run(item["paper_id"], question_count=question_count)
        final = "done" if not result.error else "failed"
        self._ms.update_queue_status(item["queue_id"], final, result.error)
        return result

    def run_all(self, limit: int = 100, question_count: int = 20) -> list[MasteryRunResult]:
        results = []
        for _ in range(limit):
            r = self.run_next(question_count=question_count)
            if r is None:
                break
            results.append(r)
        return results
