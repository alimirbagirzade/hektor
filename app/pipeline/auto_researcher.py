"""auto_researcher.py — Tam otomatik araştırma pipeline'ı.

Zincir:
  1. Onaylı bilgi kartlarından strateji hipotezlerini çıkar
  2. Her hipotezi araştırma sorusuna dönüştür
  3. ToolUseTrainer ile tool-use seansı çalıştır
  4. Reward sinyali hesapla ve kaydet
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.memory.sqlite_store import SqliteStore
from app.training.dpo_dataset_builder import score_and_save_sessions
from app.training.tool_use_trainer import ToolUseSession, ToolUseTrainer

logger = logging.getLogger(__name__)

_MAX_QUESTIONS = 10
_Q_TEMPLATE = "Bu hipotezi backtest ile test et: {hypothesis}"


@dataclass
class PipelineRun:
    n_cards_scanned: int = 0
    n_questions: int = 0
    n_sessions: int = 0
    n_scored: int = 0
    sessions: list[ToolUseSession] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        pass_n = sum(1 for s in self.sessions if s.final_verdict == "pass")
        return (
            f"Kart: {self.n_cards_scanned} | Soru: {self.n_questions} | "
            f"Seans: {self.n_sessions} (pass={pass_n}) | "
            f"Ödül: {self.n_scored} | Hata: {len(self.errors)}"
        )


def _extract_questions(
    store: SqliteStore,
    max_questions: int,
    only_approved: bool,
) -> list[str]:
    # only_approved=False → TÜM kartlar (onaysız dahil). Eski list_training_examples
    # card_json İÇERMEZ → hipotez çıkmıyordu (sessizce 0 soru); list_cards card_json döndürür.
    card_records = store.list_approved_cards() if only_approved else store.list_cards(limit=200)

    questions: list[str] = []
    seen: set[str] = set()

    for rec in card_records:
        card: dict[str, Any] = rec.get("card_json") or rec.get("card") or {}
        for hyp in card.get("possible_strategy_hypotheses", []):
            if hyp and hyp not in seen:
                seen.add(hyp)
                questions.append(_Q_TEMPLATE.format(hypothesis=hyp))
                if len(questions) >= max_questions:
                    return questions
    return questions


def run_pipeline(
    store: SqliteStore | None = None,
    max_questions: int = _MAX_QUESTIONS,
    max_iterations: int = 1,
    seed: int = 42,
    only_approved: bool = True,
    dry_run: bool = False,
) -> PipelineRun:
    """Kartlar → sorular → tool-use seansları → ödül skorlama."""
    s = store or SqliteStore()
    run = PipelineRun()

    cards = s.list_approved_cards() if only_approved else s.list_training_examples(limit=200)
    run.n_cards_scanned = len(cards)

    questions = _extract_questions(s, max_questions=max_questions, only_approved=only_approved)
    run.n_questions = len(questions)

    if not questions:
        logger.info("Pipeline: soru üretilemedi.")
        return run

    if dry_run:
        return run

    trainer = ToolUseTrainer(store=s, seed=seed)
    for q in questions:
        try:
            session = trainer.run_session(q, max_iterations=max_iterations)
            run.sessions.append(session)
            run.n_sessions += 1
        except Exception as exc:
            run.errors.append(f"{q[:50]}: {exc}")
            logger.warning("Seans hatası: %s", exc)

    run.n_scored = len(score_and_save_sessions(store=s))
    return run
