"""question_generator.py — Makaleden otomatik sınav soruları üretir.

LLM gerektirmez: paper metadata + knowledge card içeriğinden şablon sorular üretir.
Abstention soruları sabit bir havuzdan gelir (makale dışı sorular).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from app.memory.sqlite_store import SqliteStore

_ABSTENTION_POOL = [
    "Bu makalenin yazarları geçen yıl kaç trilyon dolar kâr etti?",
    "Bu makale hangi kripto borsasında işlem görüyor?",
    "Yazarların şirketinin CEO'su kimdir?",
]


@dataclass
class MasteryQuestion:
    question_id: str
    test_id: str
    paper_id: str
    question_text: str
    question_type: str
    requires_abstention: bool = False
    difficulty: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "test_id": self.test_id,
            "paper_id": self.paper_id,
            "question_text": self.question_text,
            "question_type": self.question_type,
            "requires_abstention": self.requires_abstention,
            "difficulty": self.difficulty,
        }


class QuestionGenerator:
    """Makale içeriğinden şablon tabanlı sorular üretir."""

    def __init__(self, store: SqliteStore | None = None) -> None:
        self._store = store or SqliteStore()

    def generate(self, paper_id: str, test_id: str, count: int = 20) -> list[MasteryQuestion]:
        """Makale için count adet soru üret."""
        papers = {p.paper_id: p for p in self._store.list_papers()}
        paper = papers.get(paper_id)
        if paper is None:
            return []

        title = paper.title or "bu makale"
        year = paper.year or "?"
        card = self._store.get_latest_knowledge_card(paper_id) or {}

        all_qs: list[MasteryQuestion] = []
        all_qs.extend(self._structural_questions(paper_id, test_id, title, year))
        all_qs.extend(self._card_questions(paper_id, test_id, card))
        all_qs.extend(self._abstention_questions(paper_id, test_id))

        unique: list[MasteryQuestion] = []
        seen: set[str] = set()
        for q in all_qs:
            if q.question_text not in seen:
                seen.add(q.question_text)
                unique.append(q)
        return unique[:count]

    def _qid(self) -> str:
        return "qst_" + uuid.uuid4().hex[:12]

    def _make(
        self,
        paper_id: str,
        test_id: str,
        text: str,
        qtype: str,
        requires_abstention: bool = False,
        difficulty: str = "medium",
    ) -> MasteryQuestion:
        return MasteryQuestion(
            question_id=self._qid(),
            test_id=test_id,
            paper_id=paper_id,
            question_text=text,
            question_type=qtype,
            requires_abstention=requires_abstention,
            difficulty=difficulty,
        )

    def _structural_questions(
        self, paper_id: str, test_id: str, title: str, year: str
    ) -> list[MasteryQuestion]:
        m = self._make
        return [
            m(paper_id, test_id, f"'{title}' makalesinin ana iddiası nedir?", "main_claim"),
            m(
                paper_id,
                test_id,
                f"'{title}' ({year}) makalesinde hangi yöntem önerilmektedir?",
                "method",
            ),
            m(
                paper_id,
                test_id,
                f"'{title}' makalesinin özetini çıkar.",
                "summary",
                difficulty="easy",
            ),
            m(
                paper_id,
                test_id,
                f"'{title}' makalesinde hangi veri seti kullanılmıştır?",
                "dataset",
            ),
            m(paper_id, test_id, f"'{title}' makalesinin temel bulgularını açıkla.", "result"),
            m(
                paper_id,
                test_id,
                f"'{title}' makalesindeki kısıtlamalar ve sınırlamalar nelerdir?",
                "limitation",
            ),
        ]

    def _card_questions(
        self, paper_id: str, test_id: str, card: dict[str, Any]
    ) -> list[MasteryQuestion]:
        qs: list[MasteryQuestion] = []
        for hyp in (card.get("possible_strategy_hypotheses") or [])[:3]:
            qs.append(
                self._make(
                    paper_id,
                    test_id,
                    f"Bu hipotezi makaledeki bulgularla destekle: '{hyp}'",
                    "trading_hypothesis",
                    difficulty="hard",
                )
            )
        formulas = card.get("formulas") or card.get("formula_components") or []
        for f in formulas[:2]:
            name = f if isinstance(f, str) else f.get("name", "formül")
            qs.append(
                self._make(
                    paper_id,
                    test_id,
                    f"'{name}' formülünü açıkla ve değişkenlerinin anlamlarını ver.",
                    "formula",
                    difficulty="hard",
                )
            )
        claim = card.get("main_claim")
        if claim:
            qs.append(
                self._make(
                    paper_id,
                    test_id,
                    f"Makaledeki şu iddiayı destekleyen kanıtları göster: '{claim[:120]}'",
                    "main_claim",
                )
            )
        return qs

    def _abstention_questions(self, paper_id: str, test_id: str) -> list[MasteryQuestion]:
        return [
            MasteryQuestion(
                question_id=self._qid(),
                test_id=test_id,
                paper_id=paper_id,
                question_text=q,
                question_type="abstention_test",
                requires_abstention=True,
                difficulty="easy",
            )
            for q in _ABSTENTION_POOL[:2]
        ]
