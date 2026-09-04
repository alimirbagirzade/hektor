"""Golden dataset — fixed questions and expected answers for evaluation.

Can be loaded and saved in JSON format; includes hardcoded sample questions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldenQuestion:
    """Altın set sorusu — beklenen cevap ve kaynak bilgisiyle."""

    question_id: str
    question_text: str
    domain: str  # "trading" | "math" | "philosophy" | "general"
    expected_answer: str
    expected_source_ids: list[str]
    expected_chunk_ids: list[str]
    answer_type: str  # "factual" | "analytical" | "comparative"
    difficulty: str  # "easy" | "medium" | "hard"
    allow_abstention: bool = False


class GoldenDataset:
    """Altın veri seti yöneticisi.

    JSON dosyasından yükleme, kaydetme ve örnek sorular üretme işlemlerini sağlar.
    Örnek (instance) `questions` taşır → tüketiciler (retrieval_eval) enjekte edilen seti kullanır
    (eskiden örnek yok sayılıp her zaman statik `get_sample_questions()` koşulurdu).
    """

    def __init__(self, questions: list[GoldenQuestion] | None = None) -> None:
        # Enjekte edilen sorular yoksa hardcoded örnek sete düş (geriye uyumlu: GoldenDataset()).
        self.questions: list[GoldenQuestion] = (
            questions if questions is not None else self.get_sample_questions()
        )

    @staticmethod
    def load_from_json(path: Path) -> list[GoldenQuestion]:
        """JSON dosyasından altın sorular yükle.

        Args:
            path: JSON dosya yolu.

        Returns:
            GoldenQuestion listesi.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            GoldenQuestion(
                question_id=item["question_id"],
                question_text=item["question_text"],
                domain=item.get("domain", "general"),
                expected_answer=item.get("expected_answer", ""),
                expected_source_ids=item.get("expected_source_ids", []),
                expected_chunk_ids=item.get("expected_chunk_ids", []),
                answer_type=item.get("answer_type", "factual"),
                difficulty=item.get("difficulty", "medium"),
                allow_abstention=item.get("allow_abstention", False),
            )
            for item in data
        ]

    @staticmethod
    def save_to_json(questions: list[GoldenQuestion], path: Path) -> None:
        """Altın soruları JSON dosyasına kaydet.

        Args:
            questions: Kaydedilecek sorular.
            path: Hedef dosya yolu.
        """
        data = [
            {
                "question_id": q.question_id,
                "question_text": q.question_text,
                "domain": q.domain,
                "expected_answer": q.expected_answer,
                "expected_source_ids": q.expected_source_ids,
                "expected_chunk_ids": q.expected_chunk_ids,
                "answer_type": q.answer_type,
                "difficulty": q.difficulty,
                "allow_abstention": q.allow_abstention,
            }
            for q in questions
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_sample_questions() -> list[GoldenQuestion]:
        """Test için 5 adet hardcoded örnek soru döndür.

        Returns:
            GoldenQuestion listesi.
        """
        return [
            GoldenQuestion(
                question_id="gq_001",
                question_text="ATR göstergesi volatiliteyi nasıl ölçer?",
                domain="trading",
                expected_answer=(
                    "ATR (Average True Range), belirli bir dönemdeki fiyat "
                    "aralığının üssel hareketli ortalamasıdır."
                ),
                expected_source_ids=["paper_atr_001"],
                expected_chunk_ids=["paper_atr_001_c0000"],
                answer_type="factual",
                difficulty="easy",
            ),
            GoldenQuestion(
                question_id="gq_002",
                question_text="Momentum stratejilerinde look-ahead bias nasıl önlenir?",
                domain="trading",
                expected_answer=(
                    "Pozisyon sinyali bir sonraki barda işlem görür (shift(1) geciktirmesi)."
                ),
                expected_source_ids=["paper_momentum_001"],
                expected_chunk_ids=["paper_momentum_001_c0002"],
                answer_type="analytical",
                difficulty="medium",
            ),
            GoldenQuestion(
                question_id="gq_003",
                question_text="Sharpe oranı ile Sortino oranı arasındaki fark nedir?",
                domain="trading",
                expected_answer=(
                    "Sharpe toplam volatiliteyi kullanırken Sortino yalnızca "
                    "negatif getiri volatilitesini (downside risk) kullanır."
                ),
                expected_source_ids=["paper_risk_001"],
                expected_chunk_ids=["paper_risk_001_c0001"],
                answer_type="comparative",
                difficulty="medium",
            ),
            GoldenQuestion(
                question_id="gq_004",
                question_text="Volatilite kümelenmesi (clustering) nedir?",
                domain="trading",
                expected_answer=(
                    "Yüksek volatilitenin yüksek volatiliteyi izleme eğilimi; "
                    "GARCH modelleriyle modellenir."
                ),
                expected_source_ids=["paper_garch_001"],
                expected_chunk_ids=["paper_garch_001_c0003"],
                answer_type="factual",
                difficulty="medium",
            ),
            GoldenQuestion(
                question_id="gq_005",
                question_text="Bu veri setinde olmayan bir konu hakkında cevap ver.",
                domain="general",
                expected_answer="",
                expected_source_ids=[],
                expected_chunk_ids=[],
                answer_type="factual",
                difficulty="easy",
                allow_abstention=True,
            ),
        ]
