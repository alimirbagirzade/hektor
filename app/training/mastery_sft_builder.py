"""mastery_sft_builder.py — Mastery sınavından SFT eğitim örnekleri üretir.

Yalnız citation_score >= eşik ve passed=True olan cevapları alır;
bunları {instruction, input, output} formatına dönüştürür.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.memory.mastery_store import MasteryStore
from app.memory.sqlite_store import SqliteStore

_DEFAULT_MIN_SCORE = 75.0
_DEFAULT_CITATION_THRESHOLD = 0.5
_OUTPUT_DIR = Path("data/training")


@dataclass
class SFTExample:
    source: str  # "mastery:<paper_id>"
    instruction: str
    input: str  # soru metni
    output: str  # cevap metni
    quality_score: float  # citation_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "quality_score": round(self.quality_score, 3),
        }


class MasterySFTBuilder:
    """Mastery test sonuçlarından SFT eğitim örnekleri toplayan sınıf."""

    def __init__(
        self,
        sqlite_store: SqliteStore | None = None,
        mastery_store: MasteryStore | None = None,
    ) -> None:
        self._store = sqlite_store or SqliteStore()
        self._ms = mastery_store or MasteryStore()

    def collect(
        self,
        min_mastery_score: float = _DEFAULT_MIN_SCORE,
        citation_threshold: float = _DEFAULT_CITATION_THRESHOLD,
    ) -> list[SFTExample]:
        """Uygun mastery cevaplarını SFTExample listesine dönüştür."""
        examples: list[SFTExample] = []
        papers = self._store.list_papers()

        for paper in papers:
            score = self._ms.get_latest_score(paper.paper_id)
            if score is None or score["total_score"] < min_mastery_score:
                continue

            test_id = score["test_id"]
            q_map = {q["question_id"]: q for q in self._ms.list_questions(test_id)}
            answers = self._ms.list_answers(test_id)

            seen: set[str] = set()
            for ans in answers:
                if not ans["passed"]:
                    continue
                if ans["citation_score"] < citation_threshold:
                    continue
                q = q_map.get(ans["question_id"])
                if q is None:
                    continue
                dedup_key = f"{ans['question_id']}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                examples.append(
                    SFTExample(
                        source=f"mastery:{paper.paper_id}",
                        instruction=_instruction_for_type(q["question_type"]),
                        input=q["question_text"],
                        output=ans["answer_text"],
                        quality_score=ans["citation_score"],
                    )
                )

        return examples

    def build_jsonl(
        self,
        output_path: Path | None = None,
        min_mastery_score: float = _DEFAULT_MIN_SCORE,
        citation_threshold: float = _DEFAULT_CITATION_THRESHOLD,
    ) -> tuple[Path, int]:
        """JSONL dosyasını yaz; (path, satır_sayısı) döndür."""
        examples = self.collect(min_mastery_score, citation_threshold)
        out = output_path or (_OUTPUT_DIR / "mastery_sft.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in examples))
        return out, len(examples)


def _instruction_for_type(q_type: str) -> str:
    mapping = {
        "structural": "Bu akademik çalışma hakkındaki soruyu kaynaklara dayanarak yanıtla.",
        "trading_hypothesis": "Bu trading hipotezini akademik literatüre dayanarak değerlendir.",
        "abstention": "Bu soruyu yalnızca bilgi tabanındaki makalelere dayanarak yanıtla.",
        "card_field": "Bu akademik makale içeriğine dair soruyu yanıtla.",
    }
    return mapping.get(q_type, "Bu soruyu akademik kaynaklara dayanarak yanıtla.")
