"""report_generator.py — Mastery testi sonuçlarını JSON + Markdown rapor olarak kaydeder."""

from __future__ import annotations

import json
from pathlib import Path

from app.learning.mastery_scorer import MasteryScore
from app.memory.mastery_store import MasteryStore

_REPORT_DIR = Path("reports/papers/mastery")


class ReportGenerator:
    """Mastery testi için JSON ve Markdown rapor üreten sınıf."""

    def __init__(self, store: MasteryStore | None = None, report_dir: Path | None = None) -> None:
        self._store = store or MasteryStore()
        self._dir = report_dir or _REPORT_DIR

    def generate(self, paper_id: str, test_id: str, score: MasteryScore) -> tuple[Path, Path]:
        """JSON ve Markdown raporlarını yaz, yollarını döndür."""
        self._dir.mkdir(parents=True, exist_ok=True)
        json_path = self._dir / f"{paper_id}_mastery_report.json"
        md_path = self._dir / f"{paper_id}_mastery_report.md"

        answers = self._store.list_answers(test_id)
        questions = self._store.list_questions(test_id)
        q_map = {q["question_id"]: q for q in questions}

        report = {
            "paper_id": paper_id,
            "test_id": test_id,
            "score": score.to_dict(),
            "questions": len(questions),
            "passed": sum(1 for a in answers if a["passed"]),
            "failed": sum(1 for a in answers if not a["passed"]),
            "answers": answers,
        }
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

        md_lines = [
            f"# Paper Mastery Raporu — `{paper_id}`",
            "",
            f"**Test ID:** `{test_id}`",
            f"**Toplam Skor:** {score.total_score:.1f} / 100",
            f"**Durum:** `{score.final_status}`",
            "",
            "## Bileşen Skorları",
            "",
            "| Bileşen | Skor | Maks |",
            "|---------|------|------|",
            f"| Parse | {score.parse_score:.1f} | 10 |",
            f"| Metadata | {score.metadata_score:.1f} | 5 |",
            f"| Chunk Kalitesi | {score.chunk_quality_score:.1f} | 15 |",
            f"| Index | {score.index_score:.1f} | 10 |",
            f"| Retrieval | {score.retrieval_score:.1f} | 15 |",
            f"| Citation | {score.citation_score:.1f} | 15 |",
            f"| Grounding | {score.grounding_score:.1f} | 15 |",
            f"| Abstention | {score.abstention_score:.1f} | 10 |",
            f"| Formül/Argüman | {score.formula_argument_score:.1f} | 5 |",
            "",
            "## Soru Sonuçları",
            "",
        ]

        passed_n = 0
        failed_n = 0
        for ans in answers:
            q = q_map.get(ans["question_id"], {})
            status = "✅" if ans["passed"] else "❌"
            if ans["passed"]:
                passed_n += 1
            else:
                failed_n += 1
            md_lines.append(
                f"- {status} `{q.get('question_type', '?')}` — {q.get('question_text', '?')[:80]}"
            )

        md_lines += [
            "",
            f"**Sonuç:** {passed_n} geçti / {failed_n} başarısız",
        ]

        md_path.write_text("\n".join(md_lines))

        self._store.set_test_report(test_id, str(json_path))
        return json_path, md_path
