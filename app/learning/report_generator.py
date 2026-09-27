"""report_generator.py — Mastery testi sonuçlarını JSON + Markdown rapor olarak kaydeder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.learning.mastery_scorer import MasteryScore
from app.memory.mastery_store import MasteryStore

_REPORT_DIR = Path("reports/papers/mastery")

_SCORE_COMPONENTS: tuple[str, ...] = (
    "parse_score",
    "metadata_score",
    "chunk_quality_score",
    "index_score",
    "retrieval_score",
    "citation_score",
    "grounding_score",
    "abstention_score",
    "formula_argument_score",
)


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
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

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

        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        self._store.set_test_report(test_id, str(json_path))
        return json_path, md_path


# ── Veritabanından yeniden üretim ────────────────────────────────────────────


def has_valid_report(paper_id: str, report_dir: Path | None = None) -> bool:
    """JSON + MD mevcut, boş değil ve JSON ayrıştırılabilir mi."""
    d = report_dir or _REPORT_DIR
    json_path = d / f"{paper_id}_mastery_report.json"
    md_path = d / f"{paper_id}_mastery_report.md"
    for p in (json_path, md_path):
        if not p.exists() or p.stat().st_size == 0:
            return False
    try:
        json.loads(json_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    return True


def score_from_record(record: dict[str, Any]) -> MasteryScore:
    """DB skor kaydından MasteryScore kur (toplam + durum bileşenlerden hesaplanır)."""
    return MasteryScore(
        paper_id=str(record["paper_id"]),
        test_id=str(record["test_id"]),
        **{c: float(record.get(c) or 0.0) for c in _SCORE_COMPONENTS},
    )


@dataclass
class RebuildResult:
    """`rebuild_reports` özeti."""

    finished_papers: int
    rebuilt: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # "paper_id: sebep"
    failed: list[str] = field(default_factory=list)  # "paper_id: hata"
    dry_run: bool = False


def rebuild_reports(
    store: MasteryStore | None = None,
    report_dir: Path | None = None,
    *,
    paper_id: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> RebuildResult:
    """Tamamlanmış testlerin eksik/bozuk raporlarını veritabanından yeniden üret.

    Rapor yazımı best-effort olduğundan (bkz. PaperMasteryAgent) test/skor DB'de sağlam
    kalıp rapor dosyası kaybolabilir — ör. Windows'ta UTF-8'siz yazım 0 baytlık dosya
    bırakıyordu. Makale başına EN SON biten test kullanılır. Geçerli raporu olan makaleye
    dokunulmaz (``force=True`` hariç). Bileşenlerden hesaplanan toplam, DB'deki kayıtlı
    toplamla uyuşmazsa rapor YAZILMAZ (tutarsız veriyle rapor üretme).
    """
    st = store or MasteryStore()
    gen = ReportGenerator(store=st, report_dir=report_dir)
    latest: dict[str, dict[str, Any]] = {}
    for t in st.list_finished_tests():  # eskiden yeniye → son kazanır
        if paper_id is None or t["paper_id"] == paper_id:
            latest[t["paper_id"]] = t

    result = RebuildResult(finished_papers=len(latest), dry_run=dry_run)
    for pid, test in latest.items():
        if not force and has_valid_report(pid, report_dir):
            continue
        record = st.get_score_for_test(test["test_id"])
        if record is None:
            result.skipped.append(f"{pid}: skor kaydı yok")
            continue
        score = score_from_record(record)
        stored_total = float(record.get("total_score") or 0.0)
        if abs(score.total_score - stored_total) > 0.011:
            result.skipped.append(f"{pid}: toplam uyuşmuyor ({score.total_score} ≠ {stored_total})")
            continue
        if dry_run:
            result.rebuilt.append(pid)
            continue
        try:
            gen.generate(pid, test["test_id"], score)
        except (OSError, ValueError) as exc:  # ValueError ⊃ UnicodeEncodeError
            result.failed.append(f"{pid}: {exc}")
            continue
        if has_valid_report(pid, report_dir):
            result.rebuilt.append(pid)
        else:
            result.failed.append(f"{pid}: yazıldı ama doğrulanamadı")
    return result
