"""`mastery-report --rebuild`: eksik/bozuk mastery raporlarını DB'den yeniden üretme."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.learning.mastery_scorer import MasteryScore
from app.learning.report_generator import ReportGenerator, has_valid_report, rebuild_reports
from app.memory.mastery_store import MasteryStore

COMPONENTS = {
    "parse_score": 8.0,
    "metadata_score": 4.0,
    "chunk_quality_score": 12.0,
    "index_score": 8.0,
    "retrieval_score": 10.0,
    "citation_score": 10.0,
    "grounding_score": 10.0,
    "abstention_score": 7.0,
    "formula_argument_score": 4.0,
}


def _seed(store: MasteryStore, paper_id: str, *, total: float | None = None) -> str:
    """Biten bir test + soru/cevap (cp1254 dışı karakterli) + skor kaydı oluştur."""
    test_id = store.create_test(paper_id)
    q_id = f"q_{test_id}"
    store.save_questions(
        [
            {
                "question_id": q_id,
                "test_id": test_id,
                "paper_id": paper_id,
                "question_text": "σ ve → içeren soru ≤ 5?",
                "question_type": "formula",
            }
        ]
    )
    store.save_answer(
        {
            "answer_id": f"a_{test_id}",
            "question_id": q_id,
            "test_id": test_id,
            "paper_id": paper_id,
            "answer_text": "Cevap: σ ≤ 5 ✅",
            "passed": True,
        }
    )
    score = MasteryScore(paper_id, test_id, **COMPONENTS)
    d = score.to_dict()
    if total is not None:
        d["total_score"] = total
    store.save_score(d)
    store.finish_test(test_id, 1, 0)
    return test_id


@pytest.fixture
def store(tmp_path: Path) -> MasteryStore:
    return MasteryStore(db_path=tmp_path / "mastery.db")


def test_missing_and_zero_byte_reports_are_rebuilt(store: MasteryStore, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    out.mkdir()
    _seed(store, "p_missing")
    t_zero = _seed(store, "p_zero")
    (out / "p_zero_mastery_report.json").write_bytes(b"")  # eski hatanın izi

    res = rebuild_reports(store, out)
    assert sorted(res.rebuilt) == ["p_missing", "p_zero"]
    assert res.failed == [] and res.skipped == []
    assert has_valid_report("p_missing", out) and has_valid_report("p_zero", out)
    data = json.loads((out / "p_zero_mastery_report.json").read_text(encoding="utf-8"))
    assert data["test_id"] == t_zero and data["score"]["total_score"] == 73.0
    assert "✅" in (out / "p_zero_mastery_report.md").read_text(encoding="utf-8")
    # report_path DB'ye işlendi
    assert store.list_tests("p_zero")[0]["report_path"].endswith("p_zero_mastery_report.json")


def test_rebuilt_report_is_identical_to_original_generation(
    store: MasteryStore, tmp_path: Path
) -> None:
    test_id = _seed(store, "p1")
    ref_dir, out = tmp_path / "ref", tmp_path / "out"
    ReportGenerator(store=store, report_dir=ref_dir).generate(
        "p1", test_id, MasteryScore("p1", test_id, **COMPONENTS)
    )
    rebuild_reports(store, out)
    for ext in ("json", "md"):
        name = f"p1_mastery_report.{ext}"
        assert (out / name).read_bytes() == (ref_dir / name).read_bytes()


def test_valid_reports_untouched_unless_forced(store: MasteryStore, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    _seed(store, "p1")
    rebuild_reports(store, out)
    json_path = out / "p1_mastery_report.json"
    json_path.write_text('{"elle": "düzenlendi"}', encoding="utf-8")  # geçerli JSON
    assert rebuild_reports(store, out).rebuilt == []
    assert "elle" in json_path.read_text(encoding="utf-8")
    assert rebuild_reports(store, out, force=True).rebuilt == ["p1"]
    assert "elle" not in json_path.read_text(encoding="utf-8")


def test_uses_latest_finished_test_per_paper(store: MasteryStore, tmp_path: Path) -> None:
    _seed(store, "p1")
    newer = _seed(store, "p1")
    res = rebuild_reports(store, tmp_path)
    assert res.finished_papers == 1 and res.rebuilt == ["p1"]
    data = json.loads((tmp_path / "p1_mastery_report.json").read_text(encoding="utf-8"))
    assert data["test_id"] == newer


def test_dry_run_writes_nothing(store: MasteryStore, tmp_path: Path) -> None:
    out = tmp_path / "reports"
    _seed(store, "p1")
    res = rebuild_reports(store, out, dry_run=True)
    assert res.rebuilt == ["p1"] and res.dry_run
    assert not out.exists()


def test_inconsistent_total_is_skipped(store: MasteryStore, tmp_path: Path) -> None:
    _seed(store, "p_bad", total=99.0)  # bileşen toplamı 73 ≠ kayıtlı 99
    res = rebuild_reports(store, tmp_path)
    assert res.rebuilt == [] and "toplam uyuşmuyor" in res.skipped[0]
    assert not (tmp_path / "p_bad_mastery_report.json").exists()


def test_missing_score_is_skipped(store: MasteryStore, tmp_path: Path) -> None:
    test_id = store.create_test("p_noscore")
    store.finish_test(test_id, 0, 0)
    res = rebuild_reports(store, tmp_path)
    assert res.skipped == ["p_noscore: skor kaydı yok"]


def test_paper_filter_and_unfinished_ignored(store: MasteryStore, tmp_path: Path) -> None:
    _seed(store, "p1")
    _seed(store, "p2")
    store.create_test("p_running")  # bitmemiş test → rapor üretilmez
    res = rebuild_reports(store, tmp_path, paper_id="p2")
    assert res.finished_papers == 1 and res.rebuilt == ["p2"]
    assert not (tmp_path / "p1_mastery_report.json").exists()


def test_cli_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings
    from app.main import app

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)  # rapor dizini göreli: reports/papers/mastery
    get_settings.cache_clear()
    _seed(MasteryStore(), "p_cli")
    runner = CliRunner()

    r = runner.invoke(app, ["mastery-report", "--rebuild", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "yeniden üretilecek: 1" in r.output
    assert not (tmp_path / "reports/papers/mastery/p_cli_mastery_report.json").exists()

    r = runner.invoke(app, ["mastery-report", "--rebuild"])
    assert r.exit_code == 0, r.output
    assert "yeniden üretildi: 1" in r.output

    r = runner.invoke(app, ["mastery-report", "p_cli"])  # mevcut gösterim yolu bozulmadı
    assert r.exit_code == 0 and "total=73.0" in r.output

    assert runner.invoke(app, ["mastery-report"]).exit_code == 2
