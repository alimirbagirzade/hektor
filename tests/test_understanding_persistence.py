"""Anlama merdiveni KALICILIK testleri — snapshot DB + JSON rapor + tam merdiven (L5).

Çevrimdışı çalışır (LLM kapatılır); L5 deterministik olduğundan LLM olmadan bile notlanır.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.memory.sqlite_store import SqliteStore
from app.verification.exams import understanding_record as rec_mod
from app.verification.exams.l3_application import ExamResult
from app.verification.exams.understanding_record import (
    load_understanding_history,
    record_understanding,
)
from app.verification.exams.understanding_score import aggregate, score_full_ladder


@dataclass
class _Ans:
    """RagAnswerLike stub (test enjeksiyonu)."""

    question_type: str = "main_claim"
    requires_abstention: bool = False
    answer_text: str = "geçerli bir cevap"
    citation_score: float = 1.0
    grounding_score: float = 1.0
    abstention_correct: bool = False
    hallucination_detected: bool = False


def _store(tmp_path: Path) -> SqliteStore:
    return SqliteStore(db_path=tmp_path / "t.db")


def _score():
    # Bu dosya KALICILIK (snapshot save/load) testi — skor değeri keyfi test verisi,
    # eşik semantiği DEĞİL (o test_understanding_score'da). min_graded=1 ile 2-notlu
    # fixture 'scored' kalır (varsayılan eşik=3 küçük-n şişme koruması ayrı test edilir).
    return aggregate(
        [
            ExamResult("L5", "x", True, "passed", 0),
            ExamResult("L3", "y", False, "failed", 0),
        ],
        min_graded=1,
    )


def test_snapshot_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.latest_understanding_snapshot() is None
    sid = store.save_understanding_snapshot(_score(), seed=7, context={"source": "test"})
    assert sid.startswith("und_")
    rows = store.list_understanding_snapshots()
    assert len(rows) == 1
    row = rows[0]
    assert row["snapshot_id"] == sid
    assert row["seed"] == 7
    assert row["graded"] == 2
    assert row["passed"] == 1
    assert abs(row["pass_rate"] - 0.5) < 1e-9
    assert row["status"] == "scored"
    assert row["context"] == {"source": "test"}
    assert "L5" in row["by_level"]
    assert store.latest_understanding_snapshot()["snapshot_id"] == sid


def test_record_writes_db_and_json(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    monkeypatch.setattr(rec_mod, "understanding_report_dir", lambda: tmp_path)
    info = record_understanding(_score(), seed=1, store=store, context={"source": "test"})
    assert info["snapshot_id"].startswith("und_")
    assert info["report_path"] is not None
    assert Path(info["report_path"]).is_file()
    assert info["graded"] == 2
    assert len(load_understanding_history(store=store)) == 1


def test_record_no_report(tmp_path: Path) -> None:
    store = _store(tmp_path)
    info = record_understanding(_score(), store=store, write_report=False)
    assert info["report_path"] is None
    assert len(store.list_understanding_snapshots()) == 1


def test_full_ladder_includes_l5_offline(monkeypatch) -> None:
    # LLM kapalı → L3/L4 'skipped'; L5 deterministik → çevrimdışı bile notlanır.
    from app.brain.local_llm import LocalLLM

    monkeypatch.setattr(LocalLLM, "available", lambda self: False)
    score = score_full_ladder(seed=0, use_sessions_l5=False)  # sabit örnek L5 (deterministik)
    assert "L5" in score.by_level
    assert score.total >= 1
    l5 = score.by_level["L5"]
    counted = (
        l5.get("passed", 0) + l5.get("failed", 0) + l5.get("skipped", 0) + l5.get("no_data", 0)
    )
    assert counted == 1  # tek örnek kompozisyon, tam olarak bir kez sayılır


def test_full_ladder_with_injected_rag(monkeypatch) -> None:
    # rag_answers enjekte → Taban/L1/L2 merdivene katılır (L2 KALICI kayda girebilir).
    from app.brain.local_llm import LocalLLM

    monkeypatch.setattr(LocalLLM, "available", lambda self: False)
    score = score_full_ladder(
        seed=0, rag_answers=[_Ans(), _Ans(hallucination_detected=True)], use_sessions_l5=False
    )
    assert {"L1", "L2", "L5"}.issubset(score.by_level)
    # halüsinasyonlu cevap L2'yi düşürür
    assert score.by_level["L2"]["failed"] >= 1


# --------------------------------------------------------------------- yeni: L5 sinyal kalitesi
class _FakeStore:
    """l5_results_from_sessions / score_full_ladder için minimal store (gerçek DB'ye gitmez)."""

    def __init__(self, sessions: list[dict]) -> None:
        self._s = sessions

    def list_research_sessions(self, limit: int = 50) -> list[dict]:
        return self._s[:limit]

    def list_papers(self) -> list:
        return []

    def has_knowledge_card(self, pid: str) -> bool:
        return False


def test_l5_untestable_backtest_is_skipped() -> None:
    # math+novelty geçti, backtest yalnız "çok az işlem" yüzünden düştü → TEST EDİLEMEDİ → skipped.
    from app.verification.exams.l5_composition import CompositionResult, GateResult
    from app.verification.exams.understanding_score import composition_to_result

    comp = CompositionResult(
        name="x",
        candidate=False,
        verdict="rejected",
        gates=[
            GateResult("math", True),
            GateResult("novelty", True),
            GateResult("backtest", False, ["verdict=fail", "Çok az işlem (1 < 30)"]),
        ],
    )
    assert composition_to_result(comp).status == "skipped"


def test_l5_substantive_fail_is_failed() -> None:
    # math kapısı düştü → gerçek üretim başarısızlığı → failed (skipped DEĞİL).
    from app.verification.exams.l5_composition import CompositionResult, GateResult
    from app.verification.exams.understanding_score import composition_to_result

    comp = CompositionResult(
        name="x",
        candidate=False,
        verdict="rejected",
        gates=[
            GateResult("math", False, ["RSI kuralı aralık dışı"]),
            GateResult("novelty", True),
            GateResult("backtest", False, ["matematik kapısı geçilmedi"]),
        ],
    )
    assert composition_to_result(comp).status == "failed"


def test_l5_results_from_sessions_maps_verdicts() -> None:
    from app.verification.exams.understanding_score import l5_results_from_sessions

    store = _FakeStore(
        [
            {"session_id": "s1", "question": "q1", "strategy_ir": {"a": 1}, "verdict": "pass"},
            {
                "session_id": "s2",
                "question": "q2",
                "proposed_indicator": {"b": 1},
                "verdict": "fail",
            },
            {
                "session_id": "s3",
                "question": "q3",
                "strategy_ir": {"c": 1},
                "verdict": "inconclusive",
            },
            {"session_id": "s4", "question": "q4", "verdict": "pass"},  # IR yok → atlanır
        ]
    )
    res = l5_results_from_sessions(store)
    assert len(res) == 3
    by = {r.detail["session_id"]: r.status for r in res}
    assert by == {"s1": "passed", "s2": "failed", "s3": "skipped"}
    assert all(r.level == "L5" for r in res)


def test_score_full_ladder_prefers_sessions(monkeypatch) -> None:
    from app.brain.local_llm import LocalLLM
    from app.verification.exams.understanding_score import score_full_ladder

    monkeypatch.setattr(LocalLLM, "available", lambda self: False)
    store = _FakeStore(
        [
            {"session_id": "s1", "question": "q1", "strategy_ir": {"a": 1}, "verdict": "pass"},
            {"session_id": "s2", "question": "q2", "strategy_ir": {"b": 1}, "verdict": "fail"},
        ]
    )
    score = score_full_ladder(seed=0, store=store)  # use_sessions_l5 varsayılan True
    assert score.by_level["L5"]["passed"] == 1
    assert score.by_level["L5"]["failed"] == 1


# --------------------------------------------------------------------- yeni: bağlam + regresyon
def test_context_auto_capture(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record_understanding(_score(), store=store, write_report=False)
    ctx = store.list_understanding_snapshots()[0]["context"]
    assert ctx.get("model_kind") == "base"
    assert "llm_model" in ctx
    assert ctx.get("n_papers") == 0  # boş tmp DB


def test_compare_understanding_regression() -> None:
    from app.verification.exams.understanding_record import compare_understanding

    prev = {
        "pass_rate": 0.8,
        "by_level": {"L3": {"passed": 4, "failed": 1}},
        "context": {"llm_model": "qwen3:4b"},
    }
    curr = {
        "pass_rate": 0.5,
        "by_level": {"L3": {"passed": 2, "failed": 2}},
        "context": {"llm_model": "qwen3:4b"},
    }
    c = compare_understanding(prev, curr)
    assert c["comparable"] is True
    assert c["regressed"] is True
    assert abs(c["delta"] + 0.3) < 1e-9

    # farklı model → kıyas güvenilmez, regresyon hesaplanmaz
    curr2 = {**curr, "context": {"llm_model": "achilles-v6"}}
    c2 = compare_understanding(prev, curr2)
    assert c2["comparable"] is False
    assert c2["regressed"] is False


def test_ladder_ordering_taban_first() -> None:
    from app.main import _by_level_summary

    by = {
        "L5": {"passed": 1, "failed": 0},
        "Taban": {"passed": 1, "failed": 0},
        "L2": {"passed": 1, "failed": 0},
    }
    s = _by_level_summary(by)
    assert s.index("Taban") < s.index("L2") < s.index("L5")
