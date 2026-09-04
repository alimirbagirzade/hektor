"""Birleşik eval-runner testleri (dispatch + kapı + rapor, offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.evals.eval_runner import EvalGateError, EvalRunner
from app.evals.golden_dataset import GoldenQuestion

_GOOD_HYP = {
    "hypothesis_text": (
        "Volatilite yüksekse momentum zayıflayabilir; backtest ile test edilmeli, "
        "örneklem-dışı doğrulama ve komisyon+slippage maliyetleri dahil, risk stop-loss ile."
    )
}
_BAD_HYP = "Bu strateji %100 garanti kazandırır, hemen al."


@pytest.fixture
def runner(tmp_path: Path) -> EvalRunner:
    return EvalRunner(reports_dir=tmp_path / "evals")


# --- trading-hypothesis ----------------------------------------------------
def test_empty_hypotheses_not_vacuous_pass(runner: EvalRunner) -> None:
    """Boş hipotez seti + min_candidate_rate=0.0 → 'vacuous pass' (passed=True) OLMAMALI.
    Hiçbir şey değerlendirmeden 'geçti' demek Kural 2 ihlalidir."""
    res = runner.run("trading-hypothesis", hypotheses=[], min_candidate_rate=0.0)
    assert res.passed is False
    assert any("boş set" in f for f in res.failures)


def test_trading_hypothesis_pass_writes_report(runner: EvalRunner) -> None:
    res = runner.run("trading-hypothesis", hypotheses=[_GOOD_HYP])
    assert res.passed is True
    assert res.metrics["candidate_rate"] == 1.0
    assert res.report_path is not None
    assert Path(res.report_path).exists()


def test_trading_hypothesis_fail_on_advice(runner: EvalRunner) -> None:
    res = runner.run("trading-hypothesis", hypotheses=[_GOOD_HYP, _BAD_HYP])
    assert res.passed is False
    assert any("REDDEDİLDİ" in f for f in res.failures)


def test_strict_raises_on_gate_fail(runner: EvalRunner) -> None:
    with pytest.raises(EvalGateError):
        runner.run("trading-hypothesis", hypotheses=[_BAD_HYP], strict=True)


def test_deferred_type_raises_notimplemented(runner: EvalRunner) -> None:
    with pytest.raises(NotImplementedError):
        runner.run("rlm-reward")


def test_unknown_type_raises_value_error(runner: EvalRunner) -> None:
    with pytest.raises(ValueError):
        runner.run("totally-unknown")


# --- rag-retrieval (enjekte edilen sahte retriever) ------------------------
class _Chunk:
    def __init__(self, cid: str) -> None:
        self.chunk_id = cid


class _PerfectRetriever:
    """Beklenen chunk'ı dönen mükemmel retriever (recall=1)."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def retrieve(self, query: str, top_k: int = 10) -> list[_Chunk]:
        return [_Chunk(c) for c in self.mapping.get(query, [])]


class _EmptyRetriever:
    def retrieve(self, query: str, top_k: int = 10) -> list[_Chunk]:
        return []


def _golden(qid: str, text: str, chunk_ids: list[str]) -> GoldenQuestion:
    return GoldenQuestion(
        question_id=qid,
        question_text=text,
        domain="trading",
        expected_answer="",
        expected_source_ids=[],
        expected_chunk_ids=chunk_ids,
        answer_type="factual",
        difficulty="easy",
    )


def test_rag_retrieval_perfect_passes(runner: EvalRunner) -> None:
    q = _golden("q1", "ATR nedir?", ["c1"])
    retr = _PerfectRetriever({"ATR nedir?": ["c1"]})
    res = runner.run("rag-retrieval", questions=[q], retriever=retr)
    assert res.metrics["recall_at_10"] == 1.0
    assert res.passed is True


def test_rag_retrieval_empty_fails_gate(runner: EvalRunner) -> None:
    q = _golden("q1", "ATR nedir?", ["c1"])
    res = runner.run("rag-retrieval", questions=[q], retriever=_EmptyRetriever())
    assert res.metrics["recall_at_10"] == 0.0
    assert res.passed is False
    assert res.failures


# --- opt-in registry entegrasyonu (Modül 6→8) ------------------------------
def test_rag_retrieval_logs_registry_decision(runner: EvalRunner, tmp_path: Path) -> None:
    from app.memory.sqlite_store import SqliteStore
    from app.registry import RegistryStore

    store = SqliteStore(db_path=tmp_path / "evalreg.db")
    store.upsert_paper(paper_id="p1", file_hash="h1", source_path="x.pdf", title="T")
    store.add_chunks([{"chunk_id": "c1", "paper_id": "p1", "chunk_index": 0, "text": "abc"}])
    reg = RegistryStore(store)

    q = _golden("q1", "ATR nedir?", ["c1"])
    retr = _PerfectRetriever({"ATR nedir?": ["c1"]})
    res = runner.run("rag-retrieval", questions=[q], retriever=retr, registry=reg)

    assert res.registry_decision is not None
    assert res.registry_decision["decision"]["decision"] == "approved"
    # indeks sürümü + terfi kararı kalıcı yazıldı
    assert reg.list_rag_indices()
    decisions = reg.list_decisions()
    assert decisions and decisions[0]["target_type"] == "rag_index"


def test_rag_retrieval_registry_blocked_on_fail(runner: EvalRunner, tmp_path: Path) -> None:
    from app.memory.sqlite_store import SqliteStore
    from app.registry import RegistryStore

    reg = RegistryStore(SqliteStore(db_path=tmp_path / "evalreg2.db"))
    q = _golden("q1", "ATR nedir?", ["c1"])
    res = runner.run("rag-retrieval", questions=[q], retriever=_EmptyRetriever(), registry=reg)
    assert res.registry_decision["decision"]["decision"] == "blocked"  # type: ignore[index]


def test_no_registry_keeps_default(runner: EvalRunner) -> None:
    q = _golden("q1", "ATR nedir?", ["c1"])
    res = runner.run(
        "rag-retrieval", questions=[q], retriever=_PerfectRetriever({"ATR nedir?": ["c1"]})
    )
    assert res.registry_decision is None  # opt-in: verilmezse davranış değişmez
