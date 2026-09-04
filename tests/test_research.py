"""Araştırma modülü birim testleri — çevrimdışı (LLM mock)."""

from __future__ import annotations

import json
import tempfile
import unittest.mock as mock

import pytest

from app.memory.sqlite_store import SqliteStore
from app.research.chain_data_builder import ChainDataBuilder
from app.research.concept_graph import ConceptGraph
from app.research.formula_extractor import FormulaExtractor
from app.research.orchestrator import ResearchOrchestrator
from app.research.reflection_agent import ReflectionAgent
from app.research.synthesis_engine import SynthesisEngine, SynthesisResult
from app.trading.market_data_loader import generate_synthetic_ohlcv
from app.trading.strategy_ir import example_ir


@pytest.fixture
def store() -> SqliteStore:
    tmp = tempfile.mkstemp(suffix=".db")[1]
    s = SqliteStore(db_path=tmp)
    s.upsert_paper(
        paper_id="p1",
        file_hash="h1",
        source_path="/tmp/p1.pdf",
        title="RSI momentum paper",
    )
    return s


# ---------- FormulaExtractor ----------
def test_formula_extractor_rule_based(store: SqliteStore) -> None:
    extractor = FormulaExtractor(store=store)
    with mock.patch.object(extractor.llm, "available", return_value=False):
        # chunk'a RSI içeren metin ekle
        store.add_chunks(
            [
                {
                    "chunk_id": "c1",
                    "paper_id": "p1",
                    "chunk_index": 0,
                    "text": "The RSI indicator measures momentum. ATR measures volatility.",
                    "char_count": 60,
                    "token_estimate": 15,
                    "embedded": 0,
                }
            ]
        )
        formulas = extractor.extract_from_paper("p1")
    names = {f["name"] for f in formulas}
    assert "RSI" in names
    assert "ATR" in names


def test_formula_extractor_skips_duplicates(store: SqliteStore) -> None:
    store.add_chunks(
        [
            {
                "chunk_id": "c2",
                "paper_id": "p1",
                "chunk_index": 1,
                "text": "RSI is used for momentum.",
                "char_count": 30,
                "token_estimate": 8,
                "embedded": 0,
            }
        ]
    )
    extractor = FormulaExtractor(store=store)
    with mock.patch.object(extractor.llm, "available", return_value=False):
        extractor.extract_from_paper("p1")
        formulas2 = extractor.extract_from_paper("p1")
    assert len(formulas2) == 0  # ikinci çalıştırmada yeni formül yok


def test_formula_saved_to_sqlite(store: SqliteStore) -> None:
    store.add_chunks(
        [
            {
                "chunk_id": "c3",
                "paper_id": "p1",
                "chunk_index": 2,
                "text": "MACD combines two EMA values.",
                "char_count": 30,
                "token_estimate": 8,
                "embedded": 0,
            }
        ]
    )
    extractor = FormulaExtractor(store=store)
    with mock.patch.object(extractor.llm, "available", return_value=False):
        extractor.extract_from_paper("p1")
    rows = store.list_formulas(paper_id="p1")
    assert len(rows) > 0


# ---------- ConceptGraph ----------
def test_concept_graph_add_link(store: SqliteStore) -> None:
    graph = ConceptGraph(store=store)
    graph.add_link("RSI", "measures", "momentum", source_paper_id="p1")
    links = store.list_concept_links("RSI")
    assert len(links) == 1
    assert links[0]["relation"] == "measures"


def test_concept_graph_invalid_relation(store: SqliteStore) -> None:
    graph = ConceptGraph(store=store)
    with pytest.raises(ValueError, match="Geçersiz ilişki"):
        graph.add_link("RSI", "invented_relation", "momentum")


def test_concept_graph_as_text_empty(store: SqliteStore) -> None:
    graph = ConceptGraph(store=store)
    text = graph.as_text()
    assert "boş" in text


# ---------- SynthesisEngine ----------
def test_synthesis_engine_no_formulas(store: SqliteStore) -> None:
    engine = SynthesisEngine(store=store)
    result = engine.synthesize("Test sorusu")
    assert result is None  # formül yok


def test_synthesis_engine_with_mocked_llm(store: SqliteStore) -> None:
    store.save_formula(
        formula_id="f1",
        paper_id="p1",
        name="RSI",
        category="momentum",
        description="momentum indicator",
        variables_json="{}",
        latex=None,
        plain=None,
    )
    fake_output = json.dumps(
        {
            "indicator_name": "TestIndicator",
            "description": "Test",
            "source_papers": ["p1"],
            "formula_components": [{"name": "RSI", "role": "signal"}],
            "combination_reasoning": "RSI + ATR",
            "expected_edge": "low volatility",
            "failure_conditions": ["trending market"],
            "strategy_ir": {
                "name": "test_v1",
                "market": "XAUUSD",
                "timeframe": "15m",
                "indicators": [{"name": "RSI", "period": 14}, {"name": "ATR", "period": 14}],
                "entry_rules": ["rsi_14 > 55"],
                "exit_rules": ["rsi_14 < 45"],
                "risk": {"stop_loss": "2 * ATR"},
                "costs": {"commission": 0.0005, "slippage": 0.0005},
            },
        }
    )
    engine = SynthesisEngine(store=store)
    with (
        mock.patch.object(engine.llm, "available", return_value=True),
        mock.patch.object(engine.llm, "generate", return_value=fake_output),
    ):
        result = engine.synthesize("Nasıl momentum filtrelenir?")
    assert result is not None
    assert result.indicator_name == "TestIndicator"
    assert result.strategy_ir["name"] == "test_v1"


# ---------- ReflectionAgent ----------
def test_reflection_agent_no_llm() -> None:
    agent = ReflectionAgent()
    from app.brain.local_llm import LLMUnavailable

    with mock.patch.object(agent.llm, "generate", side_effect=LLMUnavailable("offline")):
        result = agent.reflect(
            indicator={"indicator_name": "RSI_test", "combination_reasoning": "test"},
            backtest_result={
                "metrics": {
                    "n_trades": 5,
                    "total_return_pct": -2.0,
                    "sharpe": -0.3,
                    "max_drawdown_pct": -15.0,
                }
            },
            verdict="fail",
            reasons=["Az işlem", "OOS negatif"],
        )
    assert result is None


def test_reflection_agent_parses_valid_json() -> None:
    agent = ReflectionAgent()
    fake = json.dumps(
        {
            "failure_analysis": "RSI çok erken sinyal veriyor",
            "changes": ["period 14→20"],
            "improvement_reasoning": "Daha uzun period gürültüyü azaltır",
            "strategy_ir": {
                "name": "rsi_v2",
                "market": "XAUUSD",
                "timeframe": "15m",
                "indicators": [{"name": "RSI", "period": 20}],
                "entry_rules": ["rsi_20 > 60"],
                "exit_rules": ["rsi_20 < 40"],
                "risk": {"stop_loss": "2 * ATR"},
                "costs": {"commission": 0.0005, "slippage": 0.0005},
            },
        }
    )
    with (
        mock.patch.object(agent.llm, "available", return_value=True),
        mock.patch.object(agent.llm, "generate", return_value=fake),
    ):
        result = agent.reflect(
            indicator={"indicator_name": "RSI_v1"},
            backtest_result={
                "metrics": {
                    "n_trades": 5,
                    "total_return_pct": -2.0,
                    "sharpe": -0.3,
                    "max_drawdown_pct": -10.0,
                }
            },
            verdict="fail",
            reasons=["OOS negatif"],
        )
    assert result is not None
    assert result["strategy_ir"]["name"] == "rsi_v2"


# ---------- ResearchOrchestrator + L5 kompozisyon kapısı ----------
def _fake_synthesis(ir_dict: dict) -> mock.MagicMock:
    """synthesize() çağrısında sabit bir SynthesisResult döndüren sahte motor."""
    result = SynthesisResult(
        indicator_name="TestComposite",
        description="test",
        source_papers=["p1"],
        formula_components=[{"name": "RSI"}, {"name": "ATR"}],
        combination_reasoning="RSI + ATR",
        expected_edge="düşük vol",
        failure_conditions=["trend"],
        strategy_ir=ir_dict,
    )
    engine = mock.MagicMock(spec=SynthesisEngine)
    engine.synthesize.return_value = result
    return engine


def test_orchestrator_records_l5_composition(store: SqliteStore) -> None:
    """Her iterasyon L5 kompozisyon sonucunu (math+novelty+backtest) kaydeder."""
    df = generate_synthetic_ohlcv(n=2000, seed=42)
    orch = ResearchOrchestrator(
        store=store,
        synthesis_engine=_fake_synthesis(example_ir().model_dump()),
        max_iterations=1,
    )
    with mock.patch.object(orch, "_load_data", return_value=(df, "synthetic-test")):
        result = orch.run("Momentum nasıl filtrelenir backtest")

    assert result.iterations
    comp = result.iterations[0].composition
    assert comp is not None
    assert comp["verdict"] in {"candidate", "rejected"}
    assert {g["gate"] for g in comp["gates"]} == {"math", "novelty", "backtest"}
    # Session'a da yazıldığını doğrula (backtest_result zaten dict olarak döner)
    rows = store.list_research_sessions(limit=5)
    saved = rows[0]["backtest_result"]
    assert saved is not None and "composition" in saved


def test_orchestrator_l5_novelty_flags_repeat(store: SqliteStore) -> None:
    """Aynı kompozisyon 2. kez önerilirse novelty kapısı kopya olarak işaretler."""
    df = generate_synthetic_ohlcv(n=2000, seed=42)  # verdict=fail → döngü sürer
    reflection = mock.MagicMock(spec=ReflectionAgent)
    reflection.reflect.return_value = None  # yansıma yok → aynı IR yeniden sentezlenir
    orch = ResearchOrchestrator(
        store=store,
        synthesis_engine=_fake_synthesis(example_ir().model_dump()),
        reflection_agent=reflection,
        max_iterations=2,
    )
    with mock.patch.object(orch, "_load_data", return_value=(df, "synthetic-test")):
        result = orch.run("Momentum nasıl filtrelenir backtest")

    assert len(result.iterations) == 2
    novelty_2 = next(g for g in result.iterations[1].composition["gates"] if g["gate"] == "novelty")
    assert novelty_2["passed"] is False
    assert any("Kopya" in d for d in novelty_2["details"])


# ---------- ChainDataBuilder ----------
def test_chain_data_builder_empty(store: SqliteStore, tmp_path) -> None:
    builder = ChainDataBuilder(store=store)
    builder.settings = type("S", (), {"jsonl_dir": tmp_path})()  # type: ignore
    result = builder.build()
    assert result["n_records"] == 0


def test_chain_data_builder_with_session(store: SqliteStore, tmp_path) -> None:
    store.save_research_session(
        session_id="rs_test1",
        question="Momentum nasıl iyileştirilir?",
        iteration=1,
        source_paper_ids_json='["p1"]',
        synthesis_reasoning="RSI + ATR kombinasyonu",
        proposed_indicator_json=json.dumps(
            {
                "indicator_name": "FilteredRSI",
                "combination_reasoning": "ATR filtresi ile RSI",
                "formula_components": [{"name": "RSI"}],
                "expected_edge": "düşük volatilite",
                "failure_conditions": ["yüksek volatilite"],
            }
        ),
        strategy_ir_json='{"name":"filtered_rsi_v1","entry_rules":["rsi_14>55"],"exit_rules":["rsi_14<45"]}',
        backtest_result_json='{"metrics":{"n_trades":40,"sharpe":0.8},"verdict":"inconclusive"}',
        verdict="inconclusive",
        status="done",
    )
    builder = ChainDataBuilder(store=store)
    builder.settings = type("S", (), {"jsonl_dir": tmp_path})()  # type: ignore
    result = builder.build()
    assert result["n_records"] == 1
    # Dosyanın geçerli JSONL içerdiğini doğrula
    with open(result["output_path"], encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f if ln.strip()]
    assert len(lines) == 1
    assert "prompt" in lines[0] and "completion" in lines[0]
