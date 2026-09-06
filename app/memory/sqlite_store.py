"""SQLite structured store (SQLAlchemy 2.0).

Holds every durable, queryable record in the system:
papers, chunks, summaries, knowledge cards, training examples,
strategies, backtests, model evaluations and adapter metadata.

Every object carries a stable string ID so it can be cross-referenced
with the ChromaDB vector memory.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from app.verification.comprehension_scorer import ComprehensionScore

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    select,
    text,
    update,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from app.config import get_settings

log = logging.getLogger(__name__)


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


_TITLE_NORM_RE = re.compile(r"[\W_]+", re.UNICODE)
_ALNUM_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]")


def _card_has_content(card_json: str | dict[str, Any]) -> bool:
    """Kart onaylanabilir içerik taşıyor mu — tamamen boş / '...' placeholder kartı eler.

    Üretim başarısız olduğunda (LLM boş/parse edilemez döndürür) kart yine de
    `title=None, main_claim=''` ile kaydedilip onaylanabiliyordu; bu 419 onaylı kartın
    198'inin (≈%47) içeriksiz olmasına ve denetimin (lora-audit) FAIL vermesine yol açtı.
    Bu guard yalnız GERÇEKTEN boş kartı reddeder: title VE main_claim'in ikisi de
    alfanümerik karakter taşımıyorsa içeriksiz sayılır. Terse-ama-gerçek kartlara
    dokunmaz; daha sıkı 'substantive' eşiği (title≥8, main≥40) RAG öğrenme döngüsünde
    `is_substantive_card` ile ayrıca uygulanır.
    """
    if isinstance(card_json, str):
        try:
            card = json.loads(card_json)
        except (json.JSONDecodeError, TypeError):
            return False
    elif isinstance(card_json, dict):
        card = card_json
    else:
        return False
    if not isinstance(card, dict):
        return False
    title = _ALNUM_RE.findall(str(card.get("title") or ""))
    main_claim = _ALNUM_RE.findall(str(card.get("main_claim") or ""))
    return bool(title or main_claim)


# Kamusal ad: KnowledgeCardBuilder aynı "gerçekten boş" tanımını kayıt ÖNCESİ uygular
# (boş kartı hiç yazmasın); tek tanım, iki kapı — eşikler ayrışmasın.
card_has_content = _card_has_content


def _normalize_title(title: str) -> str:
    """Başlığı dedup karşılaştırması için normalize et: küçük harf, alfanümerik-dışı
    karakterleri tek boşluğa indir, kırp. (Farklı PDF export'ların noktalama/boşluk
    farklarına dayanıklı; başlık aynıysa eşleşir.)"""
    return _TITLE_NORM_RE.sub(" ", title.lower()).strip()


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source_path: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text, default=None)
    authors: Mapped[str | None] = mapped_column(Text, default=None)  # JSON list
    year: Mapped[str | None] = mapped_column(String(8), default=None)
    source: Mapped[str | None] = mapped_column(String(64), default=None)  # arxiv/ssrn/manual
    n_pages: Mapped[int | None] = mapped_column(Integer, default=None)
    n_chars: Mapped[int | None] = mapped_column(Integer, default=None)
    # İçe-alım kalite skoru (compute-on-demand; NULL = henüz skorlanmadı → retrieval
    # bu alandan ETKİLENMEZ, eski makaleler çalışmaya devam eder). Bkz app/ingestion/quality_scorer.
    quality_score: Mapped[float | None] = mapped_column(Float, default=None)
    ingest_status: Mapped[str | None] = mapped_column(String(32), default=None)
    # ready_for_rag / usable / slow_but_usable / unstable / failed / None
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)

    chunks: Mapped[list[Chunk]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("paper_id", "chunk_index", name="uq_paper_chunk"),)

    chunk_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section_name: Mapped[str | None] = mapped_column(Text, default=None)
    page_number: Mapped[int | None] = mapped_column(Integer, default=None)
    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    token_estimate: Mapped[int] = mapped_column(Integer, default=0)
    embedded: Mapped[int] = mapped_column(Integer, default=0)  # 0/1 flag
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class Summary(Base):
    __tablename__ = "summaries"

    summary_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    model: Mapped[str] = mapped_column(String(64))
    summary_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class KnowledgeCard(Base):
    __tablename__ = "knowledge_cards"

    card_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    model: Mapped[str] = mapped_column(String(64))
    card_json: Mapped[str] = mapped_column(Text)  # tam JSON belgesi
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)
    trust_level: Mapped[str] = mapped_column(String(32), default="draft")
    # canonical / verified / draft / unreviewed
    review_status: Mapped[str] = mapped_column(String(32), default="pending")
    # pending / approved / rejected
    lora_eligible: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    difficulty: Mapped[float] = mapped_column(Float, default=0.0)  # 0.0–1.0
    stage: Mapped[str] = mapped_column(String(32), default="")
    # lora_phase_1 / lora_phase_2 / lora_phase_3 / lora_phase_4 / ""


class TrainingExample(Base):
    __tablename__ = "training_examples"

    example_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_paper_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    example_type: Mapped[str] = mapped_column(String(48))
    instruction: Mapped[str] = mapped_column(Text)
    input_text: Mapped[str] = mapped_column(Text, default="")
    output_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class Strategy(Base):
    __tablename__ = "strategies"

    strategy_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    market: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(16))
    ir_json: Mapped[str] = mapped_column(Text)  # Strategy IR document
    origin: Mapped[str] = mapped_column(String(32), default="manual")  # manual/generated
    source_paper_id: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)

    backtests: Mapped[list[Backtest]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan"
    )


class Backtest(Base):
    __tablename__ = "backtests"

    backtest_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.strategy_id"), index=True)
    data_file: Mapped[str] = mapped_column(Text)
    period_start: Mapped[str | None] = mapped_column(String(40), default=None)
    period_end: Mapped[str | None] = mapped_column(String(40), default=None)
    n_trades: Mapped[int] = mapped_column(Integer, default=0)
    total_return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe: Mapped[float | None] = mapped_column(Float, default=None)
    sortino: Mapped[float | None] = mapped_column(Float, default=None)
    max_drawdown_pct: Mapped[float | None] = mapped_column(Float, default=None)
    profit_factor: Mapped[float | None] = mapped_column(Float, default=None)
    win_rate_pct: Mapped[float | None] = mapped_column(Float, default=None)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    verdict: Mapped[str | None] = mapped_column(String(32), default=None)  # pass/fail/inconclusive
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)

    strategy: Mapped[Strategy] = relationship(back_populates="backtests")


class RiskReportRow(Base):
    __tablename__ = "risk_reports"

    report_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    backtest_id: Mapped[str] = mapped_column(String(80), index=True)
    strategy_name: Mapped[str] = mapped_column(String(255))
    n_trades: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    half_kelly: Mapped[float] = mapped_column(Float, default=0.0)
    capped_kelly: Mapped[float] = mapped_column(Float, default=0.0)
    scale_factor: Mapped[float] = mapped_column(Float, default=1.0)
    position_size_pct: Mapped[float] = mapped_column(Float, default=0.0)
    position_size_usd: Mapped[float] = mapped_column(Float, default=0.0)
    report_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class ModelEvaluation(Base):
    __tablename__ = "model_evaluations"

    eval_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    eval_set: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(96))
    adapter_version: Mapped[str | None] = mapped_column(String(64), default=None)
    score: Mapped[float | None] = mapped_column(Float, default=None)
    results_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class Adapter(Base):
    __tablename__ = "adapters"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    base_model: Mapped[str] = mapped_column(String(96))
    adapter_path: Mapped[str] = mapped_column(Text)
    training_data_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    notes: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class Formula(Base):
    """Bir makaleden çıkarılan matematiksel formül / gösterge."""

    __tablename__ = "formulas"

    formula_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)  # "RSI", "EMA"
    latex: Mapped[str | None] = mapped_column(Text, default=None)  # LaTeX gösterim
    plain: Mapped[str | None] = mapped_column(Text, default=None)  # okunabilir
    description: Mapped[str | None] = mapped_column(Text, default=None)  # ne ölçüyor
    variables_json: Mapped[str] = mapped_column(Text, default="{}")  # {"period": "..."}
    category: Mapped[str | None] = mapped_column(String(48), default=None)  # momentum/vol/trend
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class ConceptLink(Base):
    """Formüller / kavramlar arası ilişki (yönlü kenar)."""

    __tablename__ = "concept_links"

    link_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_concept: Mapped[str] = mapped_column(String(128), index=True)
    relation: Mapped[str] = mapped_column(String(48))  # extends/measures/limits/combines
    to_concept: Mapped[str] = mapped_column(String(128), index=True)
    source_paper_id: Mapped[str | None] = mapped_column(String(64), default=None)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class ResearchSession(Base):
    """Bir araştırma döngüsünün tam kaydı (hipotez → öneri → test → yansıma)."""

    __tablename__ = "research_sessions"

    session_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    question: Mapped[str] = mapped_column(Text)  # araştırma sorusu
    iteration: Mapped[int] = mapped_column(Integer, default=1)
    parent_session_id: Mapped[str | None] = mapped_column(String(80), default=None)
    source_paper_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    synthesis_reasoning: Mapped[str | None] = mapped_column(Text, default=None)
    proposed_indicator_json: Mapped[str | None] = mapped_column(Text, default=None)
    strategy_ir_json: Mapped[str | None] = mapped_column(Text, default=None)
    backtest_result_json: Mapped[str | None] = mapped_column(Text, default=None)
    verdict: Mapped[str | None] = mapped_column(String(32), default=None)
    reflection: Mapped[str | None] = mapped_column(Text, default=None)
    improvement_notes: Mapped[str | None] = mapped_column(Text, default=None)
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/done/failed
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


# ---------------------------------------------------------------------------
# Advanced RAG tracking models
# ---------------------------------------------------------------------------


class ArxivSavedQuery(Base):
    """Kayıtlı arXiv arama sorgusu — tekrar kullanım ve otomasyon için."""

    __tablename__ = "arxiv_saved_queries"

    query_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    max_results: Mapped[int] = mapped_column(Integer, default=5)
    auto_ingest: Mapped[int] = mapped_column(Integer, default=1)  # 0/1
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    last_run_at: Mapped[str | None] = mapped_column(String(40), default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class EvalHistory(Base):
    """Adapter versiyonu bazında eval skor geçmişi — öğrenme dinamikleri grafiği için."""

    __tablename__ = "eval_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    adapter_name: Mapped[str] = mapped_column(String(128), index=True)
    eval_set: Mapped[str] = mapped_column(String(64))
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    passed_items: Mapped[int] = mapped_column(Integer, default=0)
    scored_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RagQuery(Base):
    """Records a user query and its expanded variants."""

    __tablename__ = "rag_queries"

    query_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    original_query: Mapped[str] = mapped_column(Text)
    expanded_queries_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RetrievalRun(Base):
    """Metadata for a single retrieval run."""

    __tablename__ = "retrieval_runs"

    retrieval_run_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("rag_queries.query_id"), index=True)
    retrieval_method: Mapped[str] = mapped_column(String(64), default="semantic")
    top_k: Mapped[int] = mapped_column(Integer, default=5)
    rerank_used: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    self_refinement_used: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RetrievalResult(Base):
    """A single chunk result from a retrieval run."""

    __tablename__ = "retrieval_results"

    result_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    retrieval_run_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_runs.retrieval_run_id"), index=True
    )
    paper_id: Mapped[str] = mapped_column(String(64), index=True)
    chunk_id: Mapped[str] = mapped_column(String(80), index=True)
    semantic_score: Mapped[float | None] = mapped_column(Float, default=None)
    rerank_score: Mapped[float | None] = mapped_column(Float, default=None)
    final_rank: Mapped[int | None] = mapped_column(Integer, default=None)
    reason: Mapped[str | None] = mapped_column(Text, default=None)


class ChunkQualityFlag(Base):
    """Quality flags for a chunk (formula, table, incomplete argument, etc.)."""

    __tablename__ = "chunk_quality_flags"

    flag_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(80), index=True)
    has_formula: Mapped[int] = mapped_column(Integer, default=0)
    has_incomplete_formula: Mapped[int] = mapped_column(Integer, default=0)
    has_incomplete_argument: Mapped[int] = mapped_column(Integer, default=0)
    has_table: Mapped[int] = mapped_column(Integer, default=0)
    has_definition: Mapped[int] = mapped_column(Integer, default=0)
    has_theorem: Mapped[int] = mapped_column(Integer, default=0)
    needs_adjacent_context: Mapped[int] = mapped_column(Integer, default=0)


class KnowledgeEntity(Base):
    """Knowledge graph entity (ORM version)."""

    __tablename__ = "knowledge_entities"

    entity_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    entity_type: Mapped[str] = mapped_column(String(48))
    description: Mapped[str | None] = mapped_column(Text, default=None)
    source_paper_id: Mapped[str | None] = mapped_column(String(64), default=None)
    source_chunk_id: Mapped[str | None] = mapped_column(String(80), default=None)


class KnowledgeRelation(Base):
    """Knowledge graph relation (ORM version)."""

    __tablename__ = "knowledge_relations"

    relation_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_entity_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_entities.entity_id"), index=True
    )
    relation_type: Mapped[str] = mapped_column(String(48))
    target_entity_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_entities.entity_id"), index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source_paper_id: Mapped[str | None] = mapped_column(String(64), default=None)
    source_chunk_id: Mapped[str | None] = mapped_column(String(80), default=None)


# ---------------------------------------------------------------------------
# Reliability / verification models
# ---------------------------------------------------------------------------


class GoldenQuestion(Base):
    """Golden set question for retrieval and answer quality evaluation."""

    __tablename__ = "golden_questions"

    question_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    question_text: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(48), default="general")
    expected_answer: Mapped[str | None] = mapped_column(Text, default=None)
    expected_source_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    expected_chunk_ids_json: Mapped[str] = mapped_column(Text, default="[]")
    answer_type: Mapped[str] = mapped_column(String(32), default="factual")
    difficulty: Mapped[str] = mapped_column(String(16), default="medium")
    created_by: Mapped[str] = mapped_column(String(64), default="system")
    reviewed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class FailureLog(Base):
    """RAG answer failure record."""

    __tablename__ = "failure_logs"

    failure_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    question_text: Mapped[str] = mapped_column(Text)
    answer_text: Mapped[str] = mapped_column(Text, default="")
    failure_type: Mapped[str] = mapped_column(String(48))
    root_cause: Mapped[str | None] = mapped_column(Text, default=None)
    hallucination: Mapped[int] = mapped_column(Integer, default=0)
    wrong_citation: Mapped[int] = mapped_column(Integer, default=0)
    suggested_fix: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class VerificationRun(Base):
    """Record of a verification pass run for a single answer."""

    __tablename__ = "verification_runs"

    verification_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    answer_id: Mapped[str] = mapped_column(String(80), index=True)
    citation_passed: Mapped[int] = mapped_column(Integer, default=0)
    grounding_passed: Mapped[int] = mapped_column(Integer, default=0)
    contradiction_passed: Mapped[int] = mapped_column(Integer, default=0)
    formula_passed: Mapped[int] = mapped_column(Integer, default=0)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.0)
    final_decision: Mapped[str] = mapped_column(String(16), default="warn")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RewardSignal(Base):
    """Bir tool-use seansının doğrulanabilir ödül skoru."""

    __tablename__ = "reward_signals"

    signal_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(80), index=True, unique=True)
    composite_score: Mapped[float] = mapped_column(Float, default=0.0)
    label: Mapped[str] = mapped_column(String(16), default="neutral")
    execution_ok: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count_ok: Mapped[float] = mapped_column(Float, default=0.0)
    sharpe_ok: Mapped[float] = mapped_column(Float, default=0.0)
    drawdown_ok: Mapped[float] = mapped_column(Float, default=0.0)
    return_ok: Mapped[float] = mapped_column(Float, default=0.0)
    win_rate_ok: Mapped[float] = mapped_column(Float, default=0.0)
    notes_json: Mapped[str] = mapped_column(Text, default="[]")
    raw_metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class ToolUseExample(Base):
    """Tool-use eğitim örneği — bir araştırma döngüsündeki tek araç çağrısı adımı."""

    __tablename__ = "tool_use_examples"

    example_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(80), index=True)
    question: Mapped[str] = mapped_column(Text)
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    # think | call | observe | conclude
    step_type: Mapped[str] = mapped_column(String(20))
    tool_name: Mapped[str | None] = mapped_column(String(80), default=None)
    tool_input_json: Mapped[str] = mapped_column(Text, default="{}")
    tool_output_json: Mapped[str] = mapped_column(Text, default="{}")
    content: Mapped[str] = mapped_column(Text, default="")
    verdict: Mapped[str | None] = mapped_column(String(20), default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class PaperComprehension(Base):
    """Makale anlama skoru — 3 katmanlı hesaplama sonucu."""

    __tablename__ = "paper_comprehension"

    paper_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    extraction_score: Mapped[float] = mapped_column(Float, default=0.0)
    retrieval_score: Mapped[float] = mapped_column(Float, default=0.0)
    llm_score: Mapped[float] = mapped_column(Float, default=0.0)
    total_score: Mapped[float] = mapped_column(Float, default=0.0)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
    computed_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class UnderstandingSnapshot(Base):
    """Objektif anlama merdiveni anlık görüntüsü — KALICI zaman serisi.

    Her satır, bir zaman noktasında L(Taban/1/2/3/4/5) sınavlarının toplanmış
    OBJEKTİF sonucudur (öz-değerlendirme değil). ``score_indicator_exams`` /
    ``score_full_ladder`` çıktısı buraya yazılır → "okuduğunu anladı mı" zaman
    içinde yüzdeyle değil SINAV geçme oranıyla izlenir (CLAUDE.md Kural 2).
    """

    __tablename__ = "understanding_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow, index=True)
    seed: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="insufficient_data")
    total: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    no_data: Mapped[int] = mapped_column(Integer, default=0)
    graded: Mapped[int] = mapped_column(Integer, default=0)
    pass_rate: Mapped[float | None] = mapped_column(Float, default=None)
    by_level_json: Mapped[str] = mapped_column(Text, default="{}")  # seviye kırılımı
    context_json: Mapped[str] = mapped_column(Text, default="{}")  # llm/adapter/with_rag meta


class AgentRunRow(Base):
    """Agent runtime gözlemcisi (Phase 1) — tek bir ajan koşusu.

    Davranışı zorlamaz; yalnız KAYIT tutar (run_id, durum, tetik, zaman, özet).
    Yeni tablo olduğundan ``Base.metadata.create_all`` ile idempotent oluşturulur.
    """

    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    trigger_type: Mapped[str | None] = mapped_column(String(32), default=None)
    trigger_payload_json: Mapped[str | None] = mapped_column(Text, default=None)
    started_at: Mapped[str] = mapped_column(String(40), index=True)
    finished_at: Mapped[str | None] = mapped_column(String(40), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    summary_json: Mapped[str | None] = mapped_column(Text, default=None)
    outputs_json: Mapped[str | None] = mapped_column(Text, default=None)


class AgentEventRow(Base):
    """Agent runtime gözlemcisi (Phase 1) — bir koşu içindeki tek bir olay.

    Retention (kullanıcı kararı): 30 gün VEYA en çok 50.000 son olay
    (``SqliteStore.prune_agent_events``).
    """

    __tablename__ = "agent_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[str] = mapped_column(String(40), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    level: Mapped[str] = mapped_column(String(16), default="info")
    message: Mapped[str | None] = mapped_column(Text, default=None)
    payload_json: Mapped[str | None] = mapped_column(Text, default=None)


class AutomationTaskRow(Base):
    """Agent runtime (Phase 2) — bir otomasyon görevi (task queue).

    Zamanlama (schedule) bilgilendirici alandır; Windows Task Scheduler DIŞ cron
    olarak kalır (Phase 2'de app içine taşınmaz) — yalnız izlenebilir kayıt tutulur.
    """

    __tablename__ = "automation_tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    params_json: Mapped[str | None] = mapped_column(Text, default=None)
    schedule: Mapped[str | None] = mapped_column(String(64), default=None)
    status: Mapped[str] = mapped_column(String(32), index=True)
    requires_approval: Mapped[int] = mapped_column(Integer, default=0)  # 0/1
    created_at: Mapped[str] = mapped_column(String(40), index=True)
    updated_at: Mapped[str | None] = mapped_column(String(40), default=None)
    claimed_at: Mapped[str | None] = mapped_column(String(40), default=None)
    completed_at: Mapped[str | None] = mapped_column(String(40), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)


class ApprovalRequestRow(Base):
    """Agent runtime (Phase 2) — bir onay isteği.

    ``consumed_at``: taze-onay tek kullanımlıktır (standing/kalıcı yetki YOK) —
    onaylanmış istek bir tehlikeli aksiyonda tüketilince damgalanır.
    """

    __tablename__ = "approval_requests"

    approval_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    agent_id: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    risk: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)
    requested_at: Mapped[str] = mapped_column(String(40), index=True)
    decided_at: Mapped[str | None] = mapped_column(String(40), default=None)
    decided_by: Mapped[str | None] = mapped_column(String(64), default=None)
    decision_note: Mapped[str | None] = mapped_column(Text, default=None)
    consumed_at: Mapped[str | None] = mapped_column(String(40), default=None)


# ---------------------------------------------------------------------------
# Deney takibi & model/veri kayıt defteri (app/registry)
# ---------------------------------------------------------------------------
# Adapter yaşam döngüsü ZATEN app/lora/adapter_registry.py (JSONL) +
# app/training/adapter_registry.py (Adapter tablosu) ile kapsanıyor. Buradaki
# tablolar yalnız EKSİK olan yatay katmanı tamamlar: dataset / rag-index /
# embedding / rlm-reward sürümleri + her terfi kararının denetlenebilir kaydı.


class DatasetVersion(Base):
    """Bir eğitim/değerlendirme veri seti sürümü (içerik-hash ile sürümlenir).

    ``content_hash`` üretildiği kaynaktan (ör. ``dataset_builder``) gelir →
    aynı veri iki kez kaydedilirse aynı hash'le idempotent kalır. ``approval_status``
    yalnız ``promotion_gates`` üzerinden değişir (Kural 8: onaysız eğitime giremez).
    """

    __tablename__ = "dataset_versions"

    dataset_version_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    path: Mapped[str | None] = mapped_column(Text, default=None)
    source_type: Mapped[str] = mapped_column(String(48), default="sft")  # sft/dpo/tool_use
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    n_records: Mapped[int] = mapped_column(Integer, default=0)
    domain_distribution_json: Mapped[str] = mapped_column(Text, default="{}")
    quality_score: Mapped[float | None] = mapped_column(Float, default=None)
    approval_status: Mapped[str] = mapped_column(String(32), default="pending")
    # pending / approved / rejected
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RagIndexVersion(Base):
    """Bir RAG vektör-indeks anlık görüntüsünün sürümü (Chroma collection)."""

    __tablename__ = "rag_index_versions"

    rag_index_version_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    vector_db: Mapped[str] = mapped_column(String(32), default="chroma")
    collection_name: Mapped[str] = mapped_column(String(96), default="")
    embedding_model: Mapped[str] = mapped_column(String(96), default="")
    chunking_strategy: Mapped[str | None] = mapped_column(String(48), default=None)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_papers: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class EmbeddingModelVersion(Base):
    """Kullanılan bir embedding modelinin sürüm kaydı (boyut + sağlayıcı)."""

    __tablename__ = "embedding_model_versions"

    embedding_model_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(96), index=True)
    dimension: Mapped[int | None] = mapped_column(Integer, default=None)
    provider: Mapped[str | None] = mapped_column(String(48), default=None)  # ollama | fake
    local_path: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class RlmRewardVersion(Base):
    """RLM/DPO ödül seti sürümü — YALNIZ tablo (kayıt RLM çıktıları OKUNARAK doldurulur).

    DİKKAT: app/rlm/ eş zamanlı bir oturumun sahibidir; bu modül o paketi içe
    aktarmaz/düzenlemez. Tablo, ödül setlerinin sürümlenmesi + terfi öncesi
    sır/PII tarama bayraklarının (Kural: ödül seti gizli veri içermemeli) tutulması
    için vardır. ``pii_scanned``/``secret_scanned``: 0=taranmadı, 1=temiz, 2=bulgu.
    """

    __tablename__ = "rlm_reward_versions"

    reward_version_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    dataset_path: Mapped[str | None] = mapped_column(Text, default=None)
    method: Mapped[str | None] = mapped_column(String(32), default=None)  # dpo/grpo/reward
    n_examples: Mapped[int] = mapped_column(Integer, default=0)
    pii_scanned: Mapped[int] = mapped_column(Integer, default=0)
    secret_scanned: Mapped[int] = mapped_column(Integer, default=0)
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class PromotionDecision(Base):
    """Bir kayıt-defteri varlığının (dataset/rag_index/adapter/reward) terfi kararı.

    Append-only denetim izi: her ``promotion_gates`` çağrısı buraya bir satır yazar
    (approved / rejected / blocked) → "neden production'a alındı/alınmadı" izlenebilir.
    """

    __tablename__ = "promotion_decisions"

    decision_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    target_type: Mapped[str] = mapped_column(String(32), index=True)
    # dataset / rag_index / embedding / adapter / reward
    target_id: Mapped[str] = mapped_column(String(96), index=True)
    from_status: Mapped[str | None] = mapped_column(String(32), default=None)
    to_status: Mapped[str] = mapped_column(String(32), default="")
    decision: Mapped[str] = mapped_column(String(16), default="blocked")
    # approved / rejected / blocked
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    approved_by: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


# ---------------------------------------------------------------------------
# Bilimsel araç çalışma kayıtları (app/tools — Scientific Tool Runtime)
# ---------------------------------------------------------------------------
# LLM'in hesabı kafadan yapması yerine deterministik Python araçlarıyla
# doğrulamasının DENETLENEBİLİR izi: her araç çağrısı + ürettiği artefakt loglanır.


class ToolRun(Base):
    """Tek bir bilimsel-araç çalıştırması (Monte Carlo / istatistik / backtest vb.)."""

    __tablename__ = "tool_runs"

    tool_run_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    tool_id: Mapped[str] = mapped_column(String(64), index=True)
    tool_version: Mapped[str] = mapped_column(String(16), default="1")
    params_json: Mapped[str] = mapped_column(Text, default="{}")
    seed: Mapped[int | None] = mapped_column(Integer, default=None)  # determinizm (Kural 6)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running/ok/error
    output_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str | None] = mapped_column(Text, default=None)
    started_at: Mapped[str] = mapped_column(String(40), default=_utcnow)
    finished_at: Mapped[str | None] = mapped_column(String(40), default=None)


class ToolArtifact(Base):
    """Bir araç çalıştırmasının ürettiği artefakt (rapor/CSV/şekil) — soy-kütüğü."""

    __tablename__ = "tool_artifacts"

    artifact_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    tool_run_id: Mapped[str] = mapped_column(ForeignKey("tool_runs.tool_run_id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(32), default="json")
    path: Mapped[str | None] = mapped_column(Text, default=None)
    content_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


class PaperIngestionRun(Base):
    """Bir makalenin içe-alım kalite skorlama koşusu (compute-on-demand denetim izi).

    PaperIndexer'ın SICAK yolunu DEĞİŞTİRMEZ; ``app/ingestion/quality_scorer`` ile
    isteğe bağlı çağrılır. component_scores_json 100-puanlık rubriğin kırılımıdır.
    """

    __tablename__ = "paper_ingestion_runs"

    ingestion_run_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="")
    # ready_for_rag / usable / slow_but_usable / unstable / failed
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    component_scores_json: Mapped[str] = mapped_column(Text, default="{}")
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    n_formulas: Mapped[int] = mapped_column(Integer, default=0)
    notes_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[str] = mapped_column(String(40), default=_utcnow)


# Toplu budamada tek IN(...) ifadesinin SQLite değişken sınırını (~32766) aşmaması için
# parça boyu. Küçük tutuldu (güvenli marj + her DELETE hızlı).
_PRUNE_CHUNK = 500


def _sqlite_pragmas(dbapi_conn: object, _record: object) -> None:
    """Her bağlantıda WAL + busy_timeout. SqliteStore aynı `sqlite_file` dosyasını
    OrchestrationStore / MasteryStore / RlmStore ile paylaşır ve en yoğun yazan taraftır
    (knowledge_cards, agent_runs/agent_events, approval_requests, tool_runs...). Diğer üç
    store zaten bu listener'ı açıyor; SqliteStore eksikti → iki gerçek risk: (a) busy_timeout
    PER-BAĞLANTI olduğundan SqliteStore bağlantıları yalnız pysqlite varsayılanı (~5sn) kadar
    kilit beklerdi (diğerleri 30sn); (b) DB'yi İLK açan SqliteStore olursa WAL hiç set edilmez
    ve rollback-journal modunda kalır → eşzamanlı okuyucu+yazıcı bile bloklanır. Yavaş CPU'da
    uzun kart-yazma transaction'ı sürerken eşzamanlı yazım 'database is locked' fırlatırdı.
    (rlm_store / mastery_store / orchestration store ile birebir aynı desen.)"""
    cur = dbapi_conn.cursor()  # type: ignore[attr-defined]
    try:
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
    finally:
        cur.close()


class SqliteStore:
    """Thin wrapper around a SQLAlchemy engine + session factory."""

    def __init__(
        self, db_path: str | Path | None = None, *, check_same_thread: bool = True
    ) -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.sqlite_file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False yalnız çok-thread'li yazıcılar (agent tracker —
        # asyncio.to_thread içinden de yazabilir) için gevşetilir. Varsayılan True →
        # mevcut tüm çağıranlar için davranış BİREBİR korunur.
        # timeout=30.0: pysqlite busy_timeout'u 30sn'ye çeker (diğer store'larla hizalı);
        # ek olarak _sqlite_pragmas her bağlantıda WAL + busy_timeout=30000 uygular.
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            future=True,
            connect_args={"check_same_thread": check_same_thread, "timeout": 30.0},
        )
        event.listen(self.engine, "connect", _sqlite_pragmas)
        self._Session = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        self.create_all()
        self._migrate()

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)

    def _migrate(self) -> None:
        """Eksik kolonları SQLite'a ekle (idempotent)."""
        with self.engine.connect() as conn:
            existing = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(knowledge_cards)")).fetchall()
            }
            new_cols = [
                ("trust_level", "VARCHAR(32)", "'draft'"),
                ("review_status", "VARCHAR(32)", "'pending'"),
                ("lora_eligible", "INTEGER", "0"),
                ("difficulty", "REAL", "0.0"),
                ("stage", "VARCHAR(32)", "''"),
            ]
            for col, typ, default in new_cols:
                if col not in existing:
                    conn.execute(
                        text(
                            f"ALTER TABLE knowledge_cards ADD COLUMN {col} {typ} DEFAULT {default}"
                        )
                    )
            # papers: içe-alım kalite alanları (nullable → eski satırlar etkilenmez)
            paper_cols = {
                row[1] for row in conn.execute(text("PRAGMA table_info(papers)")).fetchall()
            }
            for col, typ in (("quality_score", "REAL"), ("ingest_status", "VARCHAR(32)")):
                if col not in paper_cols:
                    conn.execute(text(f"ALTER TABLE papers ADD COLUMN {col} {typ} DEFAULT NULL"))
            conn.commit()

    # --- agent runtime gözlemcisi (Phase 1) ------------------------------
    def create_agent_run(
        self,
        *,
        run_id: str,
        agent_id: str,
        task_id: str | None = None,
        status: str = "running",
        trigger_type: str | None = None,
        trigger_payload: dict[str, Any] | None = None,
        started_at: str | None = None,
    ) -> None:
        """Yeni bir ajan koşusu kaydı oluştur (status='running')."""
        with self.session() as s:
            s.add(
                AgentRunRow(
                    run_id=run_id,
                    agent_id=agent_id,
                    task_id=task_id,
                    status=status,
                    trigger_type=trigger_type,
                    trigger_payload_json=(
                        json.dumps(trigger_payload, ensure_ascii=False)
                        if trigger_payload is not None
                        else None
                    ),
                    started_at=started_at or _utcnow(),
                )
            )

    def finish_agent_run(
        self,
        *,
        run_id: str,
        status: str,
        finished_at: str | None = None,
        error: str | None = None,
        summary: dict[str, Any] | None = None,
        outputs: list[Any] | None = None,
    ) -> None:
        """Koşuyu sonlandır (status/finished_at/error/summary/outputs günceller)."""
        with self.session() as s:
            row = s.get(AgentRunRow, run_id)
            if row is None:
                return
            row.status = status
            row.finished_at = finished_at or _utcnow()
            if error is not None:
                row.error = error
            if summary is not None:
                row.summary_json = json.dumps(summary, ensure_ascii=False, default=str)
            if outputs is not None:
                row.outputs_json = json.dumps(outputs, ensure_ascii=False, default=str)

    def add_agent_event(
        self,
        *,
        event_id: str,
        run_id: str,
        kind: str,
        ts: str | None = None,
        level: str = "info",
        message: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Bir koşuya olay ekle."""
        with self.session() as s:
            s.add(
                AgentEventRow(
                    event_id=event_id,
                    run_id=run_id,
                    ts=ts or _utcnow(),
                    kind=kind,
                    level=level,
                    message=message,
                    payload_json=(
                        json.dumps(payload, ensure_ascii=False, default=str)
                        if payload is not None
                        else None
                    ),
                )
            )

    def list_agent_runs(
        self, limit: int = 20, agent_id: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Son ajan koşularını (en yeni önce) döndür; agent_id/status ile filtrele."""
        with self.session() as s:
            stmt = select(AgentRunRow).order_by(AgentRunRow.started_at.desc())
            if agent_id:
                stmt = stmt.where(AgentRunRow.agent_id == agent_id)
            if status:
                stmt = stmt.where(AgentRunRow.status == status)
            stmt = stmt.limit(limit)
            return [self._agent_run_to_dict(r) for r in s.scalars(stmt)]

    def get_agent_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(AgentRunRow, run_id)
            return self._agent_run_to_dict(row) if row else None

    def list_agent_events(self, run_id: str) -> list[dict[str, Any]]:
        """Bir koşunun olaylarını zaman sırasıyla döndür."""
        with self.session() as s:
            stmt = (
                select(AgentEventRow)
                .where(AgentEventRow.run_id == run_id)
                # ts benzersiz değil (Windows saat çözünürlüğü ~15ms) → eşit-ts olaylarda
                # deterministik sıra için ikincil anahtar. (CLAUDE.md Kural 6.)
                .order_by(AgentEventRow.ts.asc(), AgentEventRow.event_id.asc())
            )
            return [self._agent_event_to_dict(r) for r in s.scalars(stmt)]

    def prune_agent_events(self, max_events: int = 50_000, max_age_days: int = 30) -> int:
        """Retention: max_age_days'ten eski VEYA en yeni max_events dışındaki olayları sil.

        Silinen olay sayısını döndürür. (run kayıtları korunur; yalnız olaylar budanır.)
        """
        cutoff = (dt.datetime.now(dt.UTC) - dt.timedelta(days=max_age_days)).isoformat()
        with self.session() as s:
            old = set(s.scalars(select(AgentEventRow.event_id).where(AgentEventRow.ts < cutoff)))
            # ts benzersiz değil → eşit-ts sınırında hangi olayın budanacağı belirsiz olmasın
            # diye ikincil deterministik anahtar (event_id). (CLAUDE.md Kural 6.)
            ids_newest_first = list(
                s.scalars(
                    select(AgentEventRow.event_id).order_by(
                        AgentEventRow.ts.desc(), AgentEventRow.event_id.desc()
                    )
                )
            )
            overflow = set(ids_newest_first[max_events:])
            to_delete = old | overflow
            if not to_delete:
                return 0
            # SQLite SQLITE_MAX_VARIABLE_NUMBER (~32766) sınırı: tek IN(...) ile budama,
            # retention'ın asıl tetiklendiği büyük backlog'da OperationalError fırlatır ve
            # çağıran tarafça sessizce yutulur → retention kalıcı çalışmaz. Parçalı sil.
            ids = list(to_delete)
            for i in range(0, len(ids), _PRUNE_CHUNK):
                batch = ids[i : i + _PRUNE_CHUNK]
                s.execute(delete(AgentEventRow).where(AgentEventRow.event_id.in_(batch)))
            return len(to_delete)

    def _agent_run_to_dict(self, r: AgentRunRow) -> dict[str, Any]:
        return {
            "run_id": r.run_id,
            "agent_id": r.agent_id,
            "task_id": r.task_id,
            "status": r.status,
            "trigger_type": r.trigger_type,
            "trigger_payload": (
                json.loads(r.trigger_payload_json) if r.trigger_payload_json else None
            ),
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "error": r.error,
            "summary": json.loads(r.summary_json) if r.summary_json else None,
            "outputs": json.loads(r.outputs_json) if r.outputs_json else None,
        }

    def _agent_event_to_dict(self, r: AgentEventRow) -> dict[str, Any]:
        return {
            "event_id": r.event_id,
            "run_id": r.run_id,
            "ts": r.ts,
            "kind": r.kind,
            "level": r.level,
            "message": r.message,
            "payload": json.loads(r.payload_json) if r.payload_json else None,
        }

    def list_recent_agent_events(
        self, limit: int = 100, run_id: str | None = None, level: str | None = None
    ) -> list[dict[str, Any]]:
        """Tüm koşulardan son olaylar (genel akış — /api/events). En yeni önce."""
        with self.session() as s:
            stmt = select(AgentEventRow).order_by(
                AgentEventRow.ts.desc(), AgentEventRow.event_id.desc()
            )
            if run_id:
                stmt = stmt.where(AgentEventRow.run_id == run_id)
            if level:
                stmt = stmt.where(AgentEventRow.level == level)
            stmt = stmt.limit(limit)
            return [self._agent_event_to_dict(r) for r in s.scalars(stmt)]

    def mark_running_agent_runs_cancelled(
        self, reason: str, cutoff_iso: str | None = None
    ) -> list[str]:
        """status='running' kalan koşuları 'cancelled' yap (startup sweep).

        cutoff_iso verilirse yalnız ondan ESKİ başlamış koşular süpürülür (canlı
        eşzamanlı koşuları yanlışlıkla iptal etmemek için). Süpürülen run_id'leri döndürür.
        """
        now = _utcnow()
        run_ids: list[str] = []
        with self.session() as s:
            stmt = select(AgentRunRow).where(AgentRunRow.status == "running")
            if cutoff_iso:
                stmt = stmt.where(AgentRunRow.started_at < cutoff_iso)
            for row in s.scalars(stmt):
                row.status = "cancelled"
                row.finished_at = now
                row.error = reason
                if not row.summary_json:
                    row.summary_json = json.dumps({"note": reason}, ensure_ascii=False)
                run_ids.append(row.run_id)
        return run_ids

    # --- automation tasks (Phase 2) --------------------------------------
    def create_automation_task(
        self,
        *,
        task_id: str,
        agent_id: str,
        title: str,
        description: str | None = None,
        params: dict[str, Any] | None = None,
        schedule: str | None = None,
        status: str = "pending",
        requires_approval: bool = False,
        created_at: str | None = None,
    ) -> None:
        with self.session() as s:
            s.add(
                AutomationTaskRow(
                    task_id=task_id,
                    agent_id=agent_id,
                    title=title,
                    description=description,
                    params_json=(
                        json.dumps(params, ensure_ascii=False) if params is not None else None
                    ),
                    schedule=schedule,
                    status=status,
                    requires_approval=1 if requires_approval else 0,
                    created_at=created_at or _utcnow(),
                )
            )

    def get_automation_task(self, task_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(AutomationTaskRow, task_id)
            return self._automation_task_to_dict(row) if row else None

    def list_automation_tasks(
        self, limit: int = 50, status: str | None = None, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self.session() as s:
            stmt = select(AutomationTaskRow).order_by(AutomationTaskRow.created_at.desc())
            if status:
                stmt = stmt.where(AutomationTaskRow.status == status)
            if agent_id:
                stmt = stmt.where(AutomationTaskRow.agent_id == agent_id)
            stmt = stmt.limit(limit)
            return [self._automation_task_to_dict(r) for r in s.scalars(stmt)]

    def update_automation_task(self, task_id: str, **fields: Any) -> dict[str, Any] | None:
        """Görev alanlarını güncelle (status/error/zaman damgaları). updated_at otomatik."""
        allowed = {
            "status",
            "error",
            "claimed_at",
            "completed_at",
            "schedule",
            "title",
            "description",
        }
        with self.session() as s:
            row = s.get(AutomationTaskRow, task_id)
            if row is None:
                return None
            for k, v in fields.items():
                if k in allowed:
                    setattr(row, k, v)
            row.updated_at = _utcnow()
            return self._automation_task_to_dict(row)

    def claim_automation_task_atomic(self, task_id: str) -> dict[str, Any] | None:
        """Atomik claim: ``pending`` → ``claimed`` TEK transaction (koşullu UPDATE + rowcount).

        İki ayrı transaction (get + update) yerine CAS (consume_fresh_approval ile aynı
        desen): yalnız ``status='pending'`` iken günceller. Eşzamanlı iki çağrıdan yalnız
        BİRİ rowcount=1 alır (claim eder); diğeri None alır (yarışı kaybetti / yok / zaten
        claimed) → aynı görevin çift-çalıştırılması önlenir.
        """
        now = _utcnow()
        with self.session() as s:
            res = s.execute(
                update(AutomationTaskRow)
                .where(
                    AutomationTaskRow.task_id == task_id,
                    AutomationTaskRow.status == "pending",
                )
                .values(status="claimed", claimed_at=now, updated_at=now)
                .execution_options(synchronize_session=False)
            )
            if cast("Any", res).rowcount != 1:
                return None  # claim edilemedi: yok / pending değil / eşzamanlı çağrı aldı
            row = s.get(AutomationTaskRow, task_id)
            return self._automation_task_to_dict(row) if row else None

    def _automation_task_to_dict(self, r: AutomationTaskRow) -> dict[str, Any]:
        return {
            "task_id": r.task_id,
            "agent_id": r.agent_id,
            "title": r.title,
            "description": r.description,
            "params": json.loads(r.params_json) if r.params_json else None,
            "schedule": r.schedule,
            "status": r.status,
            "requires_approval": bool(r.requires_approval),
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "claimed_at": r.claimed_at,
            "completed_at": r.completed_at,
            "error": r.error,
        }

    # --- approval requests (Phase 2) -------------------------------------
    def create_approval_request(
        self,
        *,
        approval_id: str,
        agent_id: str,
        action: str,
        summary: str | None = None,
        risk: str = "medium",
        status: str = "pending",
        run_id: str | None = None,
        task_id: str | None = None,
        requested_at: str | None = None,
    ) -> None:
        with self.session() as s:
            s.add(
                ApprovalRequestRow(
                    approval_id=approval_id,
                    run_id=run_id,
                    task_id=task_id,
                    agent_id=agent_id,
                    action=action,
                    summary=summary,
                    risk=risk,
                    status=status,
                    requested_at=requested_at or _utcnow(),
                )
            )

    def get_approval_request(self, approval_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(ApprovalRequestRow, approval_id)
            return self._approval_to_dict(row) if row else None

    def list_approval_requests(
        self, status: str | None = None, limit: int = 50, agent_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self.session() as s:
            stmt = select(ApprovalRequestRow).order_by(ApprovalRequestRow.requested_at.desc())
            if status:
                stmt = stmt.where(ApprovalRequestRow.status == status)
            if agent_id:
                stmt = stmt.where(ApprovalRequestRow.agent_id == agent_id)
            stmt = stmt.limit(limit)
            return [self._approval_to_dict(r) for r in s.scalars(stmt)]

    def update_approval_request(self, approval_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"status", "decided_at", "decided_by", "decision_note", "consumed_at"}
        with self.session() as s:
            row = s.get(ApprovalRequestRow, approval_id)
            if row is None:
                return None
            for k, v in fields.items():
                if k in allowed:
                    setattr(row, k, v)
            return self._approval_to_dict(row)

    def find_fresh_approval(self, agent_id: str, action: str) -> dict[str, Any] | None:
        """En son ONAYLANMIŞ + TÜKETİLMEMİŞ onay (taze onay). Yoksa None.

        Tek kullanımlık: tüketim ``consumed_at`` ile damgalanır → standing yetki yok.
        """
        with self.session() as s:
            stmt = (
                select(ApprovalRequestRow)
                .where(
                    ApprovalRequestRow.agent_id == agent_id,
                    ApprovalRequestRow.action == action,
                    ApprovalRequestRow.status == "approved",
                    ApprovalRequestRow.consumed_at.is_(None),
                )
                .order_by(ApprovalRequestRow.requested_at.desc())
                .limit(1)
            )
            row = s.scalar(stmt)
            return self._approval_to_dict(row) if row else None

    def consume_fresh_approval(self, agent_id: str, action: str) -> dict[str, Any] | None:
        """Atomik: en yeni taze (approved+unconsumed) onayı BUL ve TÜKET — TEK transaction.

        ``find_fresh_approval`` + ``update_approval_request`` İKİ ayrı transaction'dı:
        iki eşzamanlı çağrı aynı onayı bulup ikisi de tüketebiliyordu (TOCTOU çift-tüketim;
        Kural 8 eğitim kapısında kritik — bir onay iki gerçek eğitimi yetkilendirirdi).
        Burada koşullu UPDATE (``consumed_at IS NULL``) + ``rowcount`` ile karşılaştır-ve-
        değiştir (CAS): araya giren eşzamanlı çağrılardan yalnız BİRİ tüketir, diğeri None alır.
        """
        with self.session() as s:
            aid = s.scalar(
                select(ApprovalRequestRow.approval_id)
                .where(
                    ApprovalRequestRow.agent_id == agent_id,
                    ApprovalRequestRow.action == action,
                    ApprovalRequestRow.status == "approved",
                    ApprovalRequestRow.consumed_at.is_(None),
                )
                .order_by(ApprovalRequestRow.requested_at.desc())
                .limit(1)
            )
            if aid is None:
                return None
            res = s.execute(
                update(ApprovalRequestRow)
                .where(
                    ApprovalRequestRow.approval_id == aid,
                    ApprovalRequestRow.consumed_at.is_(None),
                )
                .values(consumed_at=_utcnow())
                .execution_options(synchronize_session=False)
            )
            # DML sonucu CursorResult'tır (rowcount taşır); Session.execute imzası Result[Any].
            if cast("Any", res).rowcount != 1:
                return None  # araya giren eşzamanlı çağrı bu onayı tüketti → taze onay yok
            row = s.get(ApprovalRequestRow, aid)
            return self._approval_to_dict(row) if row else None

    def _approval_to_dict(self, r: ApprovalRequestRow) -> dict[str, Any]:
        return {
            "approval_id": r.approval_id,
            "run_id": r.run_id,
            "task_id": r.task_id,
            "agent_id": r.agent_id,
            "action": r.action,
            "summary": r.summary,
            "risk": r.risk,
            "status": r.status,
            "requested_at": r.requested_at,
            "decided_at": r.decided_at,
            "decided_by": r.decided_by,
            "decision_note": r.decision_note,
            "consumed_at": r.consumed_at,
        }

    # --- bilimsel araç çalışma kayıtları (app/tools) ---------------------
    @contextmanager
    def log_tool_run(
        self,
        *,
        tool_id: str,
        tool_version: str = "1",
        params: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Iterator[str]:
        """Bir araç çalıştırmasını kaydet (running→ok/error). run_id yield edilir.

        Çağıran, çıkıştan önce ``set_tool_run_output(run_id, ...)`` ile özet yazabilir.
        İstisna olursa ``error`` damgalanır ve yeniden fırlatılır.
        """
        import uuid as _uuid

        run_id = "tr_" + _uuid.uuid4().hex[:12]
        with self.session() as s:
            s.add(
                ToolRun(
                    tool_run_id=run_id,
                    tool_id=tool_id,
                    tool_version=tool_version,
                    params_json=json.dumps(params or {}, ensure_ascii=False, default=str),
                    seed=seed,
                    status="running",
                    started_at=_utcnow(),
                )
            )
        try:
            yield run_id
        except Exception as e:
            with self.session() as s:
                row = s.get(ToolRun, run_id)
                if row is not None:
                    row.status = "error"
                    row.error = str(e)
                    row.finished_at = _utcnow()
            raise
        else:
            with self.session() as s:
                row = s.get(ToolRun, run_id)
                if row is not None and row.status == "running":
                    row.status = "ok"
                    row.finished_at = _utcnow()

    def set_tool_run_output(self, run_id: str, summary: dict[str, Any]) -> None:
        with self.session() as s:
            row = s.get(ToolRun, run_id)
            if row is not None:
                row.output_summary_json = json.dumps(summary, ensure_ascii=False, default=str)

    def link_tool_artifact(
        self,
        *,
        tool_run_id: str,
        artifact_type: str = "json",
        path: str | None = None,
        content_hash: str | None = None,
        description: str | None = None,
    ) -> str:
        import uuid as _uuid

        aid = "art_" + _uuid.uuid4().hex[:12]
        with self.session() as s:
            # Uygulama düzeyi referans bütünlüğü: SQLite'ta FK enforcement varsayılan KAPALI
            # (PRAGMA global açılmıyor — diğer tablolarda regresyon riski), bu yüzden öksüz
            # artefakt yaratmamak için ebeveyni burada doğrula.
            if s.get(ToolRun, tool_run_id) is None:
                raise ValueError(f"Bilinmeyen tool_run_id: {tool_run_id}")
            s.add(
                ToolArtifact(
                    artifact_id=aid,
                    tool_run_id=tool_run_id,
                    artifact_type=artifact_type,
                    path=path,
                    content_hash=content_hash,
                    description=description,
                )
            )
        return aid

    def get_tool_run(self, run_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            r = s.get(ToolRun, run_id)
            return self._tool_run_to_dict(r) if r else None

    def list_tool_runs(self, limit: int = 50, tool_id: str | None = None) -> list[dict[str, Any]]:
        with self.session() as s:
            stmt = select(ToolRun).order_by(ToolRun.started_at.desc())
            if tool_id:
                stmt = stmt.where(ToolRun.tool_id == tool_id)
            stmt = stmt.limit(limit)
            return [self._tool_run_to_dict(r) for r in s.scalars(stmt)]

    def list_tool_artifacts(self, tool_run_id: str) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(ToolArtifact)
                .where(ToolArtifact.tool_run_id == tool_run_id)
                .order_by(ToolArtifact.created_at.asc())
            )
            return [self._tool_artifact_to_dict(r) for r in rows]

    def _tool_run_to_dict(self, r: ToolRun) -> dict[str, Any]:
        return {
            "tool_run_id": r.tool_run_id,
            "tool_id": r.tool_id,
            "tool_version": r.tool_version,
            "params": json.loads(r.params_json or "{}"),
            "seed": r.seed,
            "status": r.status,
            "output_summary": json.loads(r.output_summary_json or "{}"),
            "error": r.error,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
        }

    def _tool_artifact_to_dict(self, r: ToolArtifact) -> dict[str, Any]:
        return {
            "artifact_id": r.artifact_id,
            "tool_run_id": r.tool_run_id,
            "artifact_type": r.artifact_type,
            "path": r.path,
            "content_hash": r.content_hash,
            "description": r.description,
            "created_at": r.created_at,
        }

    # --- içe-alım kalite koşuları (app/ingestion) ------------------------
    def add_ingestion_run(
        self,
        *,
        paper_id: str,
        status: str,
        quality_score: float,
        component_scores: dict[str, float] | None = None,
        n_chunks: int = 0,
        n_formulas: int = 0,
        notes: list[str] | None = None,
        update_paper: bool = True,
    ) -> str:
        """Bir içe-alım kalite koşusu kaydet; ``update_paper`` ise Paper alanlarını da güncelle."""
        import uuid as _uuid

        run_id = "ing_" + _uuid.uuid4().hex[:12]
        with self.session() as s:
            # Önce ebeveyni doğrula (FK enforcement global KAPALI) → öksüz koşu yaratma.
            paper = s.get(Paper, paper_id)
            if paper is None:
                raise ValueError(f"Bilinmeyen paper_id: {paper_id}")
            s.add(
                PaperIngestionRun(
                    ingestion_run_id=run_id,
                    paper_id=paper_id,
                    status=status,
                    quality_score=quality_score,
                    component_scores_json=json.dumps(component_scores or {}, ensure_ascii=False),
                    n_chunks=n_chunks,
                    n_formulas=n_formulas,
                    notes_json=json.dumps(notes or [], ensure_ascii=False),
                )
            )
            if update_paper:
                paper.quality_score = quality_score
                paper.ingest_status = status
        return run_id

    def list_ingestion_runs(
        self, paper_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        with self.session() as s:
            stmt = select(PaperIngestionRun).order_by(PaperIngestionRun.created_at.desc())
            if paper_id:
                stmt = stmt.where(PaperIngestionRun.paper_id == paper_id)
            stmt = stmt.limit(limit)
            return [
                {
                    "ingestion_run_id": r.ingestion_run_id,
                    "paper_id": r.paper_id,
                    "status": r.status,
                    "quality_score": r.quality_score,
                    "component_scores": json.loads(r.component_scores_json or "{}"),
                    "n_chunks": r.n_chunks,
                    "n_formulas": r.n_formulas,
                    "notes": json.loads(r.notes_json or "[]"),
                    "created_at": r.created_at,
                }
                for r in s.scalars(stmt)
            ]

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # --- convenience helpers ---------------------------------------------
    def upsert_paper(self, **fields: Any) -> Paper:
        with self.session() as s:
            existing = s.get(Paper, fields["paper_id"])
            if existing:
                for k, v in fields.items():
                    setattr(existing, k, v)
                return existing
            paper = Paper(**fields)
            s.add(paper)
            return paper

    def get_paper_by_hash(self, file_hash: str) -> Paper | None:
        with self.session() as s:
            return s.scalar(select(Paper).where(Paper.file_hash == file_hash))

    def find_paper_by_title(self, title: str | None) -> Paper | None:
        """Normalize edilmiş başlığı eşleşen mevcut makaleyi döndür (paper-düzeyi dedup).

        Aynı makale farklı bytes'la (yeniden indirilmiş / farklı PDF export) yüklenirse
        file_hash farklı olur ama başlık aynıdır → RAG'a 2. kez girmesini engeller.
        Başlık boş/çok kısa ise None döner (güvenilmez eşleşmeyi önler).
        """
        norm = _normalize_title(title or "")
        if len(norm) < 12:  # çok kısa/boş başlık → dedup uygulama (yanlış eşleşme riski)
            return None
        with self.session() as s:
            for p in s.scalars(select(Paper)):
                if _normalize_title(p.title or "") == norm:
                    return p
        return None

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        with self.session() as s:
            for c in chunks:
                s.merge(Chunk(**c))
            return len(chunks)

    def mark_chunks_embedded(self, chunk_ids: list[str]) -> int:
        """Chroma'ya başarıyla eklenen chunk'ları embedded=1 olarak işaretle (BUG-M6 fix).

        add_chunks embedded=0 ile çağrılır → Chroma.add() başarılı olunca bu metot
        embedded=1 yazar. Chroma başarısız olursa embedded=0 kalır → sessiz kayıp önlenir.
        """
        if not chunk_ids:
            return 0
        with self.session() as s:
            s.execute(update(Chunk).where(Chunk.chunk_id.in_(chunk_ids)).values(embedded=1))
        return len(chunk_ids)

    def delete_chunks_for_paper(self, paper_id: str) -> int:
        """Bir makaleye ait tüm chunk'ları sil (force re-index'te bayat chunk bırakmamak için).

        Yeniden chunk'lama daha AZ parça üretirse eski yüksek-indeksli chunk'lar
        öksüz kalır ve retrieval'da bayat içerik döner; bu metot onu önler.
        """
        with self.session() as s:
            res = s.execute(delete(Chunk).where(Chunk.paper_id == paper_id))
            return int(getattr(res, "rowcount", 0) or 0)

    def list_papers(self) -> list[Paper]:
        with self.session() as s:
            return list(s.scalars(select(Paper).order_by(Paper.created_at.desc())))

    def list_chunks(self, paper_id: str) -> list[Chunk]:
        with self.session() as s:
            return list(
                s.scalars(
                    select(Chunk).where(Chunk.paper_id == paper_id).order_by(Chunk.chunk_index)
                )
            )

    def has_embedded_chunks(self, paper_id: str) -> bool:
        """Bir makalenin Chroma'ya gömülmüş (embedded=1) en az bir chunk'ı var mı?

        Yarım kalmış ingest tespiti için: upsert_paper kendi transaction'ında önce commit
        edilir; embed/chroma.add adımı çökerse (Ollama timeout, chromadb hatası, process kill)
        paper satırı kalır ama chunk'lar embedded=0'da takılır (Chroma'da YOK). Bu kontrol,
        force'suz re-ingest'in yarım makaleyi onarmasını sağlar (Kural 7 + ingestion idempotent
        sözleşmesi). LIMIT 1 → ucuz varlık kontrolü (paper_id indeksli; tüm satırları çekmez).
        """
        with self.session() as s:
            row = s.scalar(
                select(Chunk.chunk_id)
                .where(Chunk.paper_id == paper_id, Chunk.embedded == 1)
                .limit(1)
            )
            return row is not None

    def list_all_chunks(self) -> list[Chunk]:
        """TÜM chunk'ları döndür (BM25 korpus indeksi için — sağlam kaynak).

        Chroma `get_all()` 91k id'yi bulk okur ve eşzamanlı erişimde (öğrenme döngüsü/MCP)
        SQLite-kilidine takılıp BM25'i sessizce öldürürdü. SQLite WAL eşzamanlı-okumayı
        sorunsuz kaldırır → BM25 korpusu buradan kurulur (kaynak = doğru, dayanıklı).
        """
        with self.session() as s:
            return list(s.scalars(select(Chunk).order_by(Chunk.paper_id, Chunk.chunk_index)))

    def save_knowledge_card(
        self,
        card_id: str,
        paper_id: str,
        model: str,
        card: dict,
        *,
        trust_level: str = "draft",
        review_status: str = "pending",
        lora_eligible: int = 0,
        difficulty: float = 0.0,
        stage: str = "",
    ) -> None:
        with self.session() as s:
            s.merge(
                KnowledgeCard(
                    card_id=card_id,
                    paper_id=paper_id,
                    model=model,
                    card_json=json.dumps(card, ensure_ascii=False),
                    trust_level=trust_level,
                    review_status=review_status,
                    lora_eligible=lora_eligible,
                    difficulty=difficulty,
                    stage=stage,
                )
            )

    def has_knowledge_card(self, paper_id: str) -> bool:
        """Makalenin CANLI (rejected olmayan) ve İÇERİKLİ bir kartı var mı?

        `rejected` kart sayılmaz: reddedilen (ör. boş) kart 'kartı var' sayılsaydı makale
        bir daha kartlanmaz, sonsuza dek okunmamış kalırdı (ölçüldü: 21 boş kart reddedildi,
        makaleler kapsam dışına düştü). Reddetmek = "yeniden üret" demektir.

        İçeriksiz kart da sayılmaz (bulgu 2026-09-06): eski builder'ın yazdığı boş
        `pending` kartlar makaleyi arayüzde "✓ KARTI GÖR" gösteriyor, kart "(başlıksız)"
        açılıyor ve "BİLGİ KARTI ÜRET" düğmesi kaybolduğundan makale yeniden kartlanamıyordu.
        Aynı tek tanım (`card_has_content`) burada da kapıdır.
        """
        return self.get_latest_knowledge_card(paper_id) is not None

    def save_comprehension_score(self, score: ComprehensionScore) -> None:
        import json as _json

        with self.session() as s:
            row = s.get(PaperComprehension, score.paper_id)
            if row is None:
                row = PaperComprehension(paper_id=score.paper_id)
                s.add(row)
            row.extraction_score = score.extraction
            row.retrieval_score = score.retrieval
            row.llm_score = score.llm_verify
            row.total_score = score.total
            row.details_json = _json.dumps(score.details, ensure_ascii=False)
            row.computed_at = score.computed_at

    def get_comprehension_score(self, paper_id: str) -> PaperComprehension | None:
        with self.session() as s:
            return s.get(PaperComprehension, paper_id)

    # --- objektif anlama merdiveni (KALICI snapshot'lar) ------------------
    def save_understanding_snapshot(
        self,
        score: Any,
        *,
        seed: int = 0,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Anlama skorunu KALICI yap → ``understanding_snapshots`` (zaman serisi).

        ``score``: UnderstandingScore (``to_dict`` alanları). ``context``: serbest
        meta (ör. ``{"llm_model": ..., "with_rag": bool}``). Üretilen snapshot_id döner.
        """
        import uuid as _uuid

        d = score.to_dict() if hasattr(score, "to_dict") else dict(score)
        sid = "und_" + _uuid.uuid4().hex[:12]
        with self.session() as s:
            s.add(
                UnderstandingSnapshot(
                    snapshot_id=sid,
                    seed=seed,
                    status=str(d.get("status", "insufficient_data")),
                    total=int(d.get("total", 0)),
                    passed=int(d.get("passed", 0)),
                    failed=int(d.get("failed", 0)),
                    skipped=int(d.get("skipped", 0)),
                    no_data=int(d.get("no_data", 0)),
                    graded=int(d.get("graded", 0)),
                    pass_rate=d.get("pass_rate"),
                    by_level_json=json.dumps(d.get("by_level", {}), ensure_ascii=False),
                    context_json=json.dumps(context or {}, ensure_ascii=False),
                )
            )
        return sid

    @staticmethod
    def _snapshot_to_dict(row: UnderstandingSnapshot) -> dict[str, Any]:
        return {
            "snapshot_id": row.snapshot_id,
            "created_at": row.created_at,
            "seed": row.seed,
            "status": row.status,
            "total": row.total,
            "passed": row.passed,
            "failed": row.failed,
            "skipped": row.skipped,
            "no_data": row.no_data,
            "graded": row.graded,
            "pass_rate": row.pass_rate,
            "by_level": json.loads(row.by_level_json or "{}"),
            "context": json.loads(row.context_json or "{}"),
        }

    def list_understanding_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = s.scalars(
                select(UnderstandingSnapshot)
                .order_by(UnderstandingSnapshot.created_at.desc())
                .limit(limit)
            )
            return [self._snapshot_to_dict(r) for r in rows]

    def latest_understanding_snapshot(self) -> dict[str, Any] | None:
        snaps = self.list_understanding_snapshots(limit=1)
        return snaps[0] if snaps else None

    def get_latest_knowledge_card(self, paper_id: str) -> dict | None:
        """En son üretilmiş CANLI ve İÇERİKLİ kartın JSON içeriğini döndür (yoksa None).

        Reddedilmiş ve içeriksiz (title+main_claim boş) kartlar atlanır: aksi hâlde arayüz
        ve skorlayıcılar boş kabuğu "kart" sanır (bkz. `has_knowledge_card`). Sıralama
        created_at DESC — birden çok canlı kart varsa en yenisi.
        """
        with self.session() as s:
            rows = s.scalars(
                select(KnowledgeCard)
                .where(
                    KnowledgeCard.paper_id == paper_id,
                    KnowledgeCard.review_status != "rejected",
                )
                .order_by(KnowledgeCard.created_at.desc())
            )
            for row in rows:
                try:
                    card = json.loads(row.card_json)
                except (json.JSONDecodeError, TypeError):
                    continue
                if _card_has_content(card):
                    return card
            return None

    def approve_card(self, card_id: str) -> bool:
        """Kartı onayla: review_status=approved, lora_eligible=1.

        İçerik guard'ı: tamamen boş / '...' placeholder kart onaylanmaz (False döner,
        kart pending kalır). 198 'approved-ama-boş' kart sorununu kökten önler — üretim
        başarısızsa kart coverage'ı/denetimi çöple şişiremez (bkz. `_card_has_content`).
        """
        with self.session() as s:
            row = s.get(KnowledgeCard, card_id)
            if row is None:
                return False
            if not _card_has_content(row.card_json):
                log.warning("approve_card: içeriksiz/boş kart onaylanmadı: %s", card_id)
                return False
            row.review_status = "approved"
            row.lora_eligible = 1
            return True

    def reject_card(self, card_id: str) -> bool:
        """Kartı reddet: review_status=rejected, lora_eligible=0."""
        with self.session() as s:
            row = s.get(KnowledgeCard, card_id)
            if row is None:
                return False
            row.review_status = "rejected"
            row.lora_eligible = 0
            return True

    def set_card_lora_eligible(self, card_id: str, eligible: bool) -> bool:
        """Kartın ``lora_eligible`` bayrağını ayarla (``review_status``'a DOKUNMAZ).

        Kart kürasyonu için: orphan / fazla-versiyon kartı EĞİTİMDEN çıkar ama
        'approved' içeriğini ve coverage/mastery semantiğini koru. ``reject_card``'dan
        farkı: kartı reddetmez, yalnız eğitim uygunluğunu düşürür → GERİ ALINABİLİR
        (yeniden True yapılabilir). Kart yoksa False döner.
        """
        with self.session() as s:
            row = s.get(KnowledgeCard, card_id)
            if row is None:
                return False
            row.lora_eligible = 1 if eligible else 0
            return True

    def list_paper_ids(self) -> set[str]:
        """``papers`` tablosundaki TÜM geçerli paper_id'leri döndür (Gate 0 kaynak denetimi).

        Onaylı bir kartın paper_id'si bu sette YOKSA kart 'orphan'dır: dayandığı makale
        papers tablosunda yok (sıfırlama/temizlik artığı ya da uydurma kaynak) → iddiası
        hiçbir kaynağa bağlanamaz (CLAUDE.md kural 7), eğitime girmemeli.
        """
        with self.session() as s:
            return set(s.scalars(select(Paper.paper_id)))

    def list_pending_cards(self) -> list[dict]:
        """review_status=pending olan kartları döndür.

        Her dict: card_id, paper_id, model, trust_level, review_status,
                   lora_eligible, difficulty, stage, created_at, card_json (parsed)
        """
        with self.session() as s:
            rows = list(
                s.scalars(
                    select(KnowledgeCard)
                    .where(KnowledgeCard.review_status == "pending")
                    .order_by(KnowledgeCard.created_at.desc())
                )
            )
            return [self._card_to_dict(r) for r in rows]

    def list_approved_cards(
        self,
        difficulty_min: float = 0.0,
        difficulty_max: float = 1.0,
    ) -> list[dict]:
        """Onaylı (approved) kartları difficulty aralığına göre filtrele."""
        with self.session() as s:
            rows = list(
                s.scalars(
                    select(KnowledgeCard)
                    .where(
                        KnowledgeCard.review_status == "approved",
                        KnowledgeCard.difficulty >= difficulty_min,
                        KnowledgeCard.difficulty <= difficulty_max,
                    )
                    .order_by(KnowledgeCard.difficulty)
                )
            )
            return [self._card_to_dict(r) for r in rows]

    def list_cards(self, limit: int | None = None) -> list[dict]:
        """TÜM knowledge kartlarını döndür (review_status filtresiz; card_json dahil)."""
        with self.session() as s:
            q = select(KnowledgeCard).order_by(KnowledgeCard.difficulty)
            if limit:
                q = q.limit(limit)
            return [self._card_to_dict(r) for r in s.scalars(q)]

    def get_card_by_id(self, card_id: str) -> dict | None:
        """Tek kartı card_id ile döndür (review metadata dahil)."""
        with self.session() as s:
            row = s.get(KnowledgeCard, card_id)
            if row is None:
                return None
            return self._card_to_dict(row)

    def _card_to_dict(self, row: KnowledgeCard) -> dict:
        """KnowledgeCard ORM satırını dict'e dönüştür."""
        try:
            parsed = json.loads(row.card_json)
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        return {
            "card_id": row.card_id,
            "paper_id": row.paper_id,
            "model": row.model,
            "trust_level": row.trust_level,
            "review_status": row.review_status,
            "lora_eligible": row.lora_eligible,
            "difficulty": row.difficulty,
            "stage": row.stage,
            "created_at": row.created_at,
            "card_json": parsed,
        }

    def save_strategy(self, **fields: Any) -> None:
        with self.session() as s:
            s.merge(Strategy(**fields))

    def save_backtest(self, **fields: Any) -> None:
        with self.session() as s:
            s.merge(Backtest(**fields))

    def list_backtests(self, limit: int = 50) -> list[dict[str, Any]]:
        """Son `limit` adet backtest kaydını yeni→eski sırasıyla döndür."""
        with self.session() as s:
            rows = list(
                s.scalars(select(Backtest).order_by(Backtest.created_at.desc()).limit(limit))
            )
            out = []
            for r in rows:
                strat = s.get(Strategy, r.strategy_id)
                out.append(
                    {
                        "backtest_id": r.backtest_id,
                        "strategy_name": strat.name if strat else r.strategy_id,
                        "market": strat.market if strat else None,
                        "timeframe": strat.timeframe if strat else None,
                        "data_file": r.data_file,
                        "n_trades": r.n_trades,
                        "total_return_pct": r.total_return_pct,
                        "sharpe": r.sharpe,
                        "max_drawdown_pct": r.max_drawdown_pct,
                        "win_rate_pct": r.win_rate_pct,
                        "verdict": r.verdict,
                        "notes": r.notes,
                        "created_at": r.created_at,
                    }
                )
            return out

    def save_risk_report(self, report_id: str, backtest_id: str, report_dict: dict) -> None:
        """Risk raporunu upsert ile kaydet (idempotent)."""
        kelly = report_dict.get("kelly", {})
        dd = report_dict.get("drawdown_scale", {})
        fr = report_dict.get("fixed_risk", {})
        with self.session() as s:
            s.merge(
                RiskReportRow(
                    report_id=report_id,
                    backtest_id=backtest_id,
                    strategy_name=report_dict.get("strategy_name", ""),
                    n_trades=report_dict.get("n_trades", 0),
                    win_rate=kelly.get("win_rate", 0.0),
                    half_kelly=kelly.get("half_kelly", 0.0),
                    capped_kelly=kelly.get("capped_kelly", 0.0),
                    scale_factor=dd.get("scale_factor", 1.0),
                    position_size_pct=fr.get("position_size_pct", 0.0),
                    position_size_usd=fr.get("position_size_usd", 0.0),
                    report_json=json.dumps(report_dict, ensure_ascii=False),
                )
            )

    def list_risk_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        """Son `limit` adet risk raporunu yeni→eski sırasıyla döndür."""
        with self.session() as s:
            rows = list(
                s.scalars(
                    select(RiskReportRow).order_by(RiskReportRow.created_at.desc()).limit(limit)
                )
            )
            return [
                {
                    "report_id": r.report_id,
                    "backtest_id": r.backtest_id,
                    "strategy_name": r.strategy_name,
                    "n_trades": r.n_trades,
                    "win_rate": r.win_rate,
                    "half_kelly": r.half_kelly,
                    "capped_kelly": r.capped_kelly,
                    "scale_factor": r.scale_factor,
                    "position_size_pct": r.position_size_pct,
                    "position_size_usd": r.position_size_usd,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def get_risk_report(self, report_id: str) -> dict | None:
        """Tam rapor JSON'unu döndür (yoksa None)."""
        with self.session() as s:
            row = s.get(RiskReportRow, report_id)
            if row is None:
                return None
            try:
                return json.loads(row.report_json)
            except (json.JSONDecodeError, TypeError):
                return None

    def save_arxiv_query(
        self, query_id: str, query: str, max_results: int = 5, auto_ingest: bool = True
    ) -> None:
        with self.session() as s:
            s.merge(
                ArxivSavedQuery(
                    query_id=query_id,
                    query=query,
                    max_results=max_results,
                    auto_ingest=1 if auto_ingest else 0,
                )
            )

    def list_arxiv_saved_queries(self) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = list(
                s.scalars(select(ArxivSavedQuery).order_by(ArxivSavedQuery.created_at.desc()))
            )
            return [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "max_results": r.max_results,
                    "auto_ingest": bool(r.auto_ingest),
                    "run_count": r.run_count,
                    "last_run_at": r.last_run_at,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def delete_arxiv_query(self, query_id: str) -> bool:
        with self.session() as s:
            row = s.get(ArxivSavedQuery, query_id)
            if row is None:
                return False
            s.delete(row)
            return True

    def mark_arxiv_query_ran(self, query_id: str) -> None:
        with self.session() as s:
            row = s.get(ArxivSavedQuery, query_id)
            if row:
                row.run_count += 1
                row.last_run_at = _utcnow()

    def list_training_examples(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = list(
                s.scalars(
                    select(TrainingExample).order_by(TrainingExample.created_at.desc()).limit(limit)
                )
            )
            return [
                {
                    "example_id": r.example_id,
                    "source_paper_id": r.source_paper_id,
                    "example_type": r.example_type,
                    "instruction": r.instruction,
                    "input_text": r.input_text,
                    "output_text": r.output_text,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def delete_training_example(self, example_id: str) -> bool:
        with self.session() as s:
            row = s.get(TrainingExample, example_id)
            if row is None:
                return False
            s.delete(row)
            return True

    # ---------- formüller ----------
    def save_formula(self, **fields: Any) -> None:
        with self.session() as s:
            s.merge(Formula(**fields))

    def list_formulas(self, paper_id: str | None = None) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(Formula)
            if paper_id:
                q = q.where(Formula.paper_id == paper_id)
            rows = list(s.scalars(q.order_by(Formula.name)))
            return [
                {
                    "formula_id": r.formula_id,
                    "paper_id": r.paper_id,
                    "name": r.name,
                    "latex": r.latex,
                    "plain": r.plain,
                    "description": r.description,
                    "variables": json.loads(r.variables_json or "{}"),
                    "category": r.category,
                }
                for r in rows
            ]

    def formula_exists(self, paper_id: str, name: str) -> bool:
        with self.session() as s:
            return (
                s.scalar(
                    select(Formula.formula_id)
                    .where(Formula.paper_id == paper_id, Formula.name == name)
                    .limit(1)
                )
                is not None
            )

    # ---------- kavram grafiği ----------
    def save_concept_link(self, **fields: Any) -> None:
        with self.session() as s:
            s.add(ConceptLink(**fields))

    def delete_concept_links_for_paper(self, paper_id: str) -> int:
        """Bir makaleye ait kavram bağlantılarını sil (idempotent yeniden-inşa için).

        build_from_papers her ingest'te tüm makaleler üzerinde çalışıp link eklediğinden,
        silmeden tekrar ekleme link tablosunu tekrar-kenarlarla şişirir (CLAUDE.md
        'ingestion idempotent' ihlali). Yeniden inşadan önce çağrılır.
        """
        with self.session() as s:
            res = s.execute(delete(ConceptLink).where(ConceptLink.source_paper_id == paper_id))
            return int(getattr(res, "rowcount", 0) or 0)

    def list_concept_links(self, concept: str | None = None) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(ConceptLink)
            if concept:
                q = q.where(
                    (ConceptLink.from_concept == concept) | (ConceptLink.to_concept == concept)
                )
            rows = list(s.scalars(q))
            return [
                {
                    "from_concept": r.from_concept,
                    "relation": r.relation,
                    "to_concept": r.to_concept,
                    "source_paper_id": r.source_paper_id,
                }
                for r in rows
            ]

    # ---------- araştırma oturumları ----------
    def save_research_session(self, **fields: Any) -> None:
        with self.session() as s:
            s.merge(ResearchSession(**fields))

    def get_research_session(self, session_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.get(ResearchSession, session_id)
            if row is None:
                return None
            return self._session_to_dict(row)

    def list_research_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = list(
                s.scalars(
                    select(ResearchSession).order_by(ResearchSession.created_at.desc()).limit(limit)
                )
            )
            return [self._session_to_dict(r) for r in rows]

    def _session_to_dict(self, r: ResearchSession) -> dict[str, Any]:
        return {
            "session_id": r.session_id,
            "question": r.question,
            "iteration": r.iteration,
            "parent_session_id": r.parent_session_id,
            "source_paper_ids": json.loads(r.source_paper_ids_json or "[]"),
            "synthesis_reasoning": r.synthesis_reasoning,
            "proposed_indicator": json.loads(r.proposed_indicator_json or "null"),
            "strategy_ir": json.loads(r.strategy_ir_json or "null"),
            "backtest_result": json.loads(r.backtest_result_json or "null"),
            "verdict": r.verdict,
            "reflection": r.reflection,
            "improvement_notes": r.improvement_notes,
            "status": r.status,
            "created_at": r.created_at,
        }

    # ---- reward_signals ----

    def save_reward_signal(
        self,
        session_id: str,
        criteria: Any,  # RewardCriteria — circular import önlemek için Any
        raw_metrics: dict | None = None,
    ) -> str:
        import hashlib

        signal_id = "rs_" + hashlib.md5(session_id.encode()).hexdigest()[:16]
        with self.session() as s:
            s.merge(
                RewardSignal(
                    signal_id=signal_id,
                    session_id=session_id,
                    composite_score=criteria.composite,
                    label=criteria.label,
                    execution_ok=criteria.execution_ok,
                    trade_count_ok=criteria.trade_count_ok,
                    sharpe_ok=criteria.sharpe_ok,
                    drawdown_ok=criteria.drawdown_ok,
                    return_ok=criteria.return_ok,
                    win_rate_ok=criteria.win_rate_ok,
                    notes_json=json.dumps(criteria.notes, ensure_ascii=False),
                    raw_metrics_json=json.dumps(raw_metrics or {}, ensure_ascii=False),
                )
            )
        return signal_id

    def list_reward_signals(
        self, label: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(RewardSignal).order_by(RewardSignal.composite_score.desc())
            if label:
                q = q.where(RewardSignal.label == label)
            rows = list(s.scalars(q.limit(limit)))
            return [
                {
                    "signal_id": r.signal_id,
                    "session_id": r.session_id,
                    "composite_score": r.composite_score,
                    "label": r.label,
                    "execution_ok": r.execution_ok,
                    "trade_count_ok": r.trade_count_ok,
                    "sharpe_ok": r.sharpe_ok,
                    "drawdown_ok": r.drawdown_ok,
                    "return_ok": r.return_ok,
                    "win_rate_ok": r.win_rate_ok,
                    "notes": json.loads(r.notes_json),
                    "raw_metrics": json.loads(r.raw_metrics_json),
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    def get_reward_signal(self, session_id: str) -> dict[str, Any] | None:
        with self.session() as s:
            row = s.scalars(
                select(RewardSignal).where(RewardSignal.session_id == session_id)
            ).first()
            if not row:
                return None
            return {
                "signal_id": row.signal_id,
                "session_id": row.session_id,
                "composite_score": row.composite_score,
                "label": row.label,
                "notes": json.loads(row.notes_json),
                "raw_metrics": json.loads(row.raw_metrics_json),
            }

    # ---- tool_use_examples ----

    def save_tool_use_example(
        self,
        example_id: str,
        session_id: str,
        question: str,
        step_index: int,
        step_type: str,
        content: str,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        tool_output: dict | None = None,
        verdict: str | None = None,
    ) -> None:
        with self.session() as s:
            s.merge(
                ToolUseExample(
                    example_id=example_id,
                    session_id=session_id,
                    question=question,
                    step_index=step_index,
                    step_type=step_type,
                    tool_name=tool_name,
                    tool_input_json=json.dumps(tool_input or {}, ensure_ascii=False),
                    tool_output_json=json.dumps(tool_output or {}, ensure_ascii=False),
                    content=content,
                    verdict=verdict,
                )
            )

    def list_tool_use_examples(
        self, session_id: str | None = None, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self.session() as s:
            q = select(ToolUseExample).order_by(
                ToolUseExample.session_id, ToolUseExample.step_index
            )
            if session_id:
                q = q.where(ToolUseExample.session_id == session_id)
            rows = list(s.scalars(q.limit(limit)))
            return [
                {
                    "example_id": r.example_id,
                    "session_id": r.session_id,
                    "question": r.question,
                    "step_index": r.step_index,
                    "step_type": r.step_type,
                    "tool_name": r.tool_name,
                    "tool_input": json.loads(r.tool_input_json),
                    "tool_output": json.loads(r.tool_output_json),
                    "content": r.content,
                    "verdict": r.verdict,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    # ------------------------------------------------------------------ eval history

    def save_eval_history(
        self,
        adapter_name: str,
        eval_set: str,
        pass_rate: float,
        total_items: int = 0,
        passed_items: int = 0,
    ) -> None:
        with self.session() as s:
            s.add(
                EvalHistory(
                    adapter_name=adapter_name,
                    eval_set=eval_set,
                    pass_rate=pass_rate,
                    total_items=total_items,
                    passed_items=passed_items,
                )
            )

    def list_eval_history(self, limit: int = 200) -> list[dict[str, Any]]:
        with self.session() as s:
            rows = list(s.scalars(select(EvalHistory).order_by(EvalHistory.scored_at).limit(limit)))
            return [
                {
                    "adapter_name": r.adapter_name,
                    "eval_set": r.eval_set,
                    "pass_rate": r.pass_rate,
                    "total_items": r.total_items,
                    "passed_items": r.passed_items,
                    "scored_at": r.scored_at,
                }
                for r in rows
            ]

    def list_card_growth(self) -> list[dict[str, Any]]:
        """Günlük onaylı kart sayısı — öğrenme grafiği için."""
        with self.session() as s:
            rows = list(
                s.execute(
                    text(
                        """
                        SELECT date(created_at) AS day, COUNT(*) AS cnt
                        FROM knowledge_cards
                        WHERE review_status = 'approved'
                        GROUP BY day
                        ORDER BY day
                        """
                    )
                )
            )
            cumulative, result = 0, []
            for day, cnt in rows:
                cumulative += cnt
                result.append({"day": day, "daily": cnt, "cumulative": cumulative})
            return result


if __name__ == "__main__":  # pragma: no cover
    store = SqliteStore()
    print(f"SQLite hazır: {store.db_path}")
    print("Tablolar:", ", ".join(Base.metadata.tables))
