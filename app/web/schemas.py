"""Web API için Pydantic şemaları (girdi doğrulama + tipli yanıtlar)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StatusResponse(BaseModel):
    llm_model: str
    llm_backend: str = "auto"  # ayarlanan backend ("ollama"/"openai"/"auto")
    active_backend: str = "none"  # gerçekte kullanılan backend
    ollama_available: bool
    embedding_mode: str
    n_papers: int
    n_chunks: int
    max_upload_mb: int = 100


class VersionResponse(BaseModel):
    """Sürüm/sapma rozeti — origin/main'e göre (salt-okuma, offline)."""

    git: bool
    branch: str | None = None
    head: str | None = None
    origin_main: str | None = None
    behind: int = 0
    ahead: int = 0
    on_main: bool = False
    converged: bool = False
    last_update: str | None = None


class PaperOut(BaseModel):
    paper_id: str
    title: str | None = None
    year: str | None = None
    authors: str | None = None
    n_chunks: int = 0
    has_card: bool = False


class IngestResponse(BaseModel):
    ingested: int
    skipped: int
    papers: list[PaperOut] = Field(default_factory=list)
    message: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    adapter_version: str | None = None  # belirtilirse MLX adapter ile yanıtla


class SourceOut(BaseModel):
    paper_id: str
    chunk_id: str
    title: str | None = None
    page: int | None = None
    distance: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut] = Field(default_factory=list)
    llm_used: bool
    embedding_mode: str
    adapter_used: str | None = None  # kullanılan adapter versiyonu


# ---------- LoRA sohbet (eğitilen adapter ile lokal, PEFT — Ollama'sız) ----------
class LoraChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    adapter: str | None = None  # boş/None → yalnız base model
    max_tokens: int = Field(default=256, ge=16, le=1024)


class LoraChatResponse(BaseModel):
    answer: str
    adapter: str
    base_model: str


# ---------- Model değerlendirme ----------
class EvalSetOut(BaseModel):
    name: str
    path: str
    n_items: int


class EvalRunRequest(BaseModel):
    eval_set: str  # evals/ içindeki .jsonl dosya adı (uzantısız)
    adapter_version: str | None = None


class EvalResultRow(BaseModel):
    question: str
    answer: str
    flags: list[str]


class EvalRunResponse(BaseModel):
    eval_set: str
    model: str
    adapter_version: str | None = None
    score: float
    n_items: int
    total_flags: int
    rows: list[EvalResultRow]


# ---------- Araştırma (Research) ----------
class FormulaOut(BaseModel):
    formula_id: str
    paper_id: str
    name: str
    latex: str | None = None
    plain: str | None = None
    description: str | None = None
    variables: dict = Field(default_factory=dict)
    category: str | None = None


class ConceptLinkOut(BaseModel):
    from_concept: str
    relation: str
    to_concept: str
    source_paper_id: str | None = None


class ResearchRequest(BaseModel):
    question: str = Field(min_length=10, max_length=1000)
    max_iterations: int = Field(default=2, ge=1, le=5)
    paper_ids: list[str] | None = None


class ResearchIterationOut(BaseModel):
    session_id: str
    iteration: int
    indicator_name: str
    verdict: str
    reasons: list[str]
    metrics: dict
    reflection: str | None = None
    improvement_notes: str | None = None
    composition: dict | None = None  # L5 kompozisyon kapısı (candidate/rejected + gates)


class ResearchRunResponse(BaseModel):
    question: str
    iterations: list[ResearchIterationOut]
    final_verdict: str
    best_session_id: str | None = None
    summary: str


class ResearchSessionOut(BaseModel):
    session_id: str
    question: str
    iteration: int
    verdict: str | None = None
    indicator_name: str | None = None
    status: str
    created_at: str


class ChainDatasetResponse(BaseModel):
    n_records: int
    output_path: str
    content_hash: str


class BacktestRequest(BaseModel):
    # Boşsa örnek strateji + sentetik veri kullanılır.
    use_synthetic: bool = True
    n_bars: int = Field(default=2000, ge=200, le=20000)
    seed: int = Field(default=42, ge=0, le=10_000_000)
    strategy_ir: dict | None = None


class BacktestResponse(BaseModel):
    strategy_name: str
    metrics: dict
    verdict: str
    reasons: list[str]
    backtest_id: str | None = None
    data_source: str | None = None
    n_bars: int | None = None


class CardResponse(BaseModel):
    paper_id: str
    card: dict
    message: str


# ---------- Eğitim ----------
class AdapterOut(BaseModel):
    version: str
    base_model: str
    adapter_path: str
    created_at: str
    notes: str | None = None


class TrainingStatusResponse(BaseModel):
    n_examples: int
    adapters: list[AdapterOut]


class DatasetBuildResponse(BaseModel):
    n_train: int
    n_valid: int
    content_hash: str
    message: str


class TrainDryRunRequest(BaseModel):
    base_model: str = ""  # boş → sunucu backend'e göre seçer (PEFT: 4B brain, MLX: mlx_base_model)
    iterations: int = Field(default=300, ge=50, le=5000)
    batch_size: int = Field(default=2, ge=1, le=16)
    learning_rate: float = Field(default=1e-4, gt=0)
    num_layers: int = Field(default=8, ge=1, le=64)


class TrainDryRunResponse(BaseModel):
    command: str
    n_train: int
    n_valid: int
    content_hash: str
    message: str


# ---------- Training run (web UI) ----------
class TrainingStartRequest(BaseModel):
    base_model: str = ""  # boş → sunucu backend'e göre seçer (PEFT: 4B brain, MLX: mlx_base_model)
    adapter_name: str = "hektor_lora"
    iterations: int = 500
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_layers: int = 8


class TrainingStartResponse(BaseModel):
    ok: bool
    message: str
    # Phase 4D-1: web eğitimi de CLI ile AYNI taze-onay kapısından geçer.
    # status: started | needs_approval | blocked | error
    status: str = "started"
    approval_id: str | None = None
    approve_command: str | None = None


# ---------- Hipotez backtest ----------
class HypothesisResult(BaseModel):
    hypothesis: str
    strategy_name: str
    verdict: str
    reasons: list[str]
    metrics: dict


class HypothesisBacktestResponse(BaseModel):
    paper_id: str
    n_hypotheses: int
    results: list[HypothesisResult]


# ---------- Backtest geçmişi ----------
class BacktestRecord(BaseModel):
    backtest_id: str
    strategy_name: str
    market: str | None = None
    timeframe: str | None = None
    data_file: str | None = None
    n_trades: int = 0
    total_return_pct: float = 0.0
    sharpe: float | None = None
    max_drawdown_pct: float | None = None
    win_rate_pct: float | None = None
    verdict: str | None = None
    notes: str | None = None
    created_at: str


class BacktestHistoryResponse(BaseModel):
    records: list[BacktestRecord]
    total: int


# ---------- Eğitim örneği ----------
class TrainingExampleOut(BaseModel):
    example_id: str
    source_paper_id: str | None = None
    example_type: str
    instruction: str
    input_text: str
    output_text: str
    created_at: str


class TrainingExamplesResponse(BaseModel):
    examples: list[TrainingExampleOut]
    total: int


# ---------- Toplu kart üretimi ----------
class BatchCardResult(BaseModel):
    paper_id: str
    title: str | None = None
    status: str  # ok | skip | error
    message: str


class BatchCardResponse(BaseModel):
    produced: int
    skipped: int
    errors: int
    results: list[BatchCardResult]


class CrossSynthesisResponse(BaseModel):
    produced: int
    message: str


class BatchScoreResult(BaseModel):
    paper_id: str
    title: str | None = None
    status: str  # ok | skip | error
    score: float | None = None
    message: str


class BatchScoreResponse(BaseModel):
    computed: int
    skipped: int
    errors: int
    results: list[BatchScoreResult]


# ---------- Kart Onay (Curriculum) ----------
class CardReviewOut(BaseModel):
    card_id: str
    paper_id: str
    model: str
    trust_level: str
    review_status: str
    lora_eligible: int
    difficulty: float
    stage: str
    created_at: str
    title: str | None = None
    main_claim: str = ""


class PendingCardsResponse(BaseModel):
    cards: list[CardReviewOut]
    total: int


class ApproveCardResponse(BaseModel):
    card_id: str
    status: str  # "approved" | "rejected" | "not_found" | "empty"
    message: str


class ApprovedCardsResponse(BaseModel):
    cards: list[CardReviewOut]
    total: int
    difficulty_min: float
    difficulty_max: float


# ---------- Hektor Package (Entropia export) ----------
class PackageCodeOut(BaseModel):
    pine: str
    python: str


class PackageExportResponse(BaseModel):
    name: str
    version: str
    type: str
    source: str
    created_at: str
    backtest_verdict: str | None = None
    backtest_metrics: dict = Field(default_factory=dict)
    code: PackageCodeOut


# ---------- TradingView Pine export ----------
class PineExportResponse(BaseModel):
    backtest_id: str
    strategy_name: str
    market: str
    timeframe: str
    pine_code: str


# ---------- Risk manager ----------
class KellyOut(BaseModel):
    win_rate: float
    avg_win: float
    avg_loss: float
    odds: float
    full_kelly: float
    half_kelly: float
    quarter_kelly: float
    capped_kelly: float


class DrawdownScaleOut(BaseModel):
    current_drawdown_pct: float
    max_allowed_pct: float
    scale_factor: float
    in_drawdown_zone: bool


class FixedRiskOut(BaseModel):
    equity: float
    risk_per_trade_pct: float
    stop_distance_pct: float
    position_size_pct: float
    position_size_usd: float


class RiskReportResponse(BaseModel):
    strategy_name: str
    n_trades: int
    kelly: KellyOut
    drawdown_scale: DrawdownScaleOut
    fixed_risk: FixedRiskOut
    warnings: list[str] = Field(default_factory=list)
    recommendation: str
    report_id: str | None = None


class RiskReportRecord(BaseModel):
    report_id: str
    backtest_id: str
    strategy_name: str
    n_trades: int
    win_rate: float
    half_kelly: float
    capped_kelly: float
    scale_factor: float
    position_size_pct: float
    position_size_usd: float
    created_at: str


class RiskReportListResponse(BaseModel):
    reports: list[RiskReportRecord]
    total: int


# ---------- arXiv ----------
class ArxivEntryOut(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    pdf_url: str = ""
    published: str = ""


class ArxivSearchResponse(BaseModel):
    query: str
    results: list[ArxivEntryOut]
    total: int


class ArxivSavedQueryIn(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    max_results: int = Field(default=5, ge=1, le=20)
    auto_ingest: bool = True


class ArxivSavedQueryOut(BaseModel):
    query_id: str
    query: str
    max_results: int
    auto_ingest: bool
    run_count: int
    last_run_at: str | None
    created_at: str


class ArxivSavedQueryListResponse(BaseModel):
    queries: list[ArxivSavedQueryOut]
    total: int


class ArxivFetchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300)
    max_results: int = Field(default=5, ge=1, le=20)
    auto_ingest: bool = True


class ArxivFetchResult(BaseModel):
    arxiv_id: str
    title: str
    skipped: bool


class ArxivFetchResponse(BaseModel):
    query: str
    fetched: int
    skipped: int
    results: list[ArxivFetchResult]
    ingested: int = 0
    message: str


# ---------- RLM Controller (çok-adımlı kaynaklı cevap) ----------
class RlmAnswerRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    paper_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    max_rounds: int | None = Field(default=None, ge=1, le=5)


class RlmAnswerResponse(BaseModel):
    run_id: str
    query: str
    task_type: str
    status: str  # answered / answered_with_limitation / abstained / no_llm
    final_answer: str
    final_confidence: float
    confidence_level: str
    evidence_score: float
    retrieval_rounds: int
    n_sources: int
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)


class RlmRunOut(BaseModel):
    run_id: str
    user_query: str
    task_type: str
    status: str
    final_confidence: float
    evidence_score: float
    created_at: str
