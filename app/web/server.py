"""Hektor Trader AI — güvenlikli yerel web arayüzü (FastAPI).

Mevcut çekirdek motoru (ingestion, RAG, backtest) saran ince bir katmandır.
Motoru YENİDEN YAZMAZ; yalnız HTTP üzerinden erişilebilir kılar.

Çalıştır:
    hektor-web                     # veya: uvicorn app.web.server:app
Varsayılan: http://127.0.0.1:8765 (yalnız localhost).
"""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.config import configure_logging, get_settings
from app.web import security
from app.web.schemas import (
    AdapterOut,
    ApproveCardResponse,
    ApprovedCardsResponse,
    ArxivEntryOut,
    ArxivFetchRequest,
    ArxivFetchResponse,
    ArxivFetchResult,
    ArxivSavedQueryIn,
    ArxivSavedQueryListResponse,
    ArxivSavedQueryOut,
    ArxivSearchResponse,
    AskRequest,
    AskResponse,
    BacktestHistoryResponse,
    BacktestRecord,
    BacktestRequest,
    BacktestResponse,
    BatchCardResponse,
    BatchCardResult,
    BatchScoreResponse,
    BatchScoreResult,
    CardResponse,
    CardReviewOut,
    ChainDatasetResponse,
    ConceptLinkOut,
    CrossSynthesisResponse,
    DatasetBuildResponse,
    DrawdownScaleOut,
    EvalResultRow,
    EvalRunRequest,
    EvalRunResponse,
    EvalSetOut,
    FixedRiskOut,
    FormulaOut,
    HypothesisBacktestResponse,
    HypothesisResult,
    IngestResponse,
    KellyOut,
    LoraChatRequest,
    LoraChatResponse,
    PackageCodeOut,
    PackageExportResponse,
    PaperOut,
    PendingCardsResponse,
    PineExportResponse,
    ResearchIterationOut,
    ResearchRequest,
    ResearchRunResponse,
    ResearchSessionOut,
    RiskReportListResponse,
    RiskReportRecord,
    RiskReportResponse,
    RlmAnswerRequest,
    RlmAnswerResponse,
    RlmRunOut,
    SourceOut,
    StatusResponse,
    TrainDryRunRequest,
    TrainDryRunResponse,
    TrainingExampleOut,
    TrainingExamplesResponse,
    TrainingStartRequest,
    TrainingStartResponse,
    TrainingStatusResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_settings = get_settings()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import asyncio as _asyncio
    import os as _os

    configure_logging()
    get_settings().ensure_dirs()
    # Dashboard endpoint'leri worker thread'lerde paralel yuklenmeden once NumPy'nin
    # ilk import'unu tek thread'de tamamla. Aksi halde Windows'ta kismi import ve
    # 0xC0000005 proses cokmesi gorulebiliyor.
    import numpy as _numpy

    logger.debug("NumPy preload tamamlandi: %s", _numpy.__version__)
    logger.info("Hektor web başladı — host=%s port=%s", _settings.web_host, _settings.web_port)
    # api_token boşsa auth KAPALIDIR — bu sessiz kalmamalı (scope izolasyonu da bu
    # modda yalnız derinlemesine savunmadır, kriptografik sınır değil).
    security.warn_if_auth_disabled()
    # Phase 2 startup sweep: crash sonrası 'running' kalan bayat koşuları temizle.
    with suppress(Exception):
        from app.agents.runtime.tracker import cancel_stale_running_agent_runs

        swept = cancel_stale_running_agent_runs()
        if swept:
            logger.info("Startup sweep: %d bayat 'running' koşu iptal edildi", len(swept))
    from app.lora.auto_pipeline import get_auto_pipeline
    from app.monitoring.self_heal import get_self_healer
    from app.orchestration.unattended_supervisor import get_unattended_supervisor

    isolate_chroma = _os.environ.get("HEKTOR_WEB_ISOLATE_CHROMA") == "1"

    # Arka plan döngüleri tek bayrakla kapatılabilir (HEKTOR_BACKGROUND_LOOPS_ENABLED).
    # Testler bunu kapatır: TestClient lifespan'i tetiklediğinde unattended supervisor
    # gerçek bir abonelik motoru doğurmaya çalışıyordu (kota + Kural 8).
    if not _settings.background_loops_enabled:
        logger.info("Arka plan döngüleri KAPALI (background_loops_enabled=false)")
    else:
        _bg_task = _asyncio.create_task(get_auto_pipeline().background_loop())
        _bg_task.add_done_callback(lambda _t: None)
        # RAG öğrenme döngüsü (sunucu-taraflı; varsayılan KAPALI — yalnız web'den açılınca çalışır).
        if not isolate_chroma:
            from app.research.rag_learning_loop import get_rag_loop

            _rag_task = _asyncio.create_task(get_rag_loop().background_loop())
            _rag_task.add_done_callback(lambda _t: None)
        _heal_task = _asyncio.create_task(get_self_healer().background_loop())
        _heal_task.add_done_callback(lambda _t: None)
        _unattended_task = _asyncio.create_task(get_unattended_supervisor().background_loop())
        _unattended_task.add_done_callback(lambda _t: None)
    # BM25 warm-up: hibrit/router/rrf AÇIKSA korpus indeksini ARKA PLAN thread'inde önceden
    # kur (~170s, tek-seferlik) → ilk lexical sorgu 170s soğuk-başlatmaya takılmaz. build-lock
    # ile thundering-herd yok. Dense-only'de gereksiz → atlanır. Sunucu açılışını bloklamaz.
    with suppress(Exception):
        from app.config import get_settings as _gs

        _s = _gs()
        if not isolate_chroma and (_s.rag_router or _s.rag_hybrid or _s.rag_rrf or _s.rag_graph):
            import threading as _threading

            def _warm_bm25() -> None:
                with suppress(Exception):
                    from app.memory.bm25_corpus import get_corpus_bm25

                    get_corpus_bm25()  # SQLite'tan kurar + cache'ler (build-lock korumalı)

            _threading.Thread(target=_warm_bm25, name="bm25-warmup", daemon=True).start()
    yield


# Keşif yüzeyi: `/api/docs` ve `/api/openapi.json` FastAPI'nin yerleşik uçlarıdır ve
# route-başına `api_auth` bağımlılığı ALMAZ → kimlik doğrulamasızdırlar. Yerel modda
# (token boş) bu zararsız ve geliştirici için faydalıdır. Ama token ATANDIYSA amaç ağa
# açmaktır; o modda kimliksiz bir istemcinin tüm route/şema envanterini okuyabilmesi
# gereksiz bir keşif yüzeyidir → kapatılır.
#
# Kırılma riski yok (doğrulandı): web arayüzü bu uçlara hiç başvurmaz ve MCP sunucusu
# spec'i HTTP'den değil IN-PROCESS üretir (`hektor_app.openapi()`).
# NOT: bu bir derinlemesine-savunma önlemidir, erişim kontrolü DEĞİL — uçların kendisi
# zaten `api_auth` ile korunur; şema gizlemek onları korumaz.
_expose_schema = not _settings.api_token.strip()

app = FastAPI(
    title="Hektor Trader AI",
    description="Yerel-öncelikli trading araştırma sistemi — web arayüzü.",
    version="0.1.0",
    docs_url="/api/docs" if _expose_schema else None,
    openapi_url="/api/openapi.json" if _expose_schema else None,
    lifespan=_lifespan,
)

_rate_limiter = security.RateLimiter(_settings.rate_limit_per_min)
# Yükleme uçlarına ayrı, daha sıkı limit (ağ DoS / disk doldurma).
_upload_rate_limiter = security.RateLimiter(_settings.upload_rate_limit_per_min)
_UPLOAD_PATHS = ("/api/papers/upload", "/api/backtest/csv")

# Host-header saldırısına karşı — yalnız trusted_hosts ayarlıysa (ağa açarken).
_trusted_hosts = [h.strip() for h in _settings.trusted_hosts.split(",") if h.strip()]
if _trusted_hosts:
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_trusted_hosts)

# CORS: `cors_origins` ayarı vardı ama middleware hiç kurulmamıştı — ayar sahte bir
# güvenlik güvencesi veriyordu. Frontend aynı origin'den sunulduğu için kısıtlı liste
# normal kullanımı etkilemez; çapraz-origin API erişimini açıkça sınırlar.
_cors_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
if _cors_origins:
    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def _security_middleware(request: Request, call_next):
    # Hız sınırı yalnız API yollarına
    if request.url.path.startswith("/api/"):
        try:
            _rate_limiter.check(security.client_ip(request))
            if request.url.path in _UPLOAD_PATHS:
                _upload_rate_limiter.check(security.client_ip(request))
        except Exception as exc:  # HTTPException
            from fastapi import HTTPException

            if isinstance(exc, HTTPException):
                resp = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
                for k, v in security.SECURITY_HEADERS.items():
                    resp.headers[k] = v
                return resp
            raise
    response = await call_next(request)
    for k, v in security.SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    if _settings.hsts_enabled:  # yalnız TLS arkasında anlamlı
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=63072000; includeSubDomains"
        )
    return response


# ====================== API ======================
api_auth = Depends(security.require_auth)

# İNSAN-YALNIZ kapı (CLAUDE.md Kural 8): sürücü (driver) scope 403 alır. Hektor kendi
# `claude -p` motorunu doğurduğu için, motorun kendi eğitimini onaylaması / kill-switch'i
# temizlemesi engellenir. `api_auth` ile BİRLİKTE kullanılır (kimlik + scope).
# NOT: bu uçlarda ayrıca `include_in_schema=False` var — bu bir GÜVENLİK KONTROLÜ DEĞİL,
# yalnız OpenAPI/doc gizlemesidir. Erişim kontrolü TAMAMEN `require_human`'dan gelir;
# schema bayrağı kaldırılsa bile koruma sürer (ve tersi doğru DEĞİLDİR).
human_only = Depends(security.require_human)

# AI-brain ek-modül uçları (registry/tools/ingestion/eval) + dashboard sayfası ayrı
# router'da tutulur → server.py minimal dokunulur (çakışma yüzeyi küçük). Salt-okuma/hesap.
from app.web.agent_graph_routes import router as _agent_graph_router  # noqa: E402
from app.web.ai_brain_routes import router as _ai_brain_router  # noqa: E402
from app.web.ai_brain_routes import ui_router as _ai_brain_ui_router  # noqa: E402
from app.web.engines_routes import router as _engines_router  # noqa: E402
from app.web.feedback_routes import router as _feedback_router  # noqa: E402
from app.web.orchestration_routes import router as _orchestration_router  # noqa: E402
from app.web.sentinel_routes import router as _sentinel_router  # noqa: E402

app.include_router(_ai_brain_router)
app.include_router(_ai_brain_ui_router)
app.include_router(_orchestration_router)
app.include_router(_feedback_router)
app.include_router(_sentinel_router)
app.include_router(_agent_graph_router)
app.include_router(_engines_router)


@app.get("/api/status", response_model=StatusResponse, dependencies=[api_auth])
def api_status() -> StatusResponse:
    import os

    from app.brain.local_llm import LocalLLM
    from app.memory.chroma_store import ChromaStore
    from app.memory.embedding_service import EmbeddingService
    from app.memory.sqlite_store import SqliteStore

    s = get_settings()
    store = SqliteStore()
    emb = EmbeddingService()
    llm = LocalLLM()
    if os.environ.get("HEKTOR_WEB_ISOLATE_CHROMA") == "1":
        n_chunks = 0
    else:
        try:
            n_chunks = ChromaStore().count()
        except Exception:
            n_chunks = 0
    return StatusResponse(
        llm_model=s.llm_model,
        # Yapılandırılmış backend TEK: yerel Ollama (bulut sağlayıcı kodu yok).
        # `active_backend` canlılığı söyler: "ollama" (ayakta) veya "none".
        llm_backend="ollama",
        active_backend=llm.active_backend(),
        ollama_available=llm.available(),
        embedding_mode=emb.mode,
        n_papers=len(store.list_papers()),
        n_chunks=n_chunks,
        max_upload_mb=s.max_upload_mb,
    )


@app.get("/api/version", response_model=VersionResponse, dependencies=[api_auth])
def api_version() -> VersionResponse:
    """Sürüm/sapma rozeti — bu makine origin/main'e göre güncel mi? (salt-okuma, offline)."""
    from app.web.version_info import get_version_info

    return VersionResponse(**get_version_info())


@app.get("/api/papers", response_model=list[PaperOut], dependencies=[api_auth])
def api_papers() -> list[PaperOut]:
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    out: list[PaperOut] = []
    for p in store.list_papers():
        out.append(
            PaperOut(
                paper_id=p.paper_id,
                title=p.title,
                year=p.year,
                authors=p.authors,
                n_chunks=len(store.list_chunks(p.paper_id)),
                has_card=store.has_knowledge_card(p.paper_id),
            )
        )
    return out


@app.post("/api/papers/upload", response_model=IngestResponse, dependencies=[api_auth])
async def api_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> IngestResponse:
    """PDF yükle → kaydet → arka planda indeksle (anında yanıt döner).

    Ingestion agir oldugu icin (PDF parse + embedding) BackgroundTasks ile
    ayristiriliyor; HTTP baglantisi beklemeden kapaniyor.
    """
    # Boyut tavanını gövdeyi SINIRSIZ belleğe materyalize etmeden uygula: read(max_bytes+1)
    # en çok o kadar okur → çok-GB gövdede tek-tahsis RAM şişmesi olmaz (ağa açık dağıtımda
    # kaynak-tükenmesi yüzeyini daraltır). Boyut aşımında parse/validate beklemeden 413.
    _max_bytes = get_settings().max_upload_mb * 1024 * 1024
    content = await file.read(_max_bytes + 1)
    if len(content) > _max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Dosya çok büyük (max {get_settings().max_upload_mb} MB)."
        )
    safe_name = security.validate_pdf_upload(file.filename or "", content)

    # Senkron dedup: birebir aynı dosya (aynı file_hash) zaten yüklüyse parse beklemeden
    # hemen bildir. (Aynı makalenin FARKLI bytes'lı kopyası başlık-dedup ile arka planda
    # yakalanır — bkz. PaperIndexer.ingest_one.)
    import hashlib

    from app.memory.sqlite_store import SqliteStore

    file_hash = hashlib.sha256(content).hexdigest()
    existing = SqliteStore().get_paper_by_hash(file_hash)
    if existing is not None:
        return IngestResponse(
            ingested=0,
            skipped=1,
            papers=[],
            message=f"Bu dosya zaten yüklü, atlandı: {existing.title or safe_name}",
        )

    s = get_settings()
    dest = security.safe_destination(s.raw_pdf_dir, safe_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    logger.info("PDF yüklendi: %s (%d bayt)", dest.name, len(content))

    background_tasks.add_task(_ingest_one, dest)
    return IngestResponse(
        ingested=0, skipped=0, papers=[], message=f"{safe_name} alındı, indeksleniyor…"
    )


@app.post("/api/ingest", response_model=IngestResponse, dependencies=[api_auth])
def api_ingest() -> IngestResponse:
    """raw_pdf/ içindeki tüm PDF'leri indeksle (idempotent)."""
    return _ingest_all()


def _ingest_one(path: Path) -> None:
    """Arka planda tek bir PDF'i indeksler (BackgroundTasks için)."""
    from app.ingestion.paper_loader import DiscoveredPaper, compute_file_hash
    from app.memory.paper_indexer import PaperIndexer

    try:
        disc = DiscoveredPaper(path=path, file_hash=compute_file_hash(path))
        PaperIndexer().ingest_one(disc)
        logger.info("Arka plan indeksleme tamamlandi: %s", path.name)
    except Exception:
        logger.exception("Arka plan indeksleme hatasi: %s", path.name)


def _ingest_all() -> IngestResponse:
    from app.memory.paper_indexer import PaperIndexer

    results = PaperIndexer().ingest_directory()
    ingested = sum(1 for r in results if not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    papers = [
        PaperOut(paper_id=r.paper_id, title=r.title, n_chunks=r.n_chunks)
        for r in results
        if not r.skipped
    ]
    return IngestResponse(
        ingested=ingested,
        skipped=skipped,
        papers=papers,
        message=f"{ingested} yeni, {skipped} zaten mevcut.",
    )


@app.post("/api/ask", response_model=AskResponse, dependencies=[api_auth])
def api_ask(req: AskRequest) -> AskResponse:
    from app.brain.rag_answerer import RagAnswerer
    from app.memory.embedding_service import EmbeddingService

    adapter_used: str | None = None

    if req.adapter_version:
        # MLX adapter ile yanıtla (Ollama bypass)
        from app.brain.mlx_llm import MlxLLM, MlxLLMUnavailable
        from app.memory.reranking_retriever import RerankingRetriever
        from app.training.adapter_registry import AdapterRegistry

        entry = AdapterRegistry().get(req.adapter_version)
        if entry is None:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=404, detail=f"Adapter bulunamadı: {req.adapter_version}"
            )

        mlx = MlxLLM(base_model=entry["base_model"], adapter_path=entry["adapter_path"])
        retriever = RerankingRetriever()
        chunks = retriever.retrieve(req.question, top_k=req.top_k)
        if not chunks:
            # Kaynak yoksa LLM'i HİÇ çağırma — uydurma/halüsinasyon yasak (CLAUDE.md kural 7).
            # (Varsayılan RagAnswerer yolu da aynısını yapar; adapter yolu da yapmalı.)
            return AskResponse(
                answer="İlgili kaynak bulunamadı; bu soruyu yanıtlayacak indekslenmiş içerik yok.",
                sources=[],
                llm_used=False,
                embedding_mode=EmbeddingService().mode,
                adapter_used=None,
            )
        context = "\n\n".join(c.text[:600] for c in chunks)
        prompt = (
            f"Aşağıdaki akademik bağlamı kullanarak soruyu yanıtla.\n\n"
            f"BAĞLAM:\n{context}\n\n"
            f"SORU: {req.question}\n\nYANIT:"
        )
        try:
            answer = mlx.generate(prompt)
            llm_used = True
            adapter_used = req.adapter_version
        except MlxLLMUnavailable as exc:
            answer = f"[MLX adapter kullanılamadı: {exc}]"
            llm_used = False
            adapter_used = None  # adapter başarısız → yanıtta adapter iddia etme

        sources = [
            SourceOut(
                paper_id=c.paper_id,
                chunk_id=c.chunk_id,
                title=c.title,
                page=c.page_number,
                distance=c.distance,
            )
            for c in chunks
        ]
        return AskResponse(
            answer=answer,
            sources=sources,
            llm_used=llm_used,
            embedding_mode=EmbeddingService().mode,
            adapter_used=adapter_used,
        )

    try:
        ans = RagAnswerer().answer(req.question, top_k=req.top_k)
    except Exception as exc:  # embedding/Chroma/Ollama hatası → 503 (500 yerine)
        raise HTTPException(status_code=503, detail=f"RAG yanıtlayıcı hatası: {exc}") from exc
    sources = [
        SourceOut(
            paper_id=c.paper_id,
            chunk_id=c.chunk_id,
            title=c.title,
            page=c.page_number,
            distance=c.distance,
        )
        for c in ans.sources
    ]
    return AskResponse(
        answer=ans.answer,
        sources=sources,
        llm_used=ans.llm_used,
        embedding_mode=EmbeddingService().mode,
    )


@app.post("/api/rlm/answer", response_model=RlmAnswerResponse, dependencies=[api_auth])
def api_rlm_answer(req: RlmAnswerRequest) -> RlmAnswerResponse:
    """RLM Controller: çok-adımlı retrieval + iddia doğrulama + kaynaklı nihai cevap.

    Desteklenmeyen iddia nihai cevaba girmez; eksik kaynakta 'yeterli kaynak yok'
    der (uydurma yasak); trading sorularında yalnız hipotez + uyarı üretir.
    """
    from app.rlm.rlm_controller import RlmController

    try:
        # write_report=False: yanıt tüm alanları zaten döndürüyor; her HTTP isteğinde
        # reports/rlm_runs'a JSON yazmak sınırsız disk büyümesi olurdu (DB logu kalır).
        res = RlmController().answer(
            req.query,
            paper_ids=req.paper_ids,
            top_k=req.top_k,
            max_rounds=req.max_rounds,
            write_report=False,
        )
    except HTTPException:
        raise
    except Exception as exc:  # retrieval/embedding/Chroma hatası → 503
        raise HTTPException(status_code=503, detail=f"RLM controller hatası: {exc}") from exc

    sources = [
        SourceOut(
            paper_id=s["paper_id"],
            chunk_id=s["chunk_id"],
            title=s.get("title"),
            page=s.get("page"),
        )
        for s in res.sources
    ]
    return RlmAnswerResponse(
        run_id=res.run_id,
        query=res.query,
        task_type=res.task_type,
        status=res.status,
        final_answer=res.final_answer,
        final_confidence=res.final_confidence,
        confidence_level=res.confidence_level,
        evidence_score=res.evidence_score,
        retrieval_rounds=res.retrieval_rounds,
        n_sources=res.n_sources,
        supported_claims=res.supported_claims,
        unsupported_claims=res.unsupported_claims,
        contradictions=res.contradictions,
        sources=sources,
    )


@app.get("/api/rlm/runs", dependencies=[api_auth])
def api_rlm_runs(limit: int = 20) -> dict:
    """Son RLM koşularını listele (run_id, görev, durum, kanıt/güven)."""
    from app.rlm.rlm_store import RlmStore

    rows = RlmStore().list_runs(limit=max(1, min(limit, 100)))
    runs = [
        RlmRunOut(
            run_id=r["run_id"],
            user_query=r["user_query"],
            task_type=r["task_type"],
            status=r["status"],
            final_confidence=r["final_confidence"],
            evidence_score=r["evidence_score"],
            created_at=r["created_at"],
        )
        for r in rows
    ]
    return {"runs": [r.model_dump() for r in runs]}


@app.get("/api/rlm/runs/{run_id}", dependencies=[api_auth])
def api_rlm_run_detail(run_id: str) -> dict:
    """Tek bir RLM koşusunun tam detayı: run + adımlar + kanıt + doğrulama."""
    from app.rlm.rlm_store import RlmStore

    store = RlmStore()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"RLM koşusu bulunamadı: {run_id}")
    return {
        "run": run,
        "steps": store.get_steps(run_id),
        "evidence": store.get_evidence(run_id),
        "verification": store.get_verification(run_id),
    }


@app.get("/api/rlm/runs/{run_id}/trajectory", dependencies=[api_auth])
def api_rlm_run_trajectory(run_id: str) -> dict:
    """Bir RLM koşusunun trajektorisi (DB'deki adım izi)."""
    from app.rlm.rlm_store import RlmStore

    store = RlmStore()
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"RLM koşusu bulunamadı: {run_id}")
    return {"run_id": run_id, "steps": store.get_steps(run_id)}


@app.get("/api/rlm/config", dependencies=[api_auth])
def api_rlm_config() -> dict:
    """RLM çalışma-zamanı özeti (salt-okuma; sır YOK).

    Motor tektir ve yereldir (RlmController + Ollama); sağlayıcı seçimi, uzak uç ve
    anahtar yoktur. Dönen alanlar: motor, model, üretim modu, izinli tool'lar, seed.
    """
    from app.rlm.tool_registry import runtime_info

    return {"config": runtime_info()}


@app.get("/api/lora-adapters", dependencies=[api_auth])
def api_lora_adapters() -> dict:
    """Sohbet için kullanılabilir (tam) LoRA adapter'larını listele."""
    from app.web.lora_chat_service import list_adapters

    return {"adapters": list_adapters()}


@app.post("/api/lora-chat", response_model=LoraChatResponse, dependencies=[api_auth])
def api_lora_chat(req: LoraChatRequest) -> LoraChatResponse:
    """Eğitilen LoRA adapter'ı (veya base) ile LOKAL sohbet — PEFT, Ollama'sız.

    AĞIR: CPU'da model yükleme (ilk istek) + üretim dakikalar sürebilir. Senkron `def` →
    FastAPI bunu threadpool'da koşturur, event loop bloklanmaz.
    """
    from app.web.lora_chat_service import chat

    try:
        out = chat(req.question, req.adapter, max_tokens=req.max_tokens)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"PEFT/transformers eksik: {exc}. Kur: uv pip install torch transformers peft",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LoRA sohbet hatası: {exc}") from exc
    return LoraChatResponse(**out)


@app.get("/api/card/{paper_id}", response_model=CardResponse, dependencies=[api_auth])
def api_get_card(paper_id: str) -> CardResponse:
    """Kaydedilmiş bilgi kartını getir (LLM gerektirmez). Yoksa 404."""
    from fastapi import HTTPException

    from app.memory.sqlite_store import SqliteStore

    card = SqliteStore().get_latest_knowledge_card(paper_id)
    if card is None:
        raise HTTPException(status_code=404, detail="Bu makale için henüz kart yok.")
    return CardResponse(paper_id=paper_id, card=card, message="ok")


@app.post("/api/card/{paper_id}", response_model=CardResponse, dependencies=[api_auth])
def api_card(paper_id: str) -> CardResponse:
    from fastapi import HTTPException

    from app.brain.knowledge_card_builder import KnowledgeCardBuilder

    # paper_id beyaz-liste: yalnız bilinen makaleler
    from app.memory.sqlite_store import SqliteStore

    known = {p.paper_id for p in SqliteStore().list_papers()}
    if paper_id not in known:
        raise HTTPException(status_code=404, detail="Makale bulunamadı.")

    try:
        card = KnowledgeCardBuilder().build(paper_id)
        return CardResponse(
            paper_id=paper_id,
            card=card.model_dump() if hasattr(card, "model_dump") else dict(card),
            message="Bilgi kartı üretildi.",
        )
    except Exception as exc:
        logger.warning("Kart üretimi başarısız: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Bilgi kartı üretilemedi (LLM gerekli olabilir).",
        ) from exc


@app.get("/api/papers/{paper_id}/comprehension", dependencies=[api_auth])
def api_comprehension_get(paper_id: str) -> dict:
    """Önbellekteki anlama skoru — henüz hesaplanmamışsa null döner."""
    import json as _json

    from app.memory.sqlite_store import SqliteStore

    row = SqliteStore().get_comprehension_score(paper_id)
    if row is None:
        return {"paper_id": paper_id, "total": None, "computed_at": None}
    # details_json bozuk/yarım kalmışsa (DB bozulması) 500 yerine boş sözlüğe düş.
    try:
        details = _json.loads(row.details_json or "{}")
    except (ValueError, TypeError):
        details = {}
    return {
        "paper_id": paper_id,
        "total": row.total_score,
        "extraction": row.extraction_score,
        "retrieval": row.retrieval_score,
        "llm_verify": row.llm_score,
        "details": details,
        "computed_at": row.computed_at,
    }


@app.post("/api/papers/{paper_id}/comprehension", dependencies=[api_auth])
def api_comprehension_compute(paper_id: str) -> dict:
    """Anlama skorunu (yeniden) hesapla ve kaydet."""

    from fastapi import HTTPException

    from app.memory.sqlite_store import SqliteStore
    from app.verification.comprehension_scorer import ComprehensionScorer

    store = SqliteStore()
    known = {p.paper_id for p in store.list_papers()}
    if paper_id not in known:
        raise HTTPException(status_code=404, detail="Makale bulunamadı.")

    try:
        result = ComprehensionScorer().score(paper_id)
        store.save_comprehension_score(result)
        return {
            "paper_id": paper_id,
            "total": result.total,
            "extraction": result.extraction,
            "retrieval": result.retrieval,
            "llm_verify": result.llm_verify,
            "details": result.details,
            "computed_at": result.computed_at,
        }
    except Exception as exc:
        logger.warning("Anlama skoru hesaplanamadı: %s", exc)
        raise HTTPException(status_code=503, detail=f"Skor hesaplanamadı: {exc}") from exc


@app.get("/api/papers/comprehension/all", dependencies=[api_auth])
def api_comprehension_all() -> dict:
    """Tüm makalelerin anlama skorlarını tek seferde döner → N+1 isteği önler."""
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    scores: dict[str, int] = {}
    for paper in store.list_papers():
        try:
            result = store.get_comprehension_score(paper.paper_id)
            if result is not None:
                scores[paper.paper_id] = round(result.total_score)
        except Exception:
            pass
    return {"scores": scores}


@app.get("/api/rag-mastery", dependencies=[api_auth])
def api_rag_mastery() -> dict:
    """RAG ustalık panosu: kaç makale anlaşıldı/öğrenildi (%), LLM gerektirmez."""
    from app.verification.rag_mastery import compute_rag_mastery

    return compute_rag_mastery()


@app.get("/api/understanding-score", dependencies=[api_auth])
def api_understanding_score(
    seed: int = 0, full: bool = False, with_rag: bool = False, record: bool = False
) -> dict:
    """Objektif anlama skoru — sınav geçme oranı (kaba öz-değerlendirme %'nin yerine).

    Varsayılan: L3+L4 gösterge sınavı (LLM; çevrimdışı → 'insufficient_data', Kural 2).
    ``full=true`` → L5 kompozisyon dahil tam merdiven (L5 deterministik, çevrimdışı bile
    notlanır). ``with_rag=true`` → Taban/L1/L2 için canlı RAG sınavı da. ``record=true`` →
    skoru KALICI yap (understanding_snapshots + JSON rapor).
    """
    from app.verification.exams.understanding_score import (
        score_full_ladder,
        score_indicator_exams,
    )

    score = (
        score_full_ladder(seed=seed, with_rag=with_rag)
        if (full or with_rag)
        else score_indicator_exams(seed=seed)
    )
    out = score.to_dict()
    if record:
        from app.verification.exams.understanding_record import record_understanding

        out["recorded"] = record_understanding(
            score, seed=seed, context={"full": full, "with_rag": with_rag, "source": "web"}
        )
    return out


@app.get("/api/understanding-score/history", dependencies=[api_auth])
def api_understanding_history(limit: int = 20) -> dict:
    """KALICI anlama skoru geçmişi (zaman serisi) — understanding_snapshots.

    `limit` sunucu tarafında kıstırılır: bu uç MCP allow-list'inde olduğu için sınırsız
    bırakılsaydı, motorun tek çağrıyla tüm tabloyu çekebildiği tek liste ucu olurdu
    (komşu uçların hepsi zaten 200'de kıstırıyor).
    """
    from app.verification.exams.understanding_record import load_understanding_history

    return {"history": load_understanding_history(limit=min(max(1, limit), 200))}


@app.get("/api/synthesis/reports", dependencies=[api_auth])
def api_synthesis_reports() -> dict:
    """Üretilmiş sentez makalelerini listele (web'den incelenebilir/indirilebilir)."""
    from app.research.synthesis_paper import list_synthesis_reports

    return {"reports": list_synthesis_reports()}


@app.get("/api/synthesis/reports/{name}", dependencies=[api_auth])
def api_synthesis_report_download(name: str) -> FileResponse:
    """Bir sentez makalesini indir (yalnız reports/synthesis altındaki .md)."""
    from app.research.synthesis_paper import is_safe_report_name, synthesis_reports_dir

    if not is_safe_report_name(name):
        raise HTTPException(status_code=400, detail="Geçersiz dosya adı")
    path = synthesis_reports_dir() / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Makale bulunamadı")
    return FileResponse(path, media_type="text/markdown", filename=name)


@app.post("/api/synthesis/reports/generate", dependencies=[api_auth])
def api_synthesis_report_generate() -> dict:
    """Son araştırma oturumlarından yeni bir sentez makalesi üret."""
    from app.research.synthesis_paper import generate_synthesis_paper

    path = generate_synthesis_paper()
    if path is None:
        return {"ok": False, "message": "Araştırma oturumu yok — önce 'research' çalıştırın."}
    return {"ok": True, "name": path.name}


@app.post(
    "/api/papers/comprehension/batch",
    response_model=BatchScoreResponse,
    dependencies=[api_auth],
)
def api_comprehension_batch(
    skip_existing: bool = False, use_llm: bool = True
) -> BatchScoreResponse:
    """Tüm makaleler için anlama skoru hesapla (kartı olan makaleler için).

    `use_llm=false` → HIZLI mod (yalnız doluluk + RAG precision; LLM atlanır) →
    tüm korpus saniyeler içinde skorlanır.
    """
    from app.memory.sqlite_store import SqliteStore
    from app.verification.comprehension_scorer import ComprehensionScorer

    store = SqliteStore()
    papers = store.list_papers()
    scorer = ComprehensionScorer()
    results: list[BatchScoreResult] = []

    for paper in papers:
        pid = paper.paper_id
        if not store.has_knowledge_card(pid):
            results.append(
                BatchScoreResult(paper_id=pid, title=paper.title, status="skip", message="kart yok")
            )
            continue
        if skip_existing:
            existing = store.get_comprehension_score(pid)
            if existing is not None:
                results.append(
                    BatchScoreResult(
                        paper_id=pid,
                        title=paper.title,
                        status="skip",
                        score=existing.total_score,
                        message="zaten hesaplanmış",
                    )
                )
                continue
        try:
            result = scorer.score(pid, use_llm=use_llm)
            store.save_comprehension_score(result)  # KALICI yap (önceden kaydetmiyordu)
            results.append(
                BatchScoreResult(
                    paper_id=pid,
                    title=paper.title,
                    status="ok",
                    score=result.total,
                    message=f"{result.total:.0f}%",
                )
            )
        except Exception as exc:
            logger.warning("Toplu skor — %s başarısız: %s", pid, exc)
            results.append(
                BatchScoreResult(paper_id=pid, title=paper.title, status="error", message=str(exc))
            )

    computed = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skip")
    errors = sum(1 for r in results if r.status == "error")
    return BatchScoreResponse(computed=computed, skipped=skipped, errors=errors, results=results)


@app.post(
    "/api/synthesis/cross-paper",
    response_model=CrossSynthesisResponse,
    dependencies=[api_auth],
)
def api_cross_paper_synthesis(force: bool = False) -> CrossSynthesisResponse:
    """Farklı makalelerden formülleri birleştirerek LoRA eğitim verisi üret.

    force=True → mevcut sentez örneklerini yeniden üretir.
    """
    from app.research.cross_paper_synthesizer import CrossPaperSynthesizer

    try:
        n = CrossPaperSynthesizer().synthesize_all(force=force)
        return CrossSynthesisResponse(
            produced=n,
            message=(
                f"{n} yeni sentez eğitim örneği üretildi."
                if n
                else "Yeni örnek yok (zaten güncel)."
            ),
        )
    except Exception as exc:
        logger.warning("Çapraz sentez başarısız: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/cards/batch", response_model=BatchCardResponse, dependencies=[api_auth])
def api_cards_batch(skip_existing: bool = True) -> BatchCardResponse:
    """Kartı olmayan tüm makaleler için sırayla bilgi kartı üret (LLM gerekli)."""
    from app.brain.knowledge_card_builder import KnowledgeCardBuilder
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    papers = store.list_papers()
    builder = KnowledgeCardBuilder()
    results: list[BatchCardResult] = []

    for paper in papers:
        pid = paper.paper_id
        if skip_existing and store.has_knowledge_card(pid):
            results.append(
                BatchCardResult(paper_id=pid, title=paper.title, status="skip", message="zaten var")
            )
            continue
        try:
            builder.build(pid)
            results.append(
                BatchCardResult(paper_id=pid, title=paper.title, status="ok", message="üretildi")
            )
        except Exception as exc:
            logger.warning("Toplu kart — %s başarısız: %s", pid, exc)
            results.append(
                BatchCardResult(
                    paper_id=pid, title=paper.title, status="error", message=str(exc)[:120]
                )
            )

    produced = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skip")
    errors = sum(1 for r in results if r.status == "error")
    return BatchCardResponse(produced=produced, skipped=skipped, errors=errors, results=results)


@app.post(
    "/api/card/{paper_id}/backtest",
    response_model=HypothesisBacktestResponse,
    dependencies=[api_auth],
)
def api_card_backtest(paper_id: str) -> HypothesisBacktestResponse:
    """Bilgi kartındaki strateji hipotezlerini sentetik veride backtest et."""
    from fastapi import HTTPException

    from app.memory.sqlite_store import SqliteStore
    from app.trading.backtester import run_backtest
    from app.trading.evaluator import evaluate as eval_strategy
    from app.trading.market_data_loader import generate_synthetic_ohlcv
    from app.trading.strategy_generator import generate_from_hypothesis

    card = SqliteStore().get_latest_knowledge_card(paper_id)
    if card is None:
        raise HTTPException(status_code=404, detail="Bu makale için henüz kart yok.")

    hypotheses: list[str] = card.get("possible_strategy_hypotheses") or []
    if not hypotheses:
        raise HTTPException(status_code=422, detail="Kart test edilebilir hipotez içermiyor.")

    df = generate_synthetic_ohlcv(n=2000, seed=42)
    results: list[HypothesisResult] = []
    for i, hyp in enumerate(hypotheses):
        ir = generate_from_hypothesis(hyp, name=f"hyp_{i + 1}", market="XAUUSD", timeframe="15m")
        bt = run_backtest(df, ir)
        ev = eval_strategy(df, ir)
        results.append(
            HypothesisResult(
                hypothesis=hyp,
                strategy_name=ir.name,
                verdict=ev.verdict,
                reasons=ev.reasons,
                metrics=bt.metrics.to_dict(),
            )
        )

    return HypothesisBacktestResponse(
        paper_id=paper_id,
        n_hypotheses=len(results),
        results=results,
    )


# ---------- Araştırma (Trader Beyin) ----------
@app.get("/api/research/formulas", response_model=list[FormulaOut], dependencies=[api_auth])
def api_formulas(paper_id: str | None = None) -> list[FormulaOut]:
    """Çıkarılmış formülleri listele."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_formulas(paper_id=paper_id)
    return [FormulaOut(**r) for r in rows]


@app.get("/api/research/graph", response_model=list[ConceptLinkOut], dependencies=[api_auth])
def api_concept_graph() -> list[ConceptLinkOut]:
    """Kavram grafiği bağlantılarını listele."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_concept_links()
    return [ConceptLinkOut(**r) for r in rows]


@app.post("/api/research/extract", dependencies=[api_auth])
def api_extract_formulas(paper_id: str | None = None) -> dict:
    """PDF'lerden formül çıkar ve kavram grafiği oluştur (LLM gerekli)."""
    from app.research.concept_graph import ConceptGraph
    from app.research.formula_extractor import FormulaExtractor

    extractor = FormulaExtractor()
    try:
        if paper_id:
            formulas = extractor.extract_from_paper(paper_id)
            n_formulas = len(formulas)
        else:
            results = extractor.extract_from_all_papers()
            n_formulas = sum(len(v) for v in results.values())
        n_links = ConceptGraph().build_from_papers()
    except Exception as exc:  # LLM kapalı/hata → 503 (500 yerine)
        raise HTTPException(
            status_code=503, detail=f"Formül çıkarımı başarısız (LLM gerekli): {exc}"
        ) from exc
    return {
        "n_formulas": n_formulas,
        "n_links": n_links,
        "message": f"{n_formulas} formül, {n_links} bağlantı",
    }


@app.post("/api/research/run", response_model=ResearchRunResponse, dependencies=[api_auth])
def api_research_run(req: ResearchRequest) -> ResearchRunResponse:
    """Agentic araştırma döngüsü: sentezle → backtest → yansıt → iyileştir."""
    from app.research.orchestrator import ResearchOrchestrator

    orchestrator = ResearchOrchestrator(max_iterations=req.max_iterations)
    try:
        result = orchestrator.run(req.question, paper_ids=req.paper_ids)
    except Exception as exc:  # LLM/backtest hatası → 503 (unhandled 500 yerine)
        raise HTTPException(status_code=503, detail=f"Araştırma döngüsü başarısız: {exc}") from exc

    return ResearchRunResponse(
        question=result.question,
        iterations=[
            ResearchIterationOut(
                session_id=it.session_id,
                iteration=it.iteration,
                indicator_name=it.indicator_name,
                verdict=it.verdict,
                reasons=it.reasons,
                metrics=it.metrics,
                reflection=it.reflection,
                improvement_notes=it.improvement_notes,
                composition=it.composition,
            )
            for it in result.iterations
        ],
        final_verdict=result.final_verdict,
        best_session_id=result.best_session_id,
        summary=result.summary(),
    )


@app.get("/api/research/sessions", response_model=list[ResearchSessionOut], dependencies=[api_auth])
def api_research_sessions(limit: int = 30) -> list[ResearchSessionOut]:
    """Kayıtlı araştırma oturumlarını listele."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_research_sessions(limit=min(limit, 200))
    out: list[ResearchSessionOut] = []
    for r in rows:
        ind = r.get("proposed_indicator") or {}
        out.append(
            ResearchSessionOut(
                session_id=r["session_id"],
                question=r["question"],
                iteration=r["iteration"],
                verdict=r.get("verdict"),
                indicator_name=ind.get("indicator_name") if ind else None,
                status=r["status"],
                created_at=r["created_at"],
            )
        )
    return out


@app.post(
    "/api/research/chain-dataset", response_model=ChainDatasetResponse, dependencies=[api_auth]
)
def api_chain_dataset(only_successful: bool = False) -> ChainDatasetResponse:
    """Araştırma zincirlerinden LoRA eğitim verisi üret."""
    from app.research.chain_data_builder import ChainDataBuilder

    result = ChainDataBuilder().build(only_successful=only_successful)
    return ChainDatasetResponse(**result)


# ---------- Eğitim ----------
@app.get("/api/eval/sets", response_model=list[EvalSetOut], dependencies=[api_auth])
def api_eval_sets() -> list[EvalSetOut]:
    """Kullanılabilir eval set listesi (evals/ dizini)."""

    evals_dir = get_settings().eval_sets_dir
    out: list[EvalSetOut] = []
    if evals_dir.exists():
        for p in sorted(evals_dir.glob("*.jsonl")):
            try:
                n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
            except Exception:
                n = 0
            out.append(EvalSetOut(name=p.stem, path=str(p), n_items=n))
    return out


@app.post("/api/eval/run", response_model=EvalRunResponse, dependencies=[api_auth])
def api_eval_run(req: EvalRunRequest) -> EvalRunResponse:
    """Seçili eval setini çalıştır (Ollama gerekli, ~dakikalar sürebilir)."""
    from fastapi import HTTPException

    from app.training.evaluate_model import ModelEvaluator

    evals_dir = get_settings().eval_sets_dir
    # Yol-aşımı koruması: req.eval_set kullanıcı girdisidir. Hedefin evals/
    # içinde kaldığını doğrula (geçersiz/aşımlı ad → 400). bkz. safe_destination.
    eval_path = security.safe_destination(evals_dir, f"{req.eval_set}.jsonl")
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail=f"Eval seti bulunamadı: {req.eval_set}")

    evaluator = ModelEvaluator()
    try:
        results = evaluator.run_eval(eval_path, adapter_version=req.adapter_version)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Eval çalıştırılamadı: {exc}") from exc

    return EvalRunResponse(
        eval_set=results["eval_set"],
        model=results["model"],
        adapter_version=results.get("adapter_version"),
        score=results["score"],
        n_items=results["n_items"],
        total_flags=results["total_flags"],
        rows=[
            EvalResultRow(question=r["q"], answer=r["a"], flags=r["flags"])
            for r in results.get("rows", [])
        ],
    )


@app.get("/api/training/status", response_model=TrainingStatusResponse, dependencies=[api_auth])
def api_training_status() -> TrainingStatusResponse:
    """Eğitim örneği sayısı + kayıtlı adapter listesi."""
    from sqlalchemy import func, select

    from app.memory.sqlite_store import SqliteStore, TrainingExample
    from app.training.adapter_registry import AdapterRegistry

    store = SqliteStore()
    with store.session() as s:
        n = s.scalar(select(func.count()).select_from(TrainingExample)) or 0
    adapters = AdapterRegistry(store).list_all()
    return TrainingStatusResponse(
        n_examples=n,
        adapters=[AdapterOut(**a) for a in adapters],
    )


@app.post("/api/training/dataset", response_model=DatasetBuildResponse, dependencies=[api_auth])
def api_training_dataset() -> DatasetBuildResponse:
    """Kanonik `lora_sft.jsonl` → train/valid JSONL böl (tek kaynak; SQLite cılız hattı DEĞİL).

    Eski `DatasetBuilder` (SQLite) cılız `{prompt,completion}` üretip CLI'nin kurduğu zengin
    train.jsonl'i ezerdi (iki-hat drifti). Artık CLI `train`/`launch()` ile aynı kanonik
    kaynaktan bölünür → format/sayı tutarlı.
    """
    from app.training.detached_launch import build_training_split

    r = build_training_split()
    if r.n_train == 0:
        return DatasetBuildResponse(
            n_train=0,
            n_valid=0,
            content_hash="",
            message=(
                "Kanonik eğitim verisi yok (lora_sft.jsonl boş). Önce üret: "
                "`uv run hektor synth-qa` veya `uv run hektor lora-cloud-prep`."
            ),
        )
    return DatasetBuildResponse(
        n_train=r.n_train,
        n_valid=r.n_valid,
        content_hash=r.content_hash,
        message=f"{r.n_train} eğitim + {r.n_valid} doğrulama kaydı (kanonik lora_sft.jsonl).",
    )


@app.get("/api/backtests", response_model=BacktestHistoryResponse, dependencies=[api_auth])
def api_backtest_history(limit: int = 50) -> BacktestHistoryResponse:
    """Son backtest kayıtlarını listele (yeni → eski)."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_backtests(limit=min(limit, 200))
    return BacktestHistoryResponse(
        records=[BacktestRecord(**r) for r in rows],
        total=len(rows),
    )


@app.get("/api/training/examples", response_model=TrainingExamplesResponse, dependencies=[api_auth])
def api_training_examples(limit: int = 100) -> TrainingExamplesResponse:
    """Kayıtlı eğitim örneklerini listele."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_training_examples(limit=min(limit, 500))
    return TrainingExamplesResponse(
        examples=[TrainingExampleOut(**r) for r in rows],
        total=len(rows),
    )


@app.delete("/api/training/examples/{example_id}", dependencies=[api_auth])
def api_delete_training_example(example_id: str) -> dict:
    """Eğitim örneğini sil."""
    from fastapi import HTTPException

    from app.memory.sqlite_store import SqliteStore

    if not SqliteStore().delete_training_example(example_id):
        raise HTTPException(status_code=404, detail="Örnek bulunamadı.")
    return {"ok": True}


@app.post("/api/training/dry-run", response_model=TrainDryRunResponse, dependencies=[api_auth])
def api_training_dry_run(req: TrainDryRunRequest) -> TrainDryRunResponse:
    """Eğitim komutunu oluştur (çalıştırmaz — CLI'da --run ile başlatılır)."""
    import datetime as dt

    from app.config import get_settings
    from app.training.backend import detect_lora_backend
    from app.training.detached_launch import build_training_split

    r = build_training_split()
    s = get_settings()
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    backend = detect_lora_backend()

    if backend == "mlx":
        from app.training.mlx_lora_train import TrainConfig, build_command

        cfg = TrainConfig(
            base_model=req.base_model or s.mlx_base_model,
            train_jsonl=r.train_path,
            valid_jsonl=r.valid_path,
            adapter_output_path=s.adapters_dir / f"adapter_{ts}",
            iterations=req.iterations,
            batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            num_layers=req.num_layers,
        )
    else:
        from app.training.peft_lora_train import (  # type: ignore[assignment]
            PeftTrainConfig as TrainConfig,
        )
        from app.training.peft_lora_train import build_command  # type: ignore[assignment]

        cfg = TrainConfig(
            base_model=req.base_model or s.peft_base_model,
            train_jsonl=r.train_path,
            valid_jsonl=r.valid_path,
            adapter_output_path=s.adapters_dir / f"adapter_{ts}",
            iterations=req.iterations,
        )

    return TrainDryRunResponse(
        command=" ".join(build_command(cfg)),
        n_train=r.n_train,
        n_valid=r.n_valid,
        content_hash=r.content_hash,
        message=f"Backend: {backend} — gerçek eğitim için terminalden --run ile çalıştırın.",
    )


@app.post(
    "/api/training/run",
    response_model=TrainingStartResponse,
    dependencies=[api_auth, human_only],
    include_in_schema=False,
)
def api_training_run(req: TrainingStartRequest) -> TrainingStartResponse:
    """Gerçek LoRA eğitimini DETACHED başlat — CLI ile AYNI taze-onay kapısı (Phase 4D-1).

    Güvenlik (CLAUDE.md Kural 8): web yolu da STOP_ALL + TEK KULLANIMLIK taze manuel
    onay gerektirir; standing yetki yok. Onay yoksa eğitim BAŞLAMAZ — pending onay
    isteği oluşturulur ve ``needs_approval`` + ``approval_id`` + onay komutu döner.
    Onaylandıktan sonra istek TEKRAR çağrılınca taze onay TÜKETİLİR ve eğitim başlar.
    CLI ``train --run`` ile aynı anahtar (agent_id='lora-trainer', action='train_run')
    kullanılır → onaylar değiştirilebilir. ``launch()`` spawn edilen ``train --run``a
    SUPERVISED verir (çift onay olmasın); STOP_ALL alt süreçte yine geçerlidir.

    Detached süreç; ilerleme /api/training/live ile (log'dan) izlenir. Veri
    `lora_sft.jsonl`'den yeniden bölünür. iterations<=0 → 1 epoch.
    """
    from app.training.unattended_policy import authorize_training_action

    decision = authorize_training_action(
        "train_run",
        (f"Gerçek LoRA eğitimi (web): {req.adapter_name or 'hektor_lora'} ({req.iterations} adım)"),
        agent_id="lora-trainer",
    )
    if decision.mode == "stop_all":
        return TrainingStartResponse(
            ok=False,
            status="blocked",
            message="STOP_ALL aktif — gerçek eğitim bloklandı.",
        )
    if not decision.authorized:
        return TrainingStartResponse(
            ok=False,
            status="needs_approval",
            approval_id=decision.approval_id,
            approve_command=f"uv run hektor approval-approve {decision.approval_id}",
            message=(
                "Gerçek eğitim TAZE manuel onay gerektirir (standing yetki yok). "
                f"Onay isteği oluşturuldu: {decision.approval_id}. Onayla, sonra eğitimi "
                "yeniden başlat."
            ),
        )

    # 3) Onay tüketildi → detached eğitimi başlat.
    from app.training.detached_launch import launch

    res = launch(
        adapter_name=req.adapter_name or "hektor_lora",
        iterations=req.iterations,
        base_model=req.base_model or None,
    )
    return TrainingStartResponse(
        ok=res["ok"],
        status="started" if res["ok"] else "error",
        message=res["message"],
    )


@app.post("/api/training/stop", dependencies=[api_auth])
def api_training_stop() -> dict:
    """In-process (training_manager) VE detached eğitimi durdur (Phase 2 fix).

    Detached koşu pid ile sonlandırılır + storage/STOP_TRAINING bırakılır; süreç
    bulunamazsa hata vermez, 'stop_requested' döner.
    """
    from app.training.detached_launch import request_stop_detached_training
    from app.web.training_manager import get_training_manager

    with suppress(Exception):
        get_training_manager().stop()
    detached = request_stop_detached_training()
    return {"ok": True, "detached": detached}


@app.get("/api/training/colab-notebook", dependencies=[api_auth])
def api_training_colab_notebook() -> Response:
    """Google Colab egitim notebook'u olustur ve indir (.ipynb)."""
    import datetime as dt

    from fastapi.responses import Response as FR

    from app.config import get_settings
    from app.training.detached_launch import build_training_split
    from app.training.peft_lora_train import PeftTrainConfig, generate_colab_notebook

    s = get_settings()
    try:
        r = build_training_split()
        train_path = r.train_path
        valid_path = r.valid_path
    except Exception:
        train_path = s.jsonl_dir / "train.jsonl"
        valid_path = s.jsonl_dir / "valid.jsonl"

    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    # Tek beyin: 4B (Ollama qwen3:4b ile birebir). Eski 1.5B hardcode adapter'ı
    # uyumsuz kılıyordu — base mutlaka peft_base_model olmalı.
    cfg = PeftTrainConfig(
        base_model=s.peft_base_model,
        train_jsonl=train_path,
        valid_jsonl=valid_path,
        adapter_output_path=s.adapters_dir / f"hektor_lora_colab_{ts}",
    )
    out = s.reports_dir / f"hektor_colab_{ts}.ipynb"
    generate_colab_notebook(cfg, out)
    content = out.read_bytes()
    return FR(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=hektor_colab_{ts}.ipynb"},
    )


@app.get("/api/training/progress", dependencies=[api_auth])
def api_training_progress() -> dict:
    from app.web.training_manager import get_training_manager

    return get_training_manager().progress.to_dict()


@app.get("/api/training/live", dependencies=[api_auth])
def api_training_live() -> dict:
    """Gercek egitim durumu (ust-bar rozeti icin).

    Hem web'den baslatilani (training_manager) HEM de detached/CLI ile baslatilani
    (logs/train-full-err.log tqdm satirindan) algilar. Calismiyorsa running=False
    + hazirlik bilgisi (ready/ready_examples/ready_label) doner.
    """
    from app.training.detached_launch import training_status

    return training_status()


@app.get("/api/training/logs", dependencies=[api_auth])
def api_training_logs(lines: int = 40) -> dict:
    """Detached eğitim için son log satırları (SSE beslenmeyen durumda sekme gösterimi)."""
    s = get_settings()
    logf = s.root / "logs" / "train-full-err.log"
    if not logf.exists():
        return {"lines": []}
    try:
        raw = logf.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"lines": []}
    # tqdm \r ile satır üzerine yazar → \r ve \n'e böl, anlamlı son satırları al.
    parts = [p.strip() for p in raw.replace("\r", "\n").split("\n") if p.strip()]
    n = max(1, min(int(lines), 200))
    return {"lines": parts[-n:]}


@app.post("/api/training/stream-ticket", dependencies=[api_auth])
def api_training_stream_ticket() -> dict:
    """SSE için KISA ömürlü, TEK-kullanımlık bilet üret (insan normal auth ile alır).

    EventSource özel başlık gönderemediği için SSE ucu kimliği query'den almak zorundadır;
    ama İNSAN api_token'ının query'ye (log/proxy/geçmiş, TTL'siz) düşmesi kalıcı sır
    sızıntısıdır. Bunun yerine burada normal ``api_auth`` ile bir bilet alınır ve EventSource
    yalnız o bileti taşır (bkz. app/web/sse_tickets.py). Token boş (yerel mod) olsa da bilet
    üretilir; stream ucu token boşken bileti zorunlu tutmaz.
    """
    from app.web import sse_tickets

    return {"ticket": sse_tickets.mint(), "ttl_s": sse_tickets.DEFAULT_TTL_S}


@app.get("/api/training/stream")
async def api_training_stream(request: Request) -> Response:
    import json

    from fastapi import HTTPException as _HTTPException
    from fastapi.responses import StreamingResponse

    # ⛔ İNSAN api_token'ı ARTIK query'den KABUL EDİLMEZ (P7 — kalıcı sır sızıntısıydı).
    # EventSource başlık gönderemediği için KISA ömürlü, TEK-kullanımlık bilet kullanılır:
    # insan `/api/training/stream-ticket`'tan normal auth ile alır, burada query'de taşır.
    # Bilet loglara düşse de saniyeler içinde ölür ve ikinci kez kullanılamaz.
    _tok = get_settings().api_token.strip()
    if _tok:
        from app.web import sse_tickets

        _ticket = (request.query_params.get("ticket") or "").strip()
        if not sse_tickets.consume(_ticket):
            raise _HTTPException(
                status_code=401,
                detail="Geçersiz, süresi dolmuş veya kullanılmış SSE bileti.",
            )

    from app.web.training_manager import get_training_manager

    mgr = get_training_manager()

    async def event_gen():
        async for msg in mgr.subscribe():
            if await request.is_disconnected():
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/backtest", response_model=BacktestResponse, dependencies=[api_auth])
def api_backtest(req: BacktestRequest) -> BacktestResponse:
    from fastapi import HTTPException

    from app.trading.backtester import persist_backtest, run_backtest
    from app.trading.evaluator import evaluate as eval_strategy
    from app.trading.market_data_loader import generate_synthetic_ohlcv
    from app.trading.strategy_ir import StrategyIR, example_ir

    # Veri
    if req.use_synthetic:
        df = generate_synthetic_ohlcv(n=req.n_bars, seed=req.seed)
        data_file = f"synthetic(n={req.n_bars},seed={req.seed})"
    else:
        raise HTTPException(
            status_code=400,
            detail="Şimdilik web'den yalnız sentetik veri desteklenir; CSV için CLI kullan.",
        )

    # Strateji IR (güvenli doğrulama; kural çalıştırma YOK, yalnız regex parse)
    if req.strategy_ir:
        try:
            ir = StrategyIR.model_validate(req.strategy_ir)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Geçersiz strateji IR: {exc}") from exc
    else:
        ir = example_ir()

    result = run_backtest(df, ir)
    verdict = eval_strategy(df, ir)
    bt_id = None
    try:
        bt_id = persist_backtest(
            result, data_file, verdict=verdict.verdict, notes="; ".join(verdict.reasons)
        )
    except Exception as exc:
        logger.warning("Backtest kaydı başarısız: %s", exc)

    return BacktestResponse(
        strategy_name=ir.name,
        metrics=result.metrics.to_dict(),
        verdict=verdict.verdict,
        reasons=verdict.reasons,
        backtest_id=bt_id,
        data_source=data_file,
        n_bars=len(df),
    )


@app.post("/api/backtest/csv", response_model=BacktestResponse, dependencies=[api_auth])
async def api_backtest_csv(file: UploadFile = File(...)) -> BacktestResponse:
    """Yüklenen gerçek OHLCV CSV üzerinde backtest (katı doğrulama + look-ahead-safe)."""
    from fastapi import HTTPException

    from app.trading.backtester import persist_backtest, run_backtest
    from app.trading.evaluator import evaluate as eval_strategy
    from app.trading.market_data_loader import load_ohlcv
    from app.trading.strategy_ir import example_ir

    # Boyut tavanını gövdeyi sınırsız materyalize etmeden uygula (bkz. api_upload).
    _max_bytes = get_settings().max_upload_mb * 1024 * 1024
    content = await file.read(_max_bytes + 1)
    if len(content) > _max_bytes:
        raise HTTPException(
            status_code=413, detail=f"Dosya çok büyük (max {get_settings().max_upload_mb} MB)."
        )
    safe_name = security.validate_csv_upload(file.filename or "", content)

    s = get_settings()
    dest = security.safe_destination(s.market_raw_dir, safe_name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    logger.info("CSV yüklendi: %s (%d bayt)", dest.name, len(content))

    try:
        df = load_ohlcv(dest)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV okunamadı: {exc}") from exc
    if len(df) < 50:
        raise HTTPException(status_code=400, detail="Yetersiz veri (en az ~50 bar gerekli).")

    ir = example_ir()
    result = run_backtest(df, ir)
    verdict = eval_strategy(df, ir)
    bt_id = None
    try:
        bt_id = persist_backtest(
            result, dest.name, verdict=verdict.verdict, notes="; ".join(verdict.reasons)
        )
    except Exception as exc:
        logger.warning("Backtest kaydı başarısız: %s", exc)

    return BacktestResponse(
        strategy_name=ir.name,
        metrics=result.metrics.to_dict(),
        verdict=verdict.verdict,
        reasons=verdict.reasons,
        backtest_id=bt_id,
        data_source=dest.name,
        n_bars=len(df),
    )


# ---------- kart onay ----------
@app.get("/api/cards/pending", response_model=PendingCardsResponse, dependencies=[api_auth])
def api_cards_pending() -> PendingCardsResponse:
    """review_status=pending olan kartları listele."""
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    cards = store.list_pending_cards()
    out = []
    for c in cards:
        card_data = c.get("card_json") or {}
        out.append(
            CardReviewOut(
                card_id=c["card_id"],
                paper_id=c["paper_id"],
                model=c["model"],
                trust_level=c["trust_level"],
                review_status=c["review_status"],
                lora_eligible=c["lora_eligible"],
                difficulty=c["difficulty"],
                stage=c["stage"],
                created_at=c["created_at"],
                title=card_data.get("title"),
                main_claim=card_data.get("main_claim", ""),
            )
        )
    return PendingCardsResponse(cards=out, total=len(out))


@app.post(
    "/api/card/{card_id}/approve",
    response_model=ApproveCardResponse,
    dependencies=[api_auth, human_only],
)
def api_approve_card(card_id: str) -> ApproveCardResponse:
    """Kartı onayla: review_status=approved, lora_eligible=1. YALNIZ insan scope'u.

    Kart onayı `lora_eligible=1` yapar → doğrudan EĞİTİM KORPUSUNU besler. Kart korpus
    kalitesi v5 gerilemesinin köküydü; motorun kendi eğitim verisini onaylaması bu
    yüzden bir yetki kararıdır. (Web UI 06·ONAY akışı etkilenmez — başlıksız istekler
    human scope'tur. `rag_learning_loop` süreç-içi `approve_card` çağırır, HTTP'den
    geçmez → etkilenmez.)

    ``approve_card`` False dönerse İKİ ayrı sebep olabilir: (a) kart hiç yok, ya da
    (b) kart var ama içeriksiz/'...' placeholder (içerik guard'ı boş kartı eğitime
    sokmaz — bkz. ``_card_has_content``). Eskiden ikisi de "Kart bulunamadı" diye
    raporlanıyordu; kuyruk boş kartlarla doluyken her Onayla bu yanıltıcı mesajı verip
    ekran "çalışmıyor" görünüyordu. Artık iki durumu ayırıp gerçek sebebi gösteriyoruz.
    """
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    if store.approve_card(card_id):
        return ApproveCardResponse(
            card_id=card_id, status="approved", message="Kart onaylandı, LoRA'ya girilebilir"
        )
    if store.get_card_by_id(card_id) is None:
        return ApproveCardResponse(card_id=card_id, status="not_found", message="Kart bulunamadı")
    return ApproveCardResponse(
        card_id=card_id,
        status="empty",
        message="Kart boş/içeriksiz — onaylanamaz. Reddet'e basıp kuyruktan çıkarabilirsin.",
    )


@app.post(
    "/api/card/{card_id}/reject",
    response_model=ApproveCardResponse,
    dependencies=[api_auth, human_only],
)
def api_reject_card(card_id: str) -> ApproveCardResponse:
    """Kartı reddet: review_status=rejected, lora_eligible=0. YALNIZ insan scope'u.

    Red de korpus üzerinde bir yetki kararıdır (motor rakip kanıtı eleyebilirdi)."""
    from app.memory.sqlite_store import SqliteStore

    ok = SqliteStore().reject_card(card_id)
    if not ok:
        return ApproveCardResponse(card_id=card_id, status="not_found", message="Kart bulunamadı")
    return ApproveCardResponse(card_id=card_id, status="rejected", message="Kart reddedildi")


@app.get("/api/cards/approved", response_model=ApprovedCardsResponse, dependencies=[api_auth])
def api_cards_approved(
    difficulty_min: float = 0.0,
    difficulty_max: float = 1.0,
) -> ApprovedCardsResponse:
    """Onaylı kartları döndür (opsiyonel difficulty filtresi)."""
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    cards = store.list_approved_cards(difficulty_min=difficulty_min, difficulty_max=difficulty_max)
    out = []
    for c in cards:
        card_data = c.get("card_json") or {}
        out.append(
            CardReviewOut(
                card_id=c["card_id"],
                paper_id=c["paper_id"],
                model=c["model"],
                trust_level=c["trust_level"],
                review_status=c["review_status"],
                lora_eligible=c["lora_eligible"],
                difficulty=c["difficulty"],
                stage=c["stage"],
                created_at=c["created_at"],
                title=card_data.get("title"),
                main_claim=card_data.get("main_claim", ""),
            )
        )
    return ApprovedCardsResponse(
        cards=out,
        total=len(out),
        difficulty_min=difficulty_min,
        difficulty_max=difficulty_max,
    )


# ---------- Risk manager ----------
@app.get(
    "/api/backtest/{backtest_id}/risk",
    response_model=RiskReportResponse,
    dependencies=[api_auth],
)
def api_backtest_risk(
    backtest_id: str,
    equity_usd: float = 10_000.0,
    max_dd_pct: float = -20.0,
    risk_pct: float = 1.0,
    stop_pct: float = 2.0,
) -> RiskReportResponse:
    """Backtest ID için Kelly + drawdown ölçekleme + sabit risk raporu üret."""
    from fastapi import HTTPException
    from sqlalchemy import select

    from app.memory.sqlite_store import Backtest, SqliteStore, Strategy
    from app.trading.backtester import _compute_columns, _net_returns, _position_series
    from app.trading.market_data_loader import generate_synthetic_ohlcv
    from app.trading.risk_manager import analyze_risk
    from app.trading.strategy_ir import StrategyIR

    store = SqliteStore()
    with store.session() as s:
        bt = s.scalars(select(Backtest).where(Backtest.backtest_id == backtest_id)).first()
        if bt is None:
            raise HTTPException(status_code=404, detail=f"Backtest bulunamadı: {backtest_id}")
        strat = s.get(Strategy, bt.strategy_id)
        if strat is None:
            raise HTTPException(status_code=404, detail="Strateji bulunamadı.")
        ir = StrategyIR.model_validate_json(strat.ir_json)

    # Sentetik veriyle risk hesabı (gerçek CSV saklanmıyor, senkron hesaplama).
    # Kanonik `_net_returns` kullanılır: maliyet, GECİKMELİ pozisyondan (eff_pos.diff())
    # alınır. Eski hâl turnover'ı kaydırılmamış position.diff()'ten alıyordu → giriş
    # maliyeti bir bar erken düşüp _extract_trade_returns'ün varsaydığı hizayı bozuyordu.
    df = generate_synthetic_ohlcv(n=2000, seed=42)
    enriched = _compute_columns(df, ir)
    position = _position_series(enriched, ir)
    bar_ret = enriched["close"].pct_change().fillna(0.0)
    net_ret = _net_returns(position, bar_ret, ir.costs.commission + ir.costs.slippage)
    equity_curve = (1 + net_ret).cumprod()

    report = analyze_risk(
        strategy_name=ir.name,
        equity_curve=equity_curve,
        position=position,
        returns=net_ret,
        equity_usd=equity_usd,
        max_dd_threshold_pct=max_dd_pct,
        risk_per_trade_pct=risk_pct,
        atr_stop_pct=stop_pct,
    )

    d = report.to_dict()
    report_id = f"rr_{backtest_id}"
    store.save_risk_report(report_id=report_id, backtest_id=backtest_id, report_dict=d)
    return RiskReportResponse(
        strategy_name=d["strategy_name"],
        n_trades=d["n_trades"],
        kelly=KellyOut(**d["kelly"]),
        drawdown_scale=DrawdownScaleOut(**d["drawdown_scale"]),
        fixed_risk=FixedRiskOut(**d["fixed_risk"]),
        warnings=d["warnings"],
        recommendation=d["recommendation"],
        report_id=report_id,
    )


# ---------- Risk raporu listesi ----------
@app.get(
    "/api/risk-reports",
    response_model=RiskReportListResponse,
    dependencies=[api_auth],
)
def api_list_risk_reports(limit: int = 50) -> RiskReportListResponse:
    """Kaydedilmiş risk raporlarını yeni→eski sırasıyla listele."""
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    rows = store.list_risk_reports(limit=limit)
    records = [RiskReportRecord(**r) for r in rows]
    return RiskReportListResponse(reports=records, total=len(records))


# ---------- Pine Script export (backtest → TradingView) ----------
@app.get(
    "/api/backtest/{backtest_id}/pine",
    response_model=PineExportResponse,
    dependencies=[api_auth],
)
def api_backtest_pine(backtest_id: str) -> PineExportResponse:
    """Backtest ID'ye ait StrategyIR'i Pine Script v5'e çevir (TradingView köprüsü için)."""
    from fastapi import HTTPException
    from sqlalchemy import select

    from app.memory.sqlite_store import Backtest, SqliteStore, Strategy
    from app.trading.strategy_ir import StrategyIR

    store = SqliteStore()
    with store.session() as s:
        bt = s.scalars(select(Backtest).where(Backtest.backtest_id == backtest_id)).first()
        if bt is None:
            raise HTTPException(status_code=404, detail=f"Backtest bulunamadı: {backtest_id}")
        strat = s.get(Strategy, bt.strategy_id)
        if strat is None:
            raise HTTPException(status_code=404, detail="Bu backtest'e ait strateji bulunamadı.")
        ir = StrategyIR.model_validate_json(strat.ir_json)

    return PineExportResponse(
        backtest_id=backtest_id,
        strategy_name=ir.name,
        market=ir.market,
        timeframe=ir.timeframe,
        pine_code=ir.to_pine(),
    )


# ---------- arXiv makale arama ve indirme ----------
@app.get("/api/learning/summary", dependencies=[api_auth])
def api_learning_summary() -> dict:
    """Makale / chunk / kart istatistikleri — dashboard özeti."""
    from sqlalchemy import func
    from sqlalchemy import select as _sel

    from app.memory.sqlite_store import Chunk, KnowledgeCard, SqliteStore

    store = SqliteStore()
    papers = store.list_papers()
    approved = store.list_approved_cards()
    with store.session() as s:
        n_chunks = s.scalar(_sel(func.count()).select_from(Chunk)) or 0
        n_pending = (
            s.scalar(
                _sel(func.count())
                .select_from(KnowledgeCard)
                .where(KnowledgeCard.review_status == "pending")
            )
            or 0
        )
    return {
        "n_papers": len(papers),
        "n_chunks": n_chunks,
        "n_approved_cards": len(approved),
        "n_pending_cards": n_pending,
    }


# --- Agent runtime gözlemcisi (Phase 1) — salt-okuma; kontrol/onay Phase 2'de ---
@app.get("/api/agents", dependencies=[api_auth])
def api_agents() -> dict:
    """Kayıtlı runtime agent'lar (automation_manifest.yaml). Salt-okuma."""
    from app.agents.runtime import list_agents

    try:
        agents = list_agents()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Manifest okunamadı: {exc}") from exc
    return {"agents": [a.model_dump(mode="json") for a in agents]}


@app.get("/api/agents/runs", dependencies=[api_auth])
def api_agent_runs(limit: int = 20, agent_id: str | None = None, status: str | None = None) -> dict:
    """Son agent koşuları (en yeni önce). Salt-okuma."""
    from app.memory.sqlite_store import SqliteStore

    limit = max(1, min(limit, 200))
    runs = SqliteStore().list_agent_runs(limit=limit, agent_id=agent_id, status=status)
    return {"runs": runs}


@app.get("/api/agents/runs/{run_id}", dependencies=[api_auth])
def api_agent_run_detail(run_id: str) -> dict:
    """Tek bir agent koşusu + olay günlüğü. Salt-okuma."""
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    run = store.get_agent_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Koşu bulunamadı")
    return {"run": run, "events": store.list_agent_events(run_id)}


# --- Phase 2: task queue + approvals + supervisor (hepsi api_auth korumalı) ---
@app.get("/api/automation/tasks", dependencies=[api_auth])
def api_list_tasks(limit: int = 50, status: str | None = None, agent_id: str | None = None) -> dict:
    """Otomasyon görevleri (en yeni önce). Salt-okuma."""
    from app.agents.runtime import task_queue

    limit = max(1, min(limit, 500))
    tasks = task_queue.list_tasks(limit=limit, status=status, agent_id=agent_id)
    return {"tasks": [t.model_dump(mode="json") for t in tasks]}


@app.post("/api/automation/tasks", dependencies=[api_auth])
def api_create_task(
    request: Request,
    agent_id: str,
    title: str,
    description: str | None = None,
    requires_approval: bool = False,
) -> dict:
    """Yeni otomasyon görevi oluştur (pending).

    ÖNLEYİCİ SERTLEŞTİRME: bugün bu kuyruğu tüketen bir executor YOK, dolayısıyla
    görev yaratmak zararsızdır. Ama ileride bir executor eklenip `requires_approval=False`
    görevleri otomatik koşarsa, sürücü kendine onaysız iş yazabilirdi. Bu yüzden
    sürücü scope'u görev yaratabilir ama onay bayrağını DÜŞÜREMEZ — driver'ın açtığı
    her görev `requires_approval=True` olur. (İnsan akışı değişmez.)
    """
    from app.agents.runtime import task_queue

    if security.resolve_scope(request) == "driver":
        requires_approval = True

    t = task_queue.create_task(
        agent_id=agent_id,
        title=title,
        description=description,
        requires_approval=requires_approval,
    )
    return {"ok": True, "task": t.model_dump(mode="json")}


@app.post("/api/automation/tasks/{task_id}/cancel", dependencies=[api_auth])
def api_cancel_task(task_id: str, reason: str | None = None) -> dict:
    from app.agents.runtime import task_queue

    t = task_queue.cancel_task(task_id, reason=reason)
    if t is None:
        raise HTTPException(status_code=404, detail="Görev bulunamadı")
    return {"ok": True, "task": t.model_dump(mode="json")}


@app.get("/api/approvals", dependencies=[api_auth])
def api_list_approvals(status: str | None = None, limit: int = 50) -> dict:
    """Onay istekleri (en yeni önce). Salt-okuma."""
    from app.agents.runtime import approvals

    limit = max(1, min(limit, 500))
    items = approvals.list_approvals(status=status, limit=limit)
    return {"approvals": [a.model_dump(mode="json") for a in items]}


@app.post(
    "/api/approvals/{approval_id}/approve",
    dependencies=[api_auth, human_only],
    include_in_schema=False,
)
def api_approve(approval_id: str, note: str | None = None) -> dict:
    """Bir onay isteğini ONAYLA (tek kullanımlık taze onay). YALNIZ insan scope'u."""
    from app.agents.runtime import approvals

    a = approvals.approve(approval_id, note=note)
    if a is None:
        raise HTTPException(status_code=404, detail="Onay bulunamadı")
    return {"ok": True, "approval": a.model_dump(mode="json")}


@app.post(
    "/api/approvals/{approval_id}/reject",
    dependencies=[api_auth, human_only],
    include_in_schema=False,
)
def api_reject(approval_id: str, note: str | None = None) -> dict:
    """Bir onay isteğini REDDET. Red de bir yetki kararıdır → YALNIZ insan scope'u."""
    from app.agents.runtime import approvals

    a = approvals.reject(approval_id, note=note)
    if a is None:
        raise HTTPException(status_code=404, detail="Onay bulunamadı")
    return {"ok": True, "approval": a.model_dump(mode="json")}


@app.get("/api/events", dependencies=[api_auth])
def api_events(limit: int = 100, run_id: str | None = None, level: str | None = None) -> dict:
    """Genel olay akışı (tüm koşular + sistem olayları). Salt-okuma."""
    from app.memory.sqlite_store import SqliteStore

    limit = max(1, min(limit, 1000))
    events = SqliteStore().list_recent_agent_events(limit=limit, run_id=run_id, level=level)
    return {"events": events}


@app.get("/api/healthz", dependencies=[api_auth])
def api_healthz() -> dict:
    """Hafif sağlık probe'u (ağır LLM/Chroma init YOK — orkestrasyon için)."""
    import datetime as _dt

    from app.agents.runtime import supervisor

    return {
        "ok": True,
        "status": "healthy",
        "stop_all": supervisor.is_stop_all_active(),
        "time": _dt.datetime.now(_dt.UTC).isoformat(),
    }


@app.post("/api/supervisor/stop-all", dependencies=[api_auth])
def api_stop_all(reason: str | None = None) -> dict:
    """KÜRESEL acil-durdurma: tehlikeli aksiyonları blokla + CANLI motorları KES.

    ⛔ Bayrak yazmak yetmez: STOP_ALL dosyası yalnız YENİ tehlikeli aksiyonları bloklar,
    o an koşan `claude -p` alt-sürecine dokunmazdı. Kullanıcı DURDUR'a bastığında motor
    30 dk zaman aşımına kadar koşup abonelik kotasını yakmaya devam ediyordu — üstelik
    arayüz "durduruldu" deyip ⚡ kilidini açtığı için ÜSTÜNE ikinci motor doğurulabiliyordu.
    Bu yüzden bayrakla BİRLİKTE canlı motor süreçleri gerçekten sonlandırılır.
    """
    from app.agents.runtime import supervisor
    from app.orchestration import engine_procs

    result = supervisor.create_stop_all(reason=reason)
    # Motor kesme bayrak yazımını ASLA engellememeli (kill-switch her koşulda yazılır).
    try:
        killed = engine_procs.terminate_all()
    except Exception:  # pragma: no cover - savunmacı
        logger.exception("STOP_ALL: canlı motorlar kesilemedi")
        killed = 0
    return {**result, "engines_terminated": killed}


@app.post(
    "/api/supervisor/clear-stop-all",
    dependencies=[api_auth, human_only],
    include_in_schema=False,
)
def api_clear_stop_all() -> dict:
    """Kill-switch'i KALDIR — YALNIZ insan scope'u (motor kendi frenini çözemez)."""
    from app.agents.runtime import supervisor

    return supervisor.clear_stop_all()


@app.get("/api/supervisor/status", dependencies=[api_auth])
def api_supervisor_status() -> dict:
    from app.agents.runtime import supervisor

    return {"stop_all_active": supervisor.is_stop_all_active()}


@app.get("/api/learning/eval-history", dependencies=[api_auth])
def api_learning_eval_history() -> dict:
    """Tüm adapter versiyonlarının eval skor geçmişi."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_eval_history()
    return {"rows": rows}


@app.get("/api/learning/training-runs", dependencies=[api_auth])
def api_learning_training_runs() -> dict:
    """Kayıtlı loss curve JSON dosyalarını listele."""
    import json as _json

    runs = []
    for f in sorted(Path("reports/training").glob("*_loss.json")):
        try:
            data = _json.loads(f.read_text())
            runs.append(
                {
                    "adapter_name": data.get("adapter_name", f.stem),
                    "started_at": data.get("started_at", ""),
                    "finished_at": data.get("finished_at", ""),
                    "total_iters": data.get("total_iters", 0),
                    "final_train_loss": data["curve"][-1]["train_loss"]
                    if data.get("curve")
                    else None,
                    "final_val_loss": data["curve"][-1].get("val_loss")
                    if data.get("curve")
                    else None,
                    "curve": data.get("curve", []),
                }
            )
        except Exception:
            pass
    return {"runs": runs}


@app.get("/api/learning/card-growth", dependencies=[api_auth])
def api_learning_card_growth() -> dict:
    """Günlük onaylı kart büyüme verisi."""
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_card_growth()
    return {"rows": rows}


@app.get("/api/profile")
def api_hardware_profile() -> dict:
    """Donanım profili döndürür (auth gerekmez — kurulum popup için)."""
    from app.agents.system_profiler.profiler import collect

    p = collect()
    return {
        "os": p.os,
        "arch": p.arch,
        "cpu": p.cpu.name,
        "cores": p.cpu.cores,
        "ram_gb": round(p.memory.ram_total_gb, 1),
        "gpu": p.gpu.name,
        "gpu_vendor": p.gpu.vendor,
        "metal": p.gpu.metal,
        "cuda": p.gpu.cuda,
        "lora_supported": p.os == "macOS" and p.arch == "arm64",
        "lora_backend": (
            "mlx"
            if (p.os == "macOS" and p.arch == "arm64")
            else "peft_cuda"
            if p.gpu.cuda
            else "peft_cpu"
        ),
    }


@app.get("/api/recommend")
def api_model_recommend() -> dict:
    """RAM'e göre önerilen Ollama modellerini döndürür (auth gerekmez)."""
    from app.agents.model_advisor.advisor import recommend
    from app.agents.system_profiler.profiler import collect

    profile = collect()
    result = recommend(profile, task="general", top_k=3)
    recommended = [
        {
            "rank": r.rank,
            "name": r.display_name,
            "ollama": r.ollama_name,
            "confidence": round(r.confidence * 100),
            "reasons": r.reasons[:2],
        }
        for r in result.recommended
    ]
    rejected = [{"name": r.display_name, "reason": r.reason} for r in result.rejected[:3]]
    return {"recommended": recommended, "rejected": rejected}


@app.get("/api/auto-lora/status", dependencies=[api_auth])
async def api_auto_lora_status() -> dict:
    """Auto-LoRA pipeline durumu."""
    from app.lora.auto_pipeline import get_auto_pipeline

    return get_auto_pipeline().get_status()


@app.post("/api/auto-lora/enable", dependencies=[api_auth])
async def api_auto_lora_enable(enabled: bool = True) -> dict:
    """Otomatik periyodik kontrolü aç/kapat."""
    from app.lora.auto_pipeline import get_auto_pipeline

    get_auto_pipeline().set_enabled(enabled)
    return {"ok": True, "auto_enabled": enabled}


@app.post("/api/auto-lora/check", dependencies=[api_auth])
async def api_auto_lora_check() -> dict:
    """Gate 0-8 kontrolünü manuel tetikle."""
    from app.lora.auto_pipeline import get_auto_pipeline

    return await get_auto_pipeline().check_and_prepare()


@app.post("/api/auto-lora/train", dependencies=[api_auth, human_only], include_in_schema=False)
async def api_auto_lora_train(adapter_name: str, iters: int = 300) -> dict:
    """Kullanıcı onayıyla eğitimi başlat (READY_TO_TRAIN durumu gerekir). YALNIZ insan.

    `/api/training/run` ile AYNI onay-tüketim mekanizmasını kullanır
    (`approvals.require_fresh_approval`). Kapı olmadan sürücü, insanın onayladığı ama
    henüz tüketmediği bir onayı ÇALIP eğitimi başlatabilirdi (denetim bulgusu).
    """
    from app.lora.auto_pipeline import get_auto_pipeline

    return await get_auto_pipeline().start_training(adapter_name, iters)


@app.post("/api/auto-lora/promote", dependencies=[api_auth, human_only], include_in_schema=False)
async def api_auto_lora_promote() -> dict:
    """Kullanıcı onayıyla adapter'ı production'a terfi et (EVAL_PASSED gerekir). YALNIZ insan.

    Terfi de onay-tüketen bir yetki kararıdır → motor kendi adapter'ını terfi ettiremez.
    """
    from app.lora.auto_pipeline import get_auto_pipeline

    return await get_auto_pipeline().promote_to_production()


@app.post("/api/auto-lora/reset", dependencies=[api_auth])
async def api_auto_lora_reset() -> dict:
    """Pipeline'ı IDLE'a sıfırla."""
    from app.lora.auto_pipeline import get_auto_pipeline

    await get_auto_pipeline().reset()
    return {"ok": True}


# ---------- RAG öğrenme döngüsü (otonom korpus büyütme + öğrenme) ----------
@app.get("/api/rag-loop/status", dependencies=[api_auth])
async def api_rag_loop_status() -> dict:
    """RAG öğrenme döngüsü durumu (ayarlar + anlık çalışma + son tur özeti)."""
    from app.research.rag_learning_loop import get_rag_loop

    return get_rag_loop().get_status()


@app.post("/api/rag-loop/enable", dependencies=[api_auth])
async def api_rag_loop_enable(enabled: bool = True) -> dict:
    """Döngüyü aç/kapat. Açınca ilk turu ARALIĞI BEKLEMEDEN hemen başlatır."""
    from app.research.rag_learning_loop import get_rag_loop

    loop = get_rag_loop()
    loop.set_enabled(enabled)
    started = bool(loop.trigger_once_bg().get("ok")) if enabled else False
    return {"ok": True, "enabled": enabled, "started_cycle": started}


@app.post("/api/rag-loop/run-once", dependencies=[api_auth])
async def api_rag_loop_run_once() -> dict:
    """Tek bir öğrenme turunu hemen başlat (arka planda; anında döner)."""
    from app.research.rag_learning_loop import get_rag_loop

    return get_rag_loop().trigger_once_bg()


@app.post("/api/rag-loop/config", dependencies=[api_auth])
async def api_rag_loop_config(
    interval_min: int | None = None,
    fetch_enabled: bool | None = None,
    fetch_interval_hours: int | None = None,
    max_fetch_per_cycle: int | None = None,
    cards_per_cycle: int | None = None,
    scores_per_cycle: int | None = None,
    score_use_llm: bool | None = None,
    rebuild_empty: bool | None = None,
) -> dict:
    """Döngü ayarlarını güncelle (değerler güvenli aralığa kelepçelenir)."""
    from app.research.rag_learning_loop import get_rag_loop

    return get_rag_loop().set_config(
        interval_min=interval_min,
        fetch_enabled=fetch_enabled,
        fetch_interval_hours=fetch_interval_hours,
        max_fetch_per_cycle=max_fetch_per_cycle,
        cards_per_cycle=cards_per_cycle,
        scores_per_cycle=scores_per_cycle,
        score_use_llm=score_use_llm,
        rebuild_empty=rebuild_empty,
    )


@app.get("/api/arxiv/search", response_model=ArxivSearchResponse, dependencies=[api_auth])
def api_arxiv_search(q: str, max_results: int = 10) -> ArxivSearchResponse:
    """arXiv'de ara (indirme yok)."""
    from app.ingestion.arxiv_fetcher import search_arxiv

    n = max(1, min(max_results, 50))
    entries = search_arxiv(q, max_results=n)
    return ArxivSearchResponse(
        query=q,
        results=[
            ArxivEntryOut(
                arxiv_id=e.arxiv_id,
                title=e.title,
                authors=e.authors,
                abstract=e.abstract[:400],
                pdf_url=e.pdf_url,
                published=e.published,
            )
            for e in entries
        ],
        total=len(entries),
    )


@app.post("/api/arxiv/fetch", response_model=ArxivFetchResponse, dependencies=[api_auth])
def api_arxiv_fetch(req: ArxivFetchRequest) -> ArxivFetchResponse:
    """arXiv'de ara → PDF'leri indir → (opsiyonel) otomatik indeksle."""
    from app.ingestion.arxiv_fetcher import fetch_arxiv_papers

    fetch_results = fetch_arxiv_papers(req.query, max_results=req.max_results)
    downloaded = [r for r in fetch_results if not r.skipped]
    skipped_count = sum(1 for r in fetch_results if r.skipped)

    ingested = 0
    if req.auto_ingest and downloaded:
        from app.memory.paper_indexer import PaperIndexer

        ingest_res = PaperIndexer().ingest_directory()
        ingested = sum(1 for r in ingest_res if not r.skipped)

    return ArxivFetchResponse(
        query=req.query,
        fetched=len(downloaded),
        skipped=skipped_count,
        results=[
            ArxivFetchResult(arxiv_id=r.arxiv_id, title=r.title, skipped=r.skipped)
            for r in fetch_results
        ],
        ingested=ingested,
        message=(
            f"{len(downloaded)} PDF indirildi, {skipped_count} atlandı"
            + (f", {ingested} indekslendi" if req.auto_ingest else "")
            + "."
        ),
    )


# ---------- arXiv kayıtlı sorgu kütüphanesi ----------
@app.post("/api/arxiv/queries", response_model=ArxivSavedQueryOut, dependencies=[api_auth])
def api_save_arxiv_query(req: ArxivSavedQueryIn) -> ArxivSavedQueryOut:
    """Sorguyu kaydet (aynı sorgu tekrar kaydedilirse üzerine yaz)."""
    import hashlib

    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    qid = "aq_" + hashlib.md5(req.query.strip().lower().encode()).hexdigest()[:16]
    store.save_arxiv_query(qid, req.query, req.max_results, req.auto_ingest)
    rows = store.list_arxiv_saved_queries()
    row = next(r for r in rows if r["query_id"] == qid)
    return ArxivSavedQueryOut(**row)


@app.get("/api/arxiv/queries", response_model=ArxivSavedQueryListResponse, dependencies=[api_auth])
def api_list_arxiv_queries() -> ArxivSavedQueryListResponse:
    from app.memory.sqlite_store import SqliteStore

    rows = SqliteStore().list_arxiv_saved_queries()
    return ArxivSavedQueryListResponse(
        queries=[ArxivSavedQueryOut(**r) for r in rows], total=len(rows)
    )


@app.delete("/api/arxiv/queries/{query_id}", dependencies=[api_auth])
def api_delete_arxiv_query(query_id: str) -> dict:
    from fastapi import HTTPException

    from app.memory.sqlite_store import SqliteStore

    if not SqliteStore().delete_arxiv_query(query_id):
        raise HTTPException(status_code=404, detail="Sorgu bulunamadı.")
    return {"ok": True}


@app.post(
    "/api/arxiv/queries/{query_id}/run", response_model=ArxivFetchResponse, dependencies=[api_auth]
)
def api_run_arxiv_query(query_id: str) -> ArxivFetchResponse:
    """Kayıtlı sorguyu çalıştır: arXiv'den çek + indeksle."""
    from fastapi import HTTPException

    from app.ingestion.arxiv_fetcher import fetch_arxiv_papers
    from app.memory.sqlite_store import SqliteStore

    store = SqliteStore()
    rows = store.list_arxiv_saved_queries()
    row = next((r for r in rows if r["query_id"] == query_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail="Sorgu bulunamadı.")
    fetch_results = fetch_arxiv_papers(row["query"], max_results=row["max_results"])
    downloaded = [r for r in fetch_results if not r.skipped]
    skipped_count = len(fetch_results) - len(downloaded)
    ingested = 0
    if row["auto_ingest"] and downloaded:
        from app.ingestion.paper_loader import DiscoveredPaper, compute_file_hash
        from app.memory.paper_indexer import PaperIndexer

        for r in downloaded:
            try:
                disc = DiscoveredPaper(path=r.pdf_path, file_hash=compute_file_hash(r.pdf_path))
                PaperIndexer().ingest_one(disc)
                ingested += 1
            except Exception:
                pass
    store.mark_arxiv_query_ran(query_id)
    return ArxivFetchResponse(
        query=row["query"],
        fetched=len(downloaded),
        skipped=skipped_count,
        results=[
            ArxivFetchResult(arxiv_id=r.arxiv_id, title=r.title, skipped=r.skipped)
            for r in fetch_results
        ],
        ingested=ingested,
        message=f"{len(downloaded)} PDF indirildi, {skipped_count} atlandı"
        + (f", {ingested} indekslendi" if row["auto_ingest"] else "")
        + ".",
    )


# ---------- package export (Entropia) ----------
@app.get(
    "/api/strategy/{strategy_name}/export",
    response_model=PackageExportResponse,
    dependencies=[api_auth],
)
def api_export_package(strategy_name: str) -> PackageExportResponse:
    """StrategyIR → .achpkg formatında Entropia-uyumlu paket döndür."""
    from sqlalchemy import select

    from app.memory.sqlite_store import SqliteStore, Strategy
    from app.trading.package_exporter import export_strategy
    from app.trading.strategy_ir import StrategyIR

    store = SqliteStore()
    with store.session() as s:
        row = s.scalars(select(Strategy).where(Strategy.name == strategy_name)).first()
    if row is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Strateji bulunamadı: {strategy_name}")

    ir = StrategyIR.model_validate_json(row.ir_json)
    pkg = export_strategy(ir)
    return PackageExportResponse(
        name=pkg.name,
        version=pkg.version,
        type=pkg.package_type,
        source=pkg.source,
        created_at=pkg.created_at,
        backtest_verdict=pkg.backtest_verdict,
        backtest_metrics=pkg.backtest_metrics,
        code=PackageCodeOut(pine=pkg.pine_code, python=pkg.python_code),
    )


@app.post("/api/package/export", response_model=PackageExportResponse, dependencies=[api_auth])
def api_export_package_from_ir(ir_json: dict) -> PackageExportResponse:
    """Ham StrategyIR JSON'ından .achpkg üret (strateji DB'de kayıtlı olmak zorunda değil)."""
    from pydantic import ValidationError

    from app.trading.package_exporter import export_strategy
    from app.trading.strategy_ir import StrategyIR

    try:
        ir = StrategyIR.model_validate(ir_json)
    except ValidationError as exc:  # geçersiz IR → 422 (unhandled 500 yerine)
        raise HTTPException(status_code=422, detail=f"Geçersiz StrategyIR: {exc}") from exc
    pkg = export_strategy(ir)
    return PackageExportResponse(
        name=pkg.name,
        version=pkg.version,
        type=pkg.package_type,
        source=pkg.source,
        created_at=pkg.created_at,
        backtest_verdict=pkg.backtest_verdict,
        backtest_metrics=pkg.backtest_metrics,
        code=PackageCodeOut(pine=pkg.pine_code, python=pkg.python_code),
    )


# ---------- .achpkg dosya indirme (backtest ID'den) ----------
@app.get(
    "/api/backtest/{backtest_id}/download-pkg",
    dependencies=[api_auth],
)
def api_download_pkg(backtest_id: str) -> Response:
    """Backtest ID'ye ait stratejiyi .achpkg dosyası olarak indir."""
    import re

    from fastapi import HTTPException
    from sqlalchemy import select

    from app.memory.sqlite_store import Backtest, SqliteStore, Strategy
    from app.trading.package_exporter import export_strategy
    from app.trading.strategy_ir import StrategyIR

    store = SqliteStore()
    with store.session() as s:
        bt = s.scalars(select(Backtest).where(Backtest.backtest_id == backtest_id)).first()
        if bt is None:
            raise HTTPException(status_code=404, detail=f"Backtest bulunamadı: {backtest_id}")
        strat = s.get(Strategy, bt.strategy_id)
        if strat is None:
            raise HTTPException(status_code=404, detail="Strateji bulunamadı.")
        ir = StrategyIR.model_validate_json(strat.ir_json)

    pkg = export_strategy(ir, backtest_verdict=bt.verdict)
    safe_name = re.sub(r"[^\w\-]", "_", pkg.name)
    content = pkg.to_json()
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.achpkg"'},
    )


# ====================== Statik frontend ======================
# API yollarından SONRA monte edilir.
# index.html'deki `/assets/app.js?v=2` gibi sabit sürüm etiketleri elle bump
# edilmeyince eskiyor: dosya değişse de URL aynı kaldığı için tarayıcı ESKİ
# kopyayı önbellekten servis eder (kullanıcı eski arayüzü görür). Aşağıdaki
# regex bu etiketleri DOSYA İÇERİĞİNİN hash'iyle değiştirir → içerik değişince
# URL otomatik değişir, önbellek kendiliğinden tazelenir.
_ASSET_VERSION_RE = re.compile(r"(/assets/(app\.(?:js|css)))\?v=[\w.]+")


def _asset_version(filename: str) -> str:
    """assets/<filename> içeriğinin kısa hash'i (cache-bust anahtarı)."""
    try:
        data = (_STATIC_DIR / "assets" / filename).read_bytes()
    except OSError:
        return "0"
    return hashlib.sha256(data).hexdigest()[:12]


if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR / "assets")), name="assets")

    @app.get("/")
    def index() -> HTMLResponse:
        """index.html'i içerik-hash'li asset sürümleriyle servis et.

        app.js/app.css değişince URL otomatik değişir → tarayıcı daima güncel
        dosyayı çeker (manuel `?v=` bump'ı gerekmez). HTML'in kendisi `no-cache`
        ile döner ki yeni hash her zaman görülsün; aksi halde eski index.html
        önbellekte kalıp eski asset URL'lerine işaret eder.
        """
        html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
        html = _ASSET_VERSION_RE.sub(lambda m: f"{m.group(1)}?v={_asset_version(m.group(2))}", html)
        return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})


def run() -> None:
    """`hektor-web` giriş noktası."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "app.web.server:app",
        host=s.web_host,
        port=s.web_port,
        reload=False,
        log_level=s.log_level.lower(),
    )


if __name__ == "__main__":
    run()
