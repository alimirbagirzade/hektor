"""AI-brain ek-modül web uçları — registry / tools / ingestion / hypothesis-eval.

`server.py`'ye TEK satır (`include_router`) ile bağlanır → sıcak dosya minimal dokunulur
(eş zamanlı oturum çakışma yüzeyi küçük). Tüm uçlar **salt-okuma/hesap**tır (kalıcı mutasyon
yok); dataset terfisi gibi İNSAN-ONAYLI işlemler CLI'da kalır (Kural 8 — web'den otomatik
terfi yok). Kimlik doğrulama: `require_auth` (token boşsa lokal-açık, mevcut davranışla aynı).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.web.security import require_auth

router = APIRouter(prefix="/api", tags=["ai-brain"], dependencies=[Depends(require_auth)])

# UI sayfası ayrı router'da (auth YOK → sayfa yüklenip token sorabilsin; veri çağrıları
# yine /api/* üzerinden auth'lu). Statik tek-dosya dashboard.
ui_router = APIRouter(tags=["ai-brain-ui"])
_STATIC_DIR = Path(__file__).resolve().parent / "static"

_REGISTRY_KINDS = ("datasets", "rag-indices", "embeddings", "rewards", "decisions")


@ui_router.get("/ai-brain", response_class=HTMLResponse)
def ai_brain_dashboard() -> HTMLResponse:
    """AI-brain dashboard sayfası (registry/tools/ingestion/eval) — statik tek dosya."""
    html = (_STATIC_DIR / "ai_brain.html").read_text(encoding="utf-8")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


# --- Model/veri kayıt defteri (salt-okuma; terfi CLI'da insan-onaylı) -------
@router.get("/registry/{kind}")
def registry_list(kind: str, limit: int = 50) -> dict[str, Any]:
    """Kayıt defteri sürümlerini listele (datasets|rag-indices|embeddings|rewards|decisions)."""
    from app.registry import RegistryStore

    reg = RegistryStore()
    if kind == "datasets":
        items = reg.list_datasets(limit)
    elif kind == "rag-indices":
        items = reg.list_rag_indices(limit)
    elif kind == "embeddings":
        items = reg.list_embeddings(limit)
    elif kind == "rewards":
        items = reg.list_rewards(limit)
    elif kind == "decisions":
        items = reg.list_decisions(limit=limit)
    else:
        raise HTTPException(
            status_code=404, detail=f"bilinmeyen kind: {kind} ({'|'.join(_REGISTRY_KINDS)})"
        )
    return {"kind": kind, "items": items}


# --- Bilimsel araç çalışma zamanı (keşif + çalışma geçmişi) ----------------
@router.get("/tools")
def tools_list() -> dict[str, Any]:
    """Kayıtlı bilimsel araçları (determinizm sözleşmesiyle) listele."""
    from app.tools.tool_registry import list_tools

    return {"tools": [t.to_dict() for t in list_tools()]}


@router.get("/tools/runs")
def tool_runs(limit: int = 50) -> dict[str, Any]:
    """Son araç çalıştırmaları (tool_runs denetim izi)."""
    from app.memory.sqlite_store import SqliteStore

    return {"runs": SqliteStore().list_tool_runs(limit=limit)}


# --- İçe-alım kalite skoru (compute-on-demand; salt-skor) ------------------
@router.get("/ingestion-quality/{paper_id}")
def ingestion_quality(paper_id: str) -> dict[str, Any]:
    """Bir makalenin içe-alım kalite skorunu (100 puan) hesapla (kalıcı yazmaz)."""
    from app.ingestion.quality_scorer import score_paper
    from app.memory.sqlite_store import SqliteStore

    try:
        res = score_paper(SqliteStore(), paper_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return res.to_dict()


# --- Hipotez değerlendirici (salt-hesap; rapor yazmaz) ---------------------
class HypothesisEvalRequest(BaseModel):
    """Trading hipotez(leri) — her öğe str ya da {hypothesis_text, risk_notes, ...}."""

    hypotheses: list[Any] = Field(default_factory=list)
    strict: bool = False


@router.post("/eval/trading-hypothesis")
def eval_trading_hypothesis(req: HypothesisEvalRequest) -> dict[str, Any]:
    """Trading hipotezlerini test-edilebilirlik + ReleaseGate'ten geçir (rapor yazmaz)."""
    from app.evals.eval_runner import EvalGateError, EvalRunner

    if not req.hypotheses:
        raise HTTPException(status_code=422, detail="hypotheses boş olamaz")
    try:
        res = EvalRunner().run(
            "trading-hypothesis",
            hypotheses=req.hypotheses,
            strict=req.strict,
            write_report=False,
        )
    except EvalGateError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return res.to_dict()
