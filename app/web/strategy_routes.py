"""Sohbetten strateji testi web uçları (Faz 2B).

Akış: ``POST /draft`` (yerel model taslak; strateji kaydı yok, orijinal öneri + taslak ayrı
çeviri kaydı) → kullanıcı okunur formu düzeltir → ``POST /`` (kaydet; varyant ise ``parent_id``)
→ ``GET /{id}/review`` (orijinal → nihai, taslak → nihai farkları) → ``POST /{id}/approve``
(her fark işaretli) → ``POST /data-check`` → ``POST /{id}/run``.

Yetki: taslak (model çağrısı), kaydetme, basitleştirme, veri yükleme ve test koşturma
``require_human``; okuma uçları ``require_auth``. Hiçbir uç şablon stratejiyle ikame yapmaz.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.web.security import require_auth, require_human

strategy_router = APIRouter(
    prefix="/api/strategy", tags=["strategy"], dependencies=[Depends(require_auth)]
)
_human = Depends(require_human)


class DraftRequest(BaseModel):
    turn_id: str = Field(..., min_length=1, max_length=40)


class SaveRequest(BaseModel):
    spec: dict[str, Any]
    source: dict[str, Any] = Field(default_factory=dict)
    parent_id: str = Field(default="", max_length=40)


class SimplifyRequest(BaseModel):
    remove_rules: list[str] = Field(default_factory=list)
    drop_unsupported: list[str] = Field(default_factory=list)


class ApproveRequest(BaseModel):
    review_sha: str = Field(..., min_length=64, max_length=64)
    acknowledged: list[str] = Field(default_factory=list, max_length=500)
    note: str = Field(default="", max_length=1000)


class DataCheckRequest(BaseModel):
    data_file: str = Field(..., min_length=1, max_length=200)
    timeframe: str = Field(..., min_length=1, max_length=10)
    tz: str | None = Field(default=None, max_length=60)


class RunRequest(BaseModel):
    data_file: str = Field(..., min_length=1, max_length=200)
    tz: str | None = Field(default=None, max_length=60)
    stage: str = Field(..., pattern="^(gelistirme|dogrulama|final)$")


def _errors(exc: ValidationError) -> list[str]:
    out = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e.get("loc", ()))
        msg = str(e.get("msg", "")).removeprefix("Value error, ")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.trading.data_quality import DataQualityError
    from app.trading.strategy_draft import DraftError
    from app.trading.strategy_testing import StrategyTestError

    try:
        return fn(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'\"")) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail={"problems": _errors(exc)}) from exc
    except (DraftError, StrategyTestError, DataQualityError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@strategy_router.post("/draft", dependencies=[_human])
def strategy_draft(req: DraftRequest) -> dict[str, Any]:
    """Yerel model cevaptan taslak çıkarır (kayıt yok, şablon ikamesi yok)."""
    from app.trading.chat_strategy import draft_from_turn

    return _call(draft_from_turn, req.turn_id)


@strategy_router.post("", dependencies=[_human])
def strategy_save(req: SaveRequest) -> dict[str, Any]:
    from app.trading.chat_strategy import save_spec

    origin = "edit" if req.parent_id else "draft"
    return _call(save_spec, req.spec, source=req.source, parent_id=req.parent_id, origin=origin)


@strategy_router.get("/data-files")
def strategy_data_files() -> dict[str, Any]:
    from app.trading.strategy_testing import list_data_files, market_dir

    return {"files": list_data_files(), "dir": str(market_dir().name)}


@strategy_router.post("/data-upload", dependencies=[_human])
async def strategy_data_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    from app.config import get_settings
    from app.web import security

    limit = get_settings().max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail="Dosya çok büyük.")
    name = security.validate_csv_upload(file.filename or "", content)
    dest = security.safe_destination(get_settings().market_raw_dir, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return {"name": dest.name, "bytes": len(content)}


@strategy_router.post("/data-check")
def strategy_data_check(req: DataCheckRequest) -> dict[str, Any]:
    """Veri denetimi + dönem önizlemesi (test yok, kalıcı yazma yok)."""
    from app.trading.strategy_testing import check_data

    return _call(check_data, req.data_file, timeframe=req.timeframe, tz=req.tz or None)


@strategy_router.get("/turn/{turn_id}")
def strategy_for_turn(turn_id: str) -> dict[str, Any]:
    from app.trading.strategy_store import StrategyStore

    store = StrategyStore()
    items = store.strategies_for_turn(turn_id)
    for it in items:
        it["runs"] = store.list_runs(strategy_id=it["strategy_id"])
    return {"items": items}


@strategy_router.get("/runs/{run_id}")
def strategy_run_detail(run_id: str) -> dict[str, Any]:
    from app.trading.strategy_store import StrategyStore

    run = StrategyStore().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Koşu yok: {run_id}")
    return run


@strategy_router.get("/{strategy_id}")
def strategy_get(strategy_id: str) -> dict[str, Any]:
    from app.trading.strategy_spec import TestableStrategy
    from app.trading.strategy_store import StrategyStore

    store = StrategyStore()
    rec = store.get_strategy(strategy_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"Strateji yok: {strategy_id}")
    spec = TestableStrategy.model_validate(rec["spec"])
    return {
        **rec,
        "readable": spec.readable(),
        "testable": not spec.important_unsupported(),
        "runs": store.list_runs(strategy_id=strategy_id),
        "family": store.family_strategies(rec["family_id"]),
    }


@strategy_router.get("/{strategy_id}/review")
def strategy_review(strategy_id: str) -> dict[str, Any]:
    """Orijinal öneri → nihai ve taslak → nihai farkları + onay durumu (salt-okuma)."""
    from app.trading.chat_strategy import review

    return _call(review, strategy_id)


@strategy_router.post("/{strategy_id}/approve", dependencies=[_human])
def strategy_approve(strategy_id: str, req: ApproveRequest) -> dict[str, Any]:
    """(İnsan) Nihai stratejiyi, her farkı işaretleyerek onayla. Test bundan sonra koşar."""
    from app.trading.chat_strategy import approve

    return _call(
        approve,
        strategy_id,
        review_sha=req.review_sha,
        acknowledged=req.acknowledged,
        note=req.note,
    )


@strategy_router.post("/{strategy_id}/simplify", dependencies=[_human])
def strategy_simplify(strategy_id: str, req: SimplifyRequest) -> dict[str, Any]:
    from app.trading.chat_strategy import simplify

    return _call(
        simplify, strategy_id, remove_rules=req.remove_rules, drop_unsupported=req.drop_unsupported
    )


@strategy_router.post("/{strategy_id}/run", dependencies=[_human])
def strategy_run(strategy_id: str, req: RunRequest) -> dict[str, Any]:
    """Seçilen dönemde test (önemli desteklenmeyen kural varsa DURUR)."""
    from app.trading.strategy_testing import run_stage

    return _call(
        run_stage, strategy_id, data_file=req.data_file, tz=req.tz or None, stage=req.stage
    )
