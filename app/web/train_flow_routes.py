"""Kolay eğitim akışı web uçları (Faz 2C). Hiçbiri eğitimi kendiliğinden başlatmaz.

- ``GET  /api/train-flow/state``    : son ayarlar + SALT-OKUNUR hazırlık (yazma yok).
- ``POST /api/train-flow/snapshot`` : (insan) değişmez anlık görüntü + reçete (açık adım).
- ``POST /api/train-flow/launch``   : (insan) özet onayı → ön kontroller → reçeteye bağlı onay →
                                      başlatma. Aynı istek kimliği çift eğitim başlatmaz.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.web.security import require_auth, require_human

train_flow_router = APIRouter(
    prefix="/api/train-flow", tags=["train-flow"], dependencies=[Depends(require_auth)]
)
_human = Depends(require_human)


class SnapshotRequest(BaseModel):
    profile: str | None = Field(default=None, max_length=80)
    base_model: str | None = Field(default=None, max_length=200)
    mix_profile: str | None = Field(default=None, max_length=80)
    mix_weights: dict[str, float] | None = None
    adapter_name: str = Field(..., min_length=1, max_length=64)
    max_examples: int = Field(default=0, ge=0, le=1_000_000)


class LaunchRequest(BaseModel):
    snapshot_id: str = Field(..., min_length=5, max_length=40)
    request_id: str = Field(..., min_length=8, max_length=80)


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.training.easy_train import EasyTrainError

    try:
        return fn(*args, **kwargs)
    except EasyTrainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@train_flow_router.get("/state")
def train_flow_state() -> dict[str, Any]:
    from app.training.easy_train import state

    return state()


@train_flow_router.post("/snapshot", dependencies=[_human])
def train_flow_snapshot(req: SnapshotRequest) -> dict[str, Any]:
    from app.training.easy_train import prepare_snapshot

    data = {k: v for k, v in req.model_dump().items() if v is not None}
    return _call(prepare_snapshot, data)


@train_flow_router.post("/launch", dependencies=[_human])
def train_flow_launch(req: LaunchRequest) -> dict[str, Any]:
    from app.training.easy_train import launch

    return _call(launch, req.snapshot_id, req.request_id)
