"""Faz 3 · buluttan ikinci görüş web uçları (varsayılan KAPALI).

Okuma uçları ``require_auth``; gönderme / iptal ``require_human``. Gönderim, önizlemede
gösterilen metnin özetiyle eşleşmezse reddedilir. Bulut cevabı eğitime/öğrenme havuzuna girmez.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.web.security import require_auth, require_human

cloud_router = APIRouter(
    prefix="/api/cloud/second-opinion", tags=["cloud"], dependencies=[Depends(require_auth)]
)
_human = Depends(require_human)


class StartRequest(BaseModel):
    payload_sha256: str = Field(..., min_length=64, max_length=64)


def _call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.cloud.second_opinion import CloudError

    try:
        return fn(*args, **kwargs)
    except CloudError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@cloud_router.get("/status")
def cloud_status() -> dict[str, Any]:
    from app.cloud.second_opinion import status

    return status()


@cloud_router.get("/turn/{turn_id}/preview")
def cloud_preview(turn_id: str) -> dict[str, Any]:
    """Tek ekran: gönderilecek metnin tamamı, sağlayıcı, şart notu, kota, iptal. GÖNDERMEZ."""
    from app.cloud.second_opinion import preview

    return _call(preview, turn_id)


@cloud_router.get("/turn/{turn_id}")
def cloud_for_turn(turn_id: str) -> dict[str, Any]:
    from app.cloud.second_opinion import list_for_turn

    return {"items": list_for_turn(turn_id)}


@cloud_router.post("/turn/{turn_id}", dependencies=[_human])
def cloud_start(turn_id: str, req: StartRequest) -> dict[str, Any]:
    from app.cloud.second_opinion import start

    return _call(start, turn_id, req.payload_sha256)


@cloud_router.get("/{rec_id}")
def cloud_get(rec_id: str) -> dict[str, Any]:
    from app.cloud.second_opinion import get

    return _call(get, rec_id)


@cloud_router.post("/{rec_id}/cancel", dependencies=[_human])
def cloud_cancel(rec_id: str) -> dict[str, Any]:
    from app.cloud.second_opinion import cancel

    return _call(cancel, rec_id)


# ── Tek tıklamalı döngü: K1/K2 bulut hakem + bekleyen kararlar (docs/TASARIM_SUREKLI_DONGU.md) ──

judge_router = APIRouter(
    prefix="/api/cloud/judge", tags=["cloud"], dependencies=[Depends(require_auth)]
)
loop_router = APIRouter(prefix="/api/loop", tags=["loop"], dependencies=[Depends(require_auth)])


class K1StartRequest(BaseModel):
    part_index: int = Field(..., ge=0, le=500)
    payload_sha256: str = Field(..., min_length=64, max_length=64)


def _jcall(fn: Any, *args: Any, **kwargs: Any) -> Any:
    from app.cloud.judge import JudgeError

    try:
        return fn(*args, **kwargs)
    except JudgeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@judge_router.get("/k1/{cmp_id}/preview")
def k1_preview(cmp_id: str) -> dict[str, Any]:
    """Sıradaki kör paket parçasının TAMAMI + parça durumları. GÖNDERMEZ."""
    from app.cloud import judge

    return _jcall(judge.k1_preview, cmp_id)


@judge_router.post("/k1/{cmp_id}", dependencies=[_human])
def k1_start(cmp_id: str, req: K1StartRequest) -> dict[str, Any]:
    from app.cloud import judge

    return _jcall(judge.k1_start, cmp_id, req.part_index, req.payload_sha256)


@judge_router.get("/k2/{candidate_id}/preview")
def k2_preview(candidate_id: str) -> dict[str, Any]:
    """TEK adayın gönderilecek metni (tamamı). GÖNDERMEZ."""
    from app.cloud import judge

    return _jcall(judge.k2_preview, candidate_id)


@judge_router.post("/k2/{candidate_id}", dependencies=[_human])
def k2_start(candidate_id: str, req: StartRequest) -> dict[str, Any]:
    from app.cloud import judge

    return _jcall(judge.k2_start, candidate_id, req.payload_sha256)


@judge_router.get("/{rec_id}")
def judge_get(rec_id: str) -> dict[str, Any]:
    from app.cloud import judge

    return _jcall(judge.get, rec_id)


@judge_router.post("/{rec_id}/cancel", dependencies=[_human])
def judge_cancel(rec_id: str) -> dict[str, Any]:
    from app.cloud import judge

    return _jcall(judge.cancel, rec_id)


@loop_router.get("/pending")
def loop_pending() -> dict[str, Any]:
    """Döngünün bekleyen insan kararları (salt-okuma; hiçbir işi başlatmaz)."""
    from app.orchestration.loop_state import pending_decisions

    return pending_decisions()
