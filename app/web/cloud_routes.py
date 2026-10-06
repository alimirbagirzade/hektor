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
