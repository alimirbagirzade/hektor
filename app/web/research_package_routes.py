"""Araştırma döngüsünün insan kontrollü başlat/durdur ve salt-okunur durum uçları."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.orchestration.research_service import get_research_service
from app.web.security import require_auth, require_human

router = APIRouter(
    prefix="/api/research-package", tags=["research-package"], dependencies=[Depends(require_auth)]
)


class ResearchStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    engines: list[str] = Field(min_length=1, max_length=2)


@router.get("/status")
def research_package_status() -> dict[str, Any]:
    """Paket durumu; motor veya Ollama çağırmaz."""
    return get_research_service().status()


@router.post("/start", dependencies=[Depends(require_human)])
async def research_package_start(req: ResearchStartRequest) -> dict[str, Any]:
    """Tek tuşla kur; eğitim varsa bekler. Gerçek eğitimi yetkilendirmez."""
    service = get_research_service()
    try:
        result = await asyncio.to_thread(service.start, req.engines)
    except (ValueError, OSError) as exc:
        raise HTTPException(409, detail=str(exc)) from exc
    service.ensure_loop()
    return result


@router.post("/stop", dependencies=[Depends(require_human)])
def research_package_stop() -> dict[str, Any]:
    """Yalnız araştırma paketini durdur; çalışan eğitim korunur."""
    return get_research_service().stop()
