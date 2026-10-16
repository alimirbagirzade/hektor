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


@router.get("/knowledge-loop")
def knowledge_loop_status() -> dict[str, Any]:
    """Bilgi kalemleri (inceleme/karantina/onay), veri sürümü, eğitim ilerlemesi ve döngü
    defteri (bitiş + geliştirme değerlendirmesi). Salt-okuma: model/motor/Ollama çağırmaz;
    yalnız koşan eğitimin durum dosyasındaki adayı deftere kaydeder."""
    from app.orchestration.training_loop import overview
    from app.research.knowledge_review import summary
    from app.training.detached_launch import training_status

    try:
        progress = training_status()
    except Exception as exc:  # durum rozeti bozuksa ekranın kalanı yine görünsün
        progress = {"running": False, "error": str(exc)[:300]}
    return {
        "knowledge": summary(limit=30),
        "training": {
            k: progress.get(k)
            for k in ("running", "adapter", "step", "total", "pct", "eta", "source", "error")
            if k in progress
        },
        "loop": overview(),
        "continuous": _continuous(),
    }


def _continuous() -> dict[str, Any]:
    """Sürekli eğitim döngüsü özeti (salt-okuma; tik ATMAZ)."""
    from app.orchestration.continuous_loop import overview as loop_overview

    try:
        ov = loop_overview()
    except Exception as exc:  # durum dosyası bozuksa kartın kalanı yine görünsün
        return {"error": str(exc)[:300]}
    st = ov["state"]
    return {
        "enabled": ov["enabled"],
        "unattended": ov["unattended"],
        "kademe2_code_only": ov["kademe2_code_only"],
        "profile": ov["profile"],
        "reference": ov["reference"],
        "judge": ov["judge"],
        "paused": bool(st.get("paused")),
        "pause_reason": st.get("pause_reason", ""),
        "consecutive_failures": st.get("consecutive_failures", 0),
        "job": (st.get("job") or {}).get("kind", ""),
        "last_tick": st.get("last_tick", ""),
        "events": ov["events"][-8:],
    }
