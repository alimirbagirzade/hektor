"""Sentinel (Nöbetçi) web uçları — sistem sağlığı canlı görünüm + geçmiş.

`server.py`'ye TEK satır (`include_router`) ile bağlanır. Tüm uçlar SALT-OKUMA/rapordur:
hiçbir şeyi durdurmaz/başlatmaz (öneriler metin olarak döner, eylem insanın).

Kimlik doğrulama: `require_auth` (token boşsa lokal-açık, mevcut davranışla aynı).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.web.security import require_auth

router = APIRouter(prefix="/api/sentinel", tags=["sentinel"], dependencies=[Depends(require_auth)])


class SentinelRunRequest(BaseModel):
    persist: bool = True  # False → yalnız canlı bak, geçmişe yazma


def _sentinel() -> Any:
    from app.monitoring import Sentinel

    return Sentinel()


@router.post("/run")
def sentinel_run(req: SentinelRunRequest) -> dict[str, Any]:
    """Nöbetçiyi ŞİMDİ koş (tüm yoklamalar salt-okuma) → canlı rapor."""
    return _sentinel().run(persist=req.persist).to_dict()


@router.get("/overview")
def sentinel_overview() -> dict[str, Any]:
    """Canlı rapor + son geçmiş (dashboard tek çağrı). Koşuyu geçmişe yazar."""
    sentinel = _sentinel()
    report = sentinel.run(persist=True)
    return {"report": report.to_dict(), "history": sentinel.history(limit=10)}


@router.get("/history")
def sentinel_history(limit: int = 20) -> dict[str, Any]:
    """Geçmiş nöbetçi koşuları (en yeni önce)."""
    return {"history": _sentinel().history(limit=min(max(1, limit), 200))}


@router.get("/self-heal")
def self_heal_status() -> dict[str, Any]:
    """Otomatik tamir döngüsünün kalıcı durumu ve son eylemleri."""
    from app.monitoring.self_heal import get_self_healer

    return get_self_healer().status()


@router.get("/unattended")
def unattended_status() -> dict[str, Any]:
    """Kalıcı motor bağlantısı ve reconciliation döngüsü durumu."""
    from app.orchestration.unattended_supervisor import get_unattended_supervisor

    return get_unattended_supervisor().status()
