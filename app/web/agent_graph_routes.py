"""Ajan haritası web ucu — canlı ajan etkileşim grafiği (salt-okuma).

`server.py`'ye TEK satır (`include_router`) ile bağlanır. Grafiği (düğüm+kenar+canlı durum)
döndürür; hiçbir şeyi tetiklemez. Frontend "ışıklı yol" haritası (15·AJAN HARİTASI) bunu poll eder.

Kimlik doğrulama: `require_auth` (token boşsa lokal-açık, mevcut davranışla aynı).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.web.security import require_auth

router = APIRouter(prefix="/api/agents", tags=["agent-graph"], dependencies=[Depends(require_auth)])


@router.get("/graph")
def agents_graph() -> dict[str, Any]:
    """Ajan etkileşim grafiği: {nodes, edges, groups, main_agent} — canlı durum best-effort."""
    from app.web.agent_graph import build_agent_graph

    return build_agent_graph()
