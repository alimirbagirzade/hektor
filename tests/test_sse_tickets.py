"""SSE bileti testleri (P7 SORUN 3) — insan api_token'ı query'den ÇIKARILDI.

Kanıtlananlar:
- Bilet kısa ömürlü + TEK-kullanımlık (ikinci kullanım / TTL sonrası reddedilir).
- `/api/training/stream` insan api_token'ını query'de KABUL ETMİYOR (401).
- Geçerli bilet auth'u geçiyor; bilet yoksa/geçersizse 401.

Hepsi ÇEVRİMDIŞI ve deterministik.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.web import security, sse_tickets, training_manager
from app.web.server import app

# Lifespan'ı (ensure_dirs) tetiklemeyen modül-seviyesi client — get_settings'i global
# yamalayacağımız için `with TestClient(app)` başlangıç kancasını patlatırdı
# (test_scope_isolation.py ile aynı desen).
client = TestClient(app)


class _AyarlarSahte:
    """`get_settings()` yerine minimal sahte (yalnız api_token okunur)."""

    def __init__(self, api_token: str) -> None:
        self.api_token = api_token


@pytest.fixture(autouse=True)
def _temiz_bilet_deposu():
    sse_tickets.reset()
    yield
    sse_tickets.reset()


# ── Modül: kısa ömürlü + TEK-kullanımlık ────────────────────────────────────────────────


def test_bilet_tek_kullanimlik() -> None:
    t = sse_tickets.mint()
    assert sse_tickets.consume(t) is True
    # İkinci kullanım reddedilir (doğrulama = tüketim).
    assert sse_tickets.consume(t) is False


def test_bos_bilet_reddedilir() -> None:
    assert sse_tickets.consume("") is False
    assert sse_tickets.consume("uydurma") is False


def test_bilet_ttl_sonrasi_reddedilir(monkeypatch: pytest.MonkeyPatch) -> None:
    """TTL geçmiş bilet tüketilemez (loglara düşse de saniyeler içinde ölür)."""
    now = [1000.0]
    # mint ve consume AYNI (sahte) saati kullansın → deterministik TTL.
    monkeypatch.setattr(sse_tickets.time, "monotonic", lambda: now[0])
    t = sse_tickets.mint(ttl_s=1)  # expires_at = 1001
    now[0] = 1002.0
    assert sse_tickets.consume(t) is False


# ── Uç: insan token'ı query'de KABUL EDİLMEZ ────────────────────────────────────────────


def test_stream_insan_tokenini_querydden_kabul_etmez(monkeypatch: pytest.MonkeyPatch) -> None:
    """KRİTİK: eski `?token=<api_token>` yolu KAPATILDI (kalıcı sır sızıntısıydı)."""
    monkeypatch.setattr("app.web.server.get_settings", lambda: _AyarlarSahte("gizli_insan"))
    r = client.get("/api/training/stream?token=gizli_insan")
    assert r.status_code == 401


def test_stream_biletsiz_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.web.server.get_settings", lambda: _AyarlarSahte("gizli_insan"))
    r = client.get("/api/training/stream")
    assert r.status_code == 401


def test_stream_gecersiz_bilet_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.web.server.get_settings", lambda: _AyarlarSahte("gizli_insan"))
    r = client.get("/api/training/stream?ticket=uydurma")
    assert r.status_code == 401


def test_bilet_ucu_bearer_ile_uretir(monkeypatch: pytest.MonkeyPatch) -> None:
    """İnsan NORMAL auth (bearer) ile bilet alır; sonra EventSource onu query'de taşır."""
    monkeypatch.setattr(security, "get_settings", lambda: _AyarlarSahte("gizli_insan"))
    # Token'sız → require_auth 401
    assert client.post("/api/training/stream-ticket").status_code == 401
    r = client.post("/api/training/stream-ticket", headers={"Authorization": "Bearer gizli_insan"})
    assert r.status_code == 200
    assert r.json().get("ticket")


def test_gecerli_bilet_streami_acar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Geçerli bilet auth'u geçer → 200 (tek mesajlık sahte akışla asılmaz)."""
    monkeypatch.setattr("app.web.server.get_settings", lambda: _AyarlarSahte("gizli_insan"))

    class _SahteMgr:
        async def subscribe(self) -> AsyncIterator[dict]:
            yield {"type": "done"}

    monkeypatch.setattr(training_manager, "get_training_manager", lambda: _SahteMgr())

    bilet = sse_tickets.mint()
    r = client.get(f"/api/training/stream?ticket={bilet}")
    assert r.status_code == 200
    # Bilet tüketildi → ikinci kullanım reddedilir.
    assert sse_tickets.consume(bilet) is False
