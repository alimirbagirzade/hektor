"""Echo feedback web uçları — TestClient ile, çevrimdışı.

submit (Kural-1 reddi dahil) / list / approve / reject / summary / export.
Eğitim ASLA tetiklenmez; export yalnız aday dosyaya yazar.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.web.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_submit_clean_is_pending(client: TestClient) -> None:
    r = client.post(
        "/api/feedback/submit",
        json={
            "correction": "Slippage + komisyon dahil edilmeli; net Sharpe ayrı raporlanmalı.",
            "question": "Backtest maliyet dahil mi?",
            "correction_type": "claim_correction",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "pending"


def test_submit_guarantee_language_rejected(client: TestClient) -> None:
    r = client.post(
        "/api/feedback/submit",
        json={"correction": "Bu strateji garanti kâr sağlar."},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


def test_submit_empty_correction_422(client: TestClient) -> None:
    r = client.post("/api/feedback/submit", json={"correction": ""})
    assert r.status_code == 422  # min_length=1


def test_approve_and_export_flow(client: TestClient) -> None:
    cid = client.post(
        "/api/feedback/submit",
        json={
            "correction": "Look-ahead'a karşı pozisyon shift(1) ile gecikmeli olmalı.",
            "question": "Pozisyon ne zaman uygulanmalı?",
        },
    ).json()["correction_id"]

    ap = client.post(f"/api/feedback/approve/{cid}", json={})
    assert ap.status_code == 200
    assert ap.json()["ok"] is True

    ex = client.post("/api/feedback/export", json={})
    assert ex.status_code == 200
    assert ex.json()["n_exported"] >= 1


def test_approve_reject_unknown_404(client: TestClient) -> None:
    assert client.post("/api/feedback/approve/fb_yok", json={}).status_code == 404
    assert client.post("/api/feedback/reject/fb_yok", json={}).status_code == 404


def test_list_and_summary(client: TestClient) -> None:
    client.post("/api/feedback/submit", json={"correction": "Maliyet modellenmeli."})
    assert isinstance(client.get("/api/feedback/list?limit=5").json()["items"], list)
    s = client.get("/api/feedback/summary")
    assert s.status_code == 200
    assert "pending" in s.json()
