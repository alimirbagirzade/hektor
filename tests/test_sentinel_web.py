"""Sentinel web uçları — TestClient ile, çevrimdışı.

Gerçek probe'lar offline koşar (Ollama kapalıyken llm=fail NORMAL); testler verdict'e
değil YAPIYA ve salt-rapor sözleşmesine bakar.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.web.server import app

_VALID = {"ok", "warn", "fail", "skip"}


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_run_returns_live_report(client: TestClient) -> None:
    r = client.post("/api/sentinel/run", json={"persist": False})
    assert r.status_code == 200
    body = r.json()
    assert body["overall"] in _VALID
    assert isinstance(body["probes"], list) and body["probes"]
    assert all(p["status"] in _VALID for p in body["probes"])
    names = {p["name"] for p in body["probes"]}
    assert {"llm", "training", "disk", "sqlite", "contention"} <= names


def test_overview_returns_report_and_history(client: TestClient) -> None:
    r = client.get("/api/sentinel/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["report"]["overall"] in _VALID
    assert isinstance(body["history"], list)
    assert len(body["history"]) >= 1  # overview persist=True → en az bu koşu kayıtlı


def test_history_endpoint(client: TestClient) -> None:
    client.get("/api/sentinel/overview")  # en az bir kayıt garanti
    r = client.get("/api/sentinel/history?limit=5")
    assert r.status_code == 200
    hist = r.json()["history"]
    assert isinstance(hist, list) and len(hist) >= 1
    assert hist[0]["overall"] in _VALID
    assert isinstance(hist[0]["probes"], list)


def test_self_heal_status_endpoint(client: TestClient) -> None:
    r = client.get("/api/sentinel/self-heal")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert isinstance(body["consecutive"], dict)
    assert isinstance(body["history"], list)


def test_unattended_status_endpoint(client: TestClient) -> None:
    r = client.get("/api/sentinel/unattended")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["engine"] == "codex"
