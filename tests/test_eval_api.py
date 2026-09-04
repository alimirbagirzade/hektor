"""Model eval API endpoint testleri — çevrimdışı."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.web.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_eval_sets_returns_list(client: TestClient) -> None:
    r = client.get("/api/eval/sets")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_eval_sets_contain_known_sets(client: TestClient) -> None:
    r = client.get("/api/eval/sets")
    names = {s["name"] for s in r.json()}
    # Repoda en az bir eval seti olmalı
    assert len(names) >= 1


def test_eval_run_missing_set_404(client: TestClient) -> None:
    r = client.post("/api/eval/run", json={"eval_set": "nonexistent_set_xyz"})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "evil",
    [
        "../../etc/passwd",
        "../secrets",
        "..\\..\\windows\\system32\\config",
        "/etc/passwd",
        "foo/../../bar",
    ],
)
def test_eval_run_rejects_path_traversal(client: TestClient, evil: str) -> None:
    """Yol-aşımı adı evals/ dışına ASLA çıkamaz → 400 (aşım) veya 404 (yok).

    Önemli olan 5xx/200 ile dosya okunmaması; aşım girişimi 400 ile reddedilir.
    """
    r = client.post("/api/eval/run", json={"eval_set": evil})
    assert r.status_code in (400, 404)
    # evals/ dışındaki bir dosya hiçbir koşulda değerlendirilmiş olmamalı
    assert r.status_code != 200


def test_ask_with_unknown_adapter_404(client: TestClient) -> None:
    r = client.post(
        "/api/ask",
        json={"question": "Sharpe oranı nedir?", "adapter_version": "adapter_hayali"},
    )
    assert r.status_code == 404


def test_ask_without_adapter_still_works(client: TestClient) -> None:
    r = client.post("/api/ask", json={"question": "Test sorusu bu mu?", "adapter_version": None})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert body.get("adapter_used") is None
