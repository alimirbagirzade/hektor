"""AI-brain ek-modül web uçları testleri — TestClient ile, çevrimdışı.

registry (salt-okuma) / tools / ingestion-quality / hypothesis-eval uçları.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.memory.sqlite_store import SqliteStore
from app.web.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --- registry (salt-okuma) -------------------------------------------------
def test_registry_kinds_ok(client: TestClient) -> None:
    for kind in ("datasets", "rag-indices", "embeddings", "rewards", "decisions"):
        r = client.get(f"/api/registry/{kind}")
        assert r.status_code == 200, kind
        body = r.json()
        assert body["kind"] == kind
        assert isinstance(body["items"], list)


def test_registry_unknown_kind_404(client: TestClient) -> None:
    r = client.get("/api/registry/nope")
    assert r.status_code == 404


def test_registry_datasets_reflects_store(client: TestClient) -> None:
    from app.registry import RegistryStore

    RegistryStore(SqliteStore()).register_dataset(name="web_ds", n_records=5)
    r = client.get("/api/registry/datasets")
    assert any(it["name"] == "web_ds" for it in r.json()["items"])


# --- tools -----------------------------------------------------------------
def test_tools_list(client: TestClient) -> None:
    r = client.get("/api/tools")
    assert r.status_code == 200
    ids = {t["tool_id"] for t in r.json()["tools"]}
    assert "montecarlo" in ids and "stats-correlation" in ids


def test_tool_runs_endpoint(client: TestClient) -> None:
    r = client.get("/api/tools/runs")
    assert r.status_code == 200
    assert isinstance(r.json()["runs"], list)


# --- ingestion quality -----------------------------------------------------
def test_ingestion_quality_unknown_paper_404(client: TestClient) -> None:
    r = client.get("/api/ingestion-quality/yok_makale")
    assert r.status_code == 404


def test_ingestion_quality_scores_existing_paper(client: TestClient) -> None:
    store = SqliteStore()
    store.upsert_paper(
        paper_id="webp1", file_hash="webh1", source_path="w.pdf", title="T", n_pages=4
    )
    store.add_chunks(
        [{"chunk_id": "webp1_0", "paper_id": "webp1", "chunk_index": 0, "text": "metin " * 80}]
    )
    r = client.get("/api/ingestion-quality/webp1")
    assert r.status_code == 200
    body = r.json()
    assert "total" in body and "status" in body and "components" in body


# --- hypothesis eval -------------------------------------------------------
def test_eval_trading_hypothesis(client: TestClient) -> None:
    good = (
        "Volatilite yüksekse momentum zayıflayabilir; backtest ile test edilmeli, "
        "örneklem-dışı doğrulama ve komisyon+slippage maliyetleri dahil, risk stop-loss ile."
    )
    bad = "Bu strateji %100 garanti kazandırır, hemen al."
    r = client.post("/api/eval/trading-hypothesis", json={"hypotheses": [good, bad]})
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] is False  # bad reddedilir → kapı geçilmez
    assert body["n_items"] == 2


def test_eval_strict_gate_422(client: TestClient) -> None:
    r = client.post(
        "/api/eval/trading-hypothesis",
        json={"hypotheses": ["%100 garanti kâr"], "strict": True},
    )
    assert r.status_code == 422


def test_eval_empty_422(client: TestClient) -> None:
    r = client.post("/api/eval/trading-hypothesis", json={"hypotheses": []})
    assert r.status_code == 422


# --- dashboard sayfası -----------------------------------------------------
def test_ai_brain_dashboard_serves_html(client: TestClient) -> None:
    r = client.get("/ai-brain")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "AI Brain" in r.text
    # sayfa kendi /api uçlarına referans veriyor (entegre dashboard)
    assert "/api/registry/" in r.text and "/api/eval/trading-hypothesis" in r.text
