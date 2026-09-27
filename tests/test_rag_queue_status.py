"""Web panosu ayrı CLI sürecinin kalıcı kuyruğunu okumalıdır."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.memory.mastery_store import MasteryStore
from app.memory.sqlite_store import SqliteStore
from app.web.server import app


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "queue.db"
    sql = SqliteStore(db, check_same_thread=False)
    monkeypatch.setattr("app.verification.rag_mastery.SqliteStore", lambda: sql)
    return MasteryStore(db)


def test_queue_status_updates_between_requests(store) -> None:
    ids = [store.enqueue(f"paper_queue_{i}") for i in range(4)]
    store.update_queue_status(ids[0], "done")
    store.update_queue_status(ids[1], "running")
    store.update_queue_status(ids[2], "failed", "Geçici hata")
    client = TestClient(app)
    response = client.get("/api/rag-mastery")
    assert response.status_code == 200
    q = response.json()["learning_queue"]
    assert (q["total"], q["done"], q["running"], q["failed"], q["pending"]) == (4, 1, 1, 1, 1)
    assert q["processed_percent"] == 25
    assert q["current_papers"] == ["paper_queue_1"]
    assert q["last_updated"]
    store.update_queue_status(ids[1], "done")
    q = client.get("/api/rag-mastery").json()["learning_queue"]
    assert q["done"] == 2
    assert q["running"] == 0
    assert q["processed_percent"] == 50
    assert q["current_papers"] == []


def test_empty_queue(store) -> None:
    q = TestClient(app).get("/api/rag-mastery").json()["learning_queue"]
    assert q["total"] == 0
    assert q["processed_percent"] == 0
    assert q["last_updated"] is None
