"""Sohbet + öğrenme havuzu web uçları ve statik arayüz (TestClient, çevrimdışı)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from tests.chat_learning_helpers import (  # noqa: F401
    SUPPORTED_SENTENCE,
    FakeLLM,
    StubRetriever,
    iso,
)

STATIC = Path(__file__).resolve().parents[1] / "app" / "web" / "static"


@pytest.fixture
def client(iso, monkeypatch):  # noqa: F811
    import app.brain.local_llm as llm_mod
    import app.brain.rag_answerer as ra
    from app.web.server import app

    monkeypatch.setattr(ra, "RerankingRetriever", StubRetriever)
    monkeypatch.setattr(llm_mod, "LocalLLM", lambda model=None, transport=None: FakeLLM())
    return TestClient(app)


def test_chat_flow_end_to_end(client) -> None:
    conv = client.post("/api/chat/conversations", json={}).json()["conversation_id"]
    body = {
        "question": "Volatilite kümelenmesi nedir?",
        "conversation_id": conv,
        "client_request_id": "web-1",
    }
    r1 = client.post("/api/ask", json=body).json()
    assert r1["turn_status"] == "answered" and r1["turn_index"] == 1
    assert r1["model"]["tag"] and r1["replayed"] is False
    r1b = client.post("/api/ask", json=body).json()
    assert r1b["replayed"] is True and r1b["turn_id"] == r1["turn_id"]
    r2 = client.post(
        "/api/ask",
        json={"question": "Bunu açar mısın?", "conversation_id": conv, "client_request_id": "w2"},
    ).json()
    assert r2["history_turn_ids"] == [r1["turn_id"]]

    tid = r1["turn_id"]
    assert (
        client.post(f"/api/chat/turns/{tid}/feedback", json={"label": "useful"}).status_code == 200
    )
    assert client.get("/api/learn/candidates").json()["items"] == []  # Faydalı aday üretmez
    learn = client.post(f"/api/chat/turns/{tid}/learn", json={}).json()
    assert learn["created"] is True
    again = client.post(f"/api/chat/turns/{tid}/learn", json={}).json()
    assert again["created"] is False
    corr = client.post(
        f"/api/chat/turns/{r2['turn_id']}/correct", json={"text": SUPPORTED_SENTENCE}
    ).json()
    # Kaynak benzerliği tek başına otomatik uygunluk vermez → inceleme; gerekçeli onayla uygun.
    assert corr["candidate"]["status"] == "review"
    cid = corr["candidate"]["candidate_id"]
    short = client.post(f"/api/learn/candidates/{cid}/approve", json={"reason": "kısa"})
    assert short.status_code == 422
    ok = client.post(
        f"/api/learn/candidates/{cid}/approve",
        json={"reason": "Kaynak parçasıyla elle karşılaştırıldı."},
    ).json()
    assert ok["status"] == "eligible" and ok["verification"]["class"] == "human"

    conv_detail = client.get(f"/api/chat/conversations/{conv}").json()
    assert len(conv_detail["turns"]) == 2
    assert conv_detail["turns"][0]["candidates"][0]["kind"] == "learn"

    summary = client.get("/api/learn/summary").json()
    assert summary["collected"] == 2
    assert summary["families"]["threshold"] == 50
    prev = client.post("/api/learn/datasets/preview", json={"include_human": True}).json()
    assert prev["n_train"] + prev["n_eval"] >= 1
    created = client.post("/api/learn/datasets", json={"include_human": True}).json()
    assert created["created"] is True
    again = client.post("/api/learn/datasets", json={"include_human": True}).json()
    assert again["created"] is False  # çift veri sürümü yok
    assert client.get("/api/learn/datasets").json()["versions"][0]["version_id"] == "chat_v1"


def test_no_conversation_id_keeps_old_ask_shape(client) -> None:
    r = client.post("/api/ask", json={"question": "Volatilite kümelenmesi nedir?"}).json()
    assert r["turn_id"] is None and r["history_turn_ids"] == []


def test_conversation_rejects_mlx_adapter_and_unknown_conv(client) -> None:
    conv = client.post("/api/chat/conversations", json={}).json()["conversation_id"]
    bad = client.post(
        "/api/ask", json={"question": "Soru?", "conversation_id": conv, "adapter_version": "v1"}
    )
    assert bad.status_code == 422
    missing = client.post("/api/ask", json={"question": "Soru?", "conversation_id": "conv_yok"})
    assert missing.status_code == 404


def test_model_and_resources_endpoints(client) -> None:
    from app.config import get_settings

    m = client.get("/api/chat/model").json()
    assert m["tag"] == get_settings().effective_chat_model
    assert m["ollama_reachable"] is False  # çevrimdışı test: tahmin edilmez
    assert m["origin"] is None
    res = client.get("/api/chat/resources").json()
    assert res["allowed"] is True


def test_legacy_echo_is_read_only(client) -> None:
    r = client.get("/api/learn/legacy-echo").json()
    assert r["read_only"] is True
    paths = {getattr(rt, "path", "") for rt in client.app.routes}
    assert not any(p.startswith("/api/learn/legacy-echo/") for p in paths)


def test_candidate_endpoints_are_human_only(client) -> None:
    from app.web.security import require_human

    human = {
        "/api/chat/turns/{turn_id}/learn",
        "/api/chat/turns/{turn_id}/correct",
        "/api/learn/candidates/{candidate_id}/approve",
        "/api/learn/candidates/{candidate_id}/edit",
        "/api/learn/datasets",
        "/api/chat/measure",
    }
    seen = set()
    for rt in client.app.routes:
        path = getattr(rt, "path", "")
        if path in human and "POST" in getattr(rt, "methods", set()):
            deps = {d.call for d in rt.dependant.dependencies}
            assert require_human in deps, path
            seen.add(path)
    assert seen == human


def test_static_ui_has_chat_and_keeps_diagnostics() -> None:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "assets" / "app.js").read_text(encoding="utf-8")
    assert 'data-tab="chat"' in html and 'id="panel-chat"' in html
    assert 'data-tab="learnpool"' in html and 'id="panel-learnpool"' in html
    for label in ("Faydalı", "Hatalı", "Düzelt", "Öğrensin", "Eğitimden hariç tut"):
        assert label in js, label
    # MLX ve PEFT yolları SİLİNMEDİ, Gelişmiş/teşhis altında korunuyor.
    assert 'id="adapterSelect"' in html and 'id="loraChatForm"' in html
    assert html.count("Gelişmiş / teşhis") >= 2
    assert '"kesfet", tabs: ["chat", "learnpool"' in js
