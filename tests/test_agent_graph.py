"""Ajan etkileşim haritası — çevrimdışı testler (manifest'ten grafik + web).

Grafik yapısı (düğüm/kenar/ana-ajan) ve salt-okuma web ucu doğrulanır. Canlı durum
best-effort olduğundan yapıya bakılır, spesifik duruma değil.
"""

from __future__ import annotations

import pytest

from app.web.agent_graph import build_agent_graph

_VALID_STATUS = {"idle", "running", "blocked", "error", "done"}


def test_graph_has_nodes_and_new_agents() -> None:
    g = build_agent_graph()
    ids = {n["id"] for n in g["nodes"]}
    # Bu seansta eklenen ajanlar haritada olmalı
    assert {
        "training-orchestrator",
        "orchestration-autodrive",
        "echo-feedback",
        "sentinel-monitor",
        "self-healing-controller",
        "unattended-supervisor",
        "training-guardian",
    } <= ids
    assert len(g["nodes"]) >= 20


def test_main_agent_flagged() -> None:
    g = build_agent_graph()
    assert g["main_agent"] == "orchestration-autodrive"
    main = [n for n in g["nodes"] if n["is_main"]]
    assert len(main) == 1 and main[0]["id"] == "orchestration-autodrive"


def test_live_driver_overrides_blocked_run_on_map(monkeypatch) -> None:
    """DB kapıda blocked olsa da motor ve sürdüğü orkestratör running görünmeli."""
    from app.orchestration import engine_procs
    from app.orchestration.orchestrator import TrainingOrchestrator

    monkeypatch.setattr(
        TrainingOrchestrator,
        "list_runs",
        lambda self, limit=1: [{"run_id": "orc-live", "status": "blocked"}],
    )
    monkeypatch.setattr(engine_procs, "live_count", lambda: 1)

    nodes = build_agent_graph()["nodes"]
    main = next(n for n in nodes if n["is_main"])
    orchestrator = next(n for n in nodes if n["id"] == "training-orchestrator")
    assert main["status"] == "running"
    assert orchestrator["status"] == "running"


def test_nodes_have_valid_shape() -> None:
    g = build_agent_graph()
    for n in g["nodes"]:
        assert n["status"] in _VALID_STATUS
        assert isinstance(n["reads"], list) and isinstance(n["writes"], list)
        assert n["group"] in {grp["key"] for grp in g["groups"]}


def test_edges_reference_existing_nodes() -> None:
    g = build_agent_graph()
    ids = {n["id"] for n in g["nodes"]}
    kinds = set()
    for e in g["edges"]:
        assert e["from"] in ids and e["to"] in ids
        assert e["from"] != e["to"]  # kendine kenar yok
        kinds.add(e["kind"])
    # chain topolojisi en az bir akış kenarı üretmeli
    assert "chain" in kinds


def test_chain_flow_edge_present() -> None:
    g = build_agent_graph()
    chain_edges = {(e["from"], e["to"]) for e in g["edges"] if e["kind"] == "chain"}
    # bilinen topoloji: arxiv-fetcher → rag-learning-loop
    assert ("arxiv-fetcher", "rag-learning-loop") in chain_edges


def test_motor_controls_rag_memory_pipeline() -> None:
    g = build_agent_graph()
    assert {
        "from": "orchestration-autodrive",
        "to": "rag-learning-loop",
        "kind": "control",
    } in g["edges"]


def test_sentinel_self_healing_control_edges_present() -> None:
    edges = {(e["from"], e["to"]) for e in build_agent_graph()["edges"] if e["kind"] == "control"}
    assert ("sentinel-monitor", "self-healing-controller") in edges
    assert ("self-healing-controller", "training-orchestrator") in edges
    assert ("self-healing-controller", "rag-learning-loop") in edges
    assert ("unattended-supervisor", "orchestration-autodrive") in edges
    assert ("unattended-supervisor", "self-healing-controller") in edges
    assert ("unattended-supervisor", "auto-lora-pipeline") in edges
    assert ("unattended-supervisor", "training-guardian") in edges
    assert ("training-guardian", "auto-lora-pipeline") in edges


def test_rag_paused_for_training_is_blocked_on_map(monkeypatch, tmp_path) -> None:
    # Durum dosyası CWD'ye DEĞİL `settings.state_dir`'e bağlıdır (süreç başka dizinden
    # başlatılınca sessizce yanlış yeri okumasın diye) → veri kökünü tmp'ye al.
    from app.config import settings as settings_mod

    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "rag_learning_state.json").write_text(
        '{"stage":"paused_training"}', encoding="utf-8"
    )
    monkeypatch.setenv("ACHILLES_ROOT_PATH", str(tmp_path))
    settings_mod.get_settings.cache_clear()

    rag = next(n for n in build_agent_graph()["nodes"] if n["id"] == "rag-learning-loop")
    assert rag["status"] == "blocked"


# ── web ─────────────────────────────────────────────────────────────────────

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.web.server import app  # noqa: E402


def test_graph_endpoint_ok() -> None:
    client = TestClient(app)
    r = client.get("/api/agents/graph")
    assert r.status_code == 200
    body = r.json()
    assert body["main_agent"] == "orchestration-autodrive"
    assert isinstance(body["nodes"], list) and body["nodes"]
    assert isinstance(body["edges"], list)
