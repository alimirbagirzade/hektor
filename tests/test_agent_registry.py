"""Agent registry (Phase 1) — manifest yükleme + sorgulama testleri (offline)."""

from __future__ import annotations

import pytest

from app.agents.runtime import (
    ManifestError,
    agents_requiring_approval,
    dangerous_agents,
    get_agent,
    list_agents,
    load_agent_registry,
)

# Denetimde bulunan ve manifest'te bulunması ZORUNLU 15 ajan.
REQUIRED_AGENTS = {
    "auto-lora-pipeline",
    "rag-learning-loop",
    "research-orchestrator",
    "rag-trend-scanner",
    "reflection-agent",
    "paper-mastery-agent",
    "status-manager",
    "lora-control-plane",
    "adapter-eval",
    "dataset-quality-gate",
    "tool-use-trainer",
    "auto-researcher",
    "arxiv-fetcher",
    "rules-updater",
    "model-advisor",
}


def test_manifest_loads() -> None:
    reg = load_agent_registry()
    assert isinstance(reg, dict)
    assert len(reg) >= 15


def test_registry_has_all_required_agents() -> None:
    ids = {a.agent_id for a in list_agents()}
    missing = REQUIRED_AGENTS - ids
    assert not missing, f"Manifest'te eksik agent'lar: {missing}"


def test_dangerous_agents_marked() -> None:
    ids = {a.agent_id for a in dangerous_agents()}
    # Gerçek eğitim + terfi yapabilen tek ajan tehlikeli işaretli olmalı.
    assert "auto-lora-pipeline" in ids


def test_agents_requiring_approval() -> None:
    ids = {a.agent_id for a in agents_requiring_approval()}
    assert "auto-lora-pipeline" not in ids
    assert "training-orchestrator" not in ids
    assert "rules-updater" in ids


def test_get_agent_and_unknown() -> None:
    spec = get_agent("rag-learning-loop")
    assert spec.file.endswith("rag_learning_loop.py")
    with pytest.raises(KeyError):
        get_agent("nonexistent-agent-xyz")


def test_missing_manifest_raises(tmp_path) -> None:
    with pytest.raises(ManifestError):
        load_agent_registry(path=tmp_path / "yok.yaml")


def test_malformed_manifest_raises(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    # 'agents' bir liste ama öğeler eşleme değil → ManifestError beklenir.
    bad.write_text("agents:\n  - bu bir mapping degil\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_agent_registry(path=bad)


def test_manifest_without_agents_key_raises(tmp_path) -> None:
    bad = tmp_path / "noagents.yaml"
    bad.write_text("version: 1\n", encoding="utf-8")
    with pytest.raises(ManifestError):
        load_agent_registry(path=bad)
