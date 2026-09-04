"""Çalıştırma zinciri (chain) testleri — offline; gerçek manifest + sentetik bozuk manifest."""

from __future__ import annotations

import pytest
import yaml

from app.agents.runtime.chain import ChainError, resolve_chain


# --- gerçek manifest -----------------------------------------------------
def test_chain_resolves_without_cycle() -> None:
    steps = resolve_chain()
    assert steps
    ids = [s.step for s in steps]
    assert len(ids) == len(set(ids))  # tekrarlanan adım yok


def test_chain_is_topologically_ordered() -> None:
    steps = resolve_chain()
    pos = {s.step: s.order for s in steps}
    for s in steps:
        for dep in s.after:
            assert pos[dep] < s.order  # her bağımlılık daha önce gelir


def test_chain_steps_are_known_agents() -> None:
    from app.agents.runtime.registry import list_agents

    known = {a.agent_id for a in list_agents()}
    for s in resolve_chain():
        assert s.step in known


def test_chain_human_gates_flagged() -> None:
    steps = {s.step: s for s in resolve_chain()}
    assert steps["auto-lora-pipeline"].requires_approval is False  # tek unattended politika
    assert steps["rules-updater"].requires_approval is True
    assert steps["arxiv-fetcher"].requires_approval is False  # otonom


# --- AI-brain ek modül ajanları (talimat entegrasyonu) -------------------
_AIBRAIN_AGENTS = {
    "ingestion-quality-scorer",
    "scientific-tool-runtime",
    "model-data-registry",
    "hypothesis-evaluator",
}


def test_aibrain_agents_registered() -> None:
    from app.agents.runtime.registry import list_agents

    ids = {a.agent_id for a in list_agents()}
    missing = _AIBRAIN_AGENTS - ids
    assert not missing, f"Manifest'te eksik AI-brain ajanı: {missing}"


def test_aibrain_agents_in_chain() -> None:
    steps = {s.step: s for s in resolve_chain()}
    missing = _AIBRAIN_AGENTS - set(steps)
    assert not missing, f"Zincirde eksik AI-brain ajanı: {missing}"
    # model-data-registry insan onayı ister (Kural 8); auto-lora ona BAĞLI (sürüm-onayı önce)
    assert steps["model-data-registry"].requires_approval is True
    assert steps["model-data-registry"].order < steps["auto-lora-pipeline"].order


def test_aibrain_agent_md_and_skill_exist() -> None:
    from app.config import get_settings

    # `.claude/` depo ağacındadır (veri değil) → veri kökü değil KAYNAK kökü.
    root = get_settings().source_root
    for aid in _AIBRAIN_AGENTS:
        assert (root / ".claude" / "agents" / f"{aid}.md").exists(), aid
        assert (root / ".claude" / "skills" / aid / "SKILL.md").exists(), aid


# --- sentetik bozuk manifest (doğrulama) ---------------------------------
def _agent(aid: str) -> dict:
    return {"agent_id": aid, "name": aid, "file": "x.py", "entrypoint": "x"}


def _manifest(tmp_path, agents: list[dict], chain: list[dict]):
    p = tmp_path / "automation_manifest.yaml"
    p.write_text(
        yaml.safe_dump({"version": 1, "agents": agents, "chain": chain}, allow_unicode=True),
        encoding="utf-8",
    )
    return p


def test_cycle_raises(tmp_path) -> None:
    p = _manifest(
        tmp_path,
        [_agent("a"), _agent("b")],
        [{"step": "a", "after": ["b"]}, {"step": "b", "after": ["a"]}],
    )
    with pytest.raises(ChainError):
        resolve_chain(p)


def test_unknown_step_raises(tmp_path) -> None:
    p = _manifest(tmp_path, [_agent("a")], [{"step": "zzz", "after": []}])
    with pytest.raises(ChainError):
        resolve_chain(p)


def test_bad_after_ref_raises(tmp_path) -> None:
    p = _manifest(tmp_path, [_agent("a")], [{"step": "a", "after": ["nope"]}])
    with pytest.raises(ChainError):
        resolve_chain(p)


def test_missing_chain_section_raises(tmp_path) -> None:
    p = tmp_path / "automation_manifest.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "agents": [_agent("a")]}), encoding="utf-8")
    with pytest.raises(ChainError):
        resolve_chain(p)


def test_self_loop_raises(tmp_path) -> None:
    """Kendine bağımlılık (a→a) refleksif döngü olarak yakalanmalı."""
    p = _manifest(tmp_path, [_agent("a")], [{"step": "a", "after": ["a"]}])
    with pytest.raises(ChainError):
        resolve_chain(p)


def test_empty_chain_raises(tmp_path) -> None:
    """Boş 'chain' listesi reddedilmeli."""
    p = _manifest(tmp_path, [_agent("a")], [])
    with pytest.raises(ChainError):
        resolve_chain(p)


def test_diamond_topology_resolves(tmp_path) -> None:
    """Çoklu-ebeveyn (diamond): c hem a hem b'den SONRA; d, c'den sonra."""
    p = _manifest(
        tmp_path,
        [_agent("a"), _agent("b"), _agent("c"), _agent("d")],
        [
            {"step": "a", "after": []},
            {"step": "b", "after": []},
            {"step": "c", "after": ["a", "b"]},
            {"step": "d", "after": ["c"]},
        ],
    )
    order = {s.step: s.order for s in resolve_chain(p)}
    assert order["c"] > order["a"] and order["c"] > order["b"]
    assert order["d"] > order["c"]
