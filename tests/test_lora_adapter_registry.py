"""Adapter kayıt defteri testleri."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.lora.adapter_registry import (
    AdapterRecord,
    AdapterRegistry,
    AdapterStatus,
)


@pytest.fixture
def registry(tmp_path: Path) -> AdapterRegistry:
    return AdapterRegistry(registry_path=tmp_path / "registry.jsonl")


def test_register_generates_id(registry: AdapterRegistry) -> None:
    """adapter_id boşsa otomatik üretilmeli."""
    adapter_id = registry.register(AdapterRecord(adapter_name="test"))
    assert adapter_id.startswith("adapter_")
    assert len(registry.list_adapters()) == 1


def test_list_round_trips_record(registry: AdapterRegistry) -> None:
    """Kaydedilen kayıt aynı alanlarla okunabilmeli."""
    registry.register(AdapterRecord(adapter_id="a1", base_model="qwen", lora_r=16))
    loaded = registry.list_adapters()[0]
    assert loaded.adapter_id == "a1"
    assert loaded.base_model == "qwen"
    assert loaded.lora_r == 16


def test_promote_requires_user_approval(registry: AdapterRegistry) -> None:
    """user_approved=False ile PRODUCTION'a geçilememeli."""
    registry.register(AdapterRecord(adapter_id="a1"))
    assert registry.promote("a1", user_approved=False) is False
    assert registry.get_production() is None


def test_promote_with_approval_sets_production(registry: AdapterRegistry) -> None:
    """user_approved=True ile PRODUCTION atanmalı."""
    registry.register(AdapterRecord(adapter_id="a1"))
    assert registry.promote("a1", user_approved=True) is True
    production = registry.get_production()
    assert production is not None
    assert production.adapter_id == "a1"
    assert production.approved_by_user is True


def test_only_one_production_at_a_time(registry: AdapterRegistry) -> None:
    """Yeni production atanınca eski production APPROVED'a düşmeli."""
    registry.register(AdapterRecord(adapter_id="a1"))
    registry.register(AdapterRecord(adapter_id="a2"))
    registry.promote("a1", user_approved=True)
    registry.promote("a2", user_approved=True)

    productions = [r for r in registry.list_adapters() if r.status is AdapterStatus.PRODUCTION]
    assert len(productions) == 1
    assert productions[0].adapter_id == "a2"


def test_reject_sets_status_and_note(registry: AdapterRegistry) -> None:
    """reject çağrısı durumu REJECTED yapmalı ve sebebi nota eklemeli."""
    registry.register(AdapterRecord(adapter_id="a1"))
    assert registry.reject("a1", reason="düşük eval") is True
    record = registry.get("a1")
    assert record is not None
    assert record.status is AdapterStatus.REJECTED
    assert "düşük eval" in record.notes
