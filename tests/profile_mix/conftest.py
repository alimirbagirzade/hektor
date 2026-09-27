"""Profil karışım/eval testleri — geçici dizinde izole registry fixture'ları."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.lora.domain_adapter_registry import DomainAdapterRegistry
from app.lora.mix_common import DOMAINS
from app.lora.profile_registry import ProfileRegistry

from .mix_helpers import make_record


@pytest.fixture
def adapter_reg(tmp_path: Path) -> DomainAdapterRegistry:
    return DomainAdapterRegistry(tmp_path / "domain_adapters.jsonl")


@pytest.fixture
def full_adapter_reg(adapter_reg: DomainAdapterRegistry) -> DomainAdapterRegistry:
    for d in DOMAINS:
        adapter_reg.register(make_record(d))
    return adapter_reg


@pytest.fixture
def profile_reg(tmp_path: Path) -> ProfileRegistry:
    return ProfileRegistry(tmp_path / "profiles.jsonl")
