"""Agent registry — ``automation_manifest.yaml``'i okuyup ``AgentSpec`` döndürür.

Manifest, denetimde (audit) bulunan tüm runtime-agent benzeri bileşenlerin
tek, bildirimsel kaynağıdır. Bu modül onu okur, doğrular ve sorgular.

YAML bozuk / eksik / şemaya uymuyorsa ``ManifestError`` ile AÇIK hata verilir
(sessizce boş liste dönmez — yanlış "ajan yok" izlenimi yaratmamak için).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from app.agents.runtime.schemas import AgentAutonomy, AgentSpec
from app.config import get_settings


class ManifestError(RuntimeError):
    """``automation_manifest.yaml`` okunamadı veya geçersiz."""


def _manifest_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    # Manifest depoyla gelen KAYNAK dosyasıdır (veri değil) → `source_root`.
    return get_settings().manifest_file


def load_agent_registry(path: str | Path | None = None) -> dict[str, AgentSpec]:
    """Manifest'i oku → ``{agent_id: AgentSpec}``. Bozuksa ``ManifestError`` fırlatır."""
    p = _manifest_path(path)
    if not p.exists():
        raise ManifestError(f"Manifest bulunamadı: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"Manifest YAML ayrıştırılamadı ({p}): {exc}") from exc
    if not isinstance(raw, dict) or "agents" not in raw:
        raise ManifestError(f"Manifest bir eşleme olmalı ve 'agents' anahtarı içermeli: {p}")
    agents_raw = raw["agents"]
    if not isinstance(agents_raw, list) or not agents_raw:
        raise ManifestError("Manifest 'agents' boş olmayan bir liste olmalı")

    registry: dict[str, AgentSpec] = {}
    for i, item in enumerate(agents_raw):
        if not isinstance(item, dict):
            raise ManifestError(f"agents[{i}] bir eşleme (mapping) olmalı")
        try:
            spec = AgentSpec(**item)
        except Exception as exc:  # pydantic doğrulama hatası → açık mesaj
            raise ManifestError(f"agents[{i}] geçersiz: {exc}") from exc
        if spec.agent_id in registry:
            raise ManifestError(f"Yinelenen agent_id: {spec.agent_id}")
        registry[spec.agent_id] = spec
    return registry


@lru_cache(maxsize=1)
def _default_registry() -> dict[str, AgentSpec]:
    """Varsayılan (kök) manifest'i bir kez yükle, önbelleğe al."""
    return load_agent_registry()


def _registry(path: str | Path | None = None) -> dict[str, AgentSpec]:
    return load_agent_registry(path) if path is not None else _default_registry()


def list_agents(path: str | Path | None = None) -> list[AgentSpec]:
    """Tüm kayıtlı ajanları döndür."""
    return list(_registry(path).values())


def get_agent(agent_id: str, path: str | Path | None = None) -> AgentSpec:
    """Bir ajanı id ile getir. Yoksa ``KeyError``."""
    reg = _registry(path)
    if agent_id not in reg:
        raise KeyError(f"Bilinmeyen agent_id: {agent_id}")
    return reg[agent_id]


def dangerous_agents(path: str | Path | None = None) -> list[AgentSpec]:
    """``dangerous`` işaretli ajanlar (denetimsiz çalıştırılması riskli)."""
    return [a for a in list_agents(path) if a.dangerous]


def agents_requiring_approval(path: str | Path | None = None) -> list[AgentSpec]:
    """Tehlikeli adımları için insan onayı gereken ajanlar."""
    approval_autonomy = {
        AgentAutonomy.requires_approval,
        AgentAutonomy.dangerous_without_approval,
    }
    return [a for a in list_agents(path) if a.approval_required or a.autonomy in approval_autonomy]
