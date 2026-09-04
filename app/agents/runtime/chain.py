"""Çalıştırma zinciri (chain) — manifest'teki bildirimsel topolojik sıra.

``automation_manifest.yaml`` içindeki ``chain`` bölümü, ajanların hangi SIRADA
koşacağını (bağımlılıklarıyla) DATA olarak tanımlar — tasarım diyagramının tek
kaynağı. Bu modül onu okur, DOĞRULAR (her step manifest'te kayıtlı bir ajan mı,
``after`` referansları geçerli mi, DÖNGÜ var mı) ve topolojik sıralı adımları döndürür.

Gate/otonomi bilgisi ayrı tutulmaz: ``AgentSpec``'ten türetilir (tek kaynak). Böylece
zincir yalnız SIRAYI ekler; "tehlikeli mi / onay ister mi" bilgisi ajanın kendisinde kalır.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.agents.runtime.registry import _manifest_path, load_agent_registry
from app.agents.runtime.schemas import AgentAutonomy, AgentSpec

_APPROVAL_AUTONOMY = {AgentAutonomy.requires_approval, AgentAutonomy.dangerous_without_approval}


class ChainError(RuntimeError):
    """``chain`` bölümü eksik / geçersiz / döngülü."""


class ChainStep(BaseModel):
    """Zincirdeki tek bir adım: bir ajan + ondan önce gelmesi gereken ajanlar."""

    step: str
    after: list[str] = Field(default_factory=list)


class ResolvedStep(BaseModel):
    """Topolojik sıraya konmuş, AgentSpec'ten zenginleştirilmiş adım."""

    order: int
    step: str
    name: str
    autonomy: str
    dangerous: bool
    requires_approval: bool
    after: list[str] = Field(default_factory=list)


def _needs_approval(spec: AgentSpec) -> bool:
    return spec.approval_required or spec.autonomy in _APPROVAL_AUTONOMY


def load_chain(path: str | Path | None = None) -> list[ChainStep]:
    """Manifest'in ``chain`` bölümünü oku → ``ChainStep`` listesi. Bozuksa ``ChainError``."""
    p = _manifest_path(path)
    if not p.exists():
        raise ChainError(f"Manifest bulunamadı: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ChainError(f"Manifest YAML ayrıştırılamadı ({p}): {exc}") from exc
    if not isinstance(raw, dict) or "chain" not in raw:
        raise ChainError(f"Manifest 'chain' anahtarı içermeli: {p}")
    chain_raw = raw["chain"]
    if not isinstance(chain_raw, list) or not chain_raw:
        raise ChainError("Manifest 'chain' boş olmayan bir liste olmalı")
    steps: list[ChainStep] = []
    for i, item in enumerate(chain_raw):
        if not isinstance(item, dict):
            raise ChainError(f"chain[{i}] bir eşleme (mapping) olmalı")
        try:
            steps.append(ChainStep(**item))
        except Exception as exc:
            raise ChainError(f"chain[{i}] geçersiz: {exc}") from exc
    return steps


def _topo_sort(steps: list[ChainStep]) -> list[ChainStep]:
    """Kahn algoritması; bildirim sırasını koruyarak topolojik sırala. Döngü → ChainError."""
    by_id = {s.step: s for s in steps}
    # after referansları resolve_chain'de doğrulandığı için len(after) = in-degree.
    indeg = {s.step: len(s.after) for s in steps}
    # in-degree 0 olanları BİLDİRİM sırasında işle (kararlı, okunur çıktı)
    ready = [s.step for s in steps if indeg[s.step] == 0]
    ordered: list[str] = []
    while ready:
        cur = ready.pop(0)
        ordered.append(cur)
        for s in steps:  # cur'a bağımlı olanların in-degree'sini düşür
            if cur in s.after:
                indeg[s.step] -= 1
                if indeg[s.step] == 0:
                    ready.append(s.step)
    if len(ordered) != len(steps):
        kalan = [sid for sid in by_id if sid not in ordered]
        raise ChainError(f"Zincirde döngü var (çözülemeyen: {kalan})")
    return [by_id[sid] for sid in ordered]


def resolve_chain(path: str | Path | None = None) -> list[ResolvedStep]:
    """Zinciri yükle + doğrula + topolojik sırala + AgentSpec ile zenginleştir.

    Doğrulamalar: her ``step`` manifest registry'de kayıtlı; her ``after`` zincirde
    tanımlı bir step; tekrarlanan step yok; döngü yok. Aksi halde ``ChainError``.
    """
    steps = load_chain(path)
    registry = load_agent_registry(path)

    seen: set[str] = set()
    step_ids = {s.step for s in steps}
    for s in steps:
        if s.step in seen:
            raise ChainError(f"Zincirde yinelenen step: {s.step}")
        seen.add(s.step)
        if s.step not in registry:
            raise ChainError(f"Zincir step'i registry'de yok: {s.step}")
        for dep in s.after:
            if dep not in step_ids:
                raise ChainError(f"'{s.step}' adımının 'after' bağımlılığı zincirde yok: {dep}")

    ordered = _topo_sort(steps)
    resolved: list[ResolvedStep] = []
    for i, s in enumerate(ordered, start=1):
        spec = registry[s.step]
        resolved.append(
            ResolvedStep(
                order=i,
                step=s.step,
                name=spec.name,
                autonomy=spec.autonomy.value,
                dangerous=spec.dangerous,
                requires_approval=_needs_approval(spec),
                after=s.after,
            )
        )
    return resolved
