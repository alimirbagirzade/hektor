from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.agents.system_profiler.profiler import SystemProfile

_REGISTRY_PATH = Path(__file__).parent.parent.parent / "registry" / "model_registry.yaml"


@dataclass
class ModelRecommendation:
    rank: int
    model_id: str
    display_name: str
    backend: str
    ollama_name: str
    confidence: float
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class RejectedModel:
    model_id: str
    display_name: str
    reason: str


@dataclass
class AdvisorResult:
    recommended: list[ModelRecommendation]
    rejected: list[RejectedModel]
    system_summary: str


def _load_registry(path: Path = _REGISTRY_PATH) -> list[dict[str, Any]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def _score_model(
    model: dict[str, Any], profile: SystemProfile, task: str
) -> tuple[float, list[str], str | None]:
    """
    Modeli profille karşılaştır. (skor, nedenler, red_nedeni) döndür.
    Red_nedeni None ise model uygun demektir.
    """
    ram = profile.memory.ram_total_gb
    vram = profile.gpu.vram_gb
    has_dedicated = profile.has_dedicated_gpu
    is_apple = profile.is_apple_silicon

    min_ram: float = model.get("min_ram_gb", 0)
    rec_ram: float = model.get("recommended_ram_gb", min_ram)
    min_vram: float = model.get("min_vram_gb", 0)
    rec_vram: float = model.get("recommended_vram_gb", min_vram)

    # --- Sert reddler ---
    if ram < min_ram:
        return (
            -1.0,
            [],
            f"Yetersiz RAM: {ram:.0f} GB < minimum {min_ram:.0f} GB",
        )
    if (
        min_vram > 0
        and not has_dedicated
        and vram < min_vram
        and not (is_apple and ram >= min_vram * 2)
    ):
        return (
            -1.0,
            [],
            f"Dedicated GPU yok ve VRAM {vram:.1f} GB < minimum {min_vram:.0f} GB",
        )

    score = 0.0
    reasons: list[str] = []

    # RAM uyum puanı (0–40)
    if ram >= rec_ram:
        score += 40
        reasons.append(f"RAM yeterli: {ram:.0f} GB ≥ önerilen {rec_ram:.0f} GB")
    elif ram >= min_ram:
        ratio = (ram - min_ram) / max(rec_ram - min_ram, 1)
        score += 20 + 20 * ratio
        reasons.append(f"RAM minimum düzeyde: {ram:.0f} GB (önerilen {rec_ram:.0f} GB)")

    # GPU bonus (0–30)
    if is_apple:
        score += 25
        reasons.append("Apple Silicon Metal hızlandırma")
    elif profile.gpu.cuda and vram >= rec_vram:
        score += 30
        reasons.append(f"CUDA GPU: {vram:.1f} GB VRAM ≥ önerilen {rec_vram:.0f} GB")
    elif profile.gpu.cuda:
        score += 15
        reasons.append(f"CUDA GPU mevcut ama VRAM {vram:.1f} GB < önerilen {rec_vram:.0f} GB")

    # Görev eşleşmesi (0–20)
    task_tags: list[str] = model.get("task_tags", [])
    if task in task_tags:
        score += 20
        reasons.append(f"Görev eşleşmesi: '{task}'")
    elif "general" in task_tags:
        score += 8

    # Kalite/hız dengesi (0–10)
    q: int = model.get("quality_score", 5)
    s: int = model.get("speed_score", 5)
    score += (q + s) / 2

    # Büyük model bonusu: donanım tüm gereksinimleri karşılıyorsa kaliteyi ön plana çıkar
    if ram >= rec_ram and ((has_dedicated and vram >= rec_vram) or is_apple):
        score += q * 0.5  # tam donanım uyumu → kalite ağırlığını artır

    return score, reasons, None


def recommend(
    profile: SystemProfile,
    task: str = "general",
    top_k: int = 3,
    registry_path: Path = _REGISTRY_PATH,
) -> AdvisorResult:
    """Sistem profiline ve göreve göre model öner."""
    models = _load_registry(registry_path)

    scored: list[tuple[float, dict, list[str]]] = []
    rejected: list[RejectedModel] = []

    for m in models:
        score, reasons, reject_reason = _score_model(m, profile, task)
        if reject_reason:
            rejected.append(
                RejectedModel(
                    model_id=m["id"],
                    display_name=m.get("display_name", m["id"]),
                    reason=reject_reason,
                )
            )
        else:
            scored.append((score, m, reasons))

    scored.sort(key=lambda x: x[0], reverse=True)

    recommended: list[ModelRecommendation] = []
    for i, (score, m, reasons) in enumerate(scored[:top_k]):
        ollama_name = m.get("backends", {}).get("ollama", {}).get("name", m["id"])
        confidence = min(1.0, round(score / 100, 2))
        recommended.append(
            ModelRecommendation(
                rank=i + 1,
                model_id=m["id"],
                display_name=m.get("display_name", m["id"]),
                backend="ollama",
                ollama_name=ollama_name,
                confidence=confidence,
                score=round(score, 1),
                reasons=reasons,
            )
        )

    system_summary = (
        f"{profile.cpu.name} · {profile.memory.ram_total_gb:.0f} GB RAM · "
        f"{profile.gpu.name} · {profile.os}"
    )

    return AdvisorResult(
        recommended=recommended,
        rejected=rejected,
        system_summary=system_summary,
    )
