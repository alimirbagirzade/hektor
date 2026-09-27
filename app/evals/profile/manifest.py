"""Eval koşusu tekrar-üretilebilirlik manifesti.

``replay_key`` zaman/koşu kimliği HARİÇ tüm girdilerin hash'idir: aynı replay_key ile
(aynı model, veri, RAG, config, seed) koşu yeniden üretilebilir olmalıdır.
"""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from app.lora.mix_common import hash_obj

_SOFTWARE = ("hektor", "torch", "transformers", "peft", "chromadb", "pydantic", "httpx")


def software_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for pkg in _SOFTWARE:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    return out


def _gpu_name() -> str | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu_name(),
    }
    try:
        import psutil

        info["ram_total_mb"] = round(psutil.virtual_memory().total / 2**20)
    except ImportError:  # pragma: no cover
        pass
    return info


@dataclass
class RunManifest:
    run_id: str
    timestamp: str
    git_commit: str
    base_model: str
    base_model_hash: str
    tokenizer_hash: str
    quantization: str
    rag_version: str
    rag_index_hash: str
    embedding_model: str
    reranker: str
    adapter_versions: dict[str, str]
    profile: str | None
    merge_method: str | None
    weights: dict[str, float]
    eval_dataset: str
    eval_dataset_hash: str
    eval_config_hash: str
    generation_config: dict[str, Any]
    seed: int
    systems: list[str]
    hardware: dict[str, Any] = field(default_factory=hardware_info)
    software_versions: dict[str, str] = field(default_factory=software_versions)

    @property
    def replay_key(self) -> str:
        d = asdict(self)
        for volatile in ("run_id", "timestamp", "hardware", "software_versions"):
            d.pop(volatile, None)
        return hash_obj(d)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["replay_key"] = self.replay_key
        return d


def new_run_id(seed_material: dict[str, Any]) -> str:
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    return f"run_{ts}_{hash_obj(seed_material)[:8]}"
