"""train-doctor — gerçek eğitimden ÖNCE rakip LLM/GPU yükünü tespit et (SALT-OKUMA).

Sorun: yerel Ollama bir modeli belleğe/GPU'ya yükledikten sonra ``ollama_keep_alive``
süresi dolana kadar orada tutar. Aynı anda gerçek LoRA eğitimi (`train --run`)
başlatılırsa iki yük aynı VRAM/RAM'i paylaşır → OOM ya da sürünen hız. Bu modül,
eğitim başlamadan önce bu çakışmayı tespit eder ve GO/WARN/NO-GO kararı üretir.

Hiçbir süreci durdurmaz/kill etmez, hiçbir modeli boşaltmaz — yalnız rapor + öneri
(``ollama stop <model>``). Ölçüm iki kaynaktan gelir:

1. Ollama ``/api/ps`` (hangi model yüklü, ne kadar VRAM/RAM tutuyor) — yerel HTTP,
   ağ değil (CLAUDE.md: LLM hattı yalnız yerel Ollama).
2. ``nvidia-smi`` (varsa) — TÜM GPU süreçlerinin toplam kullanımı; yalnız Ollama değil,
   başka bir eğitim/oyun/tarayıcı sürecini de yakalar. Yoksa (Apple Silicon/AMD/GPU'suz)
   yalnız Ollama sinyaline düşülür.
"""

from __future__ import annotations

import subprocess
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from app.config import get_settings

Verdict = Literal["GO", "WARN", "NO-GO"]


class LoadedModel(BaseModel):
    name: str = "unknown"
    size_gb: float = 0.0
    vram_gb: float = 0.0
    fully_on_gpu: bool = False


class TrainDoctorReport(BaseModel):
    ollama_reachable: bool = False
    loaded_models: list[LoadedModel] = Field(default_factory=list)
    competing_vram_gb: float = 0.0
    gpu_source: str = "yok"  # "nvidia-smi" | "ollama-tahmini" | "yok"
    gpu_total_vram_gb: float | None = None
    gpu_used_vram_gb: float | None = None
    free_vram_gb: float | None = None
    min_free_vram_gb: float = 0.0
    verdict: Verdict = "GO"
    reasons: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _fetch_ollama_ps(
    host: str,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 5.0,
) -> list[dict[str, Any]] | None:
    """``/api/ps`` yüklü-model listesini döndürür. Ulaşılamazsa ``None``."""
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            r = client.get(f"{host.rstrip('/')}/api/ps")
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    models = data.get("models")
    return list(models) if isinstance(models, list) else []


def _parse_loaded_models(raw: list[dict[str, Any]]) -> list[LoadedModel]:
    parsed: list[LoadedModel] = []
    for entry in raw:
        size_bytes = float(entry.get("size") or 0)
        vram_bytes = float(entry.get("size_vram") or 0)
        parsed.append(
            LoadedModel(
                name=str(entry.get("name") or entry.get("model") or "unknown"),
                size_gb=round(size_bytes / 1024**3, 2),
                vram_gb=round(vram_bytes / 1024**3, 2),
                fully_on_gpu=vram_bytes > 0 and vram_bytes >= size_bytes * 0.99,
            )
        )
    return parsed


def _nvidia_smi_memory_gb(*, timeout: float = 5.0) -> tuple[float, float] | None:
    """``(kullanılan_gb, toplam_gb)`` — TÜM GPU süreçleri. nvidia-smi yoksa ``None``."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=timeout,
        )
        line = out.strip().splitlines()[0]
        used_mb, total_mb = (float(x.strip()) for x in line.split(","))
        return round(used_mb / 1024, 2), round(total_mb / 1024, 2)
    except Exception:
        return None


def run_train_doctor(
    *,
    host: str | None = None,
    min_free_vram_gb: float | None = None,
    transport: httpx.BaseTransport | None = None,
    nvidia_smi: tuple[float, float] | None | Literal["auto"] = "auto",
) -> TrainDoctorReport:
    """Rakip LLM/GPU yükünü tara, GO/WARN/NO-GO kararı üret. Asla fırlatmaz."""
    settings = get_settings()
    ollama_host = (host or settings.ollama_host).rstrip("/")
    min_free = (
        min_free_vram_gb if min_free_vram_gb is not None else settings.train_doctor_min_free_vram_gb
    )

    report = TrainDoctorReport(min_free_vram_gb=min_free)

    raw = _fetch_ollama_ps(ollama_host, transport=transport)
    if raw is None:
        report.ollama_reachable = False
    else:
        report.ollama_reachable = True
        report.loaded_models = _parse_loaded_models(raw)
        report.competing_vram_gb = round(sum(m.vram_gb for m in report.loaded_models), 2)

    gpu_mem = nvidia_smi if nvidia_smi != "auto" else _nvidia_smi_memory_gb()
    if gpu_mem is not None:
        used_gb, total_gb = gpu_mem
        report.gpu_source = "nvidia-smi"
        report.gpu_used_vram_gb = used_gb
        report.gpu_total_vram_gb = total_gb
        report.free_vram_gb = round(total_gb - used_gb, 2)
    elif report.ollama_reachable and report.competing_vram_gb > 0:
        report.gpu_source = "ollama-tahmini"

    if report.loaded_models:
        names = ", ".join(f"{m.name} ({m.vram_gb:.1f}GB)" for m in report.loaded_models)
        report.reasons.append(f"Ollama'da halen belleğe yüklü model: {names}.")

    if report.free_vram_gb is not None:
        if report.free_vram_gb < min_free:
            report.verdict = "NO-GO"
            report.reasons.append(
                f"Ölçülen boş VRAM {report.free_vram_gb:.1f}GB < eşik {min_free:.1f}GB "
                f"(nvidia-smi: {report.gpu_used_vram_gb:.1f}/"
                f"{report.gpu_total_vram_gb:.1f}GB kullanımda)."
            )
            if report.loaded_models:
                stop_cmds = ", ".join(f"`ollama stop {m.name}`" for m in report.loaded_models)
                report.reasons.append(f"Öneri: {stop_cmds} sonra tekrar deneyin.")
            else:
                report.reasons.append(
                    "Ollama'da yüklü model yok — yükü başka bir süreç tutuyor "
                    "(görev yöneticisi / nvidia-smi ile kontrol edin)."
                )
        elif report.loaded_models:
            report.verdict = "WARN"
    elif report.loaded_models:
        # nvidia-smi yok (Apple Silicon/AMD/GPU'suz) — yalnız Ollama sinyaline düş.
        report.verdict = "WARN"
        report.reasons.append(
            "Gerçek VRAM ölçümü yok (nvidia-smi bulunamadı) — yalnız Ollama modeli baz alındı."
        )

    if not report.ollama_reachable and report.gpu_source == "yok":
        report.reasons.append("Ollama'ya ulaşılamadı ve nvidia-smi yok — rakip yük ölçülemedi.")

    return report
