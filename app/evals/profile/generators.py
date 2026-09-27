"""Eval'de karşılaştırılan sistemlerin üretici arka uçları.

A: base model            → Ollama base modeli, RAG yok
B: base + RAG            → aynı model, aynı retrieval bağlamı
C: base + LoRA/profil    → profilin servis modeli (GGUF→Ollama adı), RAG yok
D: base + RAG + profil   → C + B'deki AYNI retrieval bağlamı

Tüm sistemler AYNI system prompt, prompt şablonu, üretim parametreleri (temperature,
max_tokens, seed) ile çağrılır. Bulut sağlayıcı YOK — yalnız yerel Ollama.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import httpx

from app.config import get_settings


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 768
    seed: int = 42
    top_p: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Generation:
    text: str
    latency_s: float
    output_tokens: int | None = None
    tokens_per_second: float | None = None


class Generator(Protocol):
    name: str

    def generate(self, prompt: str, *, system: str, config: GenerationConfig) -> Generation: ...


class OllamaGenerator:
    """Yerel Ollama /api/generate — token/sn ve gecikme ölçümüyle."""

    def __init__(self, model: str, host: str | None = None, timeout_s: float = 600.0) -> None:
        self.name = model
        self.model = model
        self.host = (host or get_settings().ollama_host).rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, prompt: str, *, system: str, config: GenerationConfig) -> Generation:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "think": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "seed": config.seed,
                "num_predict": config.max_tokens,
            },
        }
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(f"{self.host}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
        latency = time.perf_counter() - t0
        n = data.get("eval_count")
        dur_ns = data.get("eval_duration") or 0
        tps = (n / (dur_ns / 1e9)) if n and dur_ns else None
        return Generation(
            text=str(data.get("response") or ""),
            latency_s=latency,
            output_tokens=n,
            tokens_per_second=tps,
        )


@dataclass
class SystemSpec:
    """Karşılaştırılan tek konfigürasyon."""

    name: str  # ör. "A_base", "D_rag+balanced_v1_svd"
    label: str  # rapor satırı adı (ör. "Base + RAG + Balanced Profile")
    generator: Generator | None  # None → koşulamaz (unavailable_reason)
    use_rag: bool
    lora: str | None = None  # profile_id veya domain adapter id
    lora_kind: str | None = None  # "profile" | "adapter" | None
    unavailable_reason: str = ""
    base_counterpart: str | None = None  # aynı RAG ayarlı LoRA'sız sistem (teşhis)
    individual_counterparts: tuple[str, ...] = ()  # birleşik profil ↔ tekil adapter sistemleri
