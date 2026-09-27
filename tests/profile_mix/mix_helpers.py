"""Profil karışım/eval testleri için ortak yardımcılar (sahte adapter/üretici/retriever)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.evals.profile.generators import Generation, GenerationConfig
from app.lora.domain_adapter_registry import DomainAdapterRecord

BASE_HASH = "b" * 64
TOK_HASH = "t" * 64
TARGETS = ["k_proj", "o_proj", "q_proj", "v_proj"]


def make_record(domain: str, version: str = "v1", **over: Any) -> DomainAdapterRecord:
    data: dict[str, Any] = {
        "adapter_id": f"{domain}_lora",
        "adapter_version": version,
        "domain": domain,
        "adapter_path": f"models/adapters/{domain}_{version}",
        "base_model": "Qwen/Qwen3-30B-A3B-Instruct",
        "base_model_revision": "rev1",
        "base_model_hash": BASE_HASH,
        "tokenizer_hash": TOK_HASH,
        "dataset_version": f"{domain}-ds-1",
        "dataset_hash": f"{domain}-hash",
        "training_config_hash": "cfg-hash",
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": list(TARGETS),
        "learning_rate": 1e-4,
        "epochs": 1,
        "max_seq_length": 2048,
        "training_seed": 42,
        "git_commit": "abc123",
    }
    data.update(over)
    return DomainAdapterRecord(**data)


@dataclass
class FakeChunk:
    chunk_id: str
    paper_id: str
    text: str
    page_number: int | None = None
    section_name: str | None = None
    title: str | None = None
    distance: float | None = 0.1

    @property
    def citation(self) -> str:
        return f"[{self.paper_id}:{self.chunk_id}]"


class FakeRetriever:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[FakeChunk]:
        self.calls.append(query)
        return self.chunks[: top_k or len(self.chunks)]


@dataclass
class FakeGenerator:
    """Soru metnindeki anahtar → sabit cevap; tüm çağrıları kaydeder."""

    name: str
    answers: dict[str, str]
    default: str = "Bilmiyorum."
    calls: list[tuple[str, str, GenerationConfig]] = field(default_factory=list)

    def generate(self, prompt: str, *, system: str, config: GenerationConfig) -> Generation:
        self.calls.append((prompt, system, config))
        for key, ans in self.answers.items():
            if key in prompt:
                return Generation(
                    text=ans, latency_s=0.01, output_tokens=10, tokens_per_second=100.0
                )
        return Generation(
            text=self.default, latency_s=0.01, output_tokens=2, tokens_per_second=100.0
        )
