"""Profil eval veri şeması (pydantic v2) — tek eval kaydı + sonuç yapıları."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

EVAL_DOMAINS: tuple[str, ...] = ("math", "statistics", "reasoning", "trading", "coding")


class Split(StrEnum):
    """Train (LoRA eğitim verisi, burada yok) / validation / golden_test ayrımı.

    - validation : profil/ağırlık SEÇİMİNDE kullanılabilir.
    - golden_test: YALNIZ profil seçildikten sonra nihai karşılaştırmada kullanılır.
    """

    VALIDATION = "validation"
    GOLDEN_TEST = "golden_test"


class EvalItem(BaseModel):
    """Tek eval kaydı (istenen şema birebir)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    domain: str
    subdomain: str = ""
    difficulty: int = Field(default=3, ge=1, le=5)
    question: str
    reference_answer: str = ""
    accepted_answers: list[str] = Field(default_factory=list)
    expected_numeric_value: float | None = None
    numeric_tolerance: float | None = None
    required_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    requires_rag: bool = False
    expected_document_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    requires_calculation: bool = False
    requires_code_execution: bool = False
    requires_abstention: bool = False
    source_provenance: str = ""
    human_verified: bool = False
    # Kod soruları için (requires_code_execution): model cevabındaki kodla birlikte
    # çalıştırılacak test kodu (assert'ler). Şemaya ek, opsiyonel.
    code_tests: str = ""

    @field_validator("domain")
    @classmethod
    def _domain_ok(cls, v: str) -> str:
        if v not in EVAL_DOMAINS:
            raise ValueError(f"domain {v!r} geçersiz; geçerli: {EVAL_DOMAINS}")
        return v


class ItemScore(BaseModel):
    """Bir kaydın bir sistemdeki deterministik skoru + teşhis."""

    item_id: str
    domain: str
    system: str
    correct: bool | None  # None = ölçülemedi (ör. kod yürütme kapalı)
    score: float | None
    checks: dict[str, Any] = Field(default_factory=dict)
    abstained: bool = False
    hallucinated: bool = False
    grounding: float | None = None
    citation_accuracy: float | None = None
    retrieval_recall: float | None = None
    retrieval_precision: float | None = None
    context_relevance: float | None = None
    failure_categories: list[str] = Field(default_factory=list)
    answer: str = ""
    latency_s: float | None = None
    tokens_per_second: float | None = None
    retrieval: dict[str, Any] | None = None
