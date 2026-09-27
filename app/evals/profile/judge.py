"""LLM judge sözleşmesi — YALNIZ deterministik ölçülemeyen boyutlar için.

Kullanım alanı: reasoning quality, explanation quality, semantic correctness, grounding
(atıf eşleşmesinin ötesinde içerik desteği). Matematik/kod/retrieval/atıf için KULLANILMAZ.

Saklananlar (yalnız skor değil): decision, supported_claims, unsupported_claims,
contradicted_claims, evidence_chunk_ids, short_justification. Uzun chain-of-thought
İSTENMEZ ve SAKLANMAZ (justification 300 karakterle kırpılır).

Judge sonucu tek başına production terfisi VEREMEZ — regression_eval yalnız
deterministik metrikleri kullanır; judge çıktısı raporda ayrı sütundur.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

JUDGE_ASPECTS: tuple[str, ...] = (
    "reasoning_quality",
    "explanation_quality",
    "semantic_correctness",
    "grounding",
)
MAX_JUSTIFICATION = 300

JUDGE_SYSTEM_PROMPT = (
    "Sen bir değerlendiricisin. YALNIZ tek bir JSON nesnesi döndür; düşünce zinciri, "
    "açıklama paragrafı YAZMA. Alanlar: score (0-1), decision (PASS|PARTIAL|FAIL), "
    "supported_claims, unsupported_claims, contradicted_claims (kısa iddia listeleri), "
    "evidence_chunk_ids (verilen bağlamdan chunk kimlikleri), short_justification "
    "(en fazla 2 cümle)."
)


class JudgeVerdict(BaseModel):
    aspect: str
    score: float = Field(ge=0.0, le=1.0)
    decision: Literal["PASS", "PARTIAL", "FAIL"]
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradicted_claims: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    short_justification: str = ""
    judge_model: str = ""
    can_promote: Literal[False] = False  # sözleşme: judge tek başına terfi veremez

    @field_validator("aspect")
    @classmethod
    def _aspect(cls, v: str) -> str:
        if v not in JUDGE_ASPECTS:
            raise ValueError(f"judge yalnız {JUDGE_ASPECTS} için kullanılabilir: {v}")
        return v

    @field_validator("short_justification")
    @classmethod
    def _short(cls, v: str) -> str:
        return v.strip()[:MAX_JUSTIFICATION]


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str, *, aspect: str, judge_model: str = "") -> JudgeVerdict:
    """Judge ham çıktısından ilk JSON nesnesini çıkar ve doğrula (eval/exec YOK)."""
    m = _JSON_OBJ.search(raw)
    if not m:
        raise ValueError("judge çıktısında JSON nesnesi yok")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("judge çıktısı nesne değil")
    data.pop("can_promote", None)
    data.update({"aspect": aspect, "judge_model": judge_model})
    return JudgeVerdict.model_validate(data)
