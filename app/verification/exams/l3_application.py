"""L3 ApplicationExam — UYGULAMA sınavı (anlama merdiveninin en güçlü sinyali).

Modele bir göstergenin KESİN tanımı + TUTULAN (held-out) seed'li sayılar verilir;
modelin döndürdüğü sayısal dizi, ReferenceOracle'ın ürettiği referans değerle
``np.allclose`` ile karşılaştırılır. LLM-judge YOK, eval/exec YOK — saf sayısal kıyas.

Sonuçlar:
  - ``passed``  : model toleransta → anladı (uygulayabildi)
  - ``failed``  : yanlış sayı / parse hatası / eksik (uydurma cezası)
  - ``skipped`` : LLM yok → sahte 'pass' ÜRETİLMEZ (CLAUDE.md Kural 2)
  - ``no_data`` : referansın tanımlı bölgesi boş
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.verification.exams.reference_oracle import ReferenceOracle
from app.verification.exams.registry import ExamSpec, get_spec

__all__ = ["ApplicationExam", "ExamResult"]


@dataclass
class ExamResult:
    level: str
    name: str
    passed: bool
    status: str  # "passed" | "failed" | "skipped" | "no_data"
    seed: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROMPT = """\
Sen kesin bir hesap makinesisin. Aşağıdaki göstergeyi TANIMINA GÖRE hesapla.

TANIM: {definition}

GİRDİ — close fiyatları (1. değerden {n_in}. değere, sırayla):
{closes}

PERİYOT p = {period}

GÖREV: Göstergenin TANIMLI olduğu her bar için değerini hesapla — yani {start}. \
konumdan {n_in}. konuma kadar, toplam tam {n_out} sayı. Sırayı koru.

Yalnızca şu JSON'u döndür, başka HİÇBİR ŞEY yazma:
{{"values": [v1, v2, ...]}}
({n_out} adet sayı, virgülle, sırayla)
"""


class ApplicationExam:
    def __init__(self, llm: LocalLLM | None = None, oracle: ReferenceOracle | None = None) -> None:
        self.llm = llm or LocalLLM()
        self.oracle = oracle or ReferenceOracle()

    def run_by_name(
        self, name: str, *, period: int | None = None, n_points: int = 8, seed: int = 0
    ) -> ExamResult:
        return self.run(get_spec(name), period=period, n_points=n_points, seed=seed)

    def run(
        self, spec: ExamSpec, *, period: int | None = None, n_points: int = 8, seed: int = 0
    ) -> ExamResult:
        p = period or spec.default_period
        # Bazı göstergeler (ör. permütasyon entropi) geniş pencere ister; yetersiz
        # nokta → referans hep NaN → no_data (ölü sınav). spec.min_points bunu önler.
        n_points = max(n_points, spec.min_points)
        closes = self.oracle.synthetic_closes(n_points, seed)
        df = pd.DataFrame({"close": closes})
        ref = spec.reference(df, p).to_numpy(dtype=float)

        defined = ~np.isnan(ref)
        ref_vals = ref[defined]
        if ref_vals.size == 0:
            return ExamResult(
                "L3", spec.name, False, "no_data", seed, {"reason": "referans tanımlı bölge boş"}
            )

        # LLM yoksa: sahte pass YOK.
        if not self.llm.available():
            return ExamResult(
                "L3",
                spec.name,
                False,
                "skipped",
                seed,
                {
                    "reason": "LLM kullanılamıyor — sahte pass üretilmez",
                    "n_expected": ref_vals.size,
                },
            )

        start = int(np.argmax(defined)) + 1  # 1-tabanlı ilk tanımlı konum
        prompt = _PROMPT.format(
            definition=spec.definition,
            closes=closes.round(4).tolist(),
            period=p,
            n_in=n_points,
            start=start,
            n_out=ref_vals.size,
        )
        try:
            raw = self.llm.generate(
                prompt, fmt="json", temperature=0.0, max_tokens=512, timeout=240, seed=seed
            )
        except LLMUnavailable:
            return ExamResult("L3", spec.name, False, "skipped", seed, {"reason": "LLMUnavailable"})

        model_vals = _parse_values(raw)
        if model_vals is None or len(model_vals) != ref_vals.size:
            return ExamResult(
                "L3",
                spec.name,
                False,
                "failed",
                seed,
                {
                    "reason": "parse hatası ya da değer sayısı uyuşmuyor",
                    "n_expected": ref_vals.size,
                    "n_got": None if model_vals is None else len(model_vals),
                    "raw": raw[:200],
                    "reference": ref_vals.tolist(),
                },
            )

        model_arr = np.asarray(model_vals, dtype=float)
        ok = bool(np.allclose(model_arr, ref_vals, rtol=spec.rtol, atol=spec.atol))
        max_abs_err = float(np.max(np.abs(model_arr - ref_vals)))
        return ExamResult(
            "L3",
            spec.name,
            ok,
            "passed" if ok else "failed",
            seed,
            {
                "period": p,
                "reference": ref_vals.tolist(),
                "model": model_vals,
                "max_abs_err": max_abs_err,
                "rtol": spec.rtol,
                "atol": spec.atol,
            },
        )


def _parse_values(raw: str) -> list[float] | None:
    """Model çıktısından sayı listesini güvenle çıkar (kod çalıştırmadan, sadece JSON)."""
    raw = raw.strip()
    # 1) {"values": [...]} nesnesi
    obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj_match:
        try:
            obj = json.loads(obj_match.group())
            vals = obj.get("values") if isinstance(obj, dict) else None
            nums = _as_number_list(vals)
            if nums is not None:
                return nums
        except (json.JSONDecodeError, ValueError):
            pass
    # 2) çıplak [...] dizisi
    arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if arr_match:
        try:
            nums = _as_number_list(json.loads(arr_match.group()))
            if nums is not None:
                return nums
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _as_number_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[float] = []
    for x in value:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        out.append(float(x))
    return out
