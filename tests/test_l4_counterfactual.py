"""L4 CounterfactualExam testleri — referans yön KODDAN, model ona karşı puanlanır."""

from __future__ import annotations

import json

import pandas as pd

from app.verification.exams.l4_counterfactual import (
    CounterfactualExam,
    _normalize,
    _roughness,
)
from app.verification.exams.reference_oracle import ReferenceOracle
from app.verification.exams.registry import get_spec


class _StubLLM:
    def __init__(self, response: str = "", *, available: bool = True) -> None:
        self._response = response
        self._available = available

    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self._response


def test_truth_uzun_periyot_daha_pürüzsüz() -> None:
    # Kod-gerçeği: SMA periyodu artınca pürüzlülük (diff std) AZALIR.
    closes = ReferenceOracle.synthetic_closes(200, seed=0)
    df = pd.DataFrame({"close": closes})
    spec = get_spec("SMA")
    assert _roughness(spec.reference(df, 6).to_numpy(float)) < _roughness(
        spec.reference(df, 3).to_numpy(float)
    )


def test_dogru_yon_pass() -> None:
    stub = _StubLLM(json.dumps({"direction": "azalir"}))
    res = CounterfactualExam(llm=stub).run_by_name("SMA", period=3, factor=2, seed=0)
    assert res.detail["truth"] == "azalir"  # koddan
    assert res.passed is True
    assert res.status == "passed"


def test_yanlis_yon_fail() -> None:
    stub = _StubLLM(json.dumps({"direction": "artar"}))
    res = CounterfactualExam(llm=stub).run_by_name("SMA", period=3, factor=2, seed=0)
    assert res.passed is False
    assert res.status == "failed"


def test_serbest_metin_yon() -> None:
    # JSON olmadan da "azalır" kelimesi yakalanmalı
    stub = _StubLLM("Bence oynaklık azalır çünkü daha çok değer ortalanır.")
    res = CounterfactualExam(llm=stub).run_by_name("EMA", period=3, factor=3, seed=0)
    assert res.passed is True


def test_parse_edilemez_fail() -> None:
    stub = _StubLLM("bilmiyorum, emin değilim")
    res = CounterfactualExam(llm=stub).run_by_name("SMA", period=3, factor=2, seed=0)
    assert res.passed is False
    assert res.status == "failed"


def test_llm_yoksa_skipped() -> None:
    stub = _StubLLM("", available=False)
    res = CounterfactualExam(llm=stub).run_by_name("SMA", period=3, factor=2, seed=0)
    assert res.status == "skipped"


def test_belirgin_yon_yoksa_no_data() -> None:
    # factor=1 → perturbe == baseline → delta=0 → yazı-tura yapma, no_data dön
    stub = _StubLLM(json.dumps({"direction": "azalir"}))
    res = CounterfactualExam(llm=stub).run_by_name("SMA", period=3, factor=1, seed=0)
    assert res.status == "no_data"


def test_normalize_yonler() -> None:
    assert _normalize("azalır") == "azalir"
    assert _normalize("artar") == "artar"
    assert _normalize("aynı kalır") == "ayni"
    assert _normalize("decreases") == "azalir"
    assert _normalize("kuş uçtu") is None


def test_normalize_olumsuzluk_ve_registry_dili() -> None:
    # Olumsuzluk: alt-dizgi 'art'/'azal' yönü TERS çevirmemeli (eski bug).
    assert _normalize("artmaz") == "ayni"  # eskiden yanlışlıkla 'artar'
    assert _normalize("azalmaz") == "ayni"  # eskiden yanlışlıkla 'azalir'
    assert _normalize("değişmez") == "ayni"
    assert _normalize("no change") == "ayni"
    # Registry'nin kendi monoton dili artık tanınmalı (eskiden None).
    assert _normalize("daha pürüzsüz") == "azalir"  # registry SMA/EMA/ENTROPY dili
    assert _normalize("daha az oynak") == "azalir"
