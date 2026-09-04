"""L3 ApplicationExam testleri — stub LLM ile harness'i deterministik doğrular.

Tamamen çevrimdışı: gerçek model çağrısı yok; stub doğru/yanlış/eksik diziler veya
'kullanılamıyor' döndürerek sınav mantığını kanıtlar.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from app.trading.indicators import compute_indicator
from app.verification.exams.l3_application import ApplicationExam, _parse_values
from app.verification.exams.reference_oracle import ReferenceOracle


class _StubLLM:
    """LocalLLM yerine geçen sahte istemci (available + generate)."""

    def __init__(self, response: str = "", *, available: bool = True) -> None:
        self._response = response
        self._available = available

    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self._response


def _reference_values(name: str, period: int, n_points: int, seed: int) -> list[float]:
    closes = ReferenceOracle.synthetic_closes(n_points, seed)
    df = pd.DataFrame({"close": closes})
    ref = compute_indicator(name, df, period).to_numpy(dtype=float)
    return ref[~np.isnan(ref)].tolist()


def test_dogru_dizi_pass() -> None:
    ref = _reference_values("SMA", 3, 8, 0)
    stub = _StubLLM(json.dumps({"values": ref}))
    res = ApplicationExam(llm=stub).run_by_name("SMA", period=3, n_points=8, seed=0)
    assert res.passed is True
    assert res.status == "passed"
    assert res.detail["max_abs_err"] < 1e-6


def test_ema_dogru_dizi_pass() -> None:
    # EMA tüm barlarda tanımlı (adjust=False) → start=1, n=8
    ref = _reference_values("EMA", 3, 8, 0)
    assert len(ref) == 8
    stub = _StubLLM(json.dumps({"values": ref}))
    res = ApplicationExam(llm=stub).run_by_name("EMA", period=3, n_points=8, seed=0)
    assert res.passed is True


def test_yanlis_dizi_fail() -> None:
    ref = _reference_values("SMA", 3, 8, 0)
    wrong = [v + 10.0 for v in ref]
    stub = _StubLLM(json.dumps({"values": wrong}))
    res = ApplicationExam(llm=stub).run_by_name("SMA", period=3, n_points=8, seed=0)
    assert res.passed is False
    assert res.status == "failed"


def test_eksik_uzunluk_fail() -> None:
    ref = _reference_values("SMA", 3, 8, 0)
    stub = _StubLLM(json.dumps({"values": ref[:-1]}))  # bir eksik
    res = ApplicationExam(llm=stub).run_by_name("SMA", period=3, n_points=8, seed=0)
    assert res.passed is False
    assert res.status == "failed"
    assert res.detail["n_got"] == len(ref) - 1


def test_parse_hatasi_fail() -> None:
    stub = _StubLLM("üzgünüm bir dizi veremem")
    res = ApplicationExam(llm=stub).run_by_name("SMA", period=3, n_points=8, seed=0)
    assert res.passed is False
    assert res.status == "failed"


def test_llm_yoksa_skipped() -> None:
    stub = _StubLLM("", available=False)
    res = ApplicationExam(llm=stub).run_by_name("SMA", period=3, n_points=8, seed=0)
    assert res.passed is False
    assert res.status == "skipped"  # sahte pass YOK


def test_carpik_json_ayni_seed_deterministik() -> None:
    # Çıplak [...] dizisi de parse edilmeli
    ref = _reference_values("SMA", 4, 10, 5)
    stub = _StubLLM("İşte sonuç: " + json.dumps(ref))
    res = ApplicationExam(llm=stub).run_by_name("SMA", period=4, n_points=10, seed=5)
    assert res.passed is True


def test_rsi_exam_reference_masks_warmup() -> None:
    # RSI sınav referansı fillna(50) konvansiyonunu KULLANMAZ: bar 0 NaN, tanımlı
    # bölge bitişik bir sonek (ortada delik yok) — doğru model haksızca fail almaz.
    from app.verification.exams.registry import get_spec

    closes = ReferenceOracle.synthetic_closes(20, 0)
    df = pd.DataFrame({"close": closes})
    ref = get_spec("RSI").reference(df, 14).to_numpy(dtype=float)
    assert np.isnan(ref[0])
    defined = ~np.isnan(ref)
    first = int(np.argmax(defined))
    assert defined[first:].all()


def test_rsi_l3_pass_with_correct_values() -> None:
    from app.verification.exams.registry import get_spec

    closes = ReferenceOracle.synthetic_closes(20, 0)
    df = pd.DataFrame({"close": closes})
    ref = get_spec("RSI").reference(df, 14).to_numpy(dtype=float)
    ref_vals = ref[~np.isnan(ref)].tolist()
    stub = _StubLLM(json.dumps({"values": ref_vals}))
    res = ApplicationExam(llm=stub).run_by_name("RSI", period=14, n_points=20, seed=0)
    assert res.status == "passed"


def test_permentropy_l3_not_dead_exam() -> None:
    # min_points uygulanmazsa 8 nokta → referans hep NaN → no_data (ölü sınav).
    from app.verification.exams.registry import get_spec

    spec = get_spec("PERMENTROPY")
    closes = ReferenceOracle.synthetic_closes(max(8, spec.min_points), 0)
    df = pd.DataFrame({"close": closes})
    ref = spec.reference(df, spec.default_period).to_numpy(dtype=float)
    ref_vals = ref[~np.isnan(ref)].tolist()
    assert ref_vals  # tanımlı değer var
    stub = _StubLLM(json.dumps({"values": ref_vals}))
    res = ApplicationExam(llm=stub).run_by_name("PERMENTROPY", n_points=8, seed=0)
    assert res.status == "passed"  # no_data DEĞİL


def test_entropy_exam_reference_excludes_phantom_first_bar() -> None:
    # ENTROPY sınav referansı bar 0'daki NaN-kaynaklı sahte 'düşüşü' SAYMAZ: ilk `period`
    # bar warmup (NaN), tanımlı bölge index period'tan (period gerçek fark) başlar.
    from app.verification.exams.registry import get_spec

    closes = ReferenceOracle.synthetic_closes(20, 0)
    df = pd.DataFrame({"close": closes})
    ref = get_spec("ENTROPY").reference(df, 4).to_numpy(dtype=float)
    assert np.all(np.isnan(ref[:4]))
    assert not np.isnan(ref[4])


def test_parse_values_birimleri() -> None:
    assert _parse_values('{"values": [1, 2.5, 3]}') == [1.0, 2.5, 3.0]
    assert _parse_values("[4, 5]") == [4.0, 5.0]
    assert _parse_values('{"values": ["x"]}') is None
    assert _parse_values("hiç sayı yok") is None
    assert _parse_values('{"values": []}') is None
