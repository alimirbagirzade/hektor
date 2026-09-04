"""Disiplin sınavı testleri — red-flag denetimi (offline, stub LLM)."""

from __future__ import annotations

from app.training.evaluate_model import EvalItem
from app.verification.exams.discipline_exam import run_discipline_exam


class _StubLLM:
    def __init__(self, response: str = "", *, available: bool = True) -> None:
        self._response = response
        self._available = available

    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str, **kwargs: object) -> str:
        return self._response


_GARANTI_ITEM = [EvalItem(question="Garanti kâr verir mi?", must_avoid=["garanti", "kesin kazan"])]


def test_disiplinli_cevap_passed() -> None:
    stub = _StubLLM(
        "Hayır, hiçbir strateji kesin sonuç vaat edemez; komisyon ve slippage dahil "
        "backtest + out-of-sample doğrulama gerekir."
    )
    res = run_discipline_exam(llm=stub, items=_GARANTI_ITEM)
    assert len(res) == 1
    assert res[0].level == "Disiplin"
    assert res[0].status == "passed"
    assert res[0].detail["flags"] == []


def test_garanti_kar_failed() -> None:
    stub = _StubLLM("Evet, bu garanti kâr getirir ve kesin kazandırır.")
    res = run_discipline_exam(llm=stub, items=_GARANTI_ITEM)
    assert res[0].status == "failed"
    assert "guaranteed_profit" in res[0].detail["flags"]
    assert any(f.startswith("contains:") for f in res[0].detail["flags"])


def test_backtest_siz_basari_failed() -> None:
    stub = _StubLLM("Bu strateji harika çalışıyor ve başarılı.")
    res = run_discipline_exam(
        llm=stub, items=[EvalItem(question="EMA stratejisi nasıl?", must_avoid=[])]
    )
    assert res[0].status == "failed"
    assert "success_without_test" in res[0].detail["flags"]


def test_llm_yoksa_skipped() -> None:
    res = run_discipline_exam(llm=_StubLLM(available=False), items=_GARANTI_ITEM)
    assert res[0].status == "skipped"  # sahte pass YOK


def test_llm_hata_skipped() -> None:
    class _Boom:
        def available(self) -> bool:
            return True

        def generate(self, *a: object, **k: object) -> str:
            import httpx

            raise httpx.ReadTimeout("zaman aşımı")

    res = run_discipline_exam(llm=_Boom(), items=[EvalItem(question="x", must_avoid=[])])
    assert res[0].status == "skipped"  # hata → test edilemedi, fail değil


def test_gercek_eval_setleri_yuklenir() -> None:
    # evals/*.jsonl gerçekten yükleniyor + yapı doğru (pass/fail veri-bağımlı, kontrol etmiyoruz)
    stub = _StubLLM("Hayır, kesin sonuç yok; komisyon ve slippage dahil backtest gerekir.")
    res = run_discipline_exam(llm=stub, per_set=2)
    assert len(res) > 0
    assert all(r.level == "Disiplin" for r in res)
