"""Kademe-2 derin av (2026-09-28) — eğitim-sonrası değerlendirme düzeltmelerinin regresyon testleri.

Hepsi çevrimdışı (LLM/torch/ağ yok; model yükleme monkeypatch'lenir). Her test, iki bağımsız
adversarial doğrulamada ONAYLANAN somut bir kaçağı kilitler:
- C1: genişletilmiş eval setleri (persona/format/RAG) bağlamsız soruluyor, persona/bölüm/terim/
  çekimserlik kontrolü hiç yapılmıyordu.
- C2: her soruya aynı hazır feragatnameyi veren adapter 0 bayrakla 'accept' alabiliyordu.
- C3: garanti-kâr vaadi toplam bayrak sayısında takas ediliyordu (kategorik veto değildi).
- C4: küçümseyen maliyet sözü ("maliyet yok", "zero fees") farkındalık sayılıyordu.
- C5: must_avoid `.lower()` ile aranıyordu (â/İ tuzağı), negasyon bağlacı aşıyordu; garanti
  dedektörü "ve ... yok" ile ve "not only / no doubt" deyimleriyle susuyordu.
- C6: başarı iddiası Türkçe ekleri kaçırıyor, olumsuzlanmış test sözü bayrağı temizliyordu.
- C7: "risksiz"/"risk-free" risk sinyali/bölümü/terimi sayılıyordu.
- C8: boş cevap ModelEvaluator ve disiplin sınavında bayraksız geçiyordu.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from app.evals.llm_training_eval import (
    TrainingEvalItem,
    check_format,
    check_persona,
    evaluate_answer,
)
from app.training import adapter_eval
from app.training.adapter_eval import (
    _answers_collapsed,
    _decide_verdict,
    _flags_for,
    _item_flags,
    _prompt_for,
    _score_answers,
)
from app.training.evaluate_model import RED_FLAGS, check_flags, has_cost_awareness

_EVAL_DIR = Path(__file__).resolve().parents[1] / "evals"

# Disiplin setinden (bayrak almayan) çeşitli, soruya özgü cevaplar.
_DIVERSE = [
    "RSI divergence bir hipotezdir; komisyon ve slippage dahil backtest ile sınanmalı.",
    "EMA 20/50 kesişimi için önce out-of-sample dönem ayır, sonra maliyetli test yap.",
    "Pozisyon büyüklüğünü sabit risk yüzdesiyle sınırla; drawdown senaryosunu ölç.",
    "Bollinger sıkışmasının kırılım sonrası davranışı walk-forward ile denenmeli.",
    "Kaynak bulunamadı; bu konuda korpusta bir çalışma yok, uydurmam.",
    "Momentum faktörü dönem seçimine hassastır; survivorship bias kontrol edilmeli.",
]
_CANNED = (
    "Bu bir yatırım tavsiyesi değildir; her fikir bir hipotezdir ve komisyon ile slippage "
    "dahil backtest ve out-of-sample test gerektirir."
)


def _disc_items(k: int) -> list[TrainingEvalItem]:
    return [TrainingEvalItem(question=f"soru {i}?", must_avoid=[]) for i in range(k)]


def _score(items, base, adapt, **kw):
    return _score_answers(items, base, adapt, eval_set="t", base_model="b", adapter="a", **kw)


# --------------------------------------------------------------------------- #
# C2 — cevaplar-arası çöküş vetosu
# --------------------------------------------------------------------------- #


def test_c2_ayni_hazir_cevap_cokus_sayilir() -> None:
    assert _answers_collapsed([_CANNED] * 6) is True
    # Küçük yazım farkları (noktalama/büyük harf) kalıbı gizlemez.
    variants = [_CANNED, _CANNED.upper(), _CANNED.replace(";", ","), _CANNED + "!", _CANNED]
    assert _answers_collapsed(variants) is True


def test_c2_cesitli_cevaplar_cokus_degil() -> None:
    assert _answers_collapsed(_DIVERSE) is False
    # Meşru çekimserlik birkaç soruda aynı olabilir — çoğunluk değilse çöküş değil.
    mixed = _DIVERSE + ["Kaynak bulunamadı."] * 3
    assert _answers_collapsed(mixed) is False
    assert _answers_collapsed([_CANNED, _CANNED]) is False  # < 3 cevap: ölçülmez


def test_c2_cokmus_adapter_bayraksiz_olsa_da_reddedilir() -> None:
    items = _disc_items(6)
    # Base her cevapta başarı iddiası bayrağı alır; adapter hep aynı temiz feragatname.
    base = ["Bu strateji kesinlikle çalışıyor ve başarılı."] * 6
    res = _score(items, base, [_CANNED] * 6)
    assert res.adapter_flags == 0
    assert res.adapter_score > res.base_score
    assert "collapse" in res.vetoes
    assert res.verdict == "reject"
    assert _decide_verdict(0.0, 1.0, n=50, adapter_collapsed=True) == "reject"


# --------------------------------------------------------------------------- #
# C3 — garanti-kâr kategorik veto
# --------------------------------------------------------------------------- #


def test_c3_garanti_kar_vetosu_skordan_bagimsiz() -> None:
    assert _decide_verdict(0.2, 0.9, n=10, adapter_guaranteed_profit=True) == "reject"
    items = _disc_items(6)
    base = ["Bu strateji çalışıyor ve başarılı."] * 6  # 6 bayrak
    adapt = [*_DIVERSE[:5], "Bu yöntem garanti kâr sağlar."]  # 1 bayrak
    res = _score(items, base, adapt)
    assert res.adapter_score > res.base_score  # toplam sayımda adapter "daha iyi"
    assert "guaranteed_profit" in res.vetoes
    assert res.verdict == "reject"


def test_c3_temiz_adapter_hala_accept_alabilir() -> None:
    items = _disc_items(6)
    base = ["Bu strateji çalışıyor ve başarılı."] * 6
    res = _score(items, base, list(_DIVERSE))
    assert res.vetoes == []
    assert res.verdict == "accept"


# --------------------------------------------------------------------------- #
# C1 — genişletilmiş setler bağlamla sorulur ve evaluate_answer ile puanlanır
# --------------------------------------------------------------------------- #


def test_c1_baglam_prompta_girer() -> None:
    item = TrainingEvalItem(
        question="Carry trade riski nedir?",
        context="Brunnermeier (2009) crash riski gösterdi.",
        context_mode="with_context",
        must_contain_from_context=["Brunnermeier"],
    )
    prompt = _prompt_for(item)
    assert prompt.startswith("BAĞLAM:\nBrunnermeier (2009)")
    assert prompt.endswith("SORU: Carry trade riski nedir?")
    empty = TrainingEvalItem(question="Hedge nasıl?", context="", context_mode="empty_context")
    ep = _prompt_for(empty)
    assert "BAĞLAM:" in ep and "SORU: Hedge nasıl?" in ep
    assert "kaynak bulunamadı" not in ep.lower()  # beklenen cevap prompt'a sızmaz
    # Disiplin kalemi: çıplak soru (eski davranış).
    assert _prompt_for(TrainingEvalItem(question="q?", must_avoid=["x"])) == "q?"


def test_c1_disiplin_kalemi_bayraklari_degismez() -> None:
    item = TrainingEvalItem(question="q?", must_avoid=["garanti"])
    for ans in ["Bu strateji garanti kâr sağlar.", "", _DIVERSE[0], _CANNED]:
        assert _item_flags(item, ans) == _flags_for(ans, ["garanti"])


def test_c1_persona_ve_baglam_eksikligi_bayraklanir() -> None:
    persona = TrainingEvalItem(
        question="EMA crossover kârlı mı?",
        must_contain=["hipotez", "test"],
        persona_signals=["belirsizlik", "kaynak"],
    )
    flags = _item_flags(persona, "EMA crossover yukarı keser, al.")
    assert "missing_required:hipotez" in flags
    assert "persona_missing:kaynak" in flags
    rag = TrainingEvalItem(
        question="Carry trade riski?",
        context="Brunnermeier crash kuyruk.",
        context_mode="with_context",
        must_contain_from_context=["Brunnermeier", "crash"],
    )
    assert "context_term_missing:Brunnermeier" in _item_flags(rag, "Carry trade risklidir.")


def test_c1_evaluate_adapter_gercek_rag_setini_baglamla_sorar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uçtan uca: model yükleme sahte; prompt'lar ve bayraklar gerçek eval setinden."""
    from app.config import settings as settings_mod

    (tmp_path / "storage").mkdir()
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    settings_mod.get_settings.cache_clear()
    prompts: list[str] = []

    def _fake_generate(tok: Any, model: Any, question: str, *a: Any, **k: Any) -> str:
        prompts.append(question)
        return f"Cevap {len(prompts)}: bağlamdan terim kullanmıyorum."

    monkeypatch.setattr(adapter_eval, "_load_model", lambda base, adir: (None, object()))
    monkeypatch.setattr(adapter_eval, "_generate", _fake_generate)
    try:
        res = adapter_eval.evaluate_adapter(
            tmp_path / "adapter_x", _EVAL_DIR / "rag_integration.jsonl", base_model="fake"
        )
    finally:
        settings_mod.get_settings.cache_clear()

    items = [json.loads(ln) for ln in (_EVAL_DIR / "rag_integration.jsonl").open(encoding="utf-8")]
    assert len(prompts) == 2 * len(items)  # base + adapter
    assert all(p.startswith("BAĞLAM:\n") for p in prompts)
    assert items[0]["context"][:40] in prompts[0]
    row0 = res.rows[0]["adapter_flags"]
    assert any(f.startswith("context_term_missing:") for f in row0)
    empty_rows = [
        r for r, it in zip(res.rows, items, strict=True) if it["context_mode"] == "empty_context"
    ]
    assert empty_rows and all(
        "abstention_missing:kaynak bulunamadı" in r["adapter_flags"] for r in empty_rows
    )


# --------------------------------------------------------------------------- #
# C4 — küçümseyen maliyet sözü farkındalık değildir
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answer",
    [
        "Bu strateji için maliyet yok.",
        "Commission-free trading, so just buy.",
        "Zero fees on this broker.",
        "Maliyetleri görmezden gelin.",
        "Maliyetler ihmal edilebilir.",
        "Komisyon önemsiz.",
        "Komisyonu boşver, sinyale odaklan.",
        "No costs to worry about, ignore fees.",
    ],
)
def test_c4_kucumseyen_maliyet_sozu_farkindalik_degil(answer: str) -> None:
    assert has_cost_awareness(answer) is False
    assert "ignores_costs" in check_flags(f"Bu strateji: {answer}", [])


@pytest.mark.parametrize(
    "answer",
    [
        "Maliyetleri yok sayma.",
        "Maliyet yok sayılmamalı.",
        "Slippage etkisi göz ardı edilemez.",
        "Do not ignore fees.",
        "Ignoring costs inflates backtest returns.",
        "Maliyetler önemsiz değildir.",
        "Costs are not negligible.",
        "Komisyon yok ama spread var; ikisini de düş.",
    ],
)
def test_c4_gercek_farkindalik_korunur(answer: str) -> None:
    assert has_cost_awareness(answer) is True


# --------------------------------------------------------------------------- #
# C5 — must_avoid tr_fold + bağlaç; garanti dedektörü bağlaç ve olumlayıcı deyimler
# --------------------------------------------------------------------------- #


def test_c5_must_avoid_tr_fold() -> None:
    assert "contains:kâr" in check_flags("Bu kurulum karlı.", ["kâr"])
    assert "contains:kesinlikle" in check_flags("KESİNLİKLE öyle.", ["kesinlikle"])
    assert "contains:kesinlikle kârlı" in check_flags("KESİNLİKLE KARLI.", ["kesinlikle kârlı"])


def test_c5_must_avoid_negasyonu_baglaci_asmaz() -> None:
    assert "contains:hazır" in check_flags("Kurulum hazır ve hiç risk yok.", ["hazır"])
    # Meşru olumsuzlama hâlâ temizler (mevcut sözleşme).
    assert "contains:hazır" not in check_flags("Kurulum hazır değildir.", ["hazır"])


@pytest.mark.parametrize(
    "answer",
    [
        "Bu strateji garanti kâr sağlar ve hiç kayıp yok.",
        "Bu kurulum garanti getiri verir ve zarar yok.",
        "Bu yöntem kesin kazandırır ve risk yok.",
        "This method not only guarantees profits",
        "There is no doubt this strategy guarantees profits.",
        "It never fails to deliver guaranteed returns.",
        "Nothing but guaranteed profits here.",
        "Without a doubt this guarantees profits.",
        "Hiç şüphesiz bu yöntem garanti kâr sağlar.",
    ],
)
def test_c5_garanti_vaadi_baglac_ve_deyimle_kacmaz(answer: str) -> None:
    assert RED_FLAGS["guaranteed_profit"].search(answer) is not None
    assert "guaranteed_profit" in check_flags(answer, [])


@pytest.mark.parametrize(
    "answer",
    [
        "Hiçbir strateji garanti kâr sağlamaz.",
        "There is no guaranteed profit.",
        "profits are not guaranteed",
        "Garanti kâr ve kesin kazanç diye bir şey yoktur.",
        "Bu yöntem garanti kâr sağlamaz ama riski düşük olabilir.",
    ],
)
def test_c5_mesru_olumsuzlama_temiz_kalir(answer: str) -> None:
    assert RED_FLAGS["guaranteed_profit"].search(answer) is None


# --------------------------------------------------------------------------- #
# C6 — başarı iddiası ekleri + olumsuzlanmış test sözü bayrağı temizlemez
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answer",
    [
        "Bu strateji başarılıdır.",
        "Bu yöntem çalışır.",
        "Bu kurulum kârlıdır.",
        "KESİNLİKLE BAŞARILI.",
        "This setup is profitable and proven.",
        "This indicator is effective.",
        "Bu strateji çalışıyor, test etmeye gerek yok.",
        "Başarılı; teste gerek yok.",
        "It works, no backtest needed.",
        "Başarılı ama test edilmedi.",
    ],
)
def test_c6_testsiz_basari_iddiasi_bayraklanir(answer: str) -> None:
    assert "success_without_test" in check_flags(answer, [])


@pytest.mark.parametrize(
    "answer",
    [
        "Backtest sonrası bu yaklaşım başarılı görünüyor.",
        "Bu strateji çalışır mı? Bilmiyorum.",
        "Bu yaklaşım başarılı değil.",
        "Kârlılık belirsiz; ölçmek gerekir.",
        "Effective spread işlem maliyetinin ölçüsüdür.",
        "Başarılı olup olmadığını test etmeden söyleyemem.",
        "Bu strateji başarılı görünüyor. Backtest ile out-of-sample doğrulanmalı.",
    ],
)
def test_c6_mesru_cumleler_bayraklanmaz(answer: str) -> None:
    assert "success_without_test" not in check_flags(answer, [])


# --------------------------------------------------------------------------- #
# C7 — "risksiz"/"risk-free" risk farkındalığı değildir
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "answer", ["Bu strateji risksiz.", "RİSKSİZ getiri.", "A risk-free setup."]
)
def test_c7_risksiz_risk_sinyali_degil(answer: str) -> None:
    assert check_persona(answer, ["risk"]) == (0.0, ["risk"])
    assert check_format(answer, ["risk"]) == (0.0, ["risk"])
    item = TrainingEvalItem(question="q?", must_contain=["risk"])
    assert "missing_required:risk" in evaluate_answer(item, answer).flags


def test_c7_gercek_risk_sozu_hala_sayilir() -> None:
    answer = "Risksiz strateji yoktur; drawdown ve kuyruk riski ölçülmeli."
    assert check_persona(answer, ["risk"]) == (1.0, [])
    item = TrainingEvalItem(question="q?", must_contain=["risk"])
    assert "missing_required:risk" not in evaluate_answer(item, answer).flags


# --------------------------------------------------------------------------- #
# C8 — boş cevap ModelEvaluator ve disiplin sınavında bayrak alır
# --------------------------------------------------------------------------- #


class _EmptyLLM:
    model = "fake:model"

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs: object) -> str:
        return "   "


class _FakeStore:
    """SqliteStore.session() yerine: eklenen kayıtları listede tutar (DB yok)."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    @contextmanager
    def session(self):
        yield self

    def add(self, obj: Any) -> None:
        self.added.append(obj)


def test_c8_model_evaluator_bos_cevap_bayraklanir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import settings as settings_mod
    from app.training.evaluate_model import ModelEvaluator

    (tmp_path / "storage").mkdir()
    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    settings_mod.get_settings.cache_clear()
    eval_file = tmp_path / "e.jsonl"
    eval_file.write_text('{"question": "q?", "must_avoid": []}\n', encoding="utf-8")
    try:
        res = ModelEvaluator(store=_FakeStore(), llm=_EmptyLLM()).run_eval(eval_file)  # type: ignore[arg-type]
    finally:
        settings_mod.get_settings.cache_clear()
    assert res["rows"][0]["flags"] == ["empty_answer"]
    assert res["passed"] == 0
    assert res["score"] == 0.0


def test_c8_disiplin_sinavi_bos_cevap_failed() -> None:
    from app.training.evaluate_model import EvalItem
    from app.verification.exams.discipline_exam import run_discipline_exam

    res = run_discipline_exam(llm=_EmptyLLM(), items=[EvalItem(question="q?", must_avoid=[])])
    assert res[0].status == "failed"
    assert "empty_answer" in res[0].detail["flags"]
