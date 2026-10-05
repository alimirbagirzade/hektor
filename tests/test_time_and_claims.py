"""Bölüm 4 — zaman alanları, strateji aileleri ve performans iddiasının kayıtlı koşuyla eşleşmesi.

Kabul şartları: bilgi zamanı fiyat verisinin bitişine eşitlenmez (veri / strateji / koşu / bilgi
zamanı ayrı); strateji varyantları aynı aileye bağlanır (train/eval'e bölünmez); performans
iddiası yalnız sayıyla değil parmak izi (strateji, veri, dönem, maliyet, metrik tanımı, motor) ve
metindeki dönemle eşleşir; eşleşme "kayıtlı hesapla eşleşti"dir, başarı iddiası değildir.
"""

from __future__ import annotations

import pytest
from tests.chat_learning_helpers import FakeLLM, StubRetriever, iso  # noqa: F401
from tests.test_strategy_flow import _spec, market, ok  # noqa: F401

from app.feedback.chat_store import ChatStore
from app.feedback.learning import LearningError, LearningService
from app.trading.chat_strategy import save_spec
from app.trading.strategy_testing import run_stage


def _turn(store, q="RSI EMA stratejisi backtest sonucu nasıl?"):
    from app.feedback.chat_service import send

    conv = store.create_conversation()["conversation_id"]
    t, _ = send(
        conv,
        q,
        f"r-{q[:10]}-{len(store.list_conversations())}",
        retriever=StubRetriever(),
        llm=FakeLLM(["Stratejinin backtest getirisi iyi."]),
        store=store,
    )
    return t


def _claim(run, **override) -> str:
    m = run["result"]["metrics"]
    vals = {
        "start": run["period_start"][:10],
        "end": run["period_end"][:10],
        "ret": m["total_return_pct"],
        "dd": m["max_drawdown_pct"],
        "n": m["n_trades"],
    }
    vals.update(override)
    return (
        f"Bu strateji {vals['start']} ile {vals['end']} arasındaki doğrulama döneminde, "
        f"maliyetler dahil toplam getiri %{vals['ret']} ve azami düşüş {vals['dd']} % verdi; "
        f"işlem sayısı {vals['n']}. Bu bir hipotez testidir."
    )


@pytest.fixture
def setup(market):  # noqa: F811
    rec = ok(save_spec(_spec()))
    run = run_stage(rec["strategy_id"], data_file=market, tz="UTC", stage="dogrulama")
    store = ChatStore()
    return store, LearningService(store), rec, run


def _checks(c):
    return {k["kind"]: k for k in c["verification"]["checks"]}


def test_matching_claim_and_separate_times(setup) -> None:
    store, svc, _rec, run = setup
    t = _turn(store)
    cand, _ = svc.correct(t["turn_id"], _claim(run), domain="trading")
    assert _checks(cand)["backtest"]["status"] == "yapilamadi"  # koşu bağlanmadan doğrulanamaz
    with pytest.raises(LearningError):
        svc.approve(cand["candidate_id"], "Grafikte gördüm, doğru görünüyor.")
    linked = svc.link_run(cand["candidate_id"], run["run_id"])
    bt = _checks(linked)["backtest"]
    assert bt["status"] == "eslesti" and "GELMEZ" in bt["detail"]
    tm = linked["time_meta"]
    assert tm["knowledge_available_at"] == max(t["created_at"], run["run_at"])
    assert linked["as_of"] == tm["knowledge_available_at"] != tm["data_end"]
    assert tm["data_start"] < tm["period_start"] <= tm["period_end"] <= tm["data_end"]
    assert tm["strategy_created_at"] and tm["backtest_run_at"] == run["run_at"]
    assert linked["status"] == "review"  # diğer ifadeler hâlâ insan incelemesi bekler
    ok = svc.approve(linked["candidate_id"], "Koşu raporu ve dönem elle karşılaştırıldı.")
    assert ok["status"] == "eligible"


@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"ret": 99.9}, "≠ kayıtlı"),
        ({"start": "2020-01-01"}, "dönemini belirtmiyor"),
    ],
)
def test_mismatching_claim_refutes(setup, override, why) -> None:
    store, svc, _rec, run = setup
    t = _turn(store, f"Backtest iddiası {why}?")
    cand, _ = svc.correct(t["turn_id"], _claim(run, **override), domain="trading")
    linked = svc.link_run(cand["candidate_id"], run["run_id"])
    assert linked["status"] == "rejected" and "backtest" in linked["reason_codes"]
    assert why in _checks(linked)["backtest"]["detail"]
    with pytest.raises(LearningError):
        svc.approve(linked["candidate_id"], "Bence doğru, onaylıyorum burada.")


def test_dev_result_presented_as_oos_and_tampering(setup, market) -> None:  # noqa: F811
    from app.trading import event_engine as ee
    from app.trading.strategy_store import StrategyRun, StrategyStore

    store, svc, rec, _run = setup
    dev = run_stage(rec["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    t = _turn(store, "Geliştirme sonucu nasıl sunulur?")
    text = _claim(dev).replace("doğrulama döneminde", "örneklem dışı dönemde")
    cand, _ = svc.correct(t["turn_id"], text, domain="trading")
    linked = svc.link_run(cand["candidate_id"], dev["run_id"])
    assert "örneklem dışı diye" in _checks(linked)["backtest"]["detail"]
    # Parmak izi / motor sürümü değişirse sayı aynı olsa da eşleşmez.
    st = StrategyStore()
    with st.session() as s:
        row = s.get(StrategyRun, dev["run_id"])
        row.fingerprint = "0" * 64
        row.engine_version = "hektor-event-0.9"
    t2 = _turn(store, "Değiştirilmiş kayıt?")
    c2, _ = svc.correct(t2["turn_id"], _claim(dev), domain="trading")
    l2 = svc.link_run(c2["candidate_id"], dev["run_id"])
    detail = _checks(l2)["backtest"]["detail"]
    assert "parmak izi" in detail and "eski motor" in detail and ee.ENGINE_VERSION in detail


def test_strategy_variants_share_family_and_side(setup, market) -> None:  # noqa: F811
    from app.feedback.chat_dataset import build_payload

    store, svc, rec, run = setup
    var = ok(
        save_spec(
            _spec(entry_rules=["ema_10 > ema_30", "rsi_14 > 50"]), parent_id=rec["strategy_id"]
        )
    )
    run2 = run_stage(var["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    ids = []
    for r, q in ((run, "Birinci varyant sorusu tamamen farklı"), (run2, "İkinci bambaşka soru")):
        t = _turn(store, q)
        c, _ = svc.correct(t["turn_id"], _claim(r), domain="trading")
        c = svc.link_run(c["candidate_id"], r["run_id"])
        c = svc.approve(c["candidate_id"], "Koşu raporu ve dönem elle karşılaştırıldı.")
        ids.append(c)
    assert ids[0]["family_id"] == ids[1]["family_id"] == "fam_" + rec["family_id"]
    p = build_payload(store, token_counter=lambda s: len(s.split()), token_method="t")
    sides = {m["split"] for m in p["members"] if m["family_id"] == ids[0]["family_id"]}
    assert len(sides) == 1  # aynı strateji ailesi hem train hem eval'e düşmez
    assert p["stats"]["trading_time_split"]["families"] >= 1
