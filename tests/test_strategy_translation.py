"""Strateji çevirisindeki değişiklikler görünür olmalı (orijinal öneri → taslak → onaylı nihai).

Kabul şartları:
- Orijinal sohbet önerisi, çıkarılan taslak ve onaylanan nihai strateji AYRI kaydedilir.
- Model ile metnin deterministik okuması çelişirse ya da model metinde olmayan bir değer verirse
  alan taslakta BOŞ kalır ("karar gerekli"); tahminle doldurulup onaylı gibi sunulmaz.
- Long için "fiyat + 2 ATR" / short için "fiyat − 1.5 ATR" stop'u ÇELİŞKİ olarak işaretlenir.
- Desteklenmeyen seans kuralı sessizce düşmez; basitleştirmede açıkça çıkarılır ve farkta görünür.
- Her fark işaretlenmeden (ve çelişkide gerekçe yazılmadan) test başlamaz.
- Öğrenme adayı test edilen NİHAİ stratejiye bağlanır; orijinal öneri farklıysa iddia, test edilen
  stratejiyi kimliğiyle anmadıkça "kayıtlı hesapla eşleşti" olamaz.
"""

from __future__ import annotations

import json

import pytest
from tests.chat_learning_helpers import FakeLLM, StubRetriever, iso  # noqa: F401
from tests.test_strategy_flow import COSTS, JsonLLM, market  # noqa: F401

from app.trading.chat_strategy import approve, draft_from_turn, review, save_spec, simplify
from app.trading.strategy_store import StrategyStore
from app.trading.strategy_testing import StrategyTestError, run_stage
from app.trading.strategy_translation import (
    compare_specs,
    compare_to_proposal,
    extract_proposal,
    reconcile_draft,
)

LONG_BAD = (
    "Uzun pozisyon (long): EMA10 EMA30'un üstündeyken gir, altına inince çık. "
    "Stop-loss: fiyat + 2 ATR. İşlem başına %1 risk al."
)
SHORT_BAD = (
    "Short stratejisi: EMA10 EMA30'un altındayken açığa sat, üstüne çıkınca kapat. "
    "Stop: fiyat - 1.5 ATR. İşlem başına %1 risk."
)
SHORT_OK = (
    "Short stratejisi: EMA10 EMA30'un altındayken açığa sat, üstüne çıkınca kapat. "
    "Stop girişin 1.5 ATR üstüne konur. İşlem başına %1 risk."
)
SESSION = (
    "Long: EMA10 EMA30'un üstündeyken gir, altına inince çık; yalnız Londra seansında işlem aç. "
    "Stop-loss girişin 2 ATR altında. Pozisyon büyüklüğü: özsermayenin tamamı (1 kat)."
)


def _model(direction: str, *, stop=None, sizing=None, unsupported=None, entry=None) -> dict:
    return {
        "name": "ema",
        "market": "TEST",
        "timeframe": "1h",
        "direction": direction,
        "indicators": [{"name": "EMA", "period": 10}, {"name": "EMA", "period": 30}],
        "entry_rules": entry
        or (["ema_10 > ema_30"] if direction == "long" else ["ema_10 < ema_30"]),
        "exit_rules": ["ema_10 < ema_30"] if direction == "long" else ["ema_10 > ema_30"],
        "stop": stop or {"type": "atr_initial", "value": 2.0, "atr_period": 14},
        "take_profit": {"type": "none"},
        "sizing": sizing or {"type": "risk_per_trade", "risk_pct": 0.01},
        "costs": {},
        "unsupported": unsupported or [],
    }


def _turn(answer: str, question: str = "Strateji öner?") -> str:
    from app.feedback.chat_service import send
    from app.feedback.chat_store import ChatStore

    store = ChatStore()
    conv = store.create_conversation()["conversation_id"]
    turn, _ = send(
        conv,
        question,
        f"rq-{abs(hash(answer)) % 10**8}",
        retriever=StubRetriever(),
        llm=FakeLLM([answer]),
        store=store,
    )
    return turn["turn_id"]


def _final(draft: dict, **kw) -> dict:
    data = json.loads(json.dumps(draft))
    data.update(kw)
    data["costs"] = COSTS
    return data


def _ack_all(sid: str, note: str = "") -> dict:
    rv = review(sid)
    return approve(
        sid, review_sha=rv["review_sha"], acknowledged=[i["key"] for i in rv["items"]], note=note
    )


# ── deterministik metin okuması ─────────────────────────────────────────────


def test_proposal_reads_signed_atr_stop_risk_session_and_indicators() -> None:
    p = extract_proposal(LONG_BAD)
    assert p["directions"] == ["long"]
    assert p["stops"][0]["kind"] == "atr" and p["stops"][0]["value"] == 2.0
    assert p["stops"][0]["side"] == "above"  # "fiyat + 2 ATR"
    assert p["risk_pct"] == pytest.approx(0.01)
    assert {"ema_10", "ema_30"} <= set(p["indicators"])
    s = extract_proposal(SESSION)
    assert s["sessions"] and "Londra" in s["sessions"][0]["text"]
    assert s["stops"][0]["side"] == "below"
    assert extract_proposal(SHORT_BAD)["stops"][0]["side"] == "below"


@pytest.mark.parametrize(
    ("answer", "direction", "stop"),
    [
        (LONG_BAD, "long", {"type": "atr_initial", "value": 2.0}),  # model "düzeltti"
        (SHORT_BAD, "short", {"type": "atr_initial", "value": 1.5}),
    ],
)
def test_conflicting_stop_is_left_blank_not_silently_fixed(answer, direction, stop) -> None:
    proposal = extract_proposal(answer)
    draft, pending = reconcile_draft(_model(direction, stop=stop), proposal)
    assert draft["stop"]["type"] == ""  # tahminle doldurulmadı
    p = next(x for x in pending if x["field"] == "stop")
    assert "ÇELİŞKİ" in p["why"] and p["model_value"]["type"] == "atr_initial"
    ovd = compare_to_proposal(proposal, draft)
    stop_items = [i for i in ovd if i["category"] == "stop"]
    assert stop_items and stop_items[0]["conflict"] and stop_items[0]["kind"] == "cikarildi"


def test_consistent_short_stop_kept() -> None:
    proposal = extract_proposal(SHORT_OK)
    draft, pending = reconcile_draft(
        _model("short", stop={"type": "atr_initial", "value": 1.5, "atr_period": 14}), proposal
    )
    assert draft["stop"]["type"] == "atr_initial" and not [
        p for p in pending if p["field"] == "stop"
    ]
    assert not [i for i in compare_to_proposal(proposal, draft) if i["category"] == "stop"]


def test_model_guesses_are_not_kept() -> None:
    text = "Long: EMA10 EMA30'un üstündeyken gir, altına inince çık."
    draft, pending = reconcile_draft(
        _model("long", sizing={"type": "fixed_fraction", "fraction": 1.0}), extract_proposal(text)
    )
    fields = {p["field"] for p in pending}
    assert draft["sizing"] is None and "sizing" in fields  # metinde boyut yok
    assert draft["stop"]["type"] == "none" and "stop" in fields  # metinde stop yok
    risk_text = "Long: EMA10 > EMA30 iken gir. İşlem başına %2 risk; stop girişin 2 ATR altında."
    d2, p2 = reconcile_draft(_model("long"), extract_proposal(risk_text))  # model %1 dedi
    assert d2["sizing"] is None and any("%2" in p["why"] for p in p2)


def test_both_directions_require_choice() -> None:
    text = "Long ya da short: trend yukarıysa long, aşağıysa short aç. Stop 2 ATR."
    draft, pending = reconcile_draft(_model("long"), extract_proposal(text))
    assert draft["direction"] is None and any(p["field"] == "direction" for p in pending)


def test_spec_diff_lists_user_edits_by_category() -> None:
    a = _model("long")
    b = json.loads(json.dumps(a))
    b["entry_rules"] = ["ema_10 > ema_30", "rsi_14 > 50"]
    b["stop"] = {"type": "atr_trailing", "value": 2.5, "atr_period": 14}
    b["sizing"] = {"type": "risk_per_trade", "risk_pct": 0.02}
    cats = {(i["category"], i["kind"]) for i in compare_specs(a, b)}
    assert (
        ("giris", "eklendi") in cats and ("stop", "degisti") in cats and ("risk", "degisti") in cats
    )


# ── uçtan uca: kayıt → onay → test → öğrenme bağlantısı ──────────────────────


def test_long_stop_conflict_end_to_end(market) -> None:  # noqa: F811
    tid = _turn(LONG_BAD)
    out = draft_from_turn(tid, llm=JsonLLM(_model("long")))
    assert out["translation_id"].startswith("tr_") and not out["valid"]
    store = StrategyStore()
    tr = store.get_translation(out["translation_id"])
    # Üç ayrı kayıt: orijinal metin + özet, model taslağı, uzlaştırılmış taslak.
    assert tr["original_text"] == LONG_BAD and len(tr["original_sha"]) == 64
    assert tr["model_draft"]["stop"]["type"] == "atr_initial" and tr["draft"]["stop"]["type"] == ""
    # Kullanıcı stop'u bilinçli seçer (girişin 2×ATR altı) ve kaydeder.
    final = _final(out["draft"], stop={"type": "atr_initial", "value": 2.0, "atr_period": 14})
    rec = save_spec(final, source=out["source"])
    assert rec["source"]["translation_id"] == out["translation_id"]
    with pytest.raises(StrategyTestError, match="onaylanmadı"):
        run_stage(rec["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    rv = review(rec["strategy_id"])
    assert rv["conflicts"] and rv["original_text"] == LONG_BAD
    stop = next(i for i in rv["original_vs_final"] if i["category"] == "stop")
    assert stop["kind"] == "degisti" and stop["conflict"] and "ÇELİŞKİ" in stop["note"]
    keys = [i["key"] for i in rv["items"]]
    with pytest.raises(ValueError, match="işaretlenmedi"):
        approve(
            rec["strategy_id"], review_sha=rv["review_sha"], acknowledged=keys[:-1], note="x" * 20
        )
    with pytest.raises(ValueError, match="çelişki"):
        approve(rec["strategy_id"], review_sha=rv["review_sha"], acknowledged=keys, note="")
    with pytest.raises(ValueError, match="değişti"):
        approve(rec["strategy_id"], review_sha="0" * 64, acknowledged=keys, note="x" * 20)
    appr = approve(
        rec["strategy_id"],
        review_sha=rv["review_sha"],
        acknowledged=keys,
        note="long için stop girişin altında olmalı; 2×ATR altını seçtim",
    )
    assert appr["n_changes"] == len(keys)
    run = run_stage(rec["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    ts = run["result"]["tested_strategy"]
    assert ts["strategy_id"] == rec["strategy_id"] and ts["approval_id"] == appr["approval_id"]
    assert ts["n_changes_from_original"] >= 1 and ts["is_original_proposal"] is False


def test_session_rule_not_dropped_and_removed_only_explicitly(market) -> None:  # noqa: F811
    tid = _turn(SESSION)
    out = draft_from_turn(
        tid,
        llm=JsonLLM(_model("long", sizing={"type": "fixed_fraction", "fraction": 1.0})),
    )
    texts = [u["text"] for u in out["draft"]["unsupported_rules"]]
    assert any("Londra" in t for t in texts)  # model atladı, deterministik kontrol yakaladı
    rec = save_spec(_final(out["draft"]), source=out["source"])
    assert rec["testable"] is False
    sess = next(u for u in rec["spec"]["unsupported_rules"] if "Londra" in u["text"])["text"]
    simp = simplify(rec["strategy_id"], remove_rules=[], drop_unsupported=[sess])
    rv = review(simp["strategy_id"])
    seans = [i for i in rv["original_vs_final"] if i["category"] == "seans"]
    assert seans and seans[0]["kind"] == "cikarildi" and "açıkça" in seans[0]["now"]
    with pytest.raises(StrategyTestError, match="onaylanmadı"):
        run_stage(simp["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    _ack_all(simp["strategy_id"])
    run = run_stage(simp["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    assert run["result"]["tested_strategy"]["is_original_proposal"] is False


def test_translation_bound_to_its_turn() -> None:
    tid = _turn(LONG_BAD)
    out = draft_from_turn(tid, llm=JsonLLM(_model("long")))
    other = _turn(SHORT_OK, question="Başka soru?")
    final = _final(out["draft"], stop={"type": "atr_initial", "value": 2.0, "atr_period": 14})
    with pytest.raises(ValueError, match="ait değil"):
        save_spec(final, source={"turn_id": other, "translation_id": out["translation_id"]})


def test_learning_candidate_linked_to_tested_final_not_original(market) -> None:  # noqa: F811
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    tid = _turn(LONG_BAD, question="EMA stratejisi backtest sonucu?")
    out = draft_from_turn(tid, llm=JsonLLM(_model("long")))
    final = _final(out["draft"], stop={"type": "atr_initial", "value": 2.0, "atr_period": 14})
    rec = save_spec(final, source=out["source"])
    _ack_all(rec["strategy_id"], note="stop girişin altına alındı, çelişki çözüldü")
    run = run_stage(rec["strategy_id"], data_file=market, tz="UTC", stage="dogrulama")
    m = run["result"]["metrics"]
    claim = (
        f"Bu strateji {run['period_start'][:10]} ile {run['period_end'][:10]} doğrulama "
        f"döneminde maliyetler dahil toplam getiri %{m['total_return_pct']} verdi; işlem sayısı "
        f"{m['n_trades']}. Bu bir hipotez testidir."
    )
    svc = LearningService(ChatStore())
    cand, _ = svc.correct(tid, claim, domain="trading")
    linked = svc.link_run(cand["candidate_id"], run["run_id"])
    tm = linked["time_meta"]
    assert tm["tested_strategy_id"] == rec["strategy_id"] and tm["is_original_proposal"] is False
    assert tm["translation_id"] == out["translation_id"] and tm["strategy_approval_id"]
    bt = {k["kind"]: k for k in linked["verification"]["checks"]}["backtest"]
    assert bt["status"] == "eslesmedi" and "orijinal öneri test EDİLMEDİ" in bt["detail"]
    # Metin sonucu test edilen nihai stratejiye açıkça bağlarsa kayıtlı hesapla eşleşebilir.
    claim2 = claim.replace("Bu strateji", f"Onaylı nihai strateji {rec['strategy_id']}")
    c2, _ = svc.correct(tid, claim2, domain="trading")
    l2 = svc.link_run(c2["candidate_id"], run["run_id"])
    assert {k["kind"]: k for k in l2["verification"]["checks"]}["backtest"]["status"] == "eslesti"


def test_review_endpoint_and_approve_requires_human(market) -> None:  # noqa: F811
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    tid = _turn(SHORT_BAD)
    out = draft_from_turn(
        tid, llm=JsonLLM(_model("short", stop={"type": "atr_initial", "value": 1.5}))
    )
    final = _final(out["draft"], stop={"type": "atr_initial", "value": 1.5, "atr_period": 14})
    rec = save_spec(final, source=out["source"])
    client = TestClient(app)
    rv = client.get(f"/api/strategy/{rec['strategy_id']}/review").json()
    assert rv["conflicts"] and not rv["approved"]
    bad = client.post(
        f"/api/strategy/{rec['strategy_id']}/approve",
        json={"review_sha": rv["review_sha"], "acknowledged": []},
    )
    assert bad.status_code == 422
    good = client.post(
        f"/api/strategy/{rec['strategy_id']}/approve",
        json={
            "review_sha": rv["review_sha"],
            "acknowledged": [i["key"] for i in rv["items"]],
            "note": "short stop girişin üstünde olmalı; seçtim",
        },
    )
    assert good.status_code == 200, good.text
    assert client.get(f"/api/strategy/{rec['strategy_id']}/review").json()["approved"] is True
    for rt in client.app.routes:
        if getattr(rt, "path", "") == "/api/strategy/{strategy_id}/approve":
            assert require_human in {d.call for d in rt.dependant.dependencies}
