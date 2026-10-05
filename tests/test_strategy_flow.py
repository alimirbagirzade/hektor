"""Faz 2B — sohbet cevabı → yerel model taslağı → okunur form → gerçek veride dönemli test.

Kabul şartları: şablonla sessiz ikame yok; desteklenmeyen kural sessizce düşmez, ÖNEMLİ ise asıl
strateji testi durur; basitleştirilmiş strateji ayrı kimlikli ve çıkarılan kurallar açık; cevapta
stop varken taslakta yoksa engellenir; maliyet varsayılanla doldurulmaz; doğrulama dönemine
bakan varyantlar sayılır ve dönem "geliştirmede kullanılmış" olur; final aile başına tek;
zaman alanları (veri / strateji / koşu / bilgi) ayrı ve bilgi zamanı veri bitişine EŞİT DEĞİL.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from tests.chat_learning_helpers import FakeLLM, StubRetriever, iso  # noqa: F401

from app.trading.chat_strategy import save_spec, simplify
from app.trading.strategy_draft import DraftError, extract_draft, normalize_draft
from app.trading.strategy_testing import StrategyTestError, check_data, run_stage

COSTS = {
    "commission_bps_per_side": 2,
    "slippage_bps_per_side": 1,
    "spread_bps": 2,
    "funding_bps_per_day": 0,
    "zero_reasons": {"funding_bps_per_day": "spot test verisi, fonlama yok"},
}


def _spec(**kw):
    base = {
        "name": "ema_trend",
        "market": "TEST",
        "timeframe": "1h",
        "direction": "long",
        "indicators": [{"name": "EMA", "period": 10}, {"name": "EMA", "period": 30}],
        "entry_rules": ["ema_10 > ema_30"],
        "exit_rules": ["ema_10 < ema_30"],
        "stop": {"type": "atr_initial", "value": 2.0, "atr_period": 14},
        "sizing": {"type": "fixed_fraction", "fraction": 1.0},
        "costs": COSTS,
    }
    base.update(kw)
    return base


@pytest.fixture
def market(iso):  # noqa: F811
    from app.config import get_settings

    d = get_settings().market_raw_dir
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n = 1200
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, n)))
    o = np.concatenate([[100.0], close[:-1]])
    hi = np.maximum(o, close) * 1.001
    lo = np.minimum(o, close) * 0.999
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    pd.DataFrame(
        {
            "time": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "open": o,
            "high": hi,
            "low": lo,
            "close": close,
            "volume": 1.0,
        }
    ).to_csv(d / "test_1h.csv", index=False)
    return "test_1h.csv"


class JsonLLM:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        assert kw.get("fmt") == "json" and kw.get("seed") == 42
        return json.dumps(self.payload)


def test_draft_moves_unsupported_and_never_fills_costs() -> None:
    payload = {
        "name": "x",
        "market": "XAUUSD",
        "timeframe": "15m",
        "direction": "long",
        "indicators": [{"name": "EMA", "period": 20}],
        "entry_rules": ["ema_20 > ema_50", "fiyat önceki zirveyi kırarsa"],
        "exit_rules": [],
        "stop": {"type": "none"},
        "sizing": {"type": "fixed_fraction", "fraction": 1},
        "costs": {},
    }
    answer = "EMA20 EMA50 üstündeyken ve zirve kırılınca al; stop-loss 2 ATR koy."
    out = extract_draft("Strateji?", answer, llm=JsonLLM(payload))
    d = out["draft"]
    texts = [u["text"] for u in d["unsupported_rules"]]
    assert "fiyat önceki zirveyi kırarsa" in texts  # sessizce düşmedi
    assert any("stop-loss geçiyor" in t for t in texts)  # cevapta stop var, taslakta yok
    assert all(v is None for v in d["costs"].values())  # maliyet uydurulmadı
    assert not out["valid"] and any("Maliyet eksik" in p for p in out["problems"])


def test_draft_failure_is_error_not_template() -> None:
    class Bad:
        def generate(self, *a, **k):
            return "üzgünüm, anlayamadım"

    with pytest.raises(DraftError, match="şablon kullanılmaz"):
        extract_draft("Soru?", "Bir cevap", llm=Bad())


def test_important_unsupported_blocks_until_explicit_simplification(market) -> None:
    rec = save_spec(_spec(unsupported_rules=[{"text": "haber sonrası girme", "why": "veri yok"}]))
    assert rec["testable"] is False
    with pytest.raises(StrategyTestError, match="basitleştirilmiş"):
        run_stage(rec["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    simp = simplify(rec["strategy_id"], remove_rules=[], drop_unsupported=["haber sonrası girme"])
    assert simp["strategy_id"] != rec["strategy_id"] and simp["family_id"] == rec["family_id"]
    assert simp["spec"]["simplified_from"] == rec["strategy_id"]
    assert simp["spec"]["removed_rules"] == ["haber sonrası girme"]
    run = run_stage(simp["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    assert run["result"]["leak_check"]["ok"]


def test_stages_protocol_trials_and_final_once(market) -> None:
    a = save_spec(_spec())
    dev = run_stage(a["strategy_id"], data_file=market, tz="UTC", stage="gelistirme")
    val = run_stage(a["strategy_id"], data_file=market, tz="UTC", stage="dogrulama")
    assert dev["period_end"] < val["period_start"]  # dönemler örtüşmez
    assert val["oos_status"] == "ilk_bakis"
    # Aynı aileden varyant (parametre değişti) aynı doğrulama dönemine bakınca:
    b = save_spec(_spec(entry_rules=["ema_10 > ema_30", "rsi_14 > 50"]), parent_id=a["strategy_id"])
    assert b["family_id"] == a["family_id"]
    val_b = run_stage(b["strategy_id"], data_file=market, tz="UTC", stage="dogrulama")
    assert val_b["oos_status"] == "gelistirmede_kullanildi"
    assert val_b["result"]["family_validation_trials"] == 2
    fin = run_stage(b["strategy_id"], data_file=market, tz="UTC", stage="final")
    assert fin["period_start"] > val_b["period_end"]
    with pytest.raises(StrategyTestError, match="ZATEN kullandı"):
        run_stage(a["strategy_id"], data_file=market, tz="UTC", stage="final")


def test_result_records_times_costs_engine_and_disclaimer(market) -> None:
    from app.trading import event_engine as ee

    a = save_spec(_spec())
    run = run_stage(a["strategy_id"], data_file=market, tz="UTC", stage="dogrulama")
    res = run["result"]
    times = res["times"]
    assert times["knowledge_available_at"] == times["backtest_run_at"]
    assert times["knowledge_available_at"] != times["data_end"]  # fiyat bitişi bilgi zamanı değil
    assert times["data_start"] < times["period_start"] <= times["period_end"] <= times["data_end"]
    assert res["costs"]["zero_reasons"]["funding_bps_per_day"]
    assert run["engine_version"] == ee.ENGINE_VERSION and res["metric_definitions"]
    assert "GELMEZ" in res["disclaimer"] and len(run["fingerprint"]) == 64
    assert res["window"]["warmup_bars_before"] > 0


def test_data_check_requires_timezone(market) -> None:
    from app.trading.data_quality import DataQualityError

    with pytest.raises(DataQualityError, match="saat dilimi"):
        check_data(market, timeframe="1h", tz=None)
    out = check_data(market, timeframe="1h", tz="UTC")
    assert out["report"]["rows_clean"] == 1200 and "Önizleme" in out["protocol"]["note"]


def test_draft_from_turn_uses_turn_answer_and_records_source(iso, monkeypatch) -> None:  # noqa: F811
    from app.feedback.chat_service import send
    from app.feedback.chat_store import ChatStore
    from app.trading.chat_strategy import draft_from_turn

    store = ChatStore()
    conv = store.create_conversation()["conversation_id"]
    turn, _ = send(
        conv,
        "EMA stratejisi?",
        "s1",
        retriever=StubRetriever(),
        llm=FakeLLM(["EMA10 EMA30'u yukarı kesince al, aşağı kesince çık."]),
        store=store,
    )
    out = draft_from_turn(
        turn["turn_id"], llm=JsonLLM({"entry_rules": ["ema_10 > ema_30"]}), store=store
    )
    assert out["source"]["turn_id"] == turn["turn_id"]
    assert out["draft"]["entry_rules"] == ["ema_10 > ema_30"] and not out["valid"]


def test_normalize_flags_short_language() -> None:
    _d, notes = normalize_draft({"direction": "long"}, answer="Fiyat düşünce açığa sat.")
    assert any("short" in n for n in notes)


def test_web_strategy_endpoints(market) -> None:
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    client = TestClient(app)
    files = client.get("/api/strategy/data-files").json()["files"]
    assert [f["name"] for f in files] == [market]
    bad = client.post("/api/strategy", json={"spec": _spec(costs={"spread_bps": 1})})
    assert bad.status_code == 422 and "Maliyet eksik" in str(bad.json()["detail"])
    rec = client.post("/api/strategy", json={"spec": _spec(), "source": {"turn_id": "t1"}}).json()
    chk = client.post(
        "/api/strategy/data-check", json={"data_file": market, "timeframe": "1h", "tz": "UTC"}
    )
    assert chk.status_code == 200
    run = client.post(
        f"/api/strategy/{rec['strategy_id']}/run",
        json={"data_file": market, "tz": "UTC", "stage": "gelistirme"},
    ).json()
    assert run["stage"] == "gelistirme" and run["result"]["leak_check"]["ok"]
    assert client.get("/api/strategy/turn/t1").json()["items"][0]["runs"]
    trav = client.post(
        f"/api/strategy/{rec['strategy_id']}/run",
        json={"data_file": "../../secret.csv", "tz": "UTC", "stage": "gelistirme"},
    )
    assert trav.status_code == 422
    human = {
        "/api/strategy/draft",
        "/api/strategy",
        "/api/strategy/data-upload",
        "/api/strategy/{strategy_id}/simplify",
        "/api/strategy/{strategy_id}/run",
    }
    seen = set()
    for rt in client.app.routes:
        if getattr(rt, "path", "") in human and "POST" in getattr(rt, "methods", set()):
            assert require_human in {d.call for d in rt.dependant.dependencies}
            seen.add(rt.path)
    assert seen == human


def test_result_warnings_flag_thin_evidence() -> None:
    from app.trading.strategy_testing import result_warnings

    w = result_warnings({"n_trades": 5, "sharpe": 6.0, "profit_factor": 10.0})
    assert len(w) == 3 and "Az işlem" in w[0]
    assert result_warnings({"n_trades": 0})[-1].startswith("Hiç işlem yok")
    assert result_warnings({"n_trades": 200, "sharpe": 1.0, "profit_factor": 1.3}) == []
