"""Gece döngüsü (Faz 1) — yerel hakem + karantina, CSV laboratuvarı, orkestratör.

Çevrimdışı: Ollama yerine sahte üretici, sentetik OHLCV, tmp veri kökü
(docs/TASARIM_GECE_DONGUSU.md). Kabul şartları: şüpheli/belirsiz aday karantinaya girer ve
eğitim seçimine girmez; karantinayı yeniden hesaplama silmez, yalnız gerekçeli insan kaldırır;
"tutarlı" hiçbir adayı onaylamaz; Ollama yoksa aday dokunulmaz; bulut etiketi reddedilir; CSV
laboratuvarı final dönemine dokunmaz, maliyetsiz koşmaz, aynı dosyayı iki kez işlemez; gece
koşusu STOP_ALL'da hiçbir şey yapmaz, eğitim başlatmaz ve bir adım düşünce diğerleri sürer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.chat_learning_helpers import iso, send  # noqa: F401

from app.orchestration import local_judge as lj
from app.orchestration import nightly
from app.trading import csv_lab as lab

REPO = Path(__file__).resolve().parents[1]


class Gen:
    def __init__(self, text: str = "KARAR: tutarli\n- iyi", fail: bool = False) -> None:
        self.text, self.fail, self.calls = text, fail, 0

    def __call__(self, prompt: str, system: str) -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("Ollama'ya ulaşılamıyor")
        return self.text


@pytest.fixture
def cand(iso):  # noqa: F811
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    store = ChatStore()
    conv = store.create_conversation("gece")["conversation_id"]
    turn = send(store, conv, "Volatilite kümelenmesi nedir?")
    c, _ = LearningService(store).learn(turn["turn_id"])
    assert c["status"] in ("eligible", "review")
    return c


def _get(cid: str) -> dict:
    from app.feedback.chat_store import ChatStore

    got = ChatStore().get_candidate(cid)
    assert got is not None
    return got


# ── yerel hakem + karantina ──────────────────────────────────────────────────


def test_suspicious_verdict_quarantines_and_survives_recompute(cand) -> None:
    from app.feedback.chat_dataset import chat_selection_blockers  # noqa: F401
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    out = lj.run_judging(generate=Gen("KARAR: şüpheli\n- kaynakta yok"), model="yerel:1")
    assert out["ran"] and out["counts"]["supheli"] == 1
    c = _get(cand["candidate_id"])
    assert c["status"] == "quarantined" and c["reason_codes"] == ["karantina"]
    assert c["quarantine"]["verdict"] == "supheli" and c["quarantine"]["model"] == "yerel:1"
    LearningService().recheck_all()
    assert _get(cand["candidate_id"])["status"] == "quarantined"
    # eğitim seçimi yalnız 'eligible' okur → karantinadaki aday girmez
    assert cand["candidate_id"] not in {
        x["candidate_id"] for x in ChatStore().list_candidates(status="eligible")
    }


def test_consistent_verdict_changes_nothing(cand) -> None:
    before = _get(cand["candidate_id"])
    out = lj.run_judging(generate=Gen("KARAR: tutarlı\n- tamam"), model="yerel:1")
    assert out["counts"]["tutarli"] == 1
    after = _get(cand["candidate_id"])
    assert after["status"] == before["status"] and after["quarantine"] == {}


def test_unparseable_verdict_is_belirsiz_and_quarantined(cand) -> None:
    out = lj.run_judging(generate=Gen("Bence güzel bir cevap."), model="yerel:1")
    assert out["counts"]["belirsiz"] == 1
    assert _get(cand["candidate_id"])["status"] == "quarantined"


def test_judge_is_cached_per_target_and_model(cand) -> None:
    g = Gen("KARAR: tutarli")
    lj.run_judging(generate=g, model="yerel:1")
    lj.run_judging(generate=g, model="yerel:1")
    assert g.calls == 1
    lj.run_judging(generate=g, model="yerel:2")
    assert g.calls == 2


def test_ollama_failure_leaves_candidate_untouched(cand) -> None:
    before = _get(cand["candidate_id"])
    out = lj.run_judging(generate=Gen(fail=True), model="yerel:1")
    assert out["counts"]["failed"] == 1
    assert _get(cand["candidate_id"])["status"] == before["status"]


def test_unavailable_ollama_skips_without_touching(cand) -> None:
    out = lj.run_judging(model="yerel:1", available=lambda: False)
    assert not out["ran"] and "Ollama" in out["skipped"]


def test_cloud_tag_judge_model_is_refused(cand) -> None:
    g = Gen("KARAR: supheli")
    out = lj.run_judging(generate=g, model="gpt-oss:120b-cloud")
    assert not out["ran"] and "bulut" in out["skipped"] and g.calls == 0


def test_only_human_lifts_quarantine_and_judge_does_not_requarantine(cand) -> None:
    from app.feedback.learning import LearningError, LearningService

    cid = cand["candidate_id"]
    lj.run_judging(generate=Gen("KARAR: supheli\n- x"), model="yerel:1")
    svc = LearningService()
    with pytest.raises(LearningError):
        svc.lift_quarantine(cid, "kısa")
    lifted = svc.lift_quarantine(cid, "Kaynağı okudum, ifade kaynakla tutarlı.")
    assert lifted["status"] != "quarantined"
    assert lifted["quarantine"]["lifted"]["reason"].startswith("Kaynağı")
    # aynı hedef için ne yeni yargı ne yeni karantina (insan kararı geçerli)
    g = Gen("KARAR: supheli")
    lj.run_judging(generate=g, model="yerel:1")
    lj.run_judging(generate=g, model="yerel:9")
    assert g.calls == 0 and _get(cid)["status"] != "quarantined"
    with pytest.raises(LearningError):
        svc.lift_quarantine(cid, "zaten kaldırılmış karantina")
    # hedef değişince yeniden yargılanır
    svc.edit(cid, "Volatilite kümelenmesi kesin olarak her zaman yüzde 90 kazandırır.")
    lj.run_judging(generate=g, model="yerel:1")
    assert g.calls == 1 and _get(cid)["status"] == "quarantined"


def test_judge_limit_leaves_rest_for_next_night(iso) -> None:  # noqa: F811
    from app.feedback.chat_store import ChatStore
    from app.feedback.learning import LearningService

    store = ChatStore()
    conv = store.create_conversation("gece")["conversation_id"]
    for q in ("Volatilite kümelenmesi nedir?", "Getiri varyansı zamanla değişir mi?"):
        t = send(store, conv, q)
        LearningService(store).learn(t["turn_id"])
    g = Gen("KARAR: tutarli")
    out = lj.run_judging(generate=g, model="yerel:1", max_candidates=1)
    assert out["judged_fresh"] == 1 and out["left_for_next_night"] == 1
    out = lj.run_judging(generate=g, model="yerel:1", max_candidates=1)
    assert out["judged_fresh"] == 1 and out["left_for_next_night"] == 0 and g.calls == 2


def test_lift_quarantine_route_requires_human() -> None:
    from app.web.security import require_human
    from app.web.server import app

    rts = [
        rt
        for rt in app.routes
        if getattr(rt, "path", "").endswith("/lift-quarantine")
        and "POST" in getattr(rt, "methods", set())
    ]
    assert len(rts) == 1
    assert require_human in {d.call for d in rts[0].dependant.dependencies}


def test_nightly_code_never_uses_cloud_or_starts_training() -> None:
    for rel in (
        "app/orchestration/local_judge.py",
        "app/orchestration/nightly.py",
        "app/trading/csv_lab.py",
    ):
        text = (REPO / rel).read_text(encoding="utf-8")
        src = "\n".join(
            ln for ln in text.splitlines() if ln.strip().startswith(("from ", "import "))
        )
        for bad in (
            "app.cloud.judge",
            "app.cloud.providers",
            "second_opinion",
            "candidate_jobs",
            "detached_launch",
            "start_training",
            "launch_training",
        ):
            assert bad not in src, f"{rel}: {bad}"


# ── CSV laboratuvarı ─────────────────────────────────────────────────────────


def _ohlcv(n: int, freq: str, seed: int = 3, trend: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = trend + rng.normal(0, 0.004, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    hi = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    lo = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    idx = pd.date_range("2022-01-03", periods=n, freq=freq)
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": 1.0}, index=idx
    )


def _write(root: Path, name: str, df: pd.DataFrame, *, tz: bool = True, meta=None) -> Path:
    d = root / "data" / "market" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    stamps = out.index.strftime("%Y-%m-%dT%H:%M:%S") + ("Z" if tz else "")
    out.insert(0, "time", stamps)
    p = d / name
    out.to_csv(p, index=False)
    if meta is not None:
        p.with_name(p.stem + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return p


@pytest.mark.parametrize(
    ("stem", "profile"),
    [
        ("BTCUSDT_perp_1h", "kripto_vadeli"),
        ("ETHUSDT_spot_4h", "kripto_spot"),
        ("XAUUSD_15m", "forex_cfd"),
        ("eurusd-1d", "forex_cfd"),
        ("THYAO.IS", "bist"),
        ("bist_xu100_gunluk", "bist"),
        ("benim_verim", "ihtiyatli"),
    ],
)
def test_profile_detection(stem: str, profile: str) -> None:
    assert lab.detect_profile(stem)[0] == profile


def test_every_profile_has_all_costs_and_conservative_is_costly() -> None:
    from app.trading.strategy_spec import COST_FIELDS, CostModel

    for key, prof in lab.PROFILES.items():
        cm = CostModel.model_validate(prof["costs"])  # sıfır yalnız gerekçeyle
        assert all(getattr(cm, f) is not None for f in COST_FIELDS), key
    cons = lab.PROFILES["ihtiyatli"]["costs"]
    assert all(cons[f] > 0 for f in COST_FIELDS)


@pytest.mark.parametrize(("freq", "tf"), [("15min", "15m"), ("1h", "1h"), ("4h", "4h")])
def test_timeframe_detected_from_data(tmp_path: Path, freq: str, tf: str) -> None:
    p = _write(tmp_path, "x.csv", _ohlcv(300, freq))
    got, _step, aware = lab.detect_timeframe(p.read_bytes())
    assert got == tf and aware


def test_timeframe_epoch_ms_and_business_days(tmp_path: Path) -> None:
    df = _ohlcv(300, "1h")
    raw = df.copy()
    raw.insert(0, "timestamp", (df.index.astype("int64") // 10**6).astype("int64"))
    p = tmp_path / "e.csv"
    raw.to_csv(p, index=False)
    assert lab.detect_timeframe(p.read_bytes())[0] == "1h"
    b = _write(tmp_path, "b.csv", _ohlcv(300, "B"), tz=False)
    assert lab.detect_timeframe(b.read_bytes())[0] == "1d"


def test_deflated_sharpe_falls_with_more_trials() -> None:
    few = lab.deflated_sharpe(0.05, 2000, 0.0, 3.0, [0.05, 0.0])
    many = lab.deflated_sharpe(0.05, 2000, 0.0, 3.0, [0.05, *np.linspace(-0.05, 0.04, 40)])
    assert 0 <= many < few <= 1


def test_pine_export_is_indicator_and_refuses_unknown() -> None:
    v = lab.build_variants("X", "1h", lab.PROFILES["forex_cfd"]["costs"], True)
    ema = next(x for x in v if x.template == "ema_trend")
    pine = lab.to_pine(ema.spec)
    assert "indicator(" in pine and "strategy(" not in pine and "ta.ema(close, 10)" in pine
    pe = next(x for x in v if x.template == "sma_kirilim_entropi")
    assert not pe.pine_ok
    with pytest.raises(ValueError):
        lab.to_pine(pe.spec)


def test_long_only_profiles_have_no_short_variants() -> None:
    spot = lab.build_variants("X", "1h", lab.PROFILES["kripto_spot"]["costs"], False)
    assert {v.spec.direction for v in spot} == {"long"}
    perp = lab.build_variants("X", "1h", lab.PROFILES["kripto_vadeli"]["costs"], True)
    assert {v.spec.direction for v in perp} == {"long", "short"}


def test_csv_lab_inbox_end_to_end_never_touches_final(iso) -> None:  # noqa: F811
    from app.trading.strategy_store import StrategyStore

    _write(iso, "BTCUSDT_perp_1h.csv", _ohlcv(3000, "1h", trend=0.0004))
    res = lab.run_inbox()
    f = res["files"][0]
    assert f["status"] == "done", f
    rep = json.loads(Path(f["report"]["json"]).read_text(encoding="utf-8"))
    assert rep["timeframe"] == "1h" and rep["profile"] == "kripto_vadeli"
    assert rep["n_trials"] == len(rep["trials"]) == 16
    assert all(v > 0 for k, v in rep["costs"].items() if k != "zero_reasons")
    assert "DOKUNULMADI" in rep["periods"]["final"]
    st = StrategyStore()
    assert st.accesses(lab.FAMILY, rep["clean_sha256"], "final") == []
    val = st.accesses(lab.FAMILY, rep["clean_sha256"], "dogrulama")
    assert len(val) == sum(1 for e in rep["selected"] if "val" in e)
    assert rep["selected"], "eğilimli sentetik veride en az bir aday seçilmeli"
    for e in rep["selected"]:
        assert e["dev"]["sharpe"] > 0 and e["verdict"] in ("aday_oos_tutarli", "aday_zayif")
        assert e["leak_check"]["ok"]
    md = Path(f["report"]["md"]).read_text(encoding="utf-8")
    assert "yatırım tavsiyesi değildir" in md.lower() or "tavsiye" in md
    # ikinci koşu aynı dosyayı yeniden işlemez
    assert lab.run_inbox()["files"][0]["status"] == "seen"


def test_csv_lab_naive_intraday_without_tz_is_skipped_with_reason(iso) -> None:  # noqa: F811
    _write(iso, "XAUUSD_15m.csv", _ohlcv(400, "15min"), tz=False)
    f = lab.run_inbox()["files"][0]
    assert f["status"] == "skipped" and "saat dilimi" in f["reason"]


def test_csv_lab_daily_naive_assumes_utc_and_sidecar_overrides(iso) -> None:  # noqa: F811
    _write(
        iso,
        "benim.csv",
        _ohlcv(400, "1D"),
        tz=False,
        meta={"profile": "bist", "costs": {"commission_bps_per_side": 20.0}},
    )
    f = lab.run_inbox()["files"][0]
    assert f["status"] == "done", f
    rep = json.loads(Path(f["report"]["json"]).read_text(encoding="utf-8"))
    assert rep["profile"] == "bist" and rep["profile_source"] == "yan dosya"
    assert rep["costs"]["commission_bps_per_side"] == 20.0
    assert any("UTC" in a for a in rep["assumptions"])
    assert {t.get("name", "").split()[-1] for t in rep["trials"]} == {"long"}


def test_csv_lab_bad_profile_in_sidecar_is_skipped(iso) -> None:  # noqa: F811
    _write(iso, "a.csv", _ohlcv(400, "1h"), meta={"profile": "uydurma"})
    f = lab.run_inbox()["files"][0]
    assert f["status"] == "skipped" and "profil" in f["reason"]


# ── orkestratör ──────────────────────────────────────────────────────────────


def test_stop_all_makes_nightly_a_noop(iso, monkeypatch) -> None:  # noqa: F811
    called = []
    monkeypatch.setattr(nightly, "STEP_FNS", {"hakem": lambda: called.append(1) or {}})
    (iso / "storage").mkdir(exist_ok=True)
    (iso / "storage" / "STOP_ALL").write_text("x")
    rep = nightly.run_nightly(("hakem",))
    assert rep["status"] == "stopped" and not called


def test_step_failure_does_not_stop_others_and_report_written(iso, monkeypatch) -> None:  # noqa: F811
    def boom() -> dict:
        raise RuntimeError("patladı")

    monkeypatch.setattr(
        nightly, "STEP_FNS", {"hakem": boom, "csv": lambda: {"files": []}, "egitim": lambda: {}}
    )
    rep = nightly.run_nightly(("hakem", "csv", "egitim"))
    assert rep["status"] == "partial"
    assert "patladı" in rep["steps"]["hakem"]["error"] and rep["steps"]["csv"]["ran"]
    assert Path(rep["report_md"]).is_file() and nightly.latest()["status"] == "partial"
    assert not (iso / "storage" / "nightly" / "gece.lock").exists()


def test_concurrent_run_is_refused(iso, monkeypatch) -> None:  # noqa: F811
    import os
    import time

    d = iso / "storage" / "nightly"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gece.lock").write_text(json.dumps({"pid": os.getpid(), "t": time.time()}))
    rep = nightly.run_nightly(("egitim",))
    assert rep["status"] == "busy"
    (d / "gece.lock").write_text(json.dumps({"pid": os.getpid(), "t": time.time() - 7 * 3600}))
    monkeypatch.setattr(nightly, "STEP_FNS", {"egitim": lambda: {"ok": 1}})
    assert nightly.run_nightly(("egitim",))["status"] == "done"


def test_training_step_only_reports(iso, monkeypatch) -> None:  # noqa: F811
    from app.training import easy_train

    monkeypatch.setattr(
        easy_train,
        "readiness",
        lambda: {"items": [{"key": "veri", "ok": True, "detail": ""}], "data_sha256": "ab"},
    )
    out = nightly._step_egitim()
    assert out["ready"] and out["started_training"] is False


def test_unknown_step_rejected(iso) -> None:  # noqa: F811
    with pytest.raises(ValueError):
        nightly.run_nightly(("egit",))


def test_pending_decisions_show_quarantine(cand) -> None:
    from app.orchestration.loop_state import _nightly_items

    assert not [i for i in _nightly_items() if i["key"] == "karantina"]
    lj.run_judging(generate=Gen("KARAR: supheli"), model="yerel:1")
    items = {i["key"]: i for i in _nightly_items()}
    assert items["karantina"]["who"] == "insan" and "1 öğrenme adayı" in items["karantina"]["title"]


def test_chat_store_concurrent_first_open_is_safe(tmp_path: Path) -> None:
    """'Bekleyen kararlar' bölümleri paralel ChatStore açar; ilk açılış yarışı patlamamalı."""
    from concurrent.futures import ThreadPoolExecutor

    from app.feedback.chat_store import ChatStore

    db = tmp_path / "yaris.db"
    with ThreadPoolExecutor(max_workers=6) as pool:
        stores = list(pool.map(lambda _i: ChatStore(db), range(6)))
    assert all(s.list_candidates() == [] for s in stores)


def test_web_lists_quarantine_and_lifts_with_reason(cand) -> None:
    from fastapi.testclient import TestClient

    from app.web.server import app

    lj.run_judging(generate=Gen("KARAR: supheli\n- kaynakta yok"), model="yerel:1")
    client = TestClient(app)
    items = client.get("/api/learn/candidates?status=quarantined").json()["items"]
    assert [i["candidate_id"] for i in items] == [cand["candidate_id"]]
    assert items[0]["quarantine"]["verdict"] == "supheli"
    path = f"/api/learn/candidates/{cand['candidate_id']}/lift-quarantine"
    assert client.post(path, json={"reason": "kısa"}).status_code >= 400
    r = client.post(path, json={"reason": "Kaynağı okudum; ifade kaynakla tutarlı."})
    assert r.status_code == 200 and r.json()["status"] != "quarantined"


# ── LLM-30 gece ölçümü ───────────────────────────────────────────────────────

from app.evals import llm30_nightly as l30  # noqa: E402


def _chat(answer: str = "Cevap 101 ve 105.", reason: str = "stop"):
    calls: list[str] = []

    def chat(model: str, prompt: str, options: dict) -> dict:
        calls.append(prompt)
        assert options["seed"] == 42 and options["temperature"] == 0.0
        return {"content": answer, "done_reason": reason, "prompt_eval_count": 50, "eval_count": 20}

    chat.calls = calls  # type: ignore[attr-defined]
    return chat


def _info(digest: str = "sha256:aaa"):
    return lambda model: {"name": model, "digest": digest}


def test_key_hits_numeric_trace_not_score() -> None:
    keys = {
        "s12_ema_alpha_0_1": 101.0,
        "s12_ema_alpha_0_5": 105.0,
        "s03_utc": "2026-01-15T07:00:00+00:00",
    }
    assert l30.key_hits("alpha=0,1 → 101; alpha=0,5 → 105,0", keys, "llm30-s12") == {
        "s12_ema_alpha_0_1": True,
        "s12_ema_alpha_0_5": True,
    }
    assert l30.key_hits("sonuç 102", keys, "llm30-s12")["s12_ema_alpha_0_1"] is False
    assert l30.key_hits("UTC 07:00", keys, "llm30-s03") == {"s03_utc": True}
    assert l30.key_hits("liste [[0, 1, 1]]", {"s22_x": [[0, 1, 1]]}, "llm30-s22") == {"s22_x": True}


def test_llm30_measures_once_per_digest_and_detects_regression(iso) -> None:  # noqa: F811
    good = _chat("Cevap: 101 ve 105 ve 0,6667.")
    out = l30.run_measurement(chat=good, info=_info("d1"), hold_lock=False)
    assert out["ran"] and out["n"] == 30 and out["regression"] == []
    assert out["key_total"] == 20 and len(good.calls) == 30
    assert Path(out["report"]).is_file()
    # aynı özet → yeniden ölçülmez
    again = l30.run_measurement(chat=good, info=_info("d1"), hold_lock=False)
    assert not again["ran"] and "değişmedi" in again["skipped"] and len(good.calls) == 30
    # yeni özet + kötüleşen cevaplar → gerileme
    bad = _chat("", reason="length")
    out2 = l30.run_measurement(chat=bad, info=_info("d2"), hold_lock=False)
    assert out2["ran"] and out2["n_flagged"] == 30
    assert any("Bayraklı" in r for r in out2["regression"])
    assert any("Sayısal" in r for r in out2["regression"])
    assert len(l30.history()) == 2


def test_llm30_skips_when_ollama_unreadable_or_busy(iso, monkeypatch) -> None:  # noqa: F811
    def boom(model: str) -> dict:
        raise RuntimeError("bağlantı yok")

    out = l30.run_measurement(chat=_chat(), info=boom)
    assert not out["ran"] and "Ollama" in out["skipped"]
    from app.training import resource_lock

    with resource_lock.hold("training", "test", check_chat=False):
        busy = l30.run_measurement(chat=_chat(), info=_info())
    assert not busy["ran"] and "Ağır iş" in busy["skipped"]


def test_llm30_stops_after_repeated_errors(iso) -> None:  # noqa: F811
    def dead(model: str, prompt: str, options: dict) -> dict:
        raise RuntimeError("Ollama düştü")

    out = l30.run_measurement(chat=dead, info=_info(), hold_lock=False)
    assert out["ran"] and out["n_errors"] == 3 and out["n"] == 3


def test_nightly_includes_measurement_and_pending_shows_regression(iso, monkeypatch) -> None:  # noqa: F811
    from app.orchestration.loop_state import _nightly_items

    assert nightly.STEPS == ("hakem", "olcum", "csv", "egitim")
    monkeypatch.setattr(
        nightly,
        "STEP_FNS",
        {"olcum": lambda: {"ran": True, "model": "m", "regression": ["Bayraklı 0 → 3"]}},
    )
    rep = nightly.run_nightly(("olcum",))
    assert "GERİLEME" in Path(rep["report_md"]).read_text(encoding="utf-8")
    keys = {i["key"] for i in _nightly_items()}
    assert "llm30_gerileme" in keys


# ── CSV adayı → final (insan, veri başına bir kez) ───────────────────────────


def _lab_with_candidates(root: Path) -> dict:
    _write(root, "BTCUSDT_perp_1h.csv", _ohlcv(3000, "1h", trend=0.0004))
    f = lab.run_inbox()["files"][0]
    rep = json.loads(Path(f["report"]["json"]).read_text(encoding="utf-8"))
    assert len([e for e in rep["selected"] if "val" in e]) >= 2
    return rep


def test_selected_candidates_saved_under_one_csvlab_family(iso) -> None:  # noqa: F811
    from app.trading.strategy_store import StrategyStore

    rep = _lab_with_candidates(iso)
    st = StrategyStore()
    for e in rep["selected"]:
        rec = st.get_strategy(e["strategy_id"])
        assert rec is not None and rec["origin"] == "csvlab" and rec["family_id"] == lab.FAMILY
        assert rec["source"]["csvlab"]["clean_sha256"] == rep["clean_sha256"]
    # seçilmeyen denemeler depoya girmez
    unselected = {t["strategy_id"] for t in rep["trials"] if "strategy_id" in t} - {
        e["strategy_id"] for e in rep["selected"]
    }
    assert unselected and all(st.get_strategy(s) is None for s in unselected)


def test_final_once_per_data_for_csvlab_family(iso) -> None:  # noqa: F811
    from app.trading.strategy_testing import StrategyTestError

    rep = _lab_with_candidates(iso)
    first, second = rep["selected"][0]["strategy_id"], rep["selected"][1]["strategy_id"]
    with pytest.raises(StrategyTestError, match="10 karakter"):
        lab.run_final(first, data_file=rep["file"], reason="kısa")
    run = lab.run_final(first, data_file=rep["file"], reason="Geliştirme+OOS en tutarlı aday.")
    assert run["stage"] == "final" and run["family_id"] == lab.FAMILY
    assert run["period_start"] >= rep["periods"]["dogrulama"][1]
    with pytest.raises(StrategyTestError, match="ZATEN"):
        lab.run_final(second, data_file=rep["file"], reason="İkinci adayı da deneyelim.")


def test_final_rejects_non_csvlab_strategy(iso) -> None:  # noqa: F811
    from app.trading.chat_strategy import save_spec
    from app.trading.strategy_testing import StrategyTestError

    rep = _lab_with_candidates(iso)
    v = lab.build_variants("X", "1h", lab.PROFILES["forex_cfd"]["costs"], True)
    rec = save_spec(v[3].spec.model_dump(mode="json"))
    with pytest.raises(StrategyTestError, match="yalnız CSV"):
        lab.run_final(rec["strategy_id"], data_file=rep["file"], reason="sohbet stratejisi deneme")


def test_judge_step_skips_while_user_chats(iso, monkeypatch) -> None:  # noqa: F811
    from app.feedback import resource_guard

    monkeypatch.setattr(resource_guard, "chat_lease_blocker", lambda root=None: "sohbet cevabı")
    out = nightly._step_hakem()
    assert not out["ran"] and "Sohbet" in out["skipped"]


def test_pending_decisions_carry_nightly_brief(iso, monkeypatch) -> None:  # noqa: F811
    from app.orchestration.loop_state import _nightly_brief

    assert _nightly_brief()["ran"] is False
    monkeypatch.setattr(
        nightly,
        "STEP_FNS",
        {
            "hakem": lambda: {
                "ran": True,
                "quarantined": [{"candidate_id": "x", "verdict": "supheli", "reason": "r"}],
            },
            "olcum": lambda: {"ran": True, "n": 30, "n_flagged": 2, "key_hits": 9, "key_total": 20},
            "csv": lambda: {
                "files": [{"file": "a.csv", "status": "done"}, {"file": "b.csv", "status": "seen"}]
            },
        },
    )
    nightly.run_nightly(("hakem", "olcum", "csv"))
    b = _nightly_brief()
    assert b["ran"] and b["status"] == "done" and b["quarantined"] == 1 and b["csv_done"] == 1
    assert b["llm30"] == "bayraklı 2/30 · iz 9/20" and b["report_md"].endswith(".md")
