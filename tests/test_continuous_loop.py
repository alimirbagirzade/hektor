"""Sürekli (gözetimsiz) eğitim döngüsü — kullanıcı kararı 2026-10-15.

Gerçek eğitim / Ollama / alt süreç YOK: yan etkili her çağrı modül kancalarıyla sahtelenir.
Sınanan sözleşmeler: politika onayı (insan kararı gibi görünmez, terfiye yetki vermez, ayar
kapalıyken hiçbir şey açmaz), Kademe 2 kod-yalnız kapsamı, merge-aware profil, yerel hakem
şeması + "tüm sorular doğru" ölçütü ve döngünün adım makinesi.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.runtime import approvals
from app.memory.sqlite_store import SqliteStore


def _settings(monkeypatch, **values):
    from app.config import settings as settings_mod

    for k, v in values.items():
        monkeypatch.setenv(f"HEKTOR_{k.upper()}", str(v))
    settings_mod.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _tracker(tmp_path):
    from app.agents.runtime.tracker import RunTracker, set_tracker

    set_tracker(RunTracker(store=SqliteStore(), jsonl_dir=tmp_path / "agent_runs"))
    yield
    set_tracker(None)


@pytest.fixture
def loop_env(monkeypatch):
    import shutil

    from app.orchestration import continuous_loop as cl
    from app.orchestration import training_loop as tl

    _settings(monkeypatch, continuous_loop_enabled="true", unattended_training_enabled="true")
    shutil.rmtree(cl.loop_dir(), ignore_errors=True)
    shutil.rmtree(tl.loop_dir(), ignore_errors=True)
    monkeypatch.setattr(cl, "busy", lambda: "")
    monkeypatch.setattr(cl, "job_alive", lambda job: False)
    monkeypatch.setattr("app.orchestration.nightly.stop_marker", lambda: "")
    monkeypatch.setattr(tl, "sync_from_status", lambda: None)
    yield cl
    shutil.rmtree(cl.loop_dir(), ignore_errors=True)
    shutil.rmtree(tl.loop_dir(), ignore_errors=True)


# ── politika onayı ───────────────────────────────────────────────────────────


def test_policy_off_creates_nothing(monkeypatch):
    _settings(monkeypatch, unattended_training_enabled="false")
    st = SqliteStore()
    before = len(approvals.list_approvals(store=st, limit=500))
    d = approvals.policy_authorize("lora-trainer", "train_run:x", "critical", "s", store=st)
    assert not d.authorized and d.approval_id == ""
    assert len(approvals.list_approvals(store=st, limit=500)) == before


def test_policy_on_is_recorded_as_policy_not_user(monkeypatch):
    _settings(monkeypatch, unattended_training_enabled="true")
    st = SqliteStore()
    d = approvals.policy_authorize("lora-trainer", "train_run:pol1", "critical", "s", store=st)
    assert d.authorized
    row = approvals.get_approval(d.approval_id, store=st)
    assert row is not None and row.decided_by == "policy:unattended_training"
    assert row.consumed_at  # tek kullanımlık: hemen tüketildi
    assert not approvals.has_fresh_approval("lora-trainer", "train_run:pol1", store=st)


def test_policy_never_authorizes_promotion(monkeypatch):
    _settings(monkeypatch, unattended_training_enabled="true")
    with pytest.raises(ValueError):
        approvals.policy_authorize("lora-trainer", "promote_adapter", "high", "s")


def test_policy_respects_stop_all(monkeypatch):
    _settings(monkeypatch, unattended_training_enabled="true")
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda *a: True)
    d = approvals.policy_authorize("lora-trainer", "train_run:stop", "critical", "s")
    assert not d.authorized


def test_easy_train_unattended_launch_blocked_when_policy_off(monkeypatch):
    from app.training import easy_train as et

    _settings(monkeypatch, unattended_training_enabled="false")
    monkeypatch.setattr(et, "_snapshot", lambda sid: {"recipe_sha": "a" * 64})
    monkeypatch.setattr(et, "precheck", lambda recipe: [])
    monkeypatch.setattr(et, "summary", lambda recipe: "özet")
    called = []
    monkeypatch.setattr(
        "app.training.detached_launch.launch", lambda **kw: called.append(kw) or {"ok": True}
    )
    res = et.launch("snap_" + "a" * 16, "loop-test-request", unattended=True)
    assert res["status"] == "blocked" and not called


def test_auto_lora_no_longer_follows_unattended_flag(monkeypatch):
    from app.lora import auto_pipeline

    _settings(monkeypatch, unattended_training_enabled="true")
    monkeypatch.setattr(auto_pipeline, "_pipeline", None)
    assert auto_pipeline.get_auto_pipeline().auto_enabled is False
    monkeypatch.setattr(auto_pipeline, "_pipeline", None)


# ── Kademe 2 kod-yalnız kapsamı ──────────────────────────────────────────────


def _k2(scope, findings=()):
    return {
        "code_sha": "c" * 64,
        "scope": scope,
        "findings": list(findings),
        "closure": {"status": "kapandi"},
    }


def test_kademe2_code_only_covers_new_data(monkeypatch):
    from app.training import easy_train as et

    monkeypatch.setattr(et, "list_kademe2", lambda: [_k2({"data_sha256": "old" * 21 + "x"})])
    assert et.latest_kademe2("c" * 64, "", "new" * 21 + "y") is None
    assert et.latest_kademe2("c" * 64, "", "new" * 21 + "y", code_only=True) is not None
    assert et.latest_kademe2("d" * 64, "", "new" * 21 + "y", code_only=True) is None


def test_kademe2_code_only_keeps_recipe_limited_acceptance(monkeypatch):
    from app.training import easy_train as et

    limited = {"status": "risk_kabul", "kapsam": "yalniz_recete"}
    monkeypatch.setattr(et, "_recipe_limited", lambda f: f.get("kapsam") == "yalniz_recete")
    monkeypatch.setattr(et, "_check_limited_bounds", lambda f, r: None)
    monkeypatch.setattr(et, "_unbounded_risk", lambda f: False)
    monkeypatch.setattr(et, "_failed_conditions", lambda f: [])
    monkeypatch.setattr(et, "list_kademe2", lambda: [_k2({"recipe_sha": "r" * 64}, [limited])])
    assert et.latest_kademe2("c" * 64, "", "z" * 64, code_only=True) is None
    assert et.latest_kademe2("c" * 64, "r" * 64, "z" * 64, code_only=True) is not None


# ── merge-aware profil ───────────────────────────────────────────────────────


def test_merge_aware_profile_loads():
    from app.training.peft_lora_train import load_lora_profile

    prof = load_lora_profile("moe30b_attn_long_ma")
    assert prof["merge_aware_bf16"] is True
    assert prof["lora_dropout"] == 0
    assert prof["neftune_noise_alpha"] == 0
    assert prof["max_seq_length"] == 6144


# ── yerel hakem ──────────────────────────────────────────────────────────────

DIMS = {"dogruluk": 4, "alt_madde": 2, "tutarlilik": 2, "dogrulama": 2}


def test_parse_judgement_accepts_only_schema():
    from app.evals.llm30_judge import parse_judgement

    ok = '{"dogruluk": 4, "alt_madde": 2, "tutarlilik": 1, "dogrulama": 0, "kritik": []}'
    assert parse_judgement(ok, DIMS, [])[0]["tutarlilik"] == 1
    assert parse_judgement(ok.replace(": 4", ": 5"), DIMS, []) is None  # aralık dışı
    assert parse_judgement('{"dogruluk": 4}', DIMS, []) is None  # eksik boyut
    bad_code = ok.replace("[]", '["ema_alpha_ters"]')
    assert parse_judgement(bad_code, DIMS, []) is None  # bu sorunun kodu değil
    assert parse_judgement(bad_code, DIMS, ["ema_alpha_ters"])[1] == ["ema_alpha_ters"]
    assert parse_judgement("puan: iyi", DIMS, []) is None


def _raw_run(tmp_path: Path, qids, seeds=(42, 43)) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "condition": "B",
            "question_id": q,
            "seed": s,
            "rag_enabled": False,
            "raw_answer": f"cevap {q}",
            "score_by_dimension": None,
            "critical_errors": None,
        }
        for q in qids
        for s in seeds
    ]
    (run / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return run


def test_score_run_resumes_and_all_correct(tmp_path):
    from app.evals import llm30_judge as j

    run = _raw_run(tmp_path, ["llm30-s01", "llm30-s04"])
    calls = []

    def gen(prompt, system):
        calls.append(prompt)
        return '{"dogruluk": 4, "alt_madde": 2, "tutarlilik": 2, "dogrulama": 2, "kritik": []}'

    out = j.score_run(run, generate=gen, model="yerel-hakem")
    assert out == {
        "path": str(j.scored_path(run, "yerel-hakem")),
        "rows": 4,
        "new": 4,
        "unscored": 0,
    }
    assert (run / "raw.jsonl").read_text("utf-8").count('"score_by_dimension": null') == 4
    assert j.all_correct(j.scored_path(run, "yerel-hakem"))["ok"] is True
    n = len(calls)
    j.score_run(run, generate=gen, model="yerel-hakem")  # sürdürme: yeniden sorulmaz
    assert len(calls) == n


def test_unparseable_judge_leaves_row_unscored_and_not_correct(tmp_path):
    from app.evals import llm30_judge as j

    run = _raw_run(tmp_path, ["llm30-s01"])
    out = j.score_run(run, generate=lambda p, s: "bilmiyorum", model="yerel-hakem")
    assert out["unscored"] == 2
    res = j.all_correct(j.scored_path(run, "yerel-hakem"))
    assert res["ok"] is False and res["wrong"] == ["llm30-s01"]


def test_judge_rejects_cloud_model(tmp_path):
    from app.evals import llm30_judge as j

    run = _raw_run(tmp_path, ["llm30-s01"])
    with pytest.raises(ValueError):
        j.score_run(run, generate=lambda p, s: "{}", model="gpt-oss:120b-cloud")


def test_one_wrong_dogruluk_is_not_all_correct(tmp_path):
    from app.evals import llm30_judge as j

    run = _raw_run(tmp_path, ["llm30-s01", "llm30-s04"])
    full = '{"dogruluk": 4, "alt_madde": 2, "tutarlilik": 2, "dogrulama": 2, "kritik": []}'
    # Satır sırası: s01/42, s01/43, s04/42, s04/43 → s04'ün TEK tohumu bile yanlışsa soru yanlış.
    answers = iter([full, full, full, full.replace('"dogruluk": 4', '"dogruluk": 3')])
    j.score_run(run, generate=lambda p, s: next(answers), model="yerel-hakem")
    res = j.all_correct(j.scored_path(run, "yerel-hakem"))
    assert res["ok"] is False and res["n_correct"] == 1


# ── adım makinesi ────────────────────────────────────────────────────────────


def test_disabled_loop_does_nothing(monkeypatch):
    from app.orchestration import continuous_loop as cl

    _settings(monkeypatch, continuous_loop_enabled="false")
    assert cl.tick()["action"] == "kapali"


def test_stop_marker_and_pause(loop_env, monkeypatch):
    cl = loop_env
    monkeypatch.setattr("app.orchestration.nightly.stop_marker", lambda: "STOP_ALL")
    assert cl.tick()["action"] == "durduruldu"
    monkeypatch.setattr("app.orchestration.nightly.stop_marker", lambda: "")
    cl.pause("insan duraklattı (test)")
    assert cl.tick()["action"] == "duraklatildi"
    cl.resume("test bitti, devam et")
    assert cl.read_state()["paused"] is False


def test_running_background_job_waits(loop_env, monkeypatch):
    cl = loop_env
    st = cl.read_state()
    st["job"] = {"kind": "llm30_ref", "pid": 1}
    cl.write_state(st)
    monkeypatch.setattr(cl, "job_alive", lambda job: True)
    assert cl.tick()["action"] == "bekliyor"


def _register(name: str):
    from app.orchestration import training_loop as tl

    tl.register(name, source="test", data_sha256="d" * 64)


def _fake_completion(monkeypatch, ok: bool):
    from app.training import candidate_checks

    monkeypatch.setattr(candidate_checks, "recipe_for_adapter", lambda a: {})
    monkeypatch.setattr(
        candidate_checks,
        "verify_run_completion",
        lambda a, r=None: {
            "ok": ok,
            "adapter_sha256": "w" * 64,
            "checks": [{"key": "cikis_kodu", "ok": ok, "detail": "test"}],
        },
    )
    monkeypatch.setattr("app.training.exit_witness.read_exit", lambda a: {})


def test_failed_completion_counts_and_policy_releases(loop_env, monkeypatch):
    cl = loop_env
    from app.orchestration import training_loop as tl

    _register("hektor_loop_a")
    _fake_completion(monkeypatch, ok=False)
    assert cl.tick()["action"] == "serbest_birakildi"
    st = cl.read_state()
    assert st["consecutive_failures"] == 1 and st["candidates"]["hektor_loop_a"]["failed"]
    rel = (tl.get("hektor_loop_a") or {}).get("release") or {}
    assert rel.get("by") == "policy:unattended_training" and rel.get("approval_id")


def test_second_consecutive_failure_pauses(loop_env, monkeypatch):
    cl = loop_env
    st = cl.read_state()
    st["consecutive_failures"] = 1
    cl.write_state(st)
    _register("hektor_loop_b")
    _fake_completion(monkeypatch, ok=False)
    assert cl.tick()["action"] == "duraklatildi"
    assert "ardışık" in cl.read_state()["pause_reason"]


def test_conversion_started_for_verified_candidate(loop_env, monkeypatch):
    cl = loop_env
    from app.training import candidate_jobs

    _register("hektor_loop_c")
    _fake_completion(monkeypatch, ok=True)
    assert cl.tick()["action"] == "bitis_dogrulandi"
    monkeypatch.setattr(cl, "ollama_digest", lambda tag: "")
    started = []
    monkeypatch.setattr(
        candidate_jobs,
        "start_job",
        lambda kind, **kw: started.append((kind, kw)) or {"job_id": "job_test0001"},
    )
    assert cl.tick()["action"] == "donusum_baslatildi"
    kind, kw = started[0]
    assert kind == "conversion" and kw["params"]["ollama_tag"] == "hektor-loop-c-aday"


def _scored(run_dir: Path, qids, dogruluk=4):
    from app.evals.llm30_judge import scored_path

    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in qids:
        for s in (42, 43):
            rows.append(
                {
                    "condition": "B",
                    "question_id": q,
                    "seed": s,
                    "split": "validation",
                    "rag_enabled": False,
                    "score_by_dimension": {
                        "dogruluk": dogruluk,
                        "alt_madde": 2,
                        "tutarlilik": 2,
                        "dogrulama": 2,
                    },
                    "critical_errors": [],
                    "judge": {"model": "t"},
                }
            )
    text = "".join(json.dumps(r) + "\n" for r in rows)
    (run_dir / "raw.jsonl").write_text(text, encoding="utf-8")
    scored_path(run_dir).write_text(text, encoding="utf-8")


def test_all_correct_pauses_for_human(loop_env, monkeypatch):
    cl = loop_env
    from app.evals import llm30
    from app.orchestration import training_loop as tl

    monkeypatch.setattr(llm30, "QUESTION_IDS", ("llm30-s01", "llm30-s04"))
    _register("hektor_loop_d")
    _fake_completion(monkeypatch, ok=True)
    cl.tick()  # bitiş
    monkeypatch.setattr(cl, "ollama_digest", lambda tag: "abc123def4567890")
    st = cl.read_state()
    seed_tag = "s42-43"
    ref_id = "loop_ref_hektor-v12-30b_abc123def456_" + seed_tag
    cand_id = f"loop_hektor_loop_d_{seed_tag}"
    _scored(cl.llm30_dir(ref_id), ["llm30-s01", "llm30-s04"], dogruluk=3)
    _scored(cl.llm30_dir(cand_id), ["llm30-s01", "llm30-s04"], dogruluk=4)
    cl.write_state(st)
    recorded = {}

    def fake_dev(name, **kw):
        recorded.update(kw)
        rec = tl.get(name) or {}
        rec["dev_eval"] = {"passed": True, "decision": "insan_onayi_bekliyor"}
        return rec

    monkeypatch.setattr(tl, "record_dev_eval", fake_dev)
    out = cl.tick()
    assert out["action"] == "duraklatildi"
    assert "TÜM SORULAR DOĞRU" in cl.read_state()["pause_reason"]
    assert recorded["min_repeats"] == 2 and recorded["candidate_condition"] == "B"


def test_same_data_waits_for_new_data(loop_env, monkeypatch):
    cl = loop_env
    from app.training import easy_train as et

    monkeypatch.setattr(
        et,
        "readiness",
        lambda: {
            "items": [
                {
                    "key": "egitim_dongusu",
                    "ok": False,
                    "detail": "Bu veri özetiyle (abc) x zaten eğitildi — yeni veri gerekir",
                }
            ]
        },
    )
    assert cl.tick()["action"] == "yeni_veri_bekleniyor"


def test_ready_starts_unattended_training(loop_env, monkeypatch):
    cl = loop_env
    from app.training import easy_train as et

    monkeypatch.setattr(
        et, "readiness", lambda: {"items": [{"key": "veri", "ok": True}], "data_sha256": "d" * 64}
    )
    monkeypatch.setattr(
        et, "prepare_snapshot", lambda s, remember=True: {**s, "recipe_sha": "b" * 64}
    )
    calls = []
    monkeypatch.setattr(
        et,
        "launch",
        lambda snap, req, unattended=False: (
            calls.append((snap, req, unattended)) or {"ok": True, "message": "başladı"}
        ),
    )
    out = cl.tick()
    assert out["action"] == "egitim_baslatildi"
    assert calls == [("snap_" + "b" * 16, "loop-snap_" + "b" * 16, True)]


def test_ready_but_policy_off_needs_human(loop_env, monkeypatch):
    cl = loop_env
    from app.training import easy_train as et

    _settings(monkeypatch, unattended_training_enabled="false")
    monkeypatch.setattr(et, "readiness", lambda: {"items": [{"key": "veri", "ok": True}]})
    assert cl.tick()["action"] == "insan_onayi_gerekli"
