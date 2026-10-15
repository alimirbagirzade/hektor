"""Sürekli döngü — Kademe 2 2026-10-15 bulgularının regresyon testleri.

Kanıt: docs/evidence/kademe2_2026-10-15.json. Gerçek süreç/Ollama/eğitim YOK.
"""

# ruff: noqa: F811 — fikstürler (loop_env) test_continuous_loop'tan içe aktarılır

from __future__ import annotations

import datetime as dt
import json

from tests.test_continuous_loop import (  # noqa: F401 — fikstürler
    _fake_completion,
    _raw_run,
    _register,
    _scored,
    _settings,
    _tracker,
    loop_env,
)

from app.agents.runtime import approvals
from app.memory.sqlite_store import SqliteStore

REF = "loop_ref_hektor-v12-30b_abc123def456_s42-43"


def _verified(cl, monkeypatch, name):
    """Bitişi doğrulanmış, Ollama'da etiketi ve referansı olan aday."""
    _register(name)
    _fake_completion(monkeypatch, ok=True)
    cl.tick()  # bitiş
    monkeypatch.setattr(cl, "ollama_digest", lambda tag: "abc123def4567890")


def test_cl2_dead_job_counts_attempts_and_fails_candidate(loop_env, monkeypatch):
    cl = loop_env
    from app.evals import llm30

    monkeypatch.setattr(llm30, "QUESTION_IDS", ("llm30-s01",))
    _verified(cl, monkeypatch, "hektor_loop_e")
    _scored(cl.llm30_dir(REF), ["llm30-s01"])
    spawned = []
    monkeypatch.setattr(cl, "spawn", lambda args, log: spawned.append(args) or {"pid": 1})
    for _ in range(4):
        cl.tick()  # iş başlar → (ölü, yarım çıktı) → deneme +1 → … → 3. denemede aday düşer
    # Her başarısız denemeden sonra TAZE koşu kimliği (sürdürülemeyen koşuya takılmaz — CL-1).
    run_ids = [a[a.index("--run-id") + 1] for a in spawned]
    assert run_ids == [
        "loop_hektor_loop_e_s42-43",
        "loop_hektor_loop_e_s42-43_d1",
        "loop_hektor_loop_e_s42-43_d2",
    ]
    cand = cl.read_state()["candidates"]["hektor_loop_e"]
    assert cand["failed"] and "3 denemede" in cand["failure"]


def test_cl2_busy_exit_is_not_an_attempt(loop_env, monkeypatch, tmp_path):
    cl = loop_env
    exit_file = tmp_path / "x.exit.json"
    exit_file.write_text(json.dumps({"exit_code": 9}), encoding="utf-8")
    st = cl.read_state()
    st["job"] = {"kind": "llm30_ref", "key": "k", "run_id": "r", "exit_file": str(exit_file)}
    cl.write_state(st)
    monkeypatch.setattr(cl, "busy", lambda: "meşgul")
    cl.tick()
    assert cl.read_state().get("attempts", {}).get("k", 0) == 0


def test_cl2_reference_job_exhaustion_pauses(loop_env, monkeypatch):
    cl = loop_env
    _verified(cl, monkeypatch, "hektor_loop_f")
    st = cl.read_state()
    st["attempts"] = {REF: 3}
    cl.write_state(st)
    assert cl.tick()["action"] == "duraklatildi"
    assert "Referans" in cl.read_state()["pause_reason"]


def test_cl3_stopped_conversion_fails_candidate(loop_env, monkeypatch):
    cl = loop_env
    from app.training import candidate_jobs

    _register("hektor_loop_g")
    _fake_completion(monkeypatch, ok=True)
    cl.tick()
    monkeypatch.setattr(cl, "ollama_digest", lambda tag: "")
    st = cl.read_state()
    st["candidates"]["hektor_loop_g"]["conversion_job"] = "job_x"
    cl.write_state(st)
    monkeypatch.setattr(candidate_jobs, "get_job", lambda j: {"status": "stopped"})
    cl.tick()
    assert cl.read_state()["candidates"]["hektor_loop_g"]["failed"]


def test_cl4_ollama_unreachable_waits_without_failing(loop_env, monkeypatch):
    cl = loop_env
    _register("hektor_loop_h")
    _fake_completion(monkeypatch, ok=True)
    cl.tick()
    monkeypatch.setattr(cl, "ollama_digest", lambda tag: None)
    assert cl.tick() == {"action": "bekliyor", "detail": "Ollama'ya erişilemiyor"}
    st = cl.read_state()
    assert not st["candidates"]["hektor_loop_h"].get("failed") and not st.get("paused")


def test_cl5_resume_releases_last_failed_candidate(loop_env, monkeypatch):
    cl = loop_env
    from app.orchestration import training_loop as tl

    st = cl.read_state()
    st["consecutive_failures"] = 1
    cl.write_state(st)
    _register("hektor_loop_i")
    _fake_completion(monkeypatch, ok=False)
    cl.tick()  # 2. başarısızlık → duraklat; serbest bırakma YOK
    assert cl.read_state()["paused"]
    assert not (tl.get("hektor_loop_i") or {}).get("release")
    cl.resume("insan inceledi, devam edilsin")
    assert (tl.get("hektor_loop_i") or {}).get("release", {}).get("approval_id")


def test_cl7_repeated_exception_pauses(loop_env, monkeypatch):
    cl = loop_env

    def boom(state):
        raise RuntimeError("kalıcı hata")

    monkeypatch.setattr(cl, "_advance_candidate", boom)
    assert cl.tick()["action"] == "hata"
    assert cl.tick()["action"] == "hata"
    assert cl.tick()["action"] == "duraklatildi"
    assert "kalıcı hata" in cl.read_state()["pause_reason"]


def test_auth2_snapshot_name_bound_to_data_and_settings_untouched(loop_env, monkeypatch):
    cl = loop_env
    from app.training import easy_train as et

    monkeypatch.setattr(
        et, "readiness", lambda: {"items": [{"key": "veri", "ok": True}], "data_sha256": "f" * 64}
    )
    seen = []

    def snap(settings, remember=True):
        seen.append((settings["adapter_name"], remember))
        return {**settings, "recipe_sha": "c" * 64}

    monkeypatch.setattr(et, "prepare_snapshot", snap)
    monkeypatch.setattr(
        et,
        "launch",
        lambda s, r, unattended=False: {"ok": False, "status": "blocked", "problems": ["RAM"]},
    )
    for _ in range(cl.MAX_BLOCKED_LAUNCHES - 1):
        assert cl.tick()["action"] == "egitim_baslatilamadi"
    assert cl.tick()["action"] == "duraklatildi"
    assert set(seen) == {("hektor_loop_" + "f" * 12, False)}


def test_loop_job_records_exit_code(tmp_path):
    from app.orchestration import loop_job

    out = tmp_path / "exit.json"
    # İç çağrı (argümansız sarmalayıcı) 2 ile çıkar ve kod dosyaya yazılır.
    assert loop_job.main([str(out), "app.orchestration.loop_job"]) == 2
    assert json.loads(out.read_text("utf-8"))["exit_code"] == 2
    assert loop_job.main([str(out), "os"]) == 2  # app.* dışı modül reddedilir


def test_cl6_judge_tolerates_truncated_line(tmp_path):
    from app.evals import llm30_judge as j

    run = _raw_run(tmp_path, ["llm30-s01"])
    j.scored_path(run, "yerel-hakem").write_text('{"condition": "B", "qu', encoding="utf-8")
    full = '{"dogruluk": 4, "alt_madde": 2, "tutarlilik": 2, "dogrulama": 2, "kritik": []}'
    out = j.score_run(run, generate=lambda p, s: full, model="yerel-hakem")
    assert out["rows"] == 2 and out["new"] == 2


def test_cl9_scores_are_keyed_by_judge(tmp_path):
    from app.evals import llm30_judge as j

    run = _raw_run(tmp_path, ["llm30-s01"])
    assert j.scored_path(run, "hakem-a") != j.scored_path(run, "hakem-b")


def test_auth4_unused_policy_approval_is_not_left_fresh(monkeypatch):
    _settings(monkeypatch, unattended_training_enabled="true")
    st = SqliteStore()
    real = st.consume_fresh_approval
    monkeypatch.setattr(st, "consume_fresh_approval", lambda a, b: None)
    d = approvals.policy_authorize("lora-trainer", "train_run:race", "critical", "s", store=st)
    assert not d.authorized
    monkeypatch.setattr(st, "consume_fresh_approval", real)
    assert not approvals.has_fresh_approval("lora-trainer", "train_run:race", store=st)


def test_auth5_auto_lora_cannot_be_second_owner(monkeypatch):
    from app.lora.auto_pipeline import AutoLoRAPipeline

    _settings(monkeypatch, unattended_training_enabled="true")
    p = AutoLoRAPipeline()
    assert p.set_enabled(True) is False
    _settings(monkeypatch, unattended_training_enabled="false")
    assert p.set_enabled(True) is True  # insan onaylı eski kullanım bozulmaz


def test_auth5_orchestrator_handoff_names_real_owner(monkeypatch):
    from app.orchestration import delegates

    class Ctx:
        def __init__(self) -> None:
            self.run = {"adapter_name": "x"}
            self.params: dict = {}

    _settings(monkeypatch, unattended_training_enabled="true", continuous_loop_enabled="false")
    res = delegates.train_handoff(Ctx())
    assert res.status.value == "blocked"
    _settings(monkeypatch, unattended_training_enabled="true", continuous_loop_enabled="true")
    res = delegates.train_handoff(Ctx())
    assert res.status.value == "completed" and res.output["owner"] == "continuous-loop"


def test_auth6_policy_run_not_revived_after_policy_off(monkeypatch):
    from app.training import train_guard

    _settings(monkeypatch, unattended_training_enabled="false")
    now = dt.datetime.now(dt.UTC)
    status = {"started_at": now.isoformat(), "approval_id": "apr_p", "iterations": 10}
    appr = [
        {
            "approval_id": "apr_p",
            "action": "train_run:abc",
            "status": "approved",
            "consumed_at": now.isoformat(),
            "decided_by": "policy:unattended_training",
        }
    ]
    v = train_guard.recovery_allowed(status, appr, now=now)
    assert not v.allowed and "politika" in v.reason
