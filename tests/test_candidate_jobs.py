"""Aday hattı ağır işleri (web + CLI ortak servis) — gerçek alt süreçlerle, model YOK.

Kabul şartları: aynı istek kimliği aynı işi döndürür (çift süreç yok); bir iş koşarken ya da
ortak ağır iş kilidi tutuluyorken yeni iş başlamaz; mevcut / canlı Ollama etiketi ezilmez;
desteklenmeyen platformda açık gerekçe; başarı = çıkış 0 + iş sonrası doğrulama (aksi halde
"BAŞARISIZ"); çalıştırıcı ölürse iş "kesildi" (tamamlandı sayılmaz); güvenli durdurma süreç
ağacını sonlandırır; aşama ilerlemesi günlükten okunur; web uçları insan yetkisi ister.
"""

from __future__ import annotations

import json
import sys
import time

import pytest
from tests.chat_learning_helpers import iso  # noqa: F401

from app.training import candidate_jobs as cj

TAGS = [{"name": "aktif:latest", "digest": "a" * 64}, {"name": "temel:latest", "digest": "b" * 64}]


@pytest.fixture
def env(iso, monkeypatch):  # noqa: F811
    from app.config import get_settings

    root = get_settings().root
    ad = root / "models" / "adapters" / "hektor_lora_t"
    ad.mkdir(parents=True)
    (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
    tags = list(TAGS)
    monkeypatch.setattr("app.feedback.model_identity.ollama_tags", lambda *a, **k: tags)
    monkeypatch.setattr(
        cj, "capability", lambda kind, adapter="": {"kind": kind, "supported": True, "reasons": []}
    )
    monkeypatch.setattr(
        "app.feedback.model_activation.resolve_chat_tag", lambda slot="main", root=None: "aktif"
    )
    return {"root": root, "tags": tags}


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _wait(job_id: str, until=("done", "failed", "stopped", "lost"), timeout: float = 60) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        j = cj.get_job(job_id)
        if j["status"] in until:
            return j
        time.sleep(0.2)
    raise AssertionError(cj.get_job(job_id))


def _inproc(job_id: str) -> tuple[int, float | None]:
    """Çalıştırıcıyı bu süreçte (iş parçacığında) koştur: yamalı doğrulama ona da geçer."""
    import os
    import threading

    from app.training.resource_lock import process_create_time

    threading.Thread(target=cj.run, args=(job_id,), daemon=True).start()
    return os.getpid(), process_create_time(os.getpid())


def _start(
    monkeypatch,
    cmd: list[str],
    *,
    kind="conversion",
    req="req-job-0001",
    tag="yeni-aday",
    inproc=False,
    **params,
):
    monkeypatch.setattr(cj, "conversion_cmd", lambda *a: cmd)
    monkeypatch.setattr(cj, "comparison_cmd", lambda **k: cmd)
    return cj.start_job(
        kind,
        adapter="hektor_lora_t",
        request_id=req,
        params={"ollama_tag": tag, **params},
        spawn=_inproc if inproc else None,
    )


def test_progress_and_failed_verification_is_not_done(env, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.training.candidate_checks.verify_conversion",
        lambda a, t: {"ok": False, "checks": [{"key": "yarim_cikti_yok", "ok": False}]},
    )
    j = _start(
        monkeypatch, _py("print('[x] 1/5 birlestirme'); print('[x] 4/5 Modelfile')"), inproc=True
    )
    done = _wait(j["job_id"])
    assert done["status"] == "failed" and "yarim_cikti_yok" in done["error"]  # çıkış 0 yetmez
    assert done["progress"]["done"] == 4 and done["progress"]["total"] == 5


def test_success_requires_post_verification(env, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.training.candidate_checks.verify_conversion",
        lambda a, t: {"ok": True, "checks": [], "digest": "c" * 64},
    )
    j = _start(monkeypatch, _py("print('5/5 ollama create')"), inproc=True)
    done = _wait(j["job_id"])
    assert done["status"] == "done" and done["verification"]["digest"] == "c" * 64


def test_nonzero_exit_is_failed(env, monkeypatch) -> None:
    j = _start(monkeypatch, _py("import sys; print('2/5'); sys.exit(3)"))
    done = _wait(j["job_id"])
    assert done["status"] == "failed" and done["exit_code"] == 3 and "çıkış 3" in done["error"]


def test_idempotent_request_and_single_running_job(env, monkeypatch) -> None:
    j = _start(monkeypatch, _py("import time; time.sleep(30)"))
    again = cj.start_job(
        "conversion",
        adapter="hektor_lora_t",
        request_id="req-job-0001",
        params={"ollama_tag": "yeni-aday"},
    )
    assert again["job_id"] == j["job_id"] and again["replayed"]
    with pytest.raises(cj.JobError, match="Başka bir ağır iş"):
        cj.start_job(
            "conversion",
            adapter="hektor_lora_t",
            request_id="req-job-0002",
            params={"ollama_tag": "x-aday"},
        )
    stopped = cj.stop_job(j["job_id"], "test")
    assert stopped["status"] == "stopped"
    from app.training.resource_lock import pid_alive

    time.sleep(0.5)
    assert not pid_alive(j["runner_pid"])  # süreç ağacı sonlandı
    # Durdurulan iş için yeni istek → yeni iş (tekrar dene).
    retry = _start(monkeypatch, _py("print('1/5')"), req="req-job-0003")
    assert retry["job_id"] != j["job_id"]
    _wait(retry["job_id"])


def test_existing_or_live_tag_never_overwritten(env, monkeypatch) -> None:
    with pytest.raises(cj.JobError, match="zaten var"):
        _start(monkeypatch, _py("pass"), tag="temel")
    env["tags"].append({"name": "canli:latest", "digest": "d" * 64})
    monkeypatch.setattr(
        "app.feedback.model_activation.resolve_chat_tag",
        lambda slot="main", root=None: "ozel-canli",
    )
    with pytest.raises(cj.JobError, match="sohbette kullanılan"):
        _start(monkeypatch, _py("pass"), tag="ozel-canli")


def test_shared_heavy_lock_blocks_web_start(env, monkeypatch) -> None:
    from app.training import resource_lock

    info, _why = resource_lock.acquire("training", "cli:test")
    assert info is not None
    try:
        with pytest.raises(cj.JobError, match="eğitim"):
            _start(monkeypatch, _py("pass"))
    finally:
        resource_lock.release(info["token"])


def test_dead_runner_reconciles_as_lost(env, monkeypatch) -> None:
    j = _start(monkeypatch, _py("import time; time.sleep(30)"))
    _wait(j["job_id"], until=("running",))
    from app.training.candidate_jobs import _kill_tree

    _kill_tree(j["runner_pid"])  # sunucu/çalıştırıcı çöktü (sonuç yazılamadı)
    time.sleep(0.5)
    lost = cj.get_job(j["job_id"])
    assert lost["status"] == "lost" and "TAMAMLANMIŞ" in lost["error"]


def test_comparison_requires_existing_candidate_tag(env, monkeypatch) -> None:
    from app.config import get_settings

    qs = get_settings().root / "set.jsonl"
    qs.write_text(json.dumps({"id": "q1", "family": "f", "question": "?"}) + "\n", "utf-8")
    monkeypatch.setattr(cj, "DEFAULT_SET", str(qs))
    with pytest.raises(cj.JobError, match="Ollama'da yok"):
        _start(monkeypatch, _py("pass"), kind="comparison", tag="olmayan-aday", base="temel")
    env["tags"].append({"name": "yeni-aday:latest", "digest": "e" * 64})
    j = _start(monkeypatch, _py("print('İLERLEME 3/9 active')"), kind="comparison", base="temel")
    done = _wait(j["job_id"])
    assert done["status"] == "failed" and "kimliği" in done["error"]  # üretim kanıtı yok
    assert done["progress"]["label"] == "cevap 3/9"


def test_capability_reports_reason_on_unsupported_platform(iso, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(cj, "is_windows", lambda: False)
    cap = cj.capability("conversion", "hektor_lora_t")
    assert not cap["supported"] and any("Windows" in r for r in cap["reasons"])


def test_web_routes_and_human_scope(env, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    from app.web.security import require_human
    from app.web.server import app

    client = TestClient(app)
    d = client.get("/api/candidates").json()
    assert d["items"][0]["adapter"] == "hektor_lora_t" and "capabilities" in d
    v = client.get("/api/candidates/hektor_lora_t/verify").json()
    assert v["completion"]["ok"] is False  # adapter dosyaları/koşu kaydı yok → doğrulanmadı
    bad = client.post(
        "/api/candidates/hektor_lora_t/prepare",
        json={"ollama_tag": "temel", "request_id": "req-web-0001"},
    )
    assert bad.status_code == 422 and "zaten var" in bad.json()["detail"]
    human = {
        "/api/candidates/{adapter}/prepare",
        "/api/candidates/{adapter}/compare",
        "/api/candidates/jobs/{job_id}/stop",
    }
    seen = set()
    for rt in client.app.routes:
        if getattr(rt, "path", "") in human:
            assert require_human in {x.call for x in rt.dependant.dependencies}
            seen.add(rt.path)
    assert seen == human
