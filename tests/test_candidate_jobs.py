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


def test_concurrent_poll_and_update_never_loses_job(iso) -> None:  # noqa: F811
    """Yoklama (okuma) ↔ çalıştırıcı yazımı yarışı: Windows'ta açık dosyaya ``os.replace``
    PermissionError verir. İş ne "yok" görünmeli ne de sonuç yazımı düşmeli."""
    import threading

    job_id = "job_" + "0" * 12
    cj._write(cj._path(job_id), {"job_id": job_id, "n": 0})
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            for n in range(1, 301):
                cj._update(job_id, n=n)
        except BaseException as exc:  # yarış hatası teste taşınsın
            errors.append(exc)
        finally:
            stop.set()

    t = threading.Thread(target=writer)
    t.start()
    misses = 0
    while not stop.is_set():
        misses += cj._read(cj._path(job_id)) is None
    t.join()
    assert not errors and misses == 0
    assert (cj._read(cj._path(job_id)) or {}).get("n") == 300


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


def test_stop_does_not_kill_reused_pid(env, monkeypatch) -> None:
    """Kayıtlı PID başka bir sürece geçmişse (başlangıç zamanı tutmuyor) durdurma ona dokunmaz."""
    import os

    j = _start(monkeypatch, _py("import time; time.sleep(30)"))
    real = cj.get_job(j["job_id"])
    killed: list[int] = []
    monkeypatch.setattr(cj, "_kill_tree", lambda pid: killed.append(pid) or [pid])
    # Kaydı, canlı ama İLGİSİZ bir sürece (bu test süreci) işaret edecek şekilde boz.
    cj._update(
        j["job_id"],
        child_pid=os.getpid(),
        child_create_time=1.0,
        runner_pid=os.getpid(),
        runner_create_time=1.0,
    )
    cj.stop_job(j["job_id"], "test")
    assert killed == []
    monkeypatch.undo()  # gerçek süreci temizle
    from app.training.resource_lock import pid_alive

    for key in ("child_pid", "runner_pid"):
        pid = real.get(key)
        if isinstance(pid, int) and pid_alive(pid):
            cj._kill_tree(pid)


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
    started = _wait(j["job_id"], until=("running", "done", "failed", "lost"), timeout=150)
    assert started["status"] == "running", started
    from app.training.candidate_jobs import _kill_tree

    _kill_tree(j["runner_pid"])  # sunucu/çalıştırıcı çöktü (sonuç yazılamadı)
    time.sleep(0.5)
    lost = cj.get_job(j["job_id"])
    assert lost["status"] == "lost" and "TAMAMLANMIŞ" in lost["error"]


def _fake_running_job(job_id: str) -> None:
    cj._write(
        cj._path(job_id),
        {
            "job_id": job_id,
            "kind": "conversion",
            "status": "running",
            "runner_pid": 999_999,
            "runner_create_time": 1.0,
            "log_path": str(cj.jobs_dir() / f"{job_id}.log"),
        },
    )


def test_result_written_during_liveness_check_is_not_lost(iso, monkeypatch) -> None:  # noqa: F811
    """Yarış (2026-10-06 kararsız test): yoklayıcı "running" anlık görüntüsünü okur; çalıştırıcı
    TAM o anda "failed" sonucunu yazıp çıkar; yoklayıcı sonra pid'i ölü görür. Bayat görüntüye
    dayanıp "lost" yazmak gerçek sonucu ezer. Sıra burada zorlanır (uyku/zamanlama yok)."""
    job_id = "job_" + "1" * 12
    _fake_running_job(job_id)
    snapshot = cj._read(cj._path(job_id))
    calls = {"n": 0}

    def alive_then_runner_finishes(pid, create_time=None) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:  # canlılık kontrolü sırasında çalıştırıcı sonucu yazar ve ölür
            cj._update(job_id, status="failed", exit_code=3, error="komut başarısız (çıkış 3)")
        return False

    monkeypatch.setattr(cj, "_alive", alive_then_runner_finishes)
    got = cj.reconcile(snapshot)
    assert got["status"] == "failed" and got["exit_code"] == 3, got
    assert (cj._read(cj._path(job_id)) or {})["status"] == "failed"  # dosyada da ezilmedi


def test_job_lock_treats_delete_pending_permission_error_as_busy(iso, monkeypatch) -> None:  # noqa: F811
    """Windows: silinmekte olan kilit dosyasına ``O_EXCL`` → PermissionError (FileExistsError
    değil). Kaçarsa çalıştırıcı sonucu yazamadan çöker ve bitmiş iş "lost" görünür
    (2026-10-06 kararsız ``test_nonzero_exit_is_failed``). Meşgul sayılıp yeniden denenmeli."""
    import os

    job_id = "job_" + "4" * 12
    cj._write(cj._path(job_id), {"job_id": job_id, "status": "running"})
    real_open = os.open
    calls = {"n": 0}

    def flaky_open(path, flags, *a, **k):
        if str(path).endswith(".lock") and calls["n"] < 3:
            calls["n"] += 1
            raise PermissionError(13, "Erişim engellendi (silinme bekliyor)")
        return real_open(path, flags, *a, **k)

    monkeypatch.setattr(cj.os, "open", flaky_open)
    got = cj._update(job_id, status="failed", exit_code=3)
    assert calls["n"] == 3 and got["status"] == "failed"
    assert (cj._read(cj._path(job_id)) or {})["exit_code"] == 3
    assert not cj._path(job_id).with_suffix(".lock").exists()


def test_read_retries_transient_missing_file_during_replace(iso, monkeypatch) -> None:  # noqa: F811
    """Windows: ``os.replace`` anında okuyucu hedefi kısa süre YOK görebilir; bu "iş yok"
    ya da ``_update``'te alanları silinmiş kayıt demek olmamalı."""
    from pathlib import Path

    job_id = "job_" + "5" * 12
    cj._write(cj._path(job_id), {"job_id": job_id, "n": 1})
    real_read = Path.read_text
    calls = {"n": 0}

    def flaky_read(self, *a, **k):
        if self.name == f"{job_id}.json" and calls["n"] < 2:
            calls["n"] += 1
            raise FileNotFoundError(2, "yeniden adlandırma sürüyor")
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", flaky_read)
    assert cj._read(cj._path(job_id)) == {"job_id": job_id, "n": 1}
    assert cj._read(cj.jobs_dir() / "job_ffffffffffff.json") is None  # gerçekten yok → None


def test_unreadable_fresh_record_never_marks_lost(iso, monkeypatch) -> None:  # noqa: F811
    """Taze kayıt (kilit altında) okunamazsa bayat "running" görüntüsüyle "lost" yazılmaz."""
    job_id = "job_" + "6" * 12
    _fake_running_job(job_id)
    snapshot = cj._read(cj._path(job_id))
    cj._update(job_id, status="failed", exit_code=3)  # çalıştırıcı sonucu yazdı ve çıktı
    monkeypatch.setattr(cj, "_alive", lambda pid, create_time=None: False)
    monkeypatch.setattr(cj, "_read", lambda p: None)
    got = cj.reconcile(snapshot)
    assert got["status"] == "running"  # karar ertelendi (bayat görüntü döner)
    monkeypatch.undo()
    assert (cj._read(cj._path(job_id)) or {})["status"] == "failed"  # dosyada ezilmedi


def test_stop_while_starting_never_runs_command(iso, tmp_path) -> None:  # noqa: F811
    """Durdurma iş "starting" iken gelirse çalıştırıcı komutu HİÇ başlatmaz (arayüz
    "durduruldu" derken iş arka planda koşmaz)."""
    job_id = "job_" + "7" * 12
    marker = tmp_path / "kostu.txt"
    cj._write(
        cj._path(job_id),
        {
            "job_id": job_id,
            "kind": "conversion",
            "status": "stopping",  # stop_job "starting"i yakaladı, çalıştırıcı henüz koşmadı
            "cmd": _py(f"open({str(marker)!r}, 'w').write('x')"),
            "log_path": str(cj.jobs_dir() / f"{job_id}.log"),
        },
    )
    assert cj.run(job_id) == 0
    assert not marker.exists()
    assert (cj._read(cj._path(job_id)) or {})["status"] == "stopped"


def test_stop_never_kills_pid_without_recorded_create_time(iso, monkeypatch) -> None:  # noqa: F811
    """Başlangıç zamanı kayıtlı olmayan PID bu işe ait doğrulanamaz (yeniden kullanılmış
    olabilir) → durdurma ona dokunmaz, "doğrulanamadı" olarak kaydeder."""
    import os

    job_id = "job_" + "8" * 12
    _fake_running_job(job_id)
    cj._update(job_id, runner_pid=os.getpid(), runner_create_time=None, child_pid=os.getpid())
    monkeypatch.setattr(cj, "_kill_tree", lambda pid: pytest.fail("doğrulanmamış PID öldürüldü"))
    got = cj.stop_job(job_id, "test")
    assert got["status"] == "stopped" and got["killed_pids"] == []
    assert got["unverified_pids"] == [os.getpid(), os.getpid()]


def test_activation_refusal_is_job_error_not_500(env, monkeypatch) -> None:
    """E-8a: ayardaki pilot ana model ``resolve_chat_tag``'te ActivationError verir; iş
    başlatma bunu kullanıcıya gösterilen JobError'a çevirir (ham 500 değil)."""
    from app.feedback.model_activation import ActivationError

    def refuse(slot="main", root=None):
        raise ActivationError("Ayardaki ana model — pilot")

    monkeypatch.setattr("app.feedback.model_activation.resolve_chat_tag", refuse)
    with pytest.raises(cj.JobError, match="pilot"):
        _start(monkeypatch, _py("pass"))


def test_dead_runner_without_result_is_still_lost(iso, monkeypatch) -> None:  # noqa: F811
    job_id = "job_" + "2" * 12
    _fake_running_job(job_id)
    monkeypatch.setattr(cj, "_alive", lambda pid, create_time=None: False)
    got = cj.get_job(job_id)
    assert got["status"] == "lost" and "TAMAMLANMIŞ" in got["error"]


def test_stop_after_result_written_does_not_overwrite(iso, monkeypatch) -> None:  # noqa: F811
    """Durdurma "running" görür, o arada çalıştırıcı "done" yazar → sonuç "stopped" olmaz."""
    job_id = "job_" + "3" * 12
    _fake_running_job(job_id)
    monkeypatch.setattr(cj, "_alive", lambda pid, create_time=None: True)
    real_get = cj.get_job
    calls = {"n": 0}

    def get_then_runner_finishes(jid: str) -> dict:
        j = real_get(jid)
        calls["n"] += 1
        if calls["n"] == 1:
            cj._update(jid, status="done", exit_code=0)
        return j

    monkeypatch.setattr(cj, "get_job", get_then_runner_finishes)
    monkeypatch.setattr(cj, "_kill_tree", lambda pid: pytest.fail("bitmiş iş öldürülmemeli"))
    got = cj.stop_job(job_id, "test")
    assert got["status"] == "done" and "zaten bitmiş" in got["note"]


def test_kill_tree_spares_orphan_with_stale_ppid(monkeypatch) -> None:
    """Windows'ta yetim sürecin ppid'i bayat kalır. Başka bir işin ağacındaki torun o pid'i
    yeniden alırsa, ``children(recursive=True)`` (yalnız KÖK zamanına bakar) ilgisiz yetimi
    — ör. ayrık başlatılmış bir aday-iş çalıştırıcısını — torun sayıp öldürür; o iş "lost"
    olur. Ağaç kat kat ve doğrudan ebeveyn zamanıyla yürünmeli."""
    import psutil

    # pid: (ppid, create_time). 100 = durdurulan iş çalıştırıcısı, 200 = onun çocuğu (pid'i
    # yeniden kullanılmış), 300 = başka işin yetim çalıştırıcısı: ppid'i bayat 200, ama 200'ün
    # ŞİMDİKİ sahibinden ÖNCE doğmuş → gerçek torun değil.
    table = {100: (1, 10.0), 200: (100, 30.0), 300: (200, 20.0)}
    terminated: list[int] = []

    class FakeProc:
        def __init__(self, pid: int) -> None:
            if pid not in table:
                raise psutil.NoSuchProcess(pid)
            self.pid = pid

        def create_time(self) -> float:
            return table[self.pid][1]

        def children(self, recursive: bool = False) -> list:
            # psutil semantiği: tek katta çocuk ebeveynden sonra doğmuş olmalı; özyineli
            # taramada ise her torun yalnız KÖKÜN zamanıyla kıyaslanır (açık burada).
            out, stack = [], [self.pid]
            while stack:
                cur = stack.pop()
                ref = self.create_time() if recursive else table[cur][1]
                for c, (pp, ct) in table.items():
                    if pp == cur and ref <= ct:
                        out.append(FakeProc(c))
                        if recursive:
                            stack.append(c)
            return out

        def terminate(self) -> None:
            terminated.append(self.pid)

        def kill(self) -> None:
            terminated.append(self.pid)

    monkeypatch.setattr(psutil, "Process", FakeProc)
    monkeypatch.setattr(psutil, "wait_procs", lambda procs, timeout=None: (procs, []))
    from app.training.resource_lock import process_tree

    assert [p.pid for p in process_tree(100)] == [200, 100]
    cj._kill_tree(100)
    assert sorted(terminated) == [100, 200] and 300 not in terminated


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


def test_t3_unreadable_record_is_not_overwritten_by_update(iso, monkeypatch) -> None:  # noqa: F811
    """Kademe 2 T-3: okunamayan kayıt yalnız yeni alanlarla EZİLMEZ (kimlik/durum kaybolurdu)."""
    job_id = "job_" + "7" * 12
    cj._write(cj._path(job_id), {"job_id": job_id, "status": "running", "kind": "compare"})
    monkeypatch.setattr(cj, "_read", lambda p: None)
    with pytest.raises(RuntimeError, match="okunamadı"):
        cj._update(job_id, child_pid=1)
    monkeypatch.undo()
    assert cj._read(cj._path(job_id)) == {"job_id": job_id, "status": "running", "kind": "compare"}


def test_t5_stale_lock_broken_once_fresh_lock_restored(iso) -> None:  # noqa: F811
    """Kademe 2 T-5: bayat kilit atomik kırılır; taze kilit kırılmaz (sahibine iade)."""
    import os

    p = cj.jobs_dir() / "job_888888888888.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("", "utf-8")
    old = time.time() - 120
    os.utime(p, (old, old))
    assert cj._break_stale_lock(p) and not p.exists()
    p.write_text("", "utf-8")  # taze
    assert not cj._break_stale_lock(p) and p.exists()
    assert not list(p.parent.glob("*.break*"))


def test_t6_kill_tree_does_not_taskkill_vanished_root(monkeypatch) -> None:
    """Kademe 2 T-6: kök o arada bittiyse pid'e körlemesine taskkill /T /F yapılmaz."""
    import psutil

    from app.training import resource_lock

    def gone(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(resource_lock, "process_tree", gone)
    calls = []
    monkeypatch.setattr(cj.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(cj.os, "kill", lambda *a: calls.append(a), raising=False)
    assert cj._kill_tree(999999) == [] and calls == []
