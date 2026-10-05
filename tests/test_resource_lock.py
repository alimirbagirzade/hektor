"""Ortak ağır iş kilidi (eğitim / dönüşüm / karşılaştırma) + sohbet kirasıyla koordinasyon.

Kabul şartları: edinme atomik (eşzamanlı başlatmada tek kazanan — iş parçacığı VE ayrı süreç);
çöken sahibin kilidi bayat sayılır ve kırılır; pid yeniden kullanımı yakalanır; yarım kalan
başlatma TTL ile düşer; yanlış token başkasının kilidini silemez; bayat kilit kırılırken taze
kilit silinmez; web → alt süreç devri; doğrudan CLI eğitimi kilit + sohbet kirası kontrolü yapar.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import typer
from tests.chat_learning_helpers import iso  # noqa: F401

from app.training import resource_lock as rl


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_acquire_is_exclusive_and_release_needs_token(tmp_path: Path) -> None:
    info, why = rl.acquire("training", "a", root=tmp_path)
    assert info is not None and why == ""
    again, why2 = rl.acquire("comparison", "b", root=tmp_path)
    assert again is None and "eğitim sürüyor" in why2
    assert not rl.release("yanlis-token", root=tmp_path)
    assert rl.status(tmp_path)["held"]
    assert rl.release(info["token"], root=tmp_path)
    assert not rl.status(tmp_path)["held"]


def test_concurrent_threads_single_winner(tmp_path: Path) -> None:
    barrier = threading.Barrier(16)
    wins: list[dict] = []

    def worker() -> None:
        barrier.wait()
        info, _ = rl.acquire("training", "t", root=tmp_path)
        if info is not None:
            wins.append(info)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1


def test_concurrent_processes_single_winner(tmp_path: Path) -> None:
    # Ayrı süreçler aynı anda başlar; kilit bu (canlı) test sürecinin pid'ine bağlanır.
    code = (
        "import sys,time;from pathlib import Path;from app.training import resource_lock as rl;"
        "t0=float(sys.argv[2]);time.sleep(max(0,t0-time.time()));"
        "i,_=rl.acquire('training','p',root=Path(sys.argv[1]),pid=int(sys.argv[3]));"
        "print('WIN' if i else 'LOSE')"
    )
    start = time.time() + 2.0
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(tmp_path), str(start), str(os.getpid())],
            stdout=subprocess.PIPE,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        for _ in range(4)
    ]
    outs = [p.communicate(timeout=60)[0].strip() for p in procs]
    assert outs.count("WIN") == 1, outs


def test_crashed_owner_is_stale_and_broken(tmp_path: Path) -> None:
    info, _ = rl.acquire("training", "cokecek", root=tmp_path, pid=_dead_pid())
    assert info is not None
    st = rl.status(tmp_path)
    assert not st["held"] and "yaşamıyor" in st["stale_reason"]
    new, why = rl.acquire("conversion", "yeni", root=tmp_path)
    assert new is not None, why
    assert rl.status(tmp_path)["info"]["owner"] == "yeni"


def test_pid_reuse_is_detected(tmp_path: Path) -> None:
    info, _ = rl.acquire("training", "x", root=tmp_path)
    assert info is not None
    data = json.loads(rl.lock_path(tmp_path).read_text(encoding="utf-8"))
    if data["pid_create_time"] is None:
        pytest.skip("psutil yok — pid başlangıç zamanı ölçülemiyor")
    data["pid_create_time"] -= 1000.0  # aynı pid, başka (eski) süreç
    rl.lock_path(tmp_path).write_text(json.dumps(data), encoding="utf-8")
    assert not rl.status(tmp_path)["held"]


def test_launching_ttl_expires(tmp_path: Path) -> None:
    info, _ = rl.acquire("training", "web", root=tmp_path, state="launching")
    assert info is not None and rl.status(tmp_path)["held"]
    data = json.loads(rl.lock_path(tmp_path).read_text(encoding="utf-8"))
    data["acquired_at"] = time.time() - rl.LAUNCH_TTL_S - 5
    rl.lock_path(tmp_path).write_text(json.dumps(data), encoding="utf-8")
    st = rl.status(tmp_path)
    assert not st["held"] and "TTL" in st["stale_reason"]


def test_breaker_never_deletes_a_fresh_lock(tmp_path: Path) -> None:
    stale, _ = rl.acquire("training", "eski", root=tmp_path, pid=_dead_pid())
    assert stale is not None
    seen = dict(stale)
    # Başka bir süreç bayat kilidi kırıp KENDİ kilidini aldı...
    assert rl._break_stale(tmp_path, seen)
    fresh, _ = rl.acquire("training", "taze", root=tmp_path)
    assert fresh is not None
    # ...geç kalan kırıcı eski (bayat) okumayla gelir: taze kilit silinmemeli.
    assert not rl._break_stale(tmp_path, seen)
    assert rl.status(tmp_path)["info"]["owner"] == "taze"


def test_partial_file_young_held_old_stale(tmp_path: Path) -> None:
    p = rl.lock_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")
    assert rl.status(tmp_path)["held"]  # yazılıyor olabilir
    old = time.time() - 60
    os.utime(p, (old, old))
    assert not rl.status(tmp_path)["held"]
    info, _ = rl.acquire("training", "x", root=tmp_path)
    assert info is not None


def test_transfer_and_claim(tmp_path: Path) -> None:
    info, _ = rl.acquire("training", "web", root=tmp_path, state="launching")
    assert info is not None
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert rl.transfer(info["token"], child.pid, root=tmp_path)
        st = rl.status(tmp_path)
        assert st["held"] and st["info"]["pid"] == child.pid and st["info"]["state"] == "running"
        assert not rl.claim("baska", root=tmp_path)
        assert rl.claim(info["token"], root=tmp_path)
        assert rl.status(tmp_path)["info"]["pid"] == os.getpid()
    finally:
        child.kill()
        child.wait()


def test_hold_inherits_parent_token_and_releases(tmp_path: Path, monkeypatch) -> None:
    parent, _ = rl.acquire("conversion", "ps1", root=tmp_path)
    assert parent is not None
    monkeypatch.setenv(rl.TOKEN_ENV, parent["token"])
    monkeypatch.setattr("app.feedback.resource_guard.chat_lease_blocker", lambda root=None: None)
    with rl.hold("conversion", "merge", root=tmp_path) as tok:
        assert tok == parent["token"]
    assert rl.status(tmp_path)["held"]  # üst sürecin kilidi bırakılmadı
    monkeypatch.delenv(rl.TOKEN_ENV)
    with pytest.raises(rl.HeavyJobBusy), rl.hold("comparison", "v15", root=tmp_path):
        pass
    rl.release(parent["token"], root=tmp_path)
    with rl.hold("comparison", "v15", root=tmp_path):
        assert rl.held_kind(tmp_path) == "comparison"
    assert not rl.status(tmp_path)["held"]


# ── sohbet koordinasyonu ─────────────────────────────────────────────────────


def test_chat_lease_refused_during_conversion_or_comparison(iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.feedback.resource_guard import acquire_chat_lease, active_chat_leases, evaluate

    root = get_settings().root
    for kind in ("conversion", "comparison"):
        info, _ = rl.acquire(kind, "x", root=root)
        assert info is not None
        lease, why = acquire_chat_lease(f"t-{kind}")
        assert lease is None and "sürüyor" in why
        assert active_chat_leases() == []
        rl.release(info["token"], root=root)
    d = evaluate(
        "m",
        activity={"active": False, "starting": False, "heavy": "comparison", "heavy_detail": "x"},
        ps_entries=[],
        ram_available_gb=99.0,
        gpu_mem=None,
        footprint=None,
        margin_gb=1.0,
    )
    assert not d.allowed and "karşılaştırma koşulları" in d.reason


def test_dead_chat_lease_is_ignored_immediately(iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.feedback.resource_guard import active_chat_leases

    d = get_settings().root / "storage" / "chat_leases"
    d.mkdir(parents=True)
    (d / "lease-x.json").write_text(json.dumps({"token": "x", "pid": _dead_pid()}), "utf-8")
    assert active_chat_leases() == []


def test_cli_train_lock_conflicts_and_lease(iso, monkeypatch) -> None:  # noqa: F811
    from app.config import get_settings
    from app.feedback.resource_guard import acquire_chat_lease, release_chat_lease
    from app.main import _acquire_train_lock

    root = get_settings().root
    monkeypatch.delenv(rl.TOKEN_ENV, raising=False)
    # 1) Başka ağır iş varken → çıkış 8, kilit alınmaz.
    other, _ = rl.acquire("conversion", "ps1", root=root)
    holder: dict[str, str] = {}
    with pytest.raises(typer.Exit) as exc:
        _acquire_train_lock("a", holder)
    assert exc.value.exit_code == 8 and not holder
    rl.release(other["token"], root=root)
    # 2) Sohbet cevabı üretilirken → kilit alınır ama çıkış 8 (çağıran finally'de bırakır).
    lease, _ = acquire_chat_lease("uretim")
    assert lease is not None
    with pytest.raises(typer.Exit):
        _acquire_train_lock("a", holder)
    assert holder.get("token") and rl.release(holder["token"], root=root)
    release_chat_lease(lease)
    # 3) Web başlatmasından devralma: token eşleşmezse çıkış 8.
    monkeypatch.setenv(rl.TOKEN_ENV, "yok")
    with pytest.raises(typer.Exit):
        _acquire_train_lock("a", {})
    web, _ = rl.acquire("training", "launch", root=root, state="launching")
    monkeypatch.setenv(rl.TOKEN_ENV, web["token"])
    h2: dict[str, str] = {}
    _acquire_train_lock("a", h2)
    assert h2["token"] == web["token"] and rl.status(root)["info"]["state"] == "running"


def test_web_launch_lock_shared_with_cli(iso) -> None:  # noqa: F811
    from app.config import get_settings
    from app.training import detached_launch as dl

    root = get_settings().root
    info, _ = rl.acquire("training", "cli:x", root=root)
    assert info is not None
    assert not dl._acquire_launch_lock(root)  # terminalden eğitim varken web başlatamaz
    rl.release(info["token"], root=root)
    assert dl._acquire_launch_lock(root)
    from app.feedback.resource_guard import training_activity

    act = training_activity(root)
    assert act["starting"] and not act["active"]
    dl._release_launch_lock(root)
    assert not rl.status(root)["held"]
