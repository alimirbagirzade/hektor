"""⚡ RUN hattı uçtan-uca doğrulama — motor doğurma, onay kapısı, scope, ⛔ DURDUR.

Bu dosyanın ASIL DERDİ: "DURDUR butonu koşan motoru gerçekten kesiyor mu?" sorusu.
Eskiden CEVAP HAYIRDI: `AutoDriver` motoru bloklayan `subprocess.run` ile doğuruyordu,
süreç tutamacı hiçbir yerde saklanmıyordu ve STOP_ALL yalnız bir bayrak DOSYASI yazıyordu.
Kullanıcı DURDUR'a bastığında motor 30 dk zaman aşımına kadar koşmaya (ve abonelik kotası
yakmaya) devam ediyordu — üstelik arayüz "durduruldu" deyip ⚡ kilidini açtığı için ÜSTÜNE
ikinci bir motor doğurulabiliyordu.

Testler ÇEVRİMDIŞI: gerçek `claude` doğurulmaz. "Motor" yerine zararsız, uzun ömürlü bir
Python `sleep` süreci kullanılır — kesilme davranışı için gerçek bir süreç şarttır (sahte
nesne, `subprocess.run`'ın kesilemezliğini asla yakalayamazdı; v5 dersi: stub ≠ runtime).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.orchestration import driver, engine_procs, engines
from app.web import driver_scope

# Uzun ömürlü zararsız süreç — "koşan motor"un yerine geçer.
_SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]


@pytest.fixture(autouse=True)
def _temiz_kayit():
    """Her testten önce/sonra süreç kaydı ve sürücü token deposu temiz olsun."""
    engine_procs.reset()
    driver_scope.reset()
    yield
    engine_procs.reset()
    driver_scope.reset()


def _bekle_olene(proc: subprocess.Popen, timeout: float = 10.0) -> bool:
    son = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < son:
        time.sleep(0.05)
    return proc.poll() is not None


# ── ⛔ DURDUR gerçekten kesiyor mu ────────────────────────────────────────────────


def test_stop_all_kosan_motoru_gercekten_keser(monkeypatch):
    """REGRESYON: STOP_ALL etkinleşince `_default_runner` motoru KESER (bayrak yetmez).

    Eski davranış: `subprocess.run` bloklardı, süreç 30 dk yaşardı. Bu test gerçek bir
    süreç doğurup STOP_ALL'ı etkinleştirir ve sürecin SANİYELER içinde öldüğünü kanıtlar.
    """
    monkeypatch.setattr(driver, "_stop_all_active", lambda: True)
    monkeypatch.setattr(driver, "STOP_POLL_S", 0.05)

    basla = time.monotonic()
    rc, _out = driver._default_runner(_SLEEPER, timeout=60, run_id="test-stop")
    gecen = time.monotonic() - basla

    assert rc == driver.STOPPED_RC, "kesilen süreç STOPPED_RC döndürmeli"
    # 60 sn uyuyan süreç saniyeler içinde ölmeli — zaman aşımını BEKLEMEMELİ.
    assert gecen < 20, f"DURDUR {gecen:.1f}s sürdü — süreç kesilmiyor, zaman aşımı bekleniyor"
    assert engine_procs.live_count() == 0, "kayıt sızdırdı"


def test_stop_all_yokken_surec_normal_biter(monkeypatch):
    """STOP_ALL etkin değilken süreç normal tamamlanır (yoklama yanlış-kesme yapmaz)."""
    monkeypatch.setattr(driver, "_stop_all_active", lambda: False)
    monkeypatch.setattr(driver, "STOP_POLL_S", 0.05)

    rc, out = driver._default_runner(
        [sys.executable, "-c", "print('MERHABA')"], timeout=30, run_id="test-normal"
    )
    assert rc == 0
    assert "MERHABA" in out
    assert engine_procs.live_count() == 0


def test_cikti_yoklama_donguve_kaybolmaz(monkeypatch):
    """Yoklama döngüsü çıktıyı KAYBETMEZ (communicate timeout'u yeniden çağrılabilir)."""
    monkeypatch.setattr(driver, "_stop_all_active", lambda: False)
    monkeypatch.setattr(driver, "STOP_POLL_S", 0.02)

    kod = "import time,sys; time.sleep(0.3); print('GEC_CIKTI'); sys.stderr.write('HATA_AKISI')"
    rc, out = driver._default_runner([sys.executable, "-c", kod], timeout=30, run_id="test-cikti")
    assert rc == 0
    assert "GEC_CIKTI" in out, "stdout kayboldu"
    assert "HATA_AKISI" in out, "stderr kayboldu"


def test_terminate_all_tum_canli_motorlari_keser():
    """`terminate_all` birden çok koşudaki motorları da keser (⛔ DURDUR yolu)."""
    procs = []
    try:
        for i in range(3):
            p = subprocess.Popen(_SLEEPER, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            procs.append(p)
            engine_procs.register(f"kosu-{i}", p)
        assert engine_procs.live_count() == 3

        kesilen = engine_procs.terminate_all()
        assert kesilen == 3
        for p in procs:
            assert _bekle_olene(p), "süreç kesilmedi"
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()


def test_terminate_run_yalniz_o_kosuyu_keser():
    """`terminate_run` yalnız hedef koşuyu keser — diğer koşu etkilenmez."""
    a = subprocess.Popen(_SLEEPER, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    b = subprocess.Popen(_SLEEPER, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        engine_procs.register("kosu-a", a)
        engine_procs.register("kosu-b", b)

        assert engine_procs.terminate_run("kosu-a") == 1
        assert _bekle_olene(a), "hedef koşu kesilmedi"
        assert b.poll() is None, "ilgisiz koşu da kesildi"
    finally:
        for p in (a, b):
            if p.poll() is None:
                p.kill()


def test_stop_all_ucu_canli_motorlari_keser():
    """POST /api/supervisor/stop-all bayrağı yazar VE canlı motorları keser."""
    from app.web.server import app

    proc = subprocess.Popen(_SLEEPER, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    engine_procs.register("web-kosu", proc)
    try:
        with TestClient(app) as client:
            r = client.post("/api/supervisor/stop-all", params={"reason": "test"})
        assert r.status_code == 200
        # Sözleşme: kaç motorun kesildiği AÇIKÇA raporlanır (sessiz "durduruldu" yok).
        assert r.json().get("engines_terminated") == 1, r.json()
        assert _bekle_olene(proc), "stop-all ucu motoru kesmedi"
    finally:
        if proc.poll() is None:
            proc.kill()
        # Kill-switch'i bırakma — sonraki testler etkilenmesin.
        from app.agents.runtime import supervisor

        supervisor.clear_stop_all()


def test_drive_durdurulunca_stopped_raporlar(monkeypatch, tmp_path):
    """Kesilen koşu 'av FAIL' değil, AÇIKÇA 'stopped' olarak raporlanır."""

    def sahte_runner(command, timeout, env=None):
        return driver.STOPPED_RC, ""

    d = driver.AutoDriver()
    run_id = d.orch.store.create_run(
        model="qwen2.5-1.5b",
        profile="discipline_safe",
        adapter_name="achilles_lora",
        params={"iters": 1},
    )
    # deep-hunt'a kadar ilerlet.
    d.orch.run_until_blocked(run_id)
    snap = d.orch.status(run_id).get("run") or {}
    if snap.get("current_stage") != "deep-hunt":
        pytest.skip(f"koşu deep-hunt'ta değil (current={snap.get('current_stage')})")

    res = d.drive(run_id, execute=True, runner=sahte_runner)
    assert res.get("stopped") is True, res
    assert res.get("hunt_passed") is False
    assert "DURDUR" in res.get("reason", "")


# ── Motor binary'si ele geçirilemez (CWD taklitçisi) ─────────────────────────────


def test_motor_mutlak_yola_cozulur(monkeypatch, tmp_path):
    """argv[0] MUTLAK yola çözülür — çalışma dizinindeki taklitçi binary koşmaz.

    REGRESYON (Kademe-2, 3/3 onay): motor `argv[0]='claude'` ile mutlak yol olmadan
    doğuruluyordu. Windows'ta arama yolu ÖNCE çalışma dizinine bakar → cwd'ye bırakılan
    sahte bir `claude.exe` gerçek CLI yerine koşar ve son satıra `PASS` yazarak ZORUNLU
    derin av kapısını düşürürdü (Kural 8).
    """
    # Binary adı platforma göre değişir: Windows'ta `.exe`, POSIX'te çalıştırma biti.
    ad = "claude.exe" if os.name == "nt" else "claude"
    sahte = tmp_path / ad
    sahte.write_text("taklitci")
    gercek = tmp_path / "gercek" / ad
    gercek.parent.mkdir()
    gercek.write_text("gercek")
    if os.name != "nt":  # `shutil.which` POSIX'te çalıştırma bitine bakar
        sahte.chmod(0o755)
        gercek.chmod(0o755)

    monkeypatch.chdir(tmp_path)  # cwd'de taklitçi var
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{gercek.parent}")

    cozulmus = driver._resolve_executable(["claude", "-p", "merhaba"])
    assert cozulmus[0] != "claude", "argv[0] mutlak yola çözülmedi"
    assert Path(cozulmus[0]).parent.resolve() == gercek.parent.resolve(), (
        f"cwd'deki taklitçi seçildi: {cozulmus[0]}"
    )
    assert cozulmus[1:] == ["-p", "merhaba"], "argv kuyruğu bozuldu"


def test_cozulemeyen_motor_komutu_oldugu_gibi_birakir(monkeypatch, tmp_path):
    """PATH'te yoksa komut değişmez → Popen temiz FileNotFoundError verir (sessiz düşüş yok)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "bos"))
    assert driver._resolve_executable(["yokboyle_motor", "-p"]) == ["yokboyle_motor", "-p"]


def test_motor_depo_kokunde_kosar(monkeypatch):
    """Motor DAİMA depo kökünde doğar — av yanlış ağacı tarayıp 'temiz' diyemez."""
    yakalanan: dict[str, object] = {}

    class SahteProc:
        returncode = 0

        def poll(self):
            return 0

        def communicate(self, timeout=None):
            return ("ACHILLES_HUNT_VERDICT: PASS", "")

    def sahte_popen(command, **kwargs):
        yakalanan.update(kwargs)
        return SahteProc()

    monkeypatch.setattr(driver.subprocess, "Popen", sahte_popen)
    monkeypatch.setattr(driver, "_stop_all_active", lambda: False)
    driver._default_runner(["echo", "x"], timeout=5, run_id="cwd-testi")

    assert yakalanan.get("cwd") == str(driver._REPO_ROOT), (
        f"cwd depo köküne pinlenmedi: {yakalanan.get('cwd')!r}"
    )


# ── Motor kurulu değilken temiz hata ─────────────────────────────────────────────


def test_kurulu_olmayan_motor_temiz_503(monkeypatch):
    """execute=true + motor PATH'te yok → 503 + Türkçe sebep (sessiz başarısızlık YOK)."""
    from app.web.server import app

    monkeypatch.setattr(engines, "available", lambda name, **kw: False)
    with TestClient(app) as client:
        # Önce bir koşu yarat (404 yerine 503 alalım).
        r0 = client.post(
            "/api/orchestration/start", json={"adapter_name": "achilles_lora", "iters": 1}
        )
        assert r0.status_code == 200, r0.text
        run_id = r0.json()["run_id"]

        r = client.post(f"/api/orchestration/autodrive/{run_id}", json={"execute": True})
    assert r.status_code == 503, r.text
    detay = r.json()["detail"]
    assert "PATH" in detay and "kurulu değil" in detay, detay


def test_bilinmeyen_motor_400():
    """Bilinmeyen motor adı sessizce varsayılana DÜŞMEZ — 400."""
    from app.web.server import app

    with TestClient(app) as client:
        r0 = client.post(
            "/api/orchestration/start", json={"adapter_name": "achilles_lora", "iters": 1}
        )
        run_id = r0.json()["run_id"]
        r = client.post(
            f"/api/orchestration/autodrive/{run_id}",
            json={"execute": True, "engine": "yokboyle"},
        )
    assert r.status_code == 400, r.text


# ── Sürücü scope izolasyonu (Kural 8) ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("metot", "yol", "govde"),
    [
        ("POST", "/api/supervisor/clear-stop-all", None),
        ("POST", "/api/orchestration/autodrive/herhangi", {"execute": True}),
    ],
)
def test_surucu_scope_insan_yalniz_uclarda_403(metot, yol, govde):
    """Sürücü kimliği insan-yalnız uçlarda 403 alır — motor kendi frenini çözemez."""
    from app.web.server import app

    token = driver_scope.mint("kosu-x")
    basliklar = {
        driver_scope.DRIVER_TOKEN_HEADER: token,
        driver_scope.RUN_ID_HEADER: "kosu-x",
    }
    with TestClient(app) as client:
        r = client.request(metot, yol, headers=basliklar, json=govde)
    assert r.status_code == 403, f"{metot} {yol} → {r.status_code}: {r.text}"


def test_surucu_token_kosu_bitince_iptal():
    """Koşu bitince sürücü token'ı geçersizleşir (TTL beklenmez)."""
    token = driver_scope.mint("kosu-y")
    assert driver_scope.verify(token, run_id="kosu-y") == "kosu-y"
    driver_scope.revoke_run("kosu-y")
    assert driver_scope.verify(token, run_id="kosu-y") is None


def test_surucu_token_baska_kosuda_gecersiz():
    """Token bağlı olduğu koşunun DIŞINDA kullanılamaz."""
    token = driver_scope.mint("kosu-a")
    assert driver_scope.verify(token, run_id="kosu-b") is None


# ── RUN sözleşme duman testi ─────────────────────────────────────────────────────


def test_run_pipeline_smoke_gecer():
    """`orchestrate-smoke` RUN bölümü yeşil olmalı (sözleşmeler yürürlükte)."""
    from app.orchestration.run_smoke import RunPipelineSmoke

    res = RunPipelineSmoke().run()
    basarisiz = [c.to_dict() for c in res.checks if c.status == "fail"]
    assert not basarisiz, f"RUN sözleşmesi ihlalleri: {basarisiz}"
    assert res.verdict == "pass"


def test_run_smoke_gercek_motor_dogurmaz():
    """Duman testi gerçek motor spawn'ı YAPMAZ ve bunu dürüstçe 'skip' der (kota yakmaz)."""
    from app.orchestration.run_smoke import RunPipelineSmoke

    res = RunPipelineSmoke().run()
    canli = next(c for c in res.checks if c.name == "live-spawn")
    assert canli.status == "skip"
    assert "KANITLANMAZ" in canli.detail


def test_drive_mode_wiring_probe_pass():
    """SÜR MODU artık bağlı → dürüstlük yoklaması 'warn' değil 'pass' vermeli (P7)."""
    from app.orchestration.run_smoke import RunPipelineSmoke

    res = RunPipelineSmoke().run()
    wiring = next(c for c in res.checks if c.name == "drive-mode-wiring")
    assert wiring.status == "pass", wiring.detail


# ── ⚡ RUN ucu VARSAYILAN olarak SÜR modunu doğuruyor ─────────────────────────────


def test_autodrive_varsayilan_sur_modu_dry_run():
    """POST /autodrive execute=false → SÜR komutu (MCP'li) döner, av değil."""
    from app.web.server import app

    with TestClient(app) as client:
        run_id = client.post(
            "/api/orchestration/start", json={"adapter_name": "achilles_lora", "iters": 1}
        ).json()["run_id"]
        r = client.post(f"/api/orchestration/autodrive/{run_id}", json={"execute": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "drive"
    assert "--mcp-config" in body["command"] and "--safe-mode" not in body["command"]


# ── ELLE tetiklenen canlı sür adımı VARSAYILAN KAPALI (kota güvenliği) ────────────


def test_live_drive_varsayilan_kapali_spawn_yok(monkeypatch):
    """--allow-live-spawn OLMADAN hiçbir süreç doğmaz (kota yakılmaz)."""
    from app.orchestration.run_smoke import LiveDriveSmoke

    # Yanlışlıkla spawn edilirse test PATLASIN diye Popen'i tuzağa düşür.
    monkeypatch.setattr(
        driver.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("kapalıyken spawn edildi!")),
    )
    res = LiveDriveSmoke().run(allow_live_spawn=False)
    assert res.verdict == "skip"
    assert res.checks[0].name == "live-drive" and res.checks[0].status == "skip"


def test_live_drive_web_yoksa_skip(monkeypatch):
    """Bayrak açık ama web kapalı → anlamlı skip, gerçek spawn YOK."""
    from app.orchestration.run_smoke import LiveDriveSmoke

    monkeypatch.setattr(engines, "available", lambda name, **kw: True)  # kurulu varsay
    monkeypatch.setattr(
        driver.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("web yokken spawn edildi!")),
    )
    res = LiveDriveSmoke().run(allow_live_spawn=True, web_check=lambda: False)
    assert res.verdict == "skip"
    assert "Web sunucusu" in res.summary


def test_live_drive_enjekte_runner_ile_mcp_gorunurlugu(monkeypatch):
    """Bayrak + web + (sahte) motor MCP araçlarını görürse → pass (gerçek spawn YOK)."""
    from app.orchestration.run_smoke import LiveDriveSmoke

    monkeypatch.setattr(engines, "available", lambda name, **kw: True)

    def sahte_runner(command, timeout, env=None):
        # Sür argv'si + MCP config gerçekten kuruluyor mu?
        assert "--mcp-config" in command and "--safe-mode" not in command
        return 0, "araçlar: mcp__achilles__status\nACHILLES_DRIVE_VERDICT: PASS\n"

    res = LiveDriveSmoke().run(allow_live_spawn=True, web_check=lambda: True, runner=sahte_runner)
    assert res.verdict == "pass", [c.to_dict() for c in res.checks]
