"""run_smoke.py — ⚡ RUN hattının uçtan-uca SÖZLEŞME duman testi.

`smoke.py` CANLI RUNTIME'ı yoklar (Ollama erişilebilir mi, üretim boş mu). Bu modül farklı
bir soruyu sorar: **RUN hattının güvenlik sözleşmeleri gerçekten yürürlükte mi?** — yani
kullanıcı ⚡'ye bastığında ne doğuyor, nerede duruyor, kim neyi onaylayabiliyor, DURDUR
gerçekten kesiyor mu.

⛔ NEDEN GERÇEK MOTOR DOĞURULMAZ: PR#122 doğrulaması sırasında odaklı bir onay butonu tek
Enter'la **5 gerçek `claude -p` süreci** doğurdu ve abonelik kotası yandı. Bir duman testi
asla kota yakmamalıdır. Bu yüzden yoklamalar KOMUT KURULUMUNU, kapıları ve süreç
sonlandırmayı doğrular; gerçek motor spawn'ı bilinçli olarak `skip`'tir ve raporda
"kanıtlanmadı" diye AÇIKÇA söylenir (Kural 7 — uydurma yok, Kural 2 — test edilmeden
"çalışıyor" denmez).

Tüm yoklamalar ÇEVRİMDIŞI ve yan etkisizdir: gerçek motor doğurulmaz, eğitim başlatılmaz,
depoya yazılmaz. Tek gerçek süreç, DURDUR yoklamasında doğurulan zararsız `sleep`
sürecidir (kendi ürettiğimiz, kendi kestiğimiz).
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class RunCheck:
    """Tek bir RUN sözleşmesi yoklaması."""

    name: str
    status: str  # "pass" | "fail" | "skip" | "warn"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class RunSmokeResult:
    """RUN hattı duman testinin bütünsel sonucu."""

    verdict: str  # "pass" | "fail"
    summary: str
    checks: list[RunCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }


class RunPipelineSmoke:
    """⚡ RUN hattının sözleşmelerini uçtan-uca yoklar (çevrimdışı, kota yakmaz)."""

    def run(self) -> RunSmokeResult:
        checks: list[RunCheck] = []
        for probe in (
            self._probe_engine_registry,
            self._probe_hardening_flags,
            self._probe_mcp_config,
            self._probe_missing_engine,
            self._probe_dry_run_default,
            self._probe_training_gate,
            self._probe_driver_scope_403,
            self._probe_stop_kills_engine,
            self._probe_drive_mode_wiring,
            self._probe_live_spawn,
        ):
            try:
                checks.append(probe())
            except Exception as exc:  # yoklamanın kendisi patlarsa KUSUR say (sessizlik yok)
                log.exception("RUN duman yoklaması patladı: %s", probe.__name__)
                checks.append(
                    RunCheck(
                        probe.__name__.removeprefix("_probe_"),
                        "fail",
                        f"Yoklama patladı: {type(exc).__name__}: {exc}",
                    )
                )
        failed = [c for c in checks if c.status == "fail"]
        if failed:
            adlar = ", ".join(c.name for c in failed)
            return RunSmokeResult("fail", f"RUN sözleşmesi İHLAL: {adlar}", checks)
        return RunSmokeResult("pass", "RUN hattı sözleşmeleri yürürlükte.", checks)

    # ── 1) Motor kayıt tablosu tutarlı mı ────────────────────────────────────────
    def _probe_engine_registry(self) -> RunCheck:
        from app.orchestration import engines

        eng = engines.get_engine(engines.DEFAULT_ENGINE)
        if not eng.hardened or not eng.spawns:
            return RunCheck(
                "engine-registry",
                "fail",
                f"Varsayılan motor '{eng.name}' hardened={eng.hardened} spawns={eng.spawns} — "
                "AutoDriver yalnız sertleştirilmiş+spawn eden motor doğurmalı.",
            )
        # Sertleştirilmemiş motorlar RUN için KAPALI olmalı (fail-closed).
        sizinti = [
            e.name
            for e in (engines.get_engine(n) for n in engines.engine_names())
            if e.spawns and not e.hardened and not engines.run_blocked_reason(e.name)
        ]
        if sizinti:
            return RunCheck(
                "engine-registry",
                "fail",
                f"Sertleştirilmemiş motor(lar) RUN'da seçilebilir: {sizinti}",
            )
        return RunCheck(
            "engine-registry",
            "pass",
            f"Varsayılan '{eng.name}' sertleştirilmiş; kısıtsız motorlar RUN'a kapalı.",
        )

    # ── 2) Sertleştirme bayrakları argv'de gerçekten var mı ──────────────────────
    def _probe_hardening_flags(self) -> RunCheck:
        from app.orchestration import driver, engines

        run = {"adapter_name": "smoke_probe"}
        av = driver.build_hunt_command(run, engines.DEFAULT_ENGINE)
        eksik_av = [
            f for f in ("--safe-mode", "--strict-mcp-config", "--disallowedTools") if f not in av
        ]
        if eksik_av:
            return RunCheck("hardening-flags", "fail", f"Av modunda eksik bayrak: {eksik_av}")

        sur = driver.build_drive_command(run, "C:/tmp/mcp.json", engines.DEFAULT_ENGINE)
        eksik_sur = [
            f
            for f in (
                "--setting-sources",
                "--disable-slash-commands",
                "--strict-mcp-config",
                "--tools",
            )
            if f not in sur
        ]
        if eksik_sur:
            return RunCheck("hardening-flags", "fail", f"Sür modunda eksik bayrak: {eksik_sur}")
        # `--safe-mode` sür modunda OLMAMALI (MCP'yi de kapatır → ajan araçsız kalır).
        if "--safe-mode" in sur:
            return RunCheck(
                "hardening-flags", "fail", "Sür modunda --safe-mode var — MCP'yi kapatır."
            )
        # VARIADIC bayrak en sonda tek argümanla durmalı (sonraki bayrağı yutmasın).
        if sur[-2] != "--mcp-config":
            return RunCheck(
                "hardening-flags",
                "fail",
                f"--mcp-config argv'nin sonunda değil (son iki öğe: {sur[-2:]}) — "
                "variadic yutma riski.",
            )
        return RunCheck(
            "hardening-flags", "pass", "Av + sür modu sertleştirme bayrakları argv'de eksiksiz."
        )

    # ── 3) MCP config kendine yeter + SIR içermez ────────────────────────────────
    def _probe_mcp_config(self) -> RunCheck:
        from app.orchestration import driver

        cfg = driver.build_mcp_config()
        servers = cfg.get("mcpServers") or {}
        if "achilles" not in servers:
            return RunCheck(
                "mcp-config", "fail", f"MCP config'te 'achilles' sunucusu yok: {list(servers)}"
            )
        blob = repr(cfg).lower()
        for sir in ("token", "api_key", "apikey", "password", "secret", "authorization"):
            if sir in blob:
                return RunCheck("mcp-config", "fail", f"MCP config'te sır benzeri alan: {sir!r}")
        return RunCheck("mcp-config", "pass", "MCP config kendine yeter ve sır taşımıyor.")

    # ── 4) Motor kurulu değilken TEMİZ hata (sessiz başarısızlık yok) ────────────
    def _probe_missing_engine(self) -> RunCheck:
        from app.orchestration import engines

        sebep = engines.run_blocked_reason(engines.DEFAULT_ENGINE, which=lambda _b: None)
        if not sebep:
            return RunCheck(
                "missing-engine",
                "fail",
                "Motor PATH'te yokken run_blocked_reason BOŞ döndü — sessiz başarısızlık.",
            )
        if "PATH" not in sebep:
            return RunCheck("missing-engine", "warn", f"Engel sebebi belirsiz: {sebep!r}")
        return RunCheck("missing-engine", "pass", f"Kurulu değilken temiz sebep: {sebep!r}")

    # ── 5) execute=false varsayılanı gerçekten DRY-RUN mu (spawn yok) ────────────
    def _probe_dry_run_default(self) -> RunCheck:
        import inspect

        from app.orchestration.driver import AutoDriver

        sig = inspect.signature(AutoDriver.drive)
        varsayilan = sig.parameters["execute"].default
        if varsayilan is not False:
            return RunCheck(
                "dry-run-default",
                "fail",
                f"AutoDriver.drive(execute=) varsayılanı {varsayilan!r} — "
                "spawn varsayılan OLMAMALI.",
            )
        return RunCheck(
            "dry-run-default", "pass", "drive(execute=False) varsayılan — spawn opt-in."
        )

    # ── 6) Eğitim adımı TAZE İNSAN ONAYI olmadan ilerlemiyor (Kural 8) ───────────
    def _probe_training_gate(self) -> RunCheck:
        from app.web import security

        gated = getattr(security, "require_human", None)
        if gated is None:
            return RunCheck("training-gate", "fail", "require_human kapısı bulunamadı.")
        # Sürücü kimliği taşıyan bir istek insan-yalnız kapıda 403 almalı.
        from app.web import driver_scope

        driver_scope.reset()
        token = driver_scope.mint("smoke-run")
        try:
            req = _FakeRequest({driver_scope.DRIVER_TOKEN_HEADER: token})
            try:
                gated(req)
            except Exception as exc:
                kod = getattr(exc, "status_code", None)
                if kod == 403:
                    return RunCheck(
                        "training-gate", "pass", "Sürücü kimliği insan-yalnız kapıda 403 alıyor."
                    )
                return RunCheck("training-gate", "fail", f"Beklenen 403 değil: {exc!r}")
            return RunCheck(
                "training-gate",
                "fail",
                "Sürücü kimliği insan-yalnız kapıdan GEÇTİ (Kural 8 ihlali).",
            )
        finally:
            driver_scope.reset()

    # ── 7) Sürücü scope'u onay/stop-all uçlarında 403 alıyor mu (gerçek HTTP) ────
    def _probe_driver_scope_403(self) -> RunCheck:
        try:
            from fastapi.testclient import TestClient

            from app.web import driver_scope
            from app.web.server import app
        except Exception as exc:
            return RunCheck("driver-scope-403", "skip", f"Web uygulaması yüklenemedi: {exc}")

        driver_scope.reset()
        token = driver_scope.mint("smoke-run")
        basliklar = {
            driver_scope.DRIVER_TOKEN_HEADER: token,
            driver_scope.RUN_ID_HEADER: "smoke-run",
        }
        insan_yalniz = [
            ("POST", "/api/supervisor/clear-stop-all", None),
            ("POST", "/api/orchestration/autodrive/smoke-run", {"execute": True}),
        ]
        try:
            with TestClient(app) as client:
                sizan = []
                for metot, yol, govde in insan_yalniz:
                    r = client.request(metot, yol, headers=basliklar, json=govde)
                    if r.status_code != 403:
                        sizan.append(f"{metot} {yol} → {r.status_code}")
                if sizan:
                    return RunCheck(
                        "driver-scope-403",
                        "fail",
                        f"Sürücü kimliği insan-yalnız uçtan 403 ALMADI: {sizan}",
                    )
        finally:
            driver_scope.reset()
        return RunCheck(
            "driver-scope-403",
            "pass",
            "Sürücü kimliği clear-stop-all ve autodrive uçlarında 403 alıyor.",
        )

    # ── 8) ⛔ DURDUR koşan motoru GERÇEKTEN kesiyor mu ───────────────────────────
    def _probe_stop_kills_engine(self) -> RunCheck:
        from app.orchestration import engine_procs

        # Zararsız, uzun ömürlü bir süreç doğur (gerçek motor DEĞİL — kota yakmaz).
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        engine_procs.register("smoke-stop", proc)
        try:
            if proc.poll() is not None:
                return RunCheck(
                    "stop-kills-engine", "warn", "Yoklama süreci beklenmedik şekilde bitti."
                )
            kesilen = engine_procs.terminate_all()
            son = time.monotonic() + 10.0
            while proc.poll() is None and time.monotonic() < son:
                time.sleep(0.05)
            if proc.poll() is None:
                return RunCheck(
                    "stop-kills-engine", "fail", "DURDUR sonrası süreç HÂLÂ yaşıyor — kesilemedi."
                )
            if kesilen < 1:
                return RunCheck(
                    "stop-kills-engine",
                    "fail",
                    f"terminate_all() {kesilen} süreç kesti (≥1 bekleniyordu).",
                )
            return RunCheck(
                "stop-kills-engine", "pass", f"DURDUR canlı motor sürecini kesti ({kesilen} süreç)."
            )
        finally:
            if proc.poll() is None:  # pragma: no cover - temizlik güvencesi
                with __import__("contextlib").suppress(Exception):
                    proc.kill()
            engine_procs.unregister("smoke-stop", proc)

    # ── 9) "Sür" (drive) modu gerçekten bağlı mı — DÜRÜSTLÜK YOKLAMASI ───────────
    def _probe_drive_mode_wiring(self) -> RunCheck:
        """⚡ RUN gerçekten MCP'li "sür" motoru mu doğuruyor, yoksa MCP'siz av motoru mu?

        Bu ayrım kritiktir: av modu `--safe-mode` ile doğar ve o bayrak MCP sunucularını
        da KAPATIR. Yani av motoru Achilles MCP araçlarını GÖREMEZ — veri hattını
        ilerletemez, yalnız rapor üretir. "RUN → ajanlar sürülüyor" iddiası ancak (a)
        AutoDriver'ın sür yolu `build_drive_command`'i çağırıyor VE (b) ⚡ RUN ucu
        varsayılan olarak sür moduna gidiyorsa doğrudur (P7).
        """
        import inspect

        from app.orchestration import driver
        from app.web.orchestration_routes import OrchestrationAutodriveRequest

        kaynak = inspect.getsource(driver.AutoDriver)
        sur_yolu_var = "build_drive_command" in kaynak and hasattr(
            driver.AutoDriver, "_drive_pipeline"
        )
        try:
            uc_varsayilani = OrchestrationAutodriveRequest().mode
        except Exception:  # pragma: no cover - model kurulamıyorsa yoklama patlar
            uc_varsayilani = "?"
        if sur_yolu_var and uc_varsayilani == "drive":
            return RunCheck(
                "drive-mode-wiring",
                "pass",
                "AutoDriver sür (MCP'li) modunu doğuruyor ve ⚡ RUN ucu varsayılan 'drive'.",
            )
        return RunCheck(
            "drive-mode-wiring",
            "warn",
            (
                "SÜR MODU TAM BAĞLI DEĞİL "
                f"(sür_yolu={sur_yolu_var}, uç_varsayılan={uc_varsayilani!r}). "
                "Av motoru `--safe-mode` ile başlar ve bu bayrak MCP'yi de KAPATIR → motor "
                "Achilles MCP araçlarını GÖRMEZ ve veri hattını İLERLETEMEZ."
            ),
        )

    # ── 10) Gerçek motor spawn'ı — BİLİNÇLİ OLARAK yoklanmaz ─────────────────────
    def _probe_live_spawn(self) -> RunCheck:
        from app.orchestration import engines

        kurulu = engines.available(engines.DEFAULT_ENGINE)
        return RunCheck(
            "live-spawn",
            "skip",
            (
                f"Gerçek motor spawn'ı YOKLANMADI (motor kurulu={kurulu}). Duman testi kota "
                "yakmaz: PR#122 doğrulamasında kazara 5 `claude -p` süreci doğmuştu. "
                "MCP araçlarının canlı görünürlüğü bu komutla KANITLANMAZ — elle, kapılı "
                "`achilles orchestrate-drive-live --allow-live-spawn` ile denetimli koşulur."
            ),
        )


# ── ELLE TETİKLENEN TEK canlı sür duman adımı (varsayılan KAPALI — kota yakar) ──────────
# ⚠️ Bu, otomatik duman testinin PARÇASI DEĞİLDİR ve CI'da ASLA koşmaz. Kullanıcı bir kez
# elle çalıştırıp "sür motoru gerçekten MCP araçlarını görüyor mu" kanıtını görür. Gerçek
# `claude -p` doğurur → abonelik kotası yakar; bu yüzden açık `--allow-live-spawn` şarttır.
_LIVE_PROBE_PROMPT = (
    "Sen bir Achilles SÜR-MODU DUMAN TEST ajanısın. YALNIZ şunları yap, BAŞKA HİÇBİR ŞEY: "
    "(1) sana MCP üzerinden sunulan Achilles araçlarını (adları `mcp__` ile başlar) say ve "
    "adlarını yaz; (2) salt-okuma `mcp__achilles__*status*` benzeri bir durum aracını BİR kez "
    "çağırıp döndüğünü doğrula. EĞİTİM BAŞLATMA, ONAY VERME, DOSYAYA YAZMA. Çıktının SON "
    "SATIRI tam olarak şu olmalı:\n"
    "ACHILLES_DRIVE_VERDICT: PASS   (en az bir mcp__ aracı gördüysen)\n"
    "ACHILLES_DRIVE_VERDICT: FAIL   (hiç MCP aracı görünmüyorsa)"
)


class LiveDriveSmoke:
    """Kapılı, elle tetiklenen canlı sür duman adımı — TEK motor doğurur, sonra biter.

    Kanıtladığı: (a) sür (drive) argv'siyle doğan motor Achilles MCP araçlarını GERÇEKTEN
    görüyor (çıktıda `mcp__` + ``ACHILLES_DRIVE_VERDICT: PASS``); (b) motor süreci
    ``engine_procs``'a KAYITLI doğar → ⛔ DURDUR onu kesebilir (kesme mekanizması otomatik
    smoke'un ``stop-kills-engine`` yoklamasıyla ayrıca kanıtlanır). ``runner``/``web_check``
    enjekte edilebilir → gerçek spawn olmadan test edilebilir.
    """

    def run(
        self,
        *,
        allow_live_spawn: bool = False,
        runner: Any = None,
        web_check: Any = None,
        timeout_s: int = 240,
    ) -> RunSmokeResult:
        checks: list[RunCheck] = []
        if not allow_live_spawn:
            checks.append(
                RunCheck(
                    "live-drive",
                    "skip",
                    "Kapılı: gerçek motor doğurmak (kota yakar) için --allow-live-spawn ver.",
                )
            )
            return RunSmokeResult("skip", "Canlı sür adımı KAPALI (varsayılan).", checks)

        from app.orchestration import driver, engines
        from app.orchestration.orchestrator import TrainingOrchestrator
        from app.web import driver_scope

        eng = engines.DEFAULT_ENGINE
        # 1) Motor kurulu mu (gerçek spawn öncesi temiz kapı).
        if runner is None and not engines.available(eng):
            checks.append(
                RunCheck(
                    "live-drive", "fail", f"`{eng}` CLI PATH'te yok — kurulu değil, spawn yok."
                )
            )
            return RunSmokeResult("fail", "Motor kurulu değil.", checks)

        # 2) Web sunucusu erişilebilir mi? MCP proxy'si ona bağlanır; kapalıysa MCP çağrıları
        #    başarısız olur → yanıltıcı FAIL yerine anlamlı skip.
        reachable = web_check() if web_check is not None else self._web_reachable()
        if not reachable:
            checks.append(
                RunCheck(
                    "live-drive",
                    "skip",
                    "Web sunucusu (ACHILLES_WEB_URL) erişilemedi — önce `uv run achilles-web` "
                    "başlat; MCP araçları çalışan web'e proxy'lenir.",
                )
            )
            return RunSmokeResult("skip", "Web sunucusu kapalı — canlı adım atlandı.", checks)

        # 3) Tek koşuluk throwaway run + MCP config + sür argv'si (özel MİNİMAL probe promptu).
        orch = TrainingOrchestrator()
        run_id = orch.store.create_run(
            model="live-drive-smoke",
            profile="discipline_safe",
            adapter_name="live_smoke",
            params={},
        )
        mcp_path = driver.drive_mcp_config_path(run_id)
        driver.write_mcp_config(mcp_path)
        command = engines.build_drive_command(eng, _LIVE_PROBE_PROMPT, mcp_path)

        run_fn = runner or __import__("functools").partial(driver._default_runner, run_id=run_id)
        token = driver_scope.mint(run_id, ttl_s=driver.DRIVE_TOKEN_TTL_S)
        try:
            _rc, output = run_fn(command, timeout_s, driver.build_child_env(token, run_id))
        except Exception as exc:  # canlı adım; hata KUSUR olarak raporlanır
            checks.append(RunCheck("live-drive", "fail", f"Sür motoru çalıştırılamadı: {exc}"))
            return RunSmokeResult("fail", "Canlı sür spawn'ı patladı.", checks)
        finally:
            driver_scope.revoke_run(run_id)
            with contextlib.suppress(OSError):
                Path(mcp_path).unlink(missing_ok=True)

        gordu_mcp = "mcp__" in (output or "")
        verdict = driver.parse_drive_verdict(output or "")
        kuyruk = (output or "")[-400:]
        if gordu_mcp and verdict["passed"]:
            checks.append(
                RunCheck(
                    "live-drive",
                    "pass",
                    f"Sür motoru MCP araçlarını GÖRDÜ ve PASS verdi. Çıktı kuyruğu: {kuyruk!r}",
                )
            )
            return RunSmokeResult("pass", "Canlı sür: MCP araçları görünür.", checks)
        checks.append(
            RunCheck(
                "live-drive",
                "fail",
                f"MCP görünürlüğü KANITLANAMADI (mcp__ var mı={gordu_mcp}, "
                f"verdict={verdict['verdict']}). Çıktı kuyruğu: {kuyruk!r}",
            )
        )
        return RunSmokeResult("fail", "Canlı sür: MCP araçları görünmedi.", checks)

    @staticmethod
    def _web_reachable() -> bool:
        """ACHILLES_WEB_URL/api/healthz'e kısa bir GET dene (spawn öncesi ucuz kontrol)."""
        import os
        import urllib.request

        base = os.environ.get("ACHILLES_WEB_URL", "http://127.0.0.1:8765").rstrip("/")
        try:
            with urllib.request.urlopen(f"{base}/api/healthz", timeout=3) as resp:
                return 200 <= resp.status < 500
        except Exception:
            return False


class _FakeRequest:
    """require_human için minimal Request taklidi (yalnız başlıklar okunur)."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers
        self.client = None
