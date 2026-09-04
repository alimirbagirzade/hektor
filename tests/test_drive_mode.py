"""Sür (drive) modu testleri — prompt sözleşmesi, MCP geçişi, verdict, kimlik sınırı.

Hepsi ÇEVRİMDIŞI: gerçek spawn YOK, `fastmcp` GEREKMEZ (P6'da canlı deneme yapılacak).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from app.orchestration import engines
from app.orchestration.driver import (
    DRIVE_TIMEOUT_S,
    DRIVE_TOKEN_TTL_S,
    HUNT_TIMEOUT_S,
    build_child_env,
    build_drive_command,
    build_drive_prompt,
    build_mcp_config,
    parse_drive_verdict,
    parse_hunt_verdict,
    write_mcp_config,
)
from app.web import driver_scope

_RUN = {"adapter_name": "myad"}
_RUN_ID = "run_surus_1"
_CFG = "/tmp/hektor_mcp.json"


# ── Prompt sözleşmesi ───────────────────────────────────────────────────────────────────


def test_drive_prompt_egitimi_yasaklar() -> None:
    """Sür modu iş YAPAR ama eğitimi ASLA başlatmaz (CLAUDE.md Kural 8)."""
    p = build_drive_prompt(_RUN)
    assert "EĞİTİM BAŞLATMA" in p
    assert "/api/training/run" in p  # eğitim ucu adıyla yasaklanmış
    assert "/api/approvals/{id}/approve" in p  # onay ucu adıyla yasaklanmış
    assert "TAZE İNSAN ONAYI" in p and "DUR" in p


def test_drive_prompt_mcp_kullan_dosya_duzenleme() -> None:
    """Araçlar MCP üzerinden; doğrudan dosya düzenleme yasak."""
    p = build_drive_prompt(_RUN)
    assert "mcp__" in p
    assert "DOĞRUDAN DÜZENLEME" in p
    assert "carding" in p and "RLM" in p and "curate" in p and "assemble" in p


def test_drive_prompt_403u_asmaya_calismaz() -> None:
    """Ajana 403'ün BEKLENEN olduğu söylenir → aşmaya çalışmasın."""
    assert "403" in build_drive_prompt(_RUN)


def test_drive_prompt_claude_md_okur() -> None:
    assert "CLAUDE.md" in build_drive_prompt(_RUN)


# ── Verdict: desen aynalanır, işaretçiler KARIŞMAZ ──────────────────────────────────────


def test_hunt_verdict_parse_bozulmadi() -> None:
    """Refactor sonrası av modu verdict sözleşmesi birebir aynı."""
    assert parse_hunt_verdict("bla\nHEKTOR_HUNT_VERDICT: PASS")["passed"] is True
    assert parse_hunt_verdict("HEKTOR_HUNT_VERDICT: FAIL")["passed"] is False
    yok = parse_hunt_verdict("hiç verdict yok")
    assert yok["passed"] is False and yok["verdict"] == "unknown"
    assert "HEKTOR_HUNT_VERDICT" in yok["summary"]


def test_drive_verdict_ayni_deseni_izler() -> None:
    """Son satır + PASS|FAIL + bulunamazsa fail-closed — av moduyla aynı sözleşme."""
    assert parse_drive_verdict("rapor\nHEKTOR_DRIVE_VERDICT: PASS")["passed"] is True
    assert parse_drive_verdict("HEKTOR_DRIVE_VERDICT: FAIL")["passed"] is False
    yok = parse_drive_verdict("verdict satırı yok")
    assert yok["passed"] is False and yok["verdict"] == "unknown"


def test_verdict_isaretcileri_capraz_okunmaz() -> None:
    """KRİTİK: sür PASS'i av PASS'i sanılmamalı.

    Aynı işaretçi kullanılsaydı, bir sür koşusunun PASS'i `hunt_ack=true` yazdırıp
    derin av HİÇ yapılmadan eğitim kapısını açardı (Kural 8).
    """
    assert parse_hunt_verdict("HEKTOR_DRIVE_VERDICT: PASS")["passed"] is False
    assert parse_drive_verdict("HEKTOR_HUNT_VERDICT: PASS")["passed"] is False


def test_drive_prompt_verdict_satiri_parse_edilebilir() -> None:
    """Prompt'ta ÖRNEKLENEN biçim, gerçek parser tarafından okunabilmeli."""
    p = build_drive_prompt(_RUN)
    son = [ln for ln in p.splitlines() if "HEKTOR_DRIVE_VERDICT" in ln]
    assert son, "prompt verdict biçimini örneklemiyor"
    assert parse_drive_verdict(son[0])["verdict"] in {"PASS", "FAIL"}


# ── MCP geçişi: config ÜRETİLİR, kullanıcı kaydına bağımlı DEĞİL ────────────────────────


def test_drive_komutu_mcp_configi_gecirir() -> None:
    cmd = build_drive_command(_RUN, _CFG)
    assert "--mcp-config" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == _CFG


def test_drive_komutu_kullanici_mcp_kayitlarini_yoksayar() -> None:
    """`--strict-mcp-config` → yalnız bizim config; `claude mcp add` kaydı gerekmez."""
    assert "--strict-mcp-config" in build_drive_command(_RUN, _CFG)


def test_mcp_config_sunucuyu_tanimlar() -> None:
    cfg = build_mcp_config("/repo")
    srv = cfg["mcpServers"]["hektor"]
    assert srv["command"] == "uv"
    assert "--project" in srv["args"] and "/repo" in srv["args"]
    # fastmcp opsiyonel extra'dadır → açıkça istenmeli, yoksa sunucu ImportError ile ölür.
    assert "--extra" in srv["args"] and "mcp" in srv["args"]
    assert any("hektor_mcp.py" in a for a in srv["args"])


def test_mcp_config_sir_icermez() -> None:
    """Kısa ömürlü sürücü token'ı DİSKE yazılmaz (ortamdan miras alınır)."""
    ham = json.dumps(build_mcp_config("/repo"))
    assert driver_scope.DRIVER_TOKEN_ENV not in ham
    assert "token" not in ham.lower()


def test_write_mcp_config_dosya_uretir(tmp_path: Path) -> None:
    yol = write_mcp_config(tmp_path / "alt" / "mcp.json", root="/repo")
    assert Path(yol).is_file()
    assert json.loads(Path(yol).read_text(encoding="utf-8"))["mcpServers"]["hektor"]


# ── Sertleştirme: safe-mode YOK ama kanallar kapalı ─────────────────────────────────────


def test_drive_safe_mode_kullanamaz() -> None:
    """`--safe-mode` MCP sunucularını da kapatır → sür modunda KULLANILAMAZ.

    (`claude --help`: safe-mode "MCP servers ... disabled" der.)
    """
    assert "--safe-mode" not in build_drive_command(_RUN, _CFG)


def test_drive_ozellestirme_kanallarini_kapatir() -> None:
    """safe-mode'un yerine kanallar TEK TEK kapatılır (hook kanalı kritik)."""
    cmd = build_drive_command(_RUN, _CFG)
    assert "--setting-sources" in cmd
    # boş kaynak listesi → user/project/local ayarları YÜKLENMEZ (hook/plugin/özel ajan yok)
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--disable-slash-commands" in cmd


def test_drive_yerlesik_araclar_allow_list() -> None:
    """Allow-list deny-list'ten güçlü: varsayılan KAPALI."""
    cmd = build_drive_command(_RUN, _CFG)
    izinli = cmd[cmd.index("--tools") + 1].split(",")
    assert set(izinli) == {"Read", "Grep", "Glob"}
    for yasak in ("Bash", "Edit", "Write", "Task"):
        assert yasak not in izinli


def test_drive_variadic_bayrak_sirasi() -> None:
    """`--tools` ve `--mcp-config` VARIADIC → sonraki bayrakları yutmamalı.

    `--tools` tek virgüllü arg alır ve ardından bir bayrak gelir; `--mcp-config` EN SONDA.
    """
    cmd = build_drive_command(_RUN, _CFG)
    assert cmd[-2] == "--mcp-config"  # variadic en sonda
    assert cmd[cmd.index("--tools") + 2].startswith("--")  # --tools'un ardı bayrak


def test_drive_hardened_bayragi_desteklenen_motorlarda() -> None:
    """Sür profili doğrulanmayan motorda açılmamalı (fail-closed)."""
    assert engines.get_engine("claude").drive_hardened is True
    assert engines.get_engine("codex").drive_hardened is True
    for ad in ("gemini", "local"):
        assert engines.get_engine(ad).drive_hardened is False


def test_desteklemeyen_motor_surulemez() -> None:
    """Sessizce av moduna DÜŞMEZ — araçsız ajan doğurmak yerine ValueError."""
    assert engines.drive_supported("claude") is True
    assert engines.drive_supported("codex") is True
    for ad in ("gemini", "local"):
        assert engines.drive_supported(ad) is False
        with pytest.raises(ValueError, match="sür"):
            engines.build_drive_command(ad, "p", _CFG)


def test_bos_mcp_config_reddedilir() -> None:
    with pytest.raises(ValueError, match="mcp_config_path"):
        engines.build_drive_command("claude", "p", "")


def test_prompt_kabuga_gecmez() -> None:
    """Prompt argv ÖĞESİ olarak geçer → enjeksiyon imkânsız."""
    kotu = "; rm -rf /  $(whoami)"
    cmd = engines.build_drive_command("claude", kotu, _CFG)
    assert kotu in cmd  # tek bir öğe olarak, yorumlanmadan


# ── Zaman aşımı: sür modu AYRI sabit + token TTL hizası ────────────────────────────────


def test_drive_timeout_avdan_ayri() -> None:
    """Ayrı sabit → biri değişince diğeri sessizce kaymaz."""
    assert DRIVE_TIMEOUT_S != HUNT_TIMEOUT_S
    assert DRIVE_TIMEOUT_S > HUNT_TIMEOUT_S  # veri hattı avdan uzun


def test_drive_token_ttl_timeouttan_uzun() -> None:
    """KRİTİK: token koşunun ORTASINDA ölmemeli, yoksa MCP çağrıları 401 alır."""
    assert DRIVE_TOKEN_TTL_S > DRIVE_TIMEOUT_S
    # varsayılan TTL av moduna göre ayarlı → sür modunda YETMEZ
    assert DRIVE_TOKEN_TTL_S > driver_scope.DEFAULT_TTL_S


# ── Kimlik sınırı: sürücü token'ı geçer, insan token'ı GEÇMEZ ──────────────────────────


def test_child_env_surucu_gecer_insan_gecmez(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sür modunda da: sürücü kimliği verilir, insan sırrı BOŞA ezilir."""
    monkeypatch.setenv("HEKTOR_API_TOKEN", "insan_sirri")
    env = build_child_env("surucu_tokeni", _RUN_ID)
    assert env[driver_scope.DRIVER_TOKEN_ENV] == "surucu_tokeni"
    assert env[driver_scope.DRIVER_RUN_ID_ENV] == _RUN_ID
    assert env["HEKTOR_API_TOKEN"] == ""  # silinmiş DEĞİL, ezilmiş (dotenv geri okur)
    assert "insan_sirri" not in env.values()


# ── MCP proxy KİMLİK AKLAMASI yapmamalı (P1 sınırının can damarı) ──────────────────────


def _mcp_modulu():
    """`mcp_server/hektor_mcp.py`'yi fastmcp OLMADAN yükle (paket değil → yoldan)."""
    yol = Path(__file__).resolve().parents[1] / "mcp_server" / "hektor_mcp.py"
    spec = importlib.util.spec_from_file_location("hektor_mcp_test", yol)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mcp_proxy_surucu_basligini_tasir() -> None:
    """Sürücü kimliği proxy'de KAYBOLMAMALI.

    Kaybolsaydı istek sunucuya `human` gibi ulaşır (resolve_scope: başlık yoksa human) ve
    motor MCP üzerinden kendi eğitimini onaylayıp başlatabilirdi (Kural 8 ihlali).
    """
    mod = _mcp_modulu()
    h = mod.driver_headers(
        {
            driver_scope.DRIVER_TOKEN_ENV: "drv_abc",
            driver_scope.DRIVER_RUN_ID_ENV: _RUN_ID,
        }
    )
    assert h[driver_scope.DRIVER_TOKEN_HEADER] == "drv_abc"
    assert h[driver_scope.RUN_ID_HEADER] == _RUN_ID


def test_mcp_proxy_insan_icin_baslik_eklemez() -> None:
    """İnsan kendi oturumunda aynı sunucuyu kullanır → davranış DEĞİŞMEMELİ."""
    mod = _mcp_modulu()
    assert mod.driver_headers({}) == {}
    assert mod.driver_headers({driver_scope.DRIVER_TOKEN_ENV: "   "}) == {}


def test_mcp_proxy_insan_tokenini_gondermez() -> None:
    """İnsan API token'ı MCP başlığına ASLA sızmamalı."""
    mod = _mcp_modulu()
    h = mod.driver_headers({"HEKTOR_API_TOKEN": "insan_sirri"})
    assert h == {}


# ── SORUN 1: sür modu artık drive() spawn yoluna BAĞLI (ölü kod değil) ──────────────────


def _orch(tmp_path: Path):
    from app.orchestration.orchestrator import TrainingOrchestrator
    from app.orchestration.store import OrchestrationStore

    return TrainingOrchestrator(store=OrchestrationStore(db_path=tmp_path / "drv.db"))


def test_drive_modu_sur_argvsi_kurar_av_degil(tmp_path: Path) -> None:
    """drive(mode="drive") SÜR argv'sini kurar: --mcp-config VAR, --safe-mode YOK."""
    from app.orchestration.driver import DRIVE_TIMEOUT_S, AutoDriver

    yakalanan: dict[str, object] = {}

    def sahte_runner(command, timeout, env=None):
        yakalanan["command"] = command
        yakalanan["timeout"] = timeout
        return 0, "iş bitti\nHEKTOR_DRIVE_VERDICT: PASS\n"

    d = AutoDriver(orchestrator=_orch(tmp_path))
    run_id = d.orch.start(model="m", profile="p", adapter_name="a")
    res = d.drive(run_id, execute=True, mode="drive", runner=sahte_runner)

    assert res["ok"] is True and res["mode"] == "drive" and res["drive_passed"] is True
    cmd = yakalanan["command"]
    assert "--mcp-config" in cmd, "sür argv'si MCP config geçirmeli"
    assert "--safe-mode" not in cmd, "sür modunda --safe-mode MCP'yi kapatır"
    # Sür timeout'u AV'dan AYRI sabittir (SORUN 2 gerekçesiyle uzun).
    assert yakalanan["timeout"] == DRIVE_TIMEOUT_S


def test_drive_pass_hunt_acki_acmaz(tmp_path: Path) -> None:
    """KRİTİK (Kural 8): sür PASS'i hunt_ack YAZMAZ → zorunlu av kapısı bağımsız kalır."""
    from app.orchestration.driver import AutoDriver

    def sahte_pass(command, timeout, env=None):
        return 0, "ilerletildi\nHEKTOR_DRIVE_VERDICT: PASS\n"

    d = AutoDriver(orchestrator=_orch(tmp_path))
    run_id = d.orch.start(model="m", profile="p", adapter_name="a")
    d.drive(run_id, execute=True, mode="drive", runner=sahte_pass)

    run = d.orch.store.get_run(run_id)
    params = run.get("params") or {}
    assert params.get("hunt_ack") is not True, "sür PASS'i av kapısını AÇMAMALI"


def test_drive_dry_run_mcp_configli_komut_doner(tmp_path: Path) -> None:
    from app.orchestration.driver import AutoDriver

    d = AutoDriver(orchestrator=_orch(tmp_path))
    run_id = d.orch.start(model="m", profile="p", adapter_name="a")
    res = d.drive(run_id, execute=False, mode="drive")
    assert res["dry_run"] is True and res["mode"] == "drive"
    assert res["command"][0] == "claude" and "--mcp-config" in res["command"]
    assert res["mcp_config"].endswith(".json")


def test_drive_mint_ttl_gecer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SORUN 2 FIX: mint GERÇEKTEN ttl_s=DRIVE_TOKEN_TTL_S ile çağrılır (sabit değil, çağrı)."""
    from app.orchestration import driver
    from app.orchestration.driver import DRIVE_TOKEN_TTL_S, AutoDriver

    yakalanan: dict[str, object] = {}

    def sahte_mint(run_id, *, ttl_s=None):
        yakalanan["ttl_s"] = ttl_s
        return "sahte_token"

    monkeypatch.setattr(driver.driver_scope, "mint", sahte_mint)

    d = AutoDriver(orchestrator=_orch(tmp_path))
    run_id = d.orch.start(model="m", profile="p", adapter_name="a")
    d.drive(
        run_id,
        execute=True,
        mode="drive",
        runner=lambda c, t, e=None: (0, "HEKTOR_DRIVE_VERDICT: PASS"),
    )
    assert yakalanan["ttl_s"] == DRIVE_TOKEN_TTL_S


def test_codex_drive_guvenli_argv_kullanir(tmp_path: Path) -> None:
    """Codex ChatGPT oturumuyla, read-only ve yalnız Hektor MCP ile sürülür."""
    from app.orchestration.driver import AutoDriver

    d = AutoDriver(orchestrator=_orch(tmp_path))
    run_id = d.orch.start(model="m", profile="p", adapter_name="a")
    captured: dict[str, object] = {}

    def runner(command, timeout, env=None):
        captured["command"] = command
        return 0, "HEKTOR_DRIVE_VERDICT: PASS"

    res = d.drive(run_id, execute=True, mode="drive", engine="codex", runner=runner)
    assert res["ok"] is True
    cmd = captured["command"]
    assert cmd[:2] == ["codex", "exec"]
    assert "read-only" in cmd and 'approval_policy="never"' in cmd
    assert "--ignore-user-config" in cmd and "--ignore-rules" in cmd
    assert "mcp_servers={}" in cmd
    assert any(str(x).startswith("mcp_servers.hektor.command=") for x in cmd)
    assert "mcp_servers.hektor.required=true" in cmd


def test_codex_drive_config_yoksa_fail_closed() -> None:
    with pytest.raises(ValueError, match="Codex MCP config okunamadı"):
        engines.build_drive_command("codex", "p", "olmayan.json")


def test_drive_durdurunca_stopped(tmp_path: Path) -> None:
    from app.orchestration.driver import STOPPED_RC, AutoDriver

    d = AutoDriver(orchestrator=_orch(tmp_path))
    run_id = d.orch.start(model="m", profile="p", adapter_name="a")
    res = d.drive(run_id, execute=True, mode="drive", runner=lambda c, t, e=None: (STOPPED_RC, ""))
    assert res["stopped"] is True and res["drive_passed"] is False and res["mode"] == "drive"


def test_bilinmeyen_mod_reddedilir(tmp_path: Path) -> None:
    from app.orchestration.driver import AutoDriver

    d = AutoDriver(orchestrator=_orch(tmp_path))
    run_id = d.orch.start(model="m", profile="p", adapter_name="a")
    res = d.drive(run_id, execute=False, mode="uydurma")
    assert res["ok"] is False and "Bilinmeyen sürüş modu" in res["reason"]


# ── Sür modu MCP yüzeyi allow-list'in 19 ucunu AŞMAZ (aynı sunucu → aynı budama) ────────


def test_sur_modu_mcp_yuzeyi_allowlisti_asmaz() -> None:
    """Sür motoru allow-list'li AYNI MCP sunucusunu kullanır → yüzey 21 uç."""
    from mcp_server.allowlist import ALLOWED

    # Sür MCP config'i hektor_mcp.py'yi işaret eder; o da filter_spec ile budanır.
    assert len(ALLOWED) == 21, "allow-list yüzeyi beklenmedik biçimde değişti"
    cfg = build_mcp_config("/repo")
    assert any("hektor_mcp.py" in a for a in cfg["mcpServers"]["hektor"]["args"])
