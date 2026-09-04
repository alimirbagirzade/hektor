"""Scope izolasyonu — sürücü (driver) motoru insan yetkisinden ayıran katman.

Doğrulanan zafiyet: Hektor kendi `claude -p` motorunu doğuruyor; motor
`/api/approvals/{id}/approve` + `/api/supervisor/clear-stop-all` + eğitim
uçlarında insanla AYNI yetkiye sahipti → kendi eğitimini onaylayabiliyordu
(CLAUDE.md Kural 8 etkisiz).

Tüm testler ÇEVRİMDIŞI: gerçek `claude -p` spawn YOK, gerçek eğitim YOK, ağ YOK.
`approvals`/`supervisor`/`launch` her zaman mock'lanır.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.orchestration.driver import (  # noqa: E402
    DISALLOWED_TOOLS,
    build_child_env,
    build_hunt_command,
)
from app.web import driver_scope, security  # noqa: E402
from app.web.server import app  # noqa: E402

client = TestClient(app)

_RUN_ID = "orc_test_run_1"
_APPROVE = "/api/approvals/apr_x/approve"
_REJECT = "/api/approvals/apr_x/reject"
_CLEAR = "/api/supervisor/clear-stop-all"


@pytest.fixture(autouse=True)
def _clean_store() -> None:
    """Her test temiz token deposuyla başlar (süreç-içi global)."""
    driver_scope.reset()
    yield
    driver_scope.reset()


def _driver_headers(token: str, run_id: str = _RUN_ID) -> dict[str, str]:
    return {
        driver_scope.DRIVER_TOKEN_HEADER: token,
        driver_scope.RUN_ID_HEADER: run_id,
    }


# ── driver scope → 403 (asıl güvenlik iddiası) ────────────────────────────────


def test_driver_cannot_approve() -> None:
    """Sürücü onay VEREMEZ — motor kendi eğitimini onaylayamaz (Kural 8)."""
    token = driver_scope.mint(_RUN_ID)
    with patch("app.agents.runtime.approvals.approve") as m_approve:
        r = client.post(_APPROVE, headers=_driver_headers(token))
    assert r.status_code == 403
    m_approve.assert_not_called()  # yetki fonksiyonuna HİÇ ulaşılmadı


def test_driver_cannot_reject() -> None:
    """Red de bir yetki kararıdır → sürücüye kapalı."""
    token = driver_scope.mint(_RUN_ID)
    with patch("app.agents.runtime.approvals.reject") as m_reject:
        r = client.post(_REJECT, headers=_driver_headers(token))
    assert r.status_code == 403
    m_reject.assert_not_called()


def test_driver_cannot_clear_stop_all() -> None:
    """Sürücü kill-switch'i temizleyemez."""
    token = driver_scope.mint(_RUN_ID)
    with patch("app.agents.runtime.supervisor.clear_stop_all") as m_clear:
        r = client.post(_CLEAR, headers=_driver_headers(token))
    assert r.status_code == 403
    m_clear.assert_not_called()


def test_driver_cannot_start_training() -> None:
    """Sürücü gerçek eğitim başlatamaz — launch'a ULAŞILMAZ."""
    token = driver_scope.mint(_RUN_ID)
    with patch("app.training.detached_launch.launch") as m_launch:
        r = client.post(
            "/api/training/run",
            json={"adapter_name": "x", "iterations": 3},
            headers=_driver_headers(token),
        )
    assert r.status_code == 403
    m_launch.assert_not_called()


def test_driver_cannot_use_auto_lora_train() -> None:
    """Denetim bulgusu: auto-lora AYNI onay-tüketim yolunu kullanıyor → o da kapalı."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post(
        "/api/auto-lora/train",
        params={"adapter_name": "x"},
        headers=_driver_headers(token),
    )
    assert r.status_code == 403


def test_driver_cannot_promote_adapter() -> None:
    """Motor kendi adapter'ını production'a terfi ettiremez."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post("/api/auto-lora/promote", headers=_driver_headers(token))
    assert r.status_code == 403


# ── eğitim VERİSİNİ besleyen yetki kararları da insan-yalnız ─────────────────


def test_driver_cannot_approve_card() -> None:
    """Kart onayı `lora_eligible=1` yapar → eğitim korpusunu besler (v5'in kökü)."""
    token = driver_scope.mint(_RUN_ID)
    with patch("app.memory.sqlite_store.SqliteStore.approve_card") as m_approve:
        r = client.post("/api/card/kart_x/approve", headers=_driver_headers(token))
    assert r.status_code == 403
    m_approve.assert_not_called()


def test_driver_cannot_reject_card() -> None:
    """Red de korpus kararıdır — motor rakip kanıtı eleyemez."""
    token = driver_scope.mint(_RUN_ID)
    with patch("app.memory.sqlite_store.SqliteStore.reject_card") as m_reject:
        r = client.post("/api/card/kart_x/reject", headers=_driver_headers(token))
    assert r.status_code == 403
    m_reject.assert_not_called()


def test_driver_cannot_approve_feedback() -> None:
    """Onaylanan düzeltme SFT adayı olur → eğitim verisi."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post("/api/feedback/approve/corr_x", headers=_driver_headers(token))
    assert r.status_code == 403


def test_driver_cannot_export_feedback() -> None:
    """Export eğitim verisi DOSYASI üretir → motor kendi verisini yazamamalı."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post("/api/feedback/export", headers=_driver_headers(token))
    assert r.status_code == 403


def test_driver_task_forced_to_require_approval() -> None:
    """Önleyici: sürücü görev açabilir ama onay bayrağını DÜŞÜREMEZ."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post(
        "/api/automation/tasks",
        params={"agent_id": "a", "title": "t", "requires_approval": False},
        headers=_driver_headers(token),
    )
    assert r.status_code == 200
    assert r.json()["task"]["requires_approval"] is True  # driver isteği YÜKSELTİLDİ


def test_human_task_keeps_requested_approval_flag() -> None:
    """İnsan akışı değişmez: requires_approval=False insanda False kalır."""
    r = client.post(
        "/api/automation/tasks",
        params={"agent_id": "a", "title": "t", "requires_approval": False},
    )
    assert r.status_code == 200
    assert r.json()["task"]["requires_approval"] is False


# ── human scope → çalışmaya devam eder (geriye dönük uyum) ────────────────────


def test_human_can_approve_and_clear_stop_all() -> None:
    """İnsan akışı BOZULMAZ: başlıksız istek human scope'tur → 200."""
    from app.agents.runtime.schemas import ApprovalRequest, ApprovalStatus

    fake = ApprovalRequest(
        approval_id="apr_x",
        agent_id="lora-trainer",
        action="train_run",
        summary="t",
        status=ApprovalStatus.approved,
        requested_at="2026-07-21T00:00:00Z",
    )
    with patch("app.agents.runtime.approvals.approve", return_value=fake):
        r_ok = client.post(_APPROVE)
    assert r_ok.status_code == 200

    with patch(
        "app.agents.runtime.supervisor.clear_stop_all", return_value={"was_active": True}
    ) as m_clear:
        r_clear = client.post(_CLEAR)
    assert r_clear.status_code == 200
    m_clear.assert_called_once()


# ── token bağlama: run_id / TTL / geçersizlik ─────────────────────────────────


def test_driver_token_rejected_outside_its_run() -> None:
    """Token BAŞKA bir run_id ile sunulursa reddedilir (koşuya bağlılık)."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post(_CLEAR, headers=_driver_headers(token, run_id="orc_baska_kosu"))
    assert r.status_code == 401
    assert driver_scope.verify(token, run_id="orc_baska_kosu") is None
    assert driver_scope.verify(token, run_id=_RUN_ID) == _RUN_ID  # kendi koşusunda geçerli


def test_revoked_token_is_rejected() -> None:
    """Koşu bitince token iptal edilir → sonraki kullanım reddedilir."""
    token = driver_scope.mint(_RUN_ID)
    assert driver_scope.revoke_run(_RUN_ID) == 1
    assert driver_scope.verify(token, run_id=_RUN_ID) is None
    r = client.post(_CLEAR, headers=_driver_headers(token))
    assert r.status_code == 401


def test_expired_token_is_rejected() -> None:
    """TTL dolduğunda token geçersizdir (saat ileri sarılır — gerçek bekleme YOK)."""
    import time as _time

    token = driver_scope.mint(_RUN_ID, ttl_s=1)
    future = _time.monotonic() + 3600.0  # monotonic mutlak değeri platforma göre büyük olabilir
    with patch("app.web.driver_scope.time.monotonic", return_value=future):
        assert driver_scope.verify(token, run_id=_RUN_ID) is None


def test_remint_revokes_previous_token() -> None:
    """Koşu başına tek geçerli kimlik: yeniden mint eskisini iptal eder."""
    old = driver_scope.mint(_RUN_ID)
    new = driver_scope.mint(_RUN_ID)
    assert driver_scope.verify(old, run_id=_RUN_ID) is None
    assert driver_scope.verify(new, run_id=_RUN_ID) == _RUN_ID


def test_invalid_driver_token_does_not_fall_back_to_human() -> None:
    """KRİTİK: geçersiz sürücü token'ı human'a DÜŞMEZ (yoksa yetki yükseltmesi olurdu)."""
    with patch("app.agents.runtime.supervisor.clear_stop_all") as m_clear:
        r = client.post(_CLEAR, headers=_driver_headers("uydurma_token"))
    assert r.status_code == 401
    m_clear.assert_not_called()


def test_verify_does_not_consume_token() -> None:
    """Sürücü token'ı bir KİMLİK etiketidir; salt-okuma çağrısı kimliği tüketmez."""
    token = driver_scope.mint(_RUN_ID)
    for _ in range(3):
        assert driver_scope.verify(token, run_id=_RUN_ID) == _RUN_ID


# ── doğurulan sürecin ortamı + araç kısıtı (asıl sınır) ───────────────────────


def test_child_env_blanks_human_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """İnsan token'ı çocuğa BOŞ geçer.

    Anahtarı silmek YETMEZ: Settings `env_file='.env'` okuduğu için silinen anahtar
    dotenv'den geri gelir. Bu yüzden açıkça boş string'e ezilmelidir.
    """
    monkeypatch.setenv("HEKTOR_API_TOKEN", "insan_sirri")
    env = build_child_env("drv_token", _RUN_ID)
    assert env["HEKTOR_API_TOKEN"] == ""  # silinmiş DEĞİL, ezilmiş
    assert "insan_sirri" not in env.values()
    assert env[driver_scope.DRIVER_TOKEN_ENV] == "drv_token"
    assert env[driver_scope.DRIVER_RUN_ID_ENV] == _RUN_ID


def test_hunt_command_restricts_tools() -> None:
    """Motor araç-seviyesinde kısıtlı: Bash olmadan auth'suz CLI'yi çalıştıramaz.

    Bu kısıt olmadan scope katmanı tiyatrodur — `uv run hektor approval-approve`
    hiçbir kimlik doğrulamasından geçmez (denetim BLOCKER bulgusu).
    """
    cmd = build_hunt_command({"adapter_name": "myad"})
    assert "--disallowedTools" in cmd
    blocked = cmd[cmd.index("--disallowedTools") + 1]
    for tool in ("Bash", "Write", "Edit"):
        assert tool in blocked
    # `Task` de yasak: alt-ajan doğurup deny-list'i dolaylı aşmasın.
    assert set(DISALLOWED_TOOLS) >= {"Bash", "Edit", "Write", "Task"}
    # variadic bayrak EN SONDA olmalı (sonraki bayrakları yutmasın).
    assert cmd[-2] == "--disallowedTools"


def test_child_env_strips_settings_override_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--safe-mode`'a rağmen ayar geri getirebilecek env kanalları temizlenir.

    safe-mode "admin-managed (policy) settings still apply" der; bu değişkenler harici
    bir ayar dosyası işaret ederek hook'ları geri getirebilirdi.
    """
    monkeypatch.setenv("CLAUDE_CODE_MANAGED_SETTINGS_PATH", r"C:\kotu\settings.json")
    monkeypatch.setenv("CLAUDE_CODE_REMOTE_SETTINGS_PATH", r"C:\kotu\remote.json")
    env = build_child_env("drv_token", _RUN_ID)
    assert "CLAUDE_CODE_MANAGED_SETTINGS_PATH" not in env
    assert "CLAUDE_CODE_REMOTE_SETTINGS_PATH" not in env
    assert "CLAUDE_CODE_MOCK_REMOTE_SETTINGS" not in env


def test_only_verified_engines_are_hardened() -> None:
    """Sertleştirme bayrakları motora ÖZGÜ → yalnız doğrulanan motorlar hardened olmalı.

    Kısıtsız bir motor doğurulursa scope katmanı tamamen delinir:
    araç kısıtı olmadan auth'suz `hektor approval-approve` çağrılabilir.
    """
    from app.orchestration import engines

    assert engines.get_engine("claude").hardened is True
    assert engines.get_engine("codex").hardened is True
    for name in ("gemini", "local"):
        assert engines.get_engine(name).hardened is False, f"{name} doğrulanmadan hardened"


def test_driver_refuses_unhardened_engine(monkeypatch, tmp_path) -> None:
    """FAIL-CLOSED: sertleştirilmemiş motor için gerçek spawn REDDEDİLİR."""
    from dataclasses import replace

    from app.orchestration import engines
    from app.orchestration.driver import AutoDriver
    from app.orchestration.orchestrator import RunContext, StageResult, TrainingOrchestrator
    from app.orchestration.pipeline import StageStatus
    from app.orchestration.store import OrchestrationStore

    def blocked_hunt(ctx: RunContext) -> StageResult:
        return StageResult(StageStatus.blocked, "hunt_ack yok", {})

    def done(ctx: RunContext) -> StageResult:
        return StageResult(StageStatus.completed, "ok", {})

    orch = TrainingOrchestrator(
        store=OrchestrationStore(db_path=tmp_path / "eng.db"),
        delegates={"preflight": done, "deep-hunt": blocked_hunt},
    )
    drv = AutoDriver(orchestrator=orch)
    run_id = orch.start(model="m", profile="p", adapter_name="a")

    # Kayıt tablosundaki doğrulanmış Codex'i bu testte bilinçli olarak güvensiz yap;
    # böylece PATH/kurulum durumundan önce fail-closed sertleştirme kapısını sınarız.
    unsafe = replace(engines.get_engine("codex"), hardened=False)
    monkeypatch.setattr(engines, "get_engine", lambda name: unsafe)

    # runner enjekte EDİLMEDEN (gerçek spawn yolu) → kısıt aranır ve reddedilir.
    res = drv.drive(run_id, execute=True, engine="codex")
    assert res["ok"] is False
    assert res["hardened"] is False
    assert "sertleştirilmiş değil" in res["reason"]


def test_hunt_command_disables_customization_channels() -> None:
    """Özelleştirme kanalları (hook/plugin/MCP/özel-ajan) araç deny-list'inin DIŞINDADIR.

    İki kanıtlanmış kaçış yolu: (a) `hektor` MCP sunucusu 127.0.0.1:8765'e proxy'liyor
    ve sürücü başlığı göndermiyor → human scope'una düşerdi; (b) `.claude/settings.json`
    hook'ları Claude Code tarafından DOĞRUDAN kabukta çalıştırılır (`Bash` aracı DEĞİL)
    ve `-p` modunda güven diyaloğu atlandığı için onaysız koşar.

    `--safe-mode` bu sınıfın tamamını kapatır; kanal kanal kovalamacadan üstündür.
    """
    cmd = build_hunt_command({"adapter_name": "myad"})
    assert "--safe-mode" in cmd  # hook + plugin + MCP + özel ajan/komut/skill kapalı
    assert "--strict-mcp-config" in cmd  # kemer-askı
    assert "--mcp-config" not in cmd  # hiçbir MCP sunucusu yüklenmez


def test_hunt_prompt_tells_agent_to_read_claude_md() -> None:
    """safe-mode CLAUDE.md oto-keşfini kapatır → prompt açıkça okumasını söylemeli."""
    from app.orchestration.driver import build_hunt_prompt

    prompt = build_hunt_prompt({"adapter_name": "x"})
    assert "CLAUDE.md" in prompt
    assert "OKU" in prompt


# ── hunt_ack: doğrulanmayan insan yetki beyanı → sürücüye kapalı ─────────────


def test_driver_cannot_self_ack_deep_hunt_on_start() -> None:
    """Motor zorunlu Kademe-2 avını `hunt_ack=true` göndererek ATLAYAMAZ (v5 kökü)."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post(
        "/api/orchestration/start",
        json={"hunt_ack": True, "auto_run": False},
        headers=_driver_headers(token),
    )
    assert r.status_code == 403


def test_driver_cannot_self_ack_deep_hunt_on_resume() -> None:
    token = driver_scope.mint(_RUN_ID)
    r = client.post(
        "/api/orchestration/resume/orc_hicbir_kosu",
        json={"hunt_ack": True},
        headers=_driver_headers(token),
    )
    # 403 (scope kapısı), 404'ten (koşu yok) ÖNCE gelmeli
    assert r.status_code == 403


def test_driver_may_start_run_without_ack() -> None:
    """Kapı yalnız yetki BEYANINA takılır — avsız başlatma sürücüye serbesttir."""
    token = driver_scope.mint(_RUN_ID)
    r = client.post(
        "/api/orchestration/start",
        json={"hunt_ack": False, "auto_run": False},
        headers=_driver_headers(token),
    )
    assert r.status_code == 200


# ── api_token boşken gürültülü uyarı (sessiz "auth kapalı" olmasın) ───────────


def test_warns_when_api_token_empty(caplog: pytest.LogCaptureFixture) -> None:
    from app.config.settings import Settings

    empty = Settings(api_token="")
    with (
        patch("app.web.security.get_settings", return_value=empty),
        caplog.at_level(logging.WARNING, logger="app.web.security"),
    ):
        enabled = security.warn_if_auth_disabled()
    assert enabled is False
    assert any("HEKTOR_API_TOKEN" in r.message for r in caplog.records)
    assert any("KAPALI" in r.message for r in caplog.records)


def test_no_warning_when_api_token_set(caplog: pytest.LogCaptureFixture) -> None:
    from app.config.settings import Settings

    with (
        patch("app.web.security.get_settings", return_value=Settings(api_token="gizli")),
        caplog.at_level(logging.WARNING, logger="app.web.security"),
    ):
        enabled = security.warn_if_auth_disabled()
    assert enabled is True
    assert not caplog.records


# ── insan-yalnız uçlar OpenAPI'de görünmez (doc gizleme; koruma değil) ────────


def test_human_only_endpoints_hidden_from_openapi() -> None:
    paths = client.get("/api/openapi.json").json()["paths"]
    assert "/api/supervisor/clear-stop-all" not in paths
    assert "/api/approvals/{approval_id}/approve" not in paths
    assert "/api/training/run" not in paths
    # ...ama gizleme koruma DEĞİL: uç hâlâ var ve scope kapısı çalışıyor.
    token = driver_scope.mint(_RUN_ID)
    assert client.post(_CLEAR, headers=_driver_headers(token)).status_code == 403


# ── Sür (drive) modu: MCP erişimli motorun açtığı iki delik ─────────────────────


def test_driver_cannot_autodrive() -> None:
    """Sürücü YENİ sürücü doğuramaz — özyinelemeli spawn + kota yakma kapalı.

    Bu uç gerçek bir `claude -p` alt-süreci başlatır ve OpenAPI şemasındadır → sür
    modunda motora MCP aracı olarak görünür. `human_only` olmasaydı motor kendi
    kendini çoğaltabilirdi.
    """
    token = driver_scope.mint(_RUN_ID)
    with patch("app.orchestration.driver.AutoDriver") as m_drv:
        r = client.post(
            f"/api/orchestration/autodrive/{_RUN_ID}",
            json={"execute": True},
            headers=_driver_headers(token),
        )
    assert r.status_code == 403
    m_drv.assert_not_called()  # sürücüye HİÇ ulaşılmadı (404 kontrolünden bile önce)


def test_require_auth_accepts_driver_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_token AYARLIYKEN sürücü token'ı KİMLİK olarak kabul edilir (401 değil).

    Motorun HEKTOR_API_TOKEN'ı bilinçli boşaltılır; bu kapı yalnız insan sırrını
    tanısaydı sür modu token'lı kurulumda tamamen çalışmazdı (her MCP çağrısı 401).
    """
    monkeypatch.setattr(security, "get_settings", lambda: _AyarlarSahte("gizli_insan"))
    token = driver_scope.mint(_RUN_ID)
    # salt-okuma bir uç: sürücü kimliğiyle GEÇMELİ (403/401 değil)
    r = client.get("/api/orchestration/runs", headers=_driver_headers(token))
    assert r.status_code == 200


def test_require_auth_driver_token_is_identity_not_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KRİTİK: sürücü token'ı auth'u geçer ama YETKİ VERMEZ — human_only yine 403.

    Bu ayrım bozulursa, auth kapısını gevşetmek Kural 8'i de gevşetmiş olurdu.
    """
    monkeypatch.setattr(security, "get_settings", lambda: _AyarlarSahte("gizli_insan"))
    token = driver_scope.mint(_RUN_ID)
    with patch("app.training.detached_launch.launch") as m_launch:
        r = client.post(
            "/api/training/run",
            json={"adapter_name": "x", "iterations": 3},
            headers=_driver_headers(token),
        )
    assert r.status_code == 403  # 401 DEĞİL: kimlik tanındı, yetki reddedildi
    m_launch.assert_not_called()


def test_require_auth_still_rejects_tokenless(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gevşetme yalnız sürücüye özgü: kimliksiz istek hâlâ 401."""
    monkeypatch.setattr(security, "get_settings", lambda: _AyarlarSahte("gizli_insan"))
    assert client.get("/api/orchestration/runs").status_code == 401


def test_require_auth_rejects_bogus_driver_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uydurma sürücü token'ı auth'u AÇMAZ (yoksa herkes 'sürücüyüm' derdi)."""
    monkeypatch.setattr(security, "get_settings", lambda: _AyarlarSahte("gizli_insan"))
    r = client.get("/api/orchestration/runs", headers=_driver_headers("uydurma_token"))
    assert r.status_code == 401


class _AyarlarSahte:
    """`get_settings()` yerine geçen minimal sahte (yalnız api_token okunur)."""

    def __init__(self, api_token: str) -> None:
        self.api_token = api_token
