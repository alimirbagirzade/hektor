"""/api/engines + ⚡ RUN motor kapısı — çevrimdışı TestClient testleri.

Üç sözleşmeyi sabitler (kullanıcı isteğinin doğrudan karşılığı):
1. `/api/engines` KİMLİK BİLGİSİ SIZDIRMAZ (token/mail/anahtar/çerez asla).
2. `/api/engines` SALT-OKUMADIR — süreç doğurmaz.
3. Kurulu-olmayan / sertleştirilemeyen motor ⚡ RUN'da SEÇİLEMEZ — uç 503 ile reddeder
   (UI'daki gri seçenek bir güvenlik sınırı değildir; asıl kapı burasıdır).

Ayrıca `execute=true`'nun insan-yalnız kaldığını doğrular (Kural 8): sürücü scope 403 alır.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from app.orchestration import engines
from app.web.server import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _offline_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Testler çevrimdışı: 'smoke' aşaması gerçek Ollama istemesin (bkz. test_orchestration_web)."""
    import app.brain.local_llm as llm_mod

    monkeypatch.setattr(llm_mod.LocalLLM, "active_backend", lambda self: "none")
    monkeypatch.setattr(llm_mod.LocalLLM, "available", lambda self: False)


@pytest.fixture(autouse=True)
def _clean_probe_cache() -> Iterator[None]:
    """PATH yoklama cache'i testler arası sızmasın (deterministik sonuç)."""
    engines.reset_probe_cache()
    yield
    engines.reset_probe_cache()


# ── 1) kimlik bilgisi sızıntısı ─────────────────────────────────────────────────────────
# Alan ADI bazlı yasak: yeni bir alan eklenirse ve adı bunlardan birini içeriyorsa test
# PATLAR — "kimlik bilgisi döndürme" kuralı böylece kod büyüdükçe de korunur.
_FORBIDDEN_KEY_FRAGMENTS = (
    "token",
    "key",
    "secret",
    "password",
    "passwd",
    "mail",
    "credential",
    "cookie",
    "session",
    "auth",
    "bearer",
)


def _walk_keys(obj: object) -> list[str]:
    """İç içe yapıdaki TÜM sözlük anahtarlarını topla (alan adı denetimi için)."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.append(str(k))
            found.extend(_walk_keys(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_walk_keys(item))
    return found


def test_engines_response_has_no_credential_fields(client: TestClient) -> None:
    """Yanıtta kimlik bilgisi ÇAĞRIŞTIRAN hiçbir alan adı bulunmamalı."""
    body = client.get("/api/engines").json()
    for key in _walk_keys(body):
        low = key.lower()
        for bad in _FORBIDDEN_KEY_FRAGMENTS:
            assert bad not in low, f"/api/engines kimlik alanı sızdırıyor: {key!r}"


def test_engines_does_not_leak_env_secret_values(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ortamdaki bir sır yanıtın GÖVDESİNDE hiçbir biçimde geçmemeli."""
    sentinel = "SIR_DEGER_ASLA_SIZMASIN_9f3a2b"
    monkeypatch.setenv("ACHILLES_API_TOKEN", sentinel)
    monkeypatch.setenv("ANTHROPIC_API_KEY", sentinel)
    raw = client.get("/api/engines").text
    assert sentinel not in raw


def test_engines_login_state_is_honest_unknown(client: TestClient) -> None:
    """`logged_in` UYDURULMAZ: giriş yalnız motor çalıştırılınca anlaşılır → daima None."""
    body = client.get("/api/engines").json()
    assert body["engines"], "motor listesi boş olmamalı"
    for eng in body["engines"]:
        assert eng["logged_in"] is None
        assert eng["login_note"], "giriş durumunun neden bilinmediği açıklanmalı"


def test_engines_exposes_fields_ui_needs(client: TestClient) -> None:
    """Ad, etiket, kurulu mu, girişli mi, kota uyarısı — istenen alanların hepsi var."""
    body = client.get("/api/engines").json()
    assert body["default"] == engines.DEFAULT_ENGINE
    for eng in body["engines"]:
        for field in (
            "name",
            "label",
            "installed",
            "logged_in",
            "quota_warning",
            "install_hint",
            "selectable",
            "blocked_reason",
        ):
            assert field in eng, f"eksik alan: {field}"


# ── 2) salt-okuma ───────────────────────────────────────────────────────────────────────
def test_engines_spawns_no_process(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Uç hiçbir alt-süreç başlatmamalı (kota yakmaz, yan etkisi yok)."""
    import subprocess

    def _boom(*_a: object, **_kw: object) -> None:
        raise AssertionError("/api/engines süreç başlattı — salt-okuma sözleşmesi ihlali!")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    assert client.get("/api/engines").status_code == 200
    assert client.post("/api/engines/rescan").status_code == 200


# ── 3) kurulu-değil motor ⚡ RUN'da seçilemez ────────────────────────────────────────────
def _fresh_run(client: TestClient) -> str:
    return client.post(
        "/api/orchestration/start",
        json={"adapter_name": "engine_gate", "hunt_ack": False, "auto_run": True},
    ).json()["run_id"]


def test_unavailable_engine_marked_unselectable(monkeypatch: pytest.MonkeyPatch) -> None:
    """PATH'te olmayan motor `selectable=false` + sebep taşır (UI onu gri gösterir)."""
    desc = engines.describe("claude", which=lambda _b: None)  # kurulu değilmiş gibi
    assert desc["installed"] is False
    assert desc["selectable"] is False
    assert "PATH" in str(desc["blocked_reason"])


def test_execute_rejects_uninstalled_engine(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kurulu olmayan motorla execute=true → 503 (gri butonu kurcalamak işe yaramaz)."""
    monkeypatch.setattr(engines.shutil, "which", lambda _b: None)
    engines.reset_probe_cache()
    run_id = _fresh_run(client)
    r = client.post(
        f"/api/orchestration/autodrive/{run_id}",
        json={"execute": True, "engine": "claude"},
    )
    assert r.status_code == 503
    assert "PATH" in r.json()["detail"]


def test_execute_rejects_unhardened_engine(client: TestClient) -> None:
    """Sertleştirilemeyen motor (codex/gemini) execute=true ile doğurulamaz — Kural 8."""
    run_id = _fresh_run(client)
    r = client.post(
        f"/api/orchestration/autodrive/{run_id}",
        json={"execute": True, "engine": "gemini"},
    )
    assert r.status_code == 503
    assert "kısıt" in r.json()["detail"].lower()


def test_execute_rejects_non_spawning_engine(client: TestClient) -> None:
    """`local` hattı süreç başlatmaz → otonom sürüşte seçilemez."""
    run_id = _fresh_run(client)
    r = client.post(
        f"/api/orchestration/autodrive/{run_id}",
        json={"execute": True, "engine": "local"},
    )
    assert r.status_code == 503
    assert "süreç başlatmaz" in r.json()["detail"]


def test_autodrive_rejects_unknown_engine(client: TestClient) -> None:
    """Bilinmeyen motor adı sessizce varsayılana DÜŞMEZ → 400."""
    run_id = _fresh_run(client)
    r = client.post(
        f"/api/orchestration/autodrive/{run_id}",
        json={"execute": False, "engine": "sahtemotor"},
    )
    assert r.status_code == 400


def test_dry_run_uses_selected_engine(client: TestClient) -> None:
    """Seçilen motor DRY-RUN komutuna yansır (seçici gerçekten bağlı).

    AV (hunt) modu birden çok motoru destekler → seçici bağlantısı burada kanıtlanır
    (gemini'nin komuta yansıması "claude'a hardcode edilmemiş" demektir). Sür (drive)
    modu yalnız sertleştirilmiş `claude`'u destekler; bu ayrım test_drive_mode.py'de.
    """
    run_id = _fresh_run(client)
    body = client.post(
        f"/api/orchestration/autodrive/{run_id}",
        json={"execute": False, "engine": "gemini", "mode": "hunt"},
    ).json()
    assert body["engine"] == "gemini"
    assert body["command"][0] == "gemini"


def test_drive_dry_run_default_mode_is_drive(client: TestClient) -> None:
    """⚡ RUN ucu varsayılan SÜR modu: claude dry-run'ı MCP'li komut döndürür."""
    run_id = _fresh_run(client)
    body = client.post(
        f"/api/orchestration/autodrive/{run_id}",
        json={"execute": False, "engine": "claude"},
    ).json()
    assert body["mode"] == "drive"
    assert "--mcp-config" in body["command"] and "--safe-mode" not in body["command"]


def test_execute_true_is_human_only(client: TestClient) -> None:
    """Sürücü scope execute=true çağıramaz (403) — motor kendi sürücüsünü doğuramaz."""
    from app.web import driver_scope

    run_id = _fresh_run(client)
    token = driver_scope.mint(run_id)
    try:
        r = client.post(
            f"/api/orchestration/autodrive/{run_id}",
            json={"execute": True, "engine": "claude"},
            headers={
                driver_scope.DRIVER_TOKEN_HEADER: token,
                driver_scope.RUN_ID_HEADER: run_id,
            },
        )
        assert r.status_code == 403
    finally:
        driver_scope.revoke_run(run_id)
