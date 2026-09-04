"""Phase 3 — Agent/Otomasyon dashboard statik-dosya smoke testleri (offline).

Frontend için ayrı JS test altyapısı yok → statik dosya içeriğini doğrularız:
sekme var mı, yalnız mevcut Phase 1/2 endpoint'leri kullanılıyor mu, tehlikeli
aksiyonlar confirm() istiyor mu ve dinamik içerik esc() ile kaçırılıyor mu (XSS).
Ağ/Ollama gerektirmez.
"""

from __future__ import annotations

from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "app" / "web" / "static"
_INDEX = _STATIC / "index.html"
_APPJS = _STATIC / "assets" / "app.js"
_APPCSS = _STATIC / "assets" / "app.css"


def _index() -> str:
    return _INDEX.read_text(encoding="utf-8")


def _appjs() -> str:
    return _APPJS.read_text(encoding="utf-8")


def test_static_files_exist() -> None:
    assert _INDEX.exists()
    assert _APPJS.exists()
    assert _APPCSS.exists()


def test_agents_tab_and_panel_present() -> None:
    html = _index()
    assert 'data-tab="agents"' in html
    assert 'id="panel-agents"' in html
    assert "Agents / Otomasyon" in html


def test_panel_has_all_sections() -> None:
    html = _index()
    for el_id in (
        "agStatusGrid",
        "agStopAllBtn",
        "agClearStopAllBtn",
        "agApprovalsTable",
        "agAgentsTable",
        "agRunsTable",
        "agRunDetail",
        "agTasksTable",
        "agEventsTable",
    ):
        assert f'id="{el_id}"' in html, f"panel eksik öğe: {el_id}"


def test_dashboard_uses_only_phase12_endpoints() -> None:
    js = _appjs()
    required = [
        '"/agents"',
        '"/agents/runs',
        "/agents/runs/",
        '"/automation/tasks',
        "/cancel",
        '"/approvals',
        "/approve",
        "/reject",
        '"/events',
        '"/supervisor/status"',
        "/supervisor/stop-all",
        "/supervisor/clear-stop-all",
        '"/healthz"',
    ]
    for token in required:
        assert token in js, f"app.js eksik endpoint: {token}"


def test_dangerous_actions_require_confirmation() -> None:
    js = _appjs()
    # STOP_ALL + approve + reject + cancel + clear → en az 5 confirm()
    assert js.count("window.confirm(") >= 5
    assert "dangerous agent action'larını durdurur" in js  # STOP_ALL
    assert "tek kullanımlıktır ve dangerous action" in js  # approve
    assert "approval request reddedilecek" in js  # reject
    assert "görev iptal edilecek" in js.lower()  # cancel task


def test_dynamic_content_is_escaped() -> None:
    js = _appjs()
    assert "function esc(" in js
    # Tehlike taşıyan API alanları esc() ile sarılmış olmalı (XSS savunması):
    for token in (
        "esc(ev.message",
        "esc(a.agent_id",
        "esc(a.action",
        "esc(r.error",
        "esc(ev.kind",
    ):
        assert token in js, f"escape edilmemiş olabilir: {token}"
    # Dashboard segmentinde esc() yoğun kullanılır.
    seg = js[js.index("AGENTS / OTOMASYON dashboard") :]
    assert seg.count("esc(") >= 25


def test_no_inline_onclick_csp_safe() -> None:
    # CSP-safe: satır butonları delegasyonla bağlanır (inline onclick yok).
    html = _index()
    assert "onclick=" not in html
    js = _appjs()
    assert "data-ag-action" in js  # delegasyon deseni


def test_run_timeline_and_task_render_present() -> None:
    js = _appjs()
    assert "loadAgRunDetail" in js
    assert "ag-timeline" in js
    assert "loadAgTasks" in js
    assert "loadAgApprovals" in js


def test_stopall_button_styled_red() -> None:
    css = _APPCSS.read_text(encoding="utf-8")
    assert ".ag-stopall-btn" in css
    assert "#dc2626" in css  # kasıtlı kırmızı kill-switch


# --- Phase 4D-1: EĞİTİM tabı eğitim-başlat onayı + needs_approval gösterimi ---
def test_training_start_requires_confirmation() -> None:
    """EĞİTİM tabındaki start butonu tek-tık değil; önce window.confirm() ister."""
    js = _appjs()
    # startTrainBtn handler'ı confirm metnini içerir (parçalar bitişik literal).
    assert "gerçek LoRA training başlatabilir" in js
    assert "Devam etmek istiyor musunuz?" in js
    # confirm, start handler'ı içinde ve training POST'undan ÖNCE olmalı.
    h = js.index("startTrainBtn.addEventListener")
    post = js.index('api("/training/run"', h)
    assert "window.confirm(" in js[h:post], (
        "start handler training POST'undan önce confirm istemiyor"
    )


def test_training_start_shows_needs_approval() -> None:
    """Backend needs_approval dönerse UI approval_id + onay komutunu gösterir."""
    js = _appjs()
    assert 'status === "needs_approval"' in js
    assert "Approval ID:" in js
    assert "approve_command" in js
    assert "approval-approve" in js
