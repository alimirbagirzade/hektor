"""15 · AJAN HARİTASI — frontend statik-dosya duman testleri (offline).

Frontend için ayrı JS test altyapısı yok → statik dosya içeriğini doğrularız:
sekme + panel var mı, yalnız mevcut salt-okuma/orkestrasyon uçları kullanılıyor mu,
tehlikeli ⚡ tetik execute'tan ÖNCE confirm() istiyor mu (Kural 8), dinamik içerik
esc()/textContent ile kaçırılıyor mu (XSS) ve CSP-güvenli mi (inline style/onclick yok).
Ağ/Ollama gerektirmez. Backend grafik zaten `test_agent_graph.py`'de test'li.
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


def _appcss() -> str:
    return _APPCSS.read_text(encoding="utf-8")


def _panel_section() -> str:
    """index.html'deki panel-agentmap <section> gövdesi."""
    html = _index()
    start = html.index('id="panel-agentmap"')
    end = html.index("</section>", start)
    return html[start:end]


# ── nav + panel ──────────────────────────────────────────────────────────────
def test_tab_and_panel_present() -> None:
    html = _index()
    assert 'data-tab="agentmap"' in html
    assert 'id="panel-agentmap"' in html
    # ETİKETE göre eşle, sekme NUMARASINA pinleme: sekme eklenip çıkarıldığında numara
    # kayabiliyor (RLM kaldırılınca 15→14 oldu, geri gelince 15). Aynı ders: PR#103 asset
    # cache-bust testi versiyona-duyarsız yapılmıştı.
    import re

    assert re.search(r'<span class="tab-num">\d+</span> · AJAN HARİTASI', html)


def test_panel_has_all_sections() -> None:
    html = _index()
    for el_id in (
        "amTriggerBtn",
        "amRefreshBtn",
        "amStatusLine",
        "amDryRun",
        "amDryRunCmd",
        "amLegend",
        "amCanvasWrap",
        "amCanvas",
        "amDetail",
    ):
        assert f'id="{el_id}"' in html, f"panel eksik öğe: {el_id}"


def test_assets_cache_busted() -> None:
    # Versiyona-DUYARSIZ: asset'ler cache-bust query'siyle sunuluyor mu (tam numara pinlenmez —
    # her UI değişikliğinde ?v artar; sabit "v7" beklemek kırılgandı, görsel sadeleştirmeyi kırdı).
    html = _index()
    assert "app.js?v=" in html
    assert "app.css?v=" in html


# ── app.js kablolaması ───────────────────────────────────────────────────────
def test_js_wiring_present() -> None:
    js = _appjs()
    assert "function loadAgentMap" in js
    assert 'if (name === "agentmap") loadAgentMap();' in js
    # izleme grubuna eklenmiş
    assert '"about", "agentmap"' in js


def test_js_uses_only_readonly_and_orchestration_endpoints() -> None:
    js = _appjs()
    for token in ('"/agents/graph"', '"/orchestration/start"', "/orchestration/autodrive/"):
        assert token in js, f"app.js eksik uç: {token}"


def _trigger_seg() -> str:
    """`amTriggerTraining` gövdesi (tetik akışının statik denetimi için)."""
    js = _appjs()
    h = js.index("function amTriggerTraining")
    return js[h : js.index("function loadAgentMap")]


def test_trigger_confirms_before_execute() -> None:
    """⚡ tetik, gerçek spawn'dan (execute:true) ÖNCE onay diyaloğunu AÇAR (Kural 8).

    `window.confirm` yerine gerçek bir modal kullanılır: diyalogda hangi motorun koşacağı
    ve abonelik kotası uyarısı GÖSTERİLEBİLSİN diye (düz confirm metin kutusu buna elverişsiz).
    Güvenlik sözleşmesi aynı, hatta daha güçlü: `execute` değeri diyaloğun çözümüdür.
    """
    seg = _trigger_seg()
    assert "amConfirmOpen(" in seg
    # Diyalog, autodrive POST'undan ÖNCE açılmalı ve execute değeri onun sonucu (go) olmalı.
    assert seg.index("amConfirmOpen(") < seg.index("/orchestration/autodrive/")
    assert "execute: go" in seg
    # DRY-RUN yolu: onaylanmazsa komut gösterilir (spawn yok).
    assert "amShowDryRun" in seg


def test_agentmap_trigger_never_hardcodes_execute_true() -> None:
    """⚡ tetikte sabit `execute: true` YOK — tek yol onay diyaloğunun sonucudur (`go`).

    Tek-tıkla geri alınamaz iş başlamasın diye: diyalog atlanamaz olmalı, yani `execute`
    değeri bu akışta diyalogdan bağımsız olarak `true` yazılamamalı.
    """
    seg = _trigger_seg()
    for bad in ("execute: true", "execute:true", '"execute": true'):
        assert bad not in seg, f"onay atlanabilir: sabit {bad!r} bulundu"


def test_every_execute_true_site_is_human_gated() -> None:
    """app.js'teki HER `execute:true` çağrısı bir insan kapısının ARDINDA olmalı.

    ⚡ butonu tek başına yeterli değil: 12·ORKESTRASYON sekmesinde de bir otonom-sürüş
    tetiği var. Gerçek spawn'a giden HİÇBİR yol kapısız kalmamalı — bu test yeni bir
    tetik eklendiğinde (kapı unutulursa) patlar.
    """
    import re

    # Yorum satırlarını at: açıklama metinlerinde geçen "execute:true" ifadesi bir çağrı
    # DEĞİLDİR (aksi halde test kendi dokümantasyonuna takılır).
    js = re.sub(r"^\s*//.*$", "", _appjs(), flags=re.MULTILINE)
    gates = ("window.confirm(", "amConfirmOpen(")
    for m in re.finditer(r"execute:\s*true", js):
        # Çağrıdan geriye doğru makul bir pencerede insan kapısı aranır.
        window = js[max(0, m.start() - 1200) : m.start()]
        assert any(g in window for g in gates), (
            f"kapısız spawn yolu: konum {m.start()} — execute:true öncesinde "
            "confirm/onay diyaloğu yok"
        )


def test_confirm_dialog_shows_engine_quota_and_safety() -> None:
    """Onay diyaloğu AÇIKÇA gösterir: hangi motor, kota uyarısı, 'eğitim başlamaz' güvencesi."""
    html = _index()
    # Modal panel DIŞINDA durur (paneller display:none ile gizlenir) → index sonundan bak.
    seg = html[html.index('id="amConfirmModal"') :]
    # üç zorunlu bilgi alanı
    for el_id in ("amConfirmEngine", "amConfirmQuota", "amConfirmSafe"):
        assert f'id="{el_id}"' in seg, f"onay diyaloğunda eksik alan: {el_id}"
    # "eğitim BAŞLAMAZ + taze insan onayında durur" güvencesi metinde geçmeli
    assert "Eğitim BAŞLAMAZ" in seg
    assert "onay" in seg and "DURUR" in seg
    # motor/kota alanları JS'te doldurulur
    js = _appjs()
    assert "quota_warning" in js


# ── REGRESYON: kazara spawn (geliştirme sırasında FİİLEN yaşandı) ────────────
# Canlı doğrulamada ⚡ diyaloğu, "başlat" butonu ODAKLANMIŞ olduğu için hiç bilinçli
# tıklama olmadan çözüldü ve 5 gerçek `claude -p` süreci doğdu (abonelik kotası yandı).
# Aşağıdaki dört test o üç savunmayı da sabitler; biri kalkarsa CI patlar.
def test_confirm_never_autofocuses_destructive_button() -> None:
    """Yıkıcı butona ODAK VERİLMEZ — odaklı buton tek Enter'la motor doğurur."""
    js = _appjs()
    h = js.index("function amConfirmOpen")
    seg = js[h : js.index("function amConfirmClose")]
    assert "amConfirmGoBtn" in seg  # buton yine de kilitlenmek için bulunur
    assert 'getElementById("amConfirmCancelBtn")' in seg and "cancel.focus()" in seg
    # "başlat" butonuna odak VERİLMEMELİ
    assert "go.focus()" not in seg


def test_confirm_requires_explicit_acknowledgement() -> None:
    """Başlat butonu, onay kutusu işaretlenmeden AÇILMAZ (iki bilinçli hareket)."""
    html = _index()
    seg = html[html.index('id="amConfirmModal"') :]
    assert 'id="amConfirmAck"' in seg, "onay kutusu yok"
    # buton HTML'de varsayılan olarak devre dışı (fail-closed)
    go_idx = seg.index('id="amConfirmGoBtn"')
    assert "disabled" in seg[go_idx : go_idx + 200], "başlat butonu varsayılan kilitli olmalı"
    js = _appjs()
    h = js.index("function amWireConfirm")
    wire = js[h : js.index("function amSetLive")]
    assert "go.disabled = !ack.checked" in wire


def test_confirm_go_rejects_untrusted_and_unacked_events() -> None:
    """Başlat: kilit açık + kutu işaretli + GERÇEK kullanıcı olayı — üçü birden şart."""
    js = _appjs()
    h = js.index("function amWireConfirm")
    wire = js[h : js.index("function amSetLive")]
    assert "if (go.disabled) return;" in wire
    assert "!ack.checked" in wire
    assert "ev.isTrusted === false" in wire


def test_confirm_resets_acknowledgement_on_open() -> None:
    """Diyalog her açılışta sıfırlanır — önceki işaret bir sonrakini açmasın."""
    js = _appjs()
    h = js.index("function amConfirmOpen")
    seg = js[h : js.index("function amConfirmClose")]
    assert "ack.checked = false" in seg
    assert "go.disabled = true" in seg


def test_trigger_has_reentrancy_lock() -> None:
    """Sürüş devredeyken ⚡ yeniden tetiklenemez (üst üste motor doğurmasın)."""
    seg = _trigger_seg()
    assert "if (amDriving)" in seg
    assert "if (amConfirmResolve) return;" in seg


def test_confirm_dialog_defaults_to_cancel() -> None:
    """Diyalog güvenli tarafa varsayılan: Esc / arka plan / kapat → İPTAL (spawn yok)."""
    js = _appjs()
    h = js.index("function amWireConfirm")
    seg = js[h : js.index("function amSetLive")]
    assert "amConfirmClose(false)" in seg
    assert 'ev.key === "Escape"' in seg
    # diyalog hiç yoksa da çalıştırma (fail-closed)
    assert "return Promise.resolve(false)" in _appjs()


# ── motor seçici ─────────────────────────────────────────────────────────────
def test_engine_selector_present_and_wired() -> None:
    """Motor seçici var, /api/engines'ten beslenir ve seçim autodrive'a gönderilir."""
    seg = _panel_section()
    for el_id in ("amEngineSel", "amEngineNote", "amEngineRescanBtn"):
        assert f'id="{el_id}"' in seg, f"motor seçicide eksik öğe: {el_id}"
    js = _appjs()
    assert '"/engines"' in js or '/engines"' in js
    assert "engine: eng.name" in js, "seçilen motor autodrive isteğine konmalı"


def test_uninstalled_engine_is_disabled_in_selector() -> None:
    """Kurulu olmayan motor <option disabled> → seçilemez; ipucu gösterilir."""
    js = _appjs()
    h = js.index("function amRenderEngines")
    seg = js[h : js.index("function amRenderEngineNote")]
    assert "o.disabled = !e.selectable" in seg
    # tetik ayrıca seçilemeyen motoru reddeder (çift kapı)
    assert "!eng.selectable" in _trigger_seg()
    # "nasıl kurulur" ipucu — kimlik formu DEĞİL
    assert "install_hint" in js


def test_selector_never_renders_credentials() -> None:
    """Seçici kimlik bilgisi göstermez: giriş durumu 'bilinmiyor' olarak sunulur."""
    js = _appjs()
    h = js.index("function amRenderEngineNote")
    seg = js[h : js.index("function amLoadEngines")]
    assert "bilinmiyor" in seg
    for bad in ("password", "api_key", "apiKey", "token"):
        assert bad not in seg, f"motor notunda kimlik alanı: {bad}"


# ── canlı durum + görünür DURDUR ─────────────────────────────────────────────
def test_live_strip_and_visible_stop_button() -> None:
    """Koşu sırasında canlı durum + GÖRÜNÜR stop-all butonu bulunur."""
    seg = _panel_section()
    for el_id in ("amLive", "amLiveText", "amStopBtn"):
        assert f'id="{el_id}"' in seg, f"canlı şeritte eksik öğe: {el_id}"
    js = _appjs()
    h = js.index("function amStopAll")
    stop = js[h : js.index("function amTriggerTraining")]
    assert '"/supervisor/stop-all"' in stop
    # durdurma da tek tık olmasın
    assert "window.confirm(" in stop


def test_live_strip_follows_real_run_status() -> None:
    """Canlı şerit DB aşamasına değil GERÇEK motor süreç kaydına bakar."""
    js = _appjs()
    h = js.index("function amRefreshGate")
    seg = js[h : js.index("function amApproveAndTrain")]
    assert "run.driver_running" in seg
    assert "amSetLive(false)" in seg


def test_control_edges_have_distinct_rag_motor_style() -> None:
    assert 'e.kind === "control"' in _appjs()
    assert ".am-edge-control" in _appcss()


def test_dynamic_content_is_escaped() -> None:
    js = _appjs()
    h = js.index("// ---------- 15 · AJAN HARİTASI")
    seg = js[h : js.index("// ---------- init ----------")]
    # API'den gelen alanlar esc() ile kaçırılmalı (XSS savunması).
    for token in ("esc(n.name", "esc(n.trigger)", "esc(x)", "esc(lm.label)", "esc(e.from)"):
        assert token in seg, f"escape edilmemiş olabilir: {token}"
    # Kuru-çalıştırma komutu innerHTML DEĞİL textContent ile yazılmalı.
    assert "el.textContent = cmd" in seg
    assert seg.count("esc(") >= 12


def test_csp_safe_no_inline_style_or_onclick() -> None:
    # Panel CSP-güvenli olmalı: satır-içi style / onclick yok (style-src 'self').
    section = _panel_section()
    assert "onclick=" not in section
    assert "style=" not in section
    # DRY-RUN gizleme inline style değil `hidden` özniteliği ile.
    assert " hidden>" in section or " hidden " in section


# ── "DERLİ TOPLU" kart yerleşimi (JS + CSS) ──────────────────────────────────
def test_js_card_layout_structure() -> None:
    js = _appjs()
    # kartlar + dik-açılı yol + ana ajan alt buton + aktivasyon yolu
    for token in ("amOrthPath", "am-card", "am-card-edge", "am-mainbar", "am-edge-activate"):
        assert token in js, f"kart-yerleşim eksik: {token}"
    # ana ajan şeritlerden çıkarılıp alt butona alınır + tıkla → ⚡ tetik
    assert "if (n.is_main)" in js
    assert 'closest(".am-mainbar")' in js and "amTriggerTraining()" in js


def test_js_running_orbit_and_linked() -> None:
    """Çalışan ajan = kart etrafında GEZEN LED; ona bağlı komşular = 'ilişkili' çerçeve."""
    js = _appjs()
    # LED halkası bir <path> üzerinde döner (pathLength=100 → kart boyutundan bağımsız)
    assert "amRoundRectPath" in js
    assert "am-card-orbit" in js and "am-mainbar-orbit" in js
    assert 'pathLength="100"' in js
    # "ilişkili" küme kenarlardan (aktivasyon yolları dahil, DOM'dan) hesaplanır
    assert "am-linked" in js
    seg = js[js.index("function amApplyStatus") : js.index("function amUpdateStatusLine")]
    assert "linked" in seg
    assert 'querySelectorAll(".am-edge")' in seg  # sentetik aktivasyon yolları da sayılsın
    assert 'status === "running"' in seg


def test_css_running_orbit_and_linked() -> None:
    css = _appcss()
    # gezen LED: yalnız çalışırken görünür + dönen dash + parıltı
    assert ".am-node.status-running .am-card-orbit" in css
    assert ".am-mainbar.status-running .am-mainbar-orbit" in css
    assert "@keyframes am-orbit" in css
    # parıltı iki-stroke ile (geniş yarı-saydam hale + ince LED)
    assert ".am-node.status-running .am-card-orbit-glow" in css
    # TERCİH KAPISI: animasyonlu LED'lerde kare-başı filtre rasterizasyonundan kaçınılır
    # (GPU'suz + eşzamanlı CPU-only eğitim) → parıltı iki-stroke ile yapılmalı, filter ile değil.
    # Yorumlar ayıklanır (gerekçe metninin kendisi "drop-shadow" kelimesini içeriyor).
    import re

    seg = re.sub(r"/\*.*?\*/", "", css[css.index('/* ── "GEZEN LED"') :], flags=re.S)
    assert "drop-shadow" not in seg, "animasyonlu LED'lerde filtre yerine iki-stroke kullan"
    # ilişkili komşu çerçevesi
    assert ".am-node.am-linked .am-card" in css
    # erişilebilirlik: hareket azaltılınca dönmesin ama "devrede" bilgisi kalsın
    seg = css[css.index("@media (prefers-reduced-motion: reduce)", css.index(".am-card-orbit")) :]
    assert "am-card-orbit" in seg


def test_css_light_path_and_status_colors() -> None:
    css = _appcss()
    # akan kenar ("ışıklı yol") animasyonu + aktivasyon (yeşil "devreye sokar") yolu
    assert ".am-edge.flowing" in css
    assert ".am-edge-activate" in css
    assert "@keyframes am-flow" in css
    # çalışan durum nabzı
    assert "@keyframes am-pulse" in css
    # kart sol-kenar durum rengi + ana ajan yeşil buton + tehlikeli kesik kenarlık
    assert ".am-node.status-running .am-card-edge" in css
    assert ".am-mainbar-box" in css
    assert ".am-node.am-danger .am-card" in css
    # Okabe-Ito CVD-güvenli palet
    assert "var(--pos)" in css and "var(--ok)" in css and "var(--warn)" in css
    # geniş grafik yatay kaydırma kabında
    assert ".am-canvas-wrap" in css and "overflow-x: auto" in css


# ── web ucu (uçtan uca, offline) ─────────────────────────────────────────────
def test_graph_endpoint_shape() -> None:
    import pytest

    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.web.server import app

    client = TestClient(app)
    r = client.get("/api/agents/graph")
    assert r.status_code == 200
    body = r.json()
    assert body["main_agent"] == "orchestration-autodrive"
    assert isinstance(body["nodes"], list) and body["nodes"]
    assert isinstance(body["edges"], list)
    assert isinstance(body["groups"], list) and body["groups"]


# ── İKİ-TIK eğitim kapısı (⚡ RUN → onay → gerçek eğitim) ─────────────────────
def test_gate_box_present() -> None:
    """Onay kapısı kutusu + butonu panelde var ve varsayılan GİZLİ (hidden)."""
    section = _panel_section()
    assert 'id="amGate"' in section
    assert 'id="amApproveTrainBtn"' in section
    assert 'id="amGateInfo"' in section
    # kutu varsayılan gizli olmalı (hat kapıda değilken görünmesin)
    gate = section[section.index('id="amGate"') :]
    assert gate[: gate.index(">")].find("hidden") != -1, "amGate varsayılan hidden olmalı"


def test_gate_only_opens_when_blocked_at_approval() -> None:
    """Kutu YALNIZ koşu approval/train aşamasında BLOCKED iken açılır (yanlışlıkla açılmasın)."""
    js = _appjs()
    seg = js[js.index("function amRefreshGate") : js.index("function amApproveAndTrain")]
    assert '"/orchestration/runs?limit=1"' in seg
    assert 'run.status === "blocked"' in seg
    assert 'run.current_stage === "approval"' in seg
    assert 'run.current_stage === "train"' in seg
    # durum okunamazsa kapı KAPALI kalmalı (güvenli taraf)
    assert "box.hidden = true" in seg


def test_training_requires_human_confirm_and_uses_gated_endpoint() -> None:
    """Gerçek eğitim: confirm() ŞART ve mevcut taze-onay kapılı /training/run kullanılır."""
    js = _appjs()
    seg = js[js.index("function amApproveAndTrain") : js.index("function amStartPoll")]
    assert "window.confirm(" in seg
    # confirm, eğitim POST'undan ÖNCE gelmeli
    assert seg.index("window.confirm(") < seg.index('"/training/run"')
    # needs_approval → insan yüzeyinden onay → tekrar çağır (onay tüketilir)
    assert 'status === "needs_approval"' in seg
    assert "/approvals/" in seg and "/approve" in seg
    # Kural 8: STOP_ALL / blocked yanıtı kullanıcıya bildirilmeli
    assert 'status === "blocked"' in seg


def test_engine_cannot_selfapprove_no_autoapprove_in_autodrive() -> None:
    """P1 güvenlik: ⚡ otonom sürüş yolu onay ucuna DOKUNMAZ — onay yalnız insan butonunda.

    Motorun kendi eğitimini onaylaması roadmap'te bloklayıcı açık; ⚡ (autodrive) akışında
    /approvals/*/approve çağrısı BULUNMAMALI.
    """
    js = _appjs()
    trigger = js[js.index("function amTriggerTraining") : js.index("function loadAgentMap")]
    assert "/approvals/" not in trigger, "⚡ otonom yol onay veremez (motor kendini onaylamasın)"
    assert '"/training/run"' not in trigger, "⚡ otonom yol gerçek eğitimi başlatamaz"


# ── Geniş panel + genişliğe uyarlanan satır kapasitesi ───────────────────────
def test_panel_is_wide_and_recenters() -> None:
    """Harita paneli `main`in 1000px kolonundan taşar (tüm ajanlar aynı anda görünsün)."""
    css = _appcss()
    seg = css[css.index("#panel-agentmap") :]
    assert "--am-wide" in seg
    assert "100vw" in seg  # viewport genişliğine göre taşar
    assert "margin-left: calc(50%" in seg  # negatif margin ile yeniden merkezlenir
    # dar ekranda taşırma kapanmalı (yatay sayfa kayması olmasın)
    assert "@media (max-width: 1060px)" in css


def test_per_row_adapts_to_canvas_width() -> None:
    """Satır başına kart sayısı tuvalin gerçek genişliğinden hesaplanır (sabit 6 değil)."""
    js = _appjs()
    seg = js[js.index("function amPerRowCap") : js.index("function amLayout")]
    assert "clientWidth" in seg
    assert "AM_CARD_W" in seg and "AM_CARD_GX" in seg
    assert "AM_MIN_PER_ROW" in seg and "AM_MAX_PER_ROW" in seg
    # yerleşim bu kapasiteyi kullanmalı (eski sabit sınır kalmasın)
    lay = js[js.index("function amLayout") : js.index("function amRoundRectPath")]
    assert "amPerRowCap()" in lay
    assert "perRowCap" in lay


def test_resize_rebuilds_layout() -> None:
    """Pencere genişleyince yerleşim yeniden kurulur (imza sıfırlanır)."""
    js = _appjs()
    assert '"resize"' in js
    seg = js[js.index('addEventListener("resize"') :]
    assert "amSig = null" in seg[:400]
