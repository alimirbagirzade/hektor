"""Gelişmiş/gözlem grubu toggle — offline, statik dosya regresyon testleri.

Bağlam: "İzleme & sağlık" grubu (öğrenme/ajanlar/nöbetçi/sistem/ajan haritası) saf
gözlem panellerini içerir. Kullanıcı bunları günlük görmek istemedi → varsayılan gizli,
"⋯ gelişmiş" toggle ile açılır. GİZLEME yalnız NAVİGASYON katmanıdır: paneller, API
uçları ve arka plan işleri (öğrenme döngüsü, nöbetçi) her hâlükârda çalışır — bu yüzden
testler yalnız görünürlük kablolamasını doğrular, panel/uç varlığına dokunmaz.

NOT: RLM sekmesi bu grupta DEĞİL (Keşfet & sor grubunda) ve bu değişiklikle
KALDIRILMADI — [[rlm-paneli-silinmesin-2026-07-21]] dersi gereği korunur.
"""

from __future__ import annotations

import re
from pathlib import Path

_STATIC = Path(__file__).resolve().parents[1] / "app" / "web" / "static"


def _index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


def _appjs() -> str:
    return (_STATIC / "assets" / "app.js").read_text(encoding="utf-8")


def _appcss() -> str:
    return (_STATIC / "assets" / "app.css").read_text(encoding="utf-8")


def test_gelismis_toggle_var() -> None:
    html = _index()
    assert 'id="advancedToggle"' in html
    # "İzleme & sağlık" grubu gelişmiş işaretli (varsayılan gizli)
    assert 'group-advanced" data-group="izleme"' in html


def test_gelismis_grubu_css_ile_gizli() -> None:
    css = _appcss()
    assert ".group-advanced {\n  display: none;" in css
    assert ".groups.show-advanced .group-advanced" in css


def test_gelismis_toggle_js_ve_localstorage() -> None:
    js = _appjs()
    assert "achilles_advanced_on" in js  # tercih saklama anahtarı
    assert "applyAdvanced" in js
    assert "show-advanced" in js


def test_cekirdek_gruplar_gelismis_isaretlenmedi() -> None:
    """Günlük iş akışı grupları görünür kalmalı — yalnız 'izleme' gelişmiş olur."""
    html = _index()
    for grp in ("kesfet", "kutuphane", "trader", "egitim"):
        m = re.search(r'class="([^"]*)" data-group="' + grp + '"', html)
        assert m, f"{grp} grup butonu bulunamadı"
        assert "group-advanced" not in m.group(1), f"{grp} yanlışlıkla gelişmiş işaretlendi"


def test_rlm_sekmesi_korundu() -> None:
    """Bu değişiklik RLM'i SİLMEZ — geçmişte kaldırılıp geri getirilmişti (PR#111)."""
    html = _index()
    assert 'data-tab="rlm"' in html
    assert 'id="panel-rlm"' in html


def test_asset_cache_busted() -> None:
    html = _index()
    assert "app.js?v=" in html
    assert "app.css?v=" in html
