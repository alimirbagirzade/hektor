"""`start-train.ps1` Kural 8 sözleşmesi — statik (betik ÇALIŞTIRILMAZ, eğitim başlamaz).

Kademe 2 bulgusu (2026-09-16, K8-b): betik `HEKTOR_TRAIN_SUPERVISED=1`'i KOŞULSUZ geçiyordu.
`app/main.py` bu değişkeni görünce taze-onay kapısını atlar ("üst katman onayı kullanıldı"),
ama betik hiçbir onay isteği açmıyor ve hiçbir onayı tüketmiyordu → belgelenen
`approval-approve <id>` adımı bu yolda hiçbir şeye bağlanmıyordu (v7/v8 koşuları böyle başladı).

Sözleşme:
- SUPERVISED yalnız açık `-Supervised` anahtarıyla geçer (nöbetçi kurtarması).
- Varsayılan yolda değişken ortamdan SİLİNİR (kabukta kalmış eski değer kapıyı atlatmasın).
- Onaysızlıkta alt süreç 3 ile çıkar; betik bunu yakalar, durum dosyasını siler (nöbetçi ölü
  koşuyu diriltmeye çalışmasın) ve onay komutunu yazdırır.
"""

from __future__ import annotations

import re
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_START_TRAIN = _SCRIPTS / "start-train.ps1"
_WATCHDOG = _SCRIPTS / "training-watchdog.ps1"


def test_supervised_anahtari_var() -> None:
    src = _START_TRAIN.read_text(encoding="utf-8")
    assert re.search(r"\[switch\]\$Supervised", src), "-Supervised anahtarı yok"


def test_supervised_kosulsuz_gecilmez() -> None:
    """`$env:HEKTOR_TRAIN_SUPERVISED = "1"` yalnız `-Supervised` dalında olabilir."""
    src = _START_TRAIN.read_text(encoding="utf-8")
    for line in src.splitlines():
        if re.search(r'\$env:HEKTOR_TRAIN_SUPERVISED\s*=\s*"1"', line):
            assert re.search(r"if\s*\(\s*\$Supervised\s*\)", line), (
                f"SUPERVISED koşulsuz atanıyor → Kural 8 kapısı atlanır: {line.strip()}"
            )


def test_varsayilan_yolda_supervised_ortamdan_silinir() -> None:
    src = _START_TRAIN.read_text(encoding="utf-8")
    assert "Remove-Item Env:HEKTOR_TRAIN_SUPERVISED" in src, (
        "kabukta kalmış HEKTOR_TRAIN_SUPERVISED temizlenmiyor → kapı sessizce atlanabilir"
    )


def test_onaysiz_cikis_yakalanir_ve_durum_dosyasi_silinir() -> None:
    """Alt süreç onaysızlıkta 3 ile çıkar; betik 'başladı' demeden durumu temizlemeli."""
    src = _START_TRAIN.read_text(encoding="utf-8")
    assert "WaitForExit" in src, "erken çıkış beklenmiyor → onaysız koşuya 'başladı' denir"
    assert re.search(r"Remove-Item\s+\$StatusFile", src), "durum dosyası temizlenmiyor"
    assert "approval-approve" in src, "onay komutu kullanıcıya gösterilmiyor"
    assert re.search(r"^\s*exit 3\s*$", src, re.MULTILINE), "onaysız durumda 3 ile çıkılmıyor"


def test_nobetci_kurtarmada_supervised_gecer() -> None:
    """Kurtarma koşusu yeni onay istemez (onay zaten tüketilmişti)."""
    src = _WATCHDOG.read_text(encoding="utf-8")
    assert "-Supervised" in src, "nöbetçi kurtarması onay bekleyip takılır"
    assert "-Resume" in src, "kurtarma sıfırdan başlatır (checkpoint kaybı)"
