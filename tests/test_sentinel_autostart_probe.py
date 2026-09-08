"""Sentinel `autostart` probe'u — ÖLÜ NÖBETÇİ tespiti (çevrimdışı).

Gerçek olay (2026-09-07): yeniden adlandırma Windows zamanlanmış görevlerini taşımadı.
Üç eski görev SİLİNMİŞ bir yolu gösteriyordu, `Hektor*` görevlerinin hiçbiri kayıtlı
değildi → eğitim nöbetçisi, günlük güncelleme ve web otomatik başlatma üçü de ölüydü.
Görevler listede "Ready" göründüğü ve dokümanlar çalıştıklarını söylediği için kimse
fark etmedi; elle bulundu. Bu probe o sınıfı sistemin KENDİSİNİN yakalaması içindir
(Kural 2: doğrulanmadan "çalışıyor" sayma).

2026-09-08: ölü görevler makineden kaldırıldı ve `Hektor*` görevlerinin üçü de kayıtlı;
bu yüzden testler artık eski görev ADLARIYLA değil, aynı ariza SINIFLARIYLA yazılıdır
(nöbetçi yok / betik yolu ölü / beklenen görev eksik).

Probe okuyucusu enjekte edilebilir → testler PowerShell çağırmaz, platform bağımsızdır.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.monitoring import sentinel
from app.monitoring.sentinel import _task_script_path, probe_autostart

_WD = "HektorTrainingWatchdog"


@pytest.fixture(autouse=True)
def _windows_varsay(monkeypatch: pytest.MonkeyPatch):
    """Probe Windows dışında skip döner; mantığı sınamak için nt varsay."""
    monkeypatch.setattr(sentinel.os, "name", "nt", raising=False)


def _args(script: str | Path) -> str:
    return f'-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "{script}"'


def test_hic_gorev_yoksa_fail() -> None:
    """Kayıt yoksa çökme-kurtarma da yoktur."""
    r = probe_autostart(lambda: {})
    assert r.status == "fail"
    assert "kayıtlı değil" in r.detail.lower()
    assert "start-server.ps1" in r.advice


def test_nobetci_eksikse_fail(tmp_path: Path) -> None:
    """ASIL OLAY sınıfı: başka görevler kayıtlı ama EĞİTİM NÖBETÇİSİ yok.

    Görev listesi dolu olduğu için yüzeyden "otomasyon kurulu" görünür; eğitim
    çökerse yeniden başlatan yoktur → fail (sessiz güvenlik ağı kaybı).
    """
    betik = tmp_path / "run-web-service.ps1"
    betik.write_text("# test", encoding="utf-8")
    r = probe_autostart(lambda: {"HektorWeb": _args(betik), "HektorUpdate": _args(betik)})
    assert r.status == "fail"
    assert "nöbetçi" in r.detail.lower()


def test_nobetci_var_olmayan_betigi_gosteriyorsa_fail(tmp_path: Path) -> None:
    """Yeniden adlandırma sonrası tam olarak bu oldu: yol artık yok."""
    yok = tmp_path / "silinmis-depo" / "scripts" / "training-watchdog.ps1"
    r = probe_autostart(lambda: {_WD: _args(yok)})
    assert r.status == "fail"
    assert "var olmayan" in r.detail.lower()


def test_eksik_gorev_varsa_warn(tmp_path: Path) -> None:
    """Nöbetçi sağlam ama beklenen görevlerden biri kayıtlı değil → uyarı.

    2026-09-08'de gerçekleşti: yükseltilmemiş kurulum `HektorUpdate`'i kaydedemedi,
    betik yine de "güncelleme her gece 03:00'de yapılır" dedi. Eksiklik görünür olmalı.
    """
    betik = tmp_path / "training-watchdog.ps1"
    betik.write_text("# test", encoding="utf-8")
    r = probe_autostart(lambda: {_WD: _args(betik), "HektorWeb": _args(betik)})
    assert r.status == "warn"
    assert "HektorUpdate" in r.detail


def test_tam_kurulum_ok(tmp_path: Path) -> None:
    """Üç görev kayıtlı, kalıntı yok, betik mevcut → ok."""
    betik = tmp_path / "training-watchdog.ps1"
    betik.write_text("# test", encoding="utf-8")
    r = probe_autostart(
        lambda: {"HektorWeb": _args(betik), "HektorUpdate": _args(betik), _WD: _args(betik)}
    )
    assert r.status == "ok"


def test_windows_disinda_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux/macOS'ta görev kaydı kavramı yok → sessiz skip, fail DEĞİL."""
    monkeypatch.setattr(sentinel.os, "name", "posix", raising=False)
    assert probe_autostart(lambda: {}).status == "skip"


def test_okuyucu_patlarsa_skip() -> None:
    """Probe savunmacı: okuma hatası nöbetçiyi düşürmemeli (_guard sözleşmesi)."""

    def _patla() -> dict[str, str]:
        raise RuntimeError("powershell yok")

    assert probe_autostart(_patla).status == "skip"


def test_betik_yolu_ayiklama() -> None:
    """-File argümanı tırnaklı ve tırnaksız biçimde çözülmeli."""
    tirnakli = r'-File "C:\\a b\\x.ps1"'
    assert _task_script_path(tirnakli) == r"C:\\a b\\x.ps1"
    tirnaksiz = r"-File C:\\a\\x.ps1"
    assert _task_script_path(tirnaksiz) == r"C:\\a\\x.ps1"
    assert _task_script_path("-NoProfile") == ""


def test_varsayilan_probe_listesinde_kayitli() -> None:
    """Probe kaydedilmezse hiç koşmaz — regresyon kilidi."""
    assert probe_autostart in sentinel.default_probes()
