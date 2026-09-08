"""Windows otomatik-başlatma kurulumunun METİN sözleşmesi (çevrimdışı).

`start-server.ps1` üç görevi kurar: `HektorWeb` (açılışta web), `HektorUpdate`
(gece 03:00 güncelleme) ve `HektorTrainingWatchdog` (eğitim çökerse yeniden başlatır).
Bu adlar kaybolursa güvenlik ağı sessizce yok olur ama dokümanlar çalıştığını söylemeye
devam eder (2026-09-07'de tam olarak bu yaşandı: görevler ölü bir yolu gösteriyordu ve
sessizce başarısız oluyordu; Kural 2 — doğrulanmadan "çalışıyor" sayma).

NOT (2026-09-08): eski adlandırma döneminden kalan üç ölü görev makineden kaldırıldı
(`Get-ScheduledTask` + `schtasks` + görev dosyalarıyla doğrulandı) ve betikteki
`Remove-LegacyAutostart` temizleyicisi ile onu denetleyen testler kaldırıldı.

Bu testler betiği METİN olarak denetler; PowerShell çalıştırmaz (çevrimdışı, platform
bağımsız) — repoda `test_script_spawn_hardening` ile aynı yaklaşım.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start-server.ps1"


@pytest.fixture(scope="module")
def betik() -> str:
    assert _SCRIPT.is_file(), f"start-server.ps1 bulunamadı: {_SCRIPT}"
    return _SCRIPT.read_text(encoding="utf-8", errors="replace")


def test_gorev_adlari_korundu(betik: str) -> None:
    """Regresyon: üç görev adı da betikte kalmalı (kaybolursa güvenlik ağı sessizce gider)."""
    for gorev in ("HektorWeb", "HektorUpdate", "HektorTrainingWatchdog"):
        assert f'"{gorev}"' in betik, f"görev adı kayboldu: {gorev}"


def test_kurulum_gorevleri_kaydediyor(betik: str) -> None:
    """`Sync-Autostart` görevleri gerçekten kaydetmeli (Registry Run + Register-ScheduledTask)."""
    govde = betik.split("function Sync-Autostart", 1)[1].split("\nfunction ", 1)[0]
    assert "Register-ScheduledTask" in govde, "kurulum hiçbir görev kaydetmiyor"
    assert (
        "Set-ItemProperty" in govde or "New-ItemProperty" in govde
    ), "açılış (Registry Run) kaydı yazılmıyor"


# --------------------------------------------------------------------------- #
# uv sync eğitim paketlerini SİLMEMELİ (2026-09-07'de yaşandı)
# --------------------------------------------------------------------------- #
def test_verify_install_uv_sync_inexact_kullanir() -> None:
    """`uv sync` varsayılan olarak istenen küme DIŞINDAKİ paketleri KALDIRIR.

    Eğitim paketleri (torch/transformers/peft/accelerate) `train-cpu` adlı AYRI
    extra'dadır. `--inexact` olmadan `uv sync --extra dev` onları sessizce siler →
    sunucuyu yeniden başlatmak makinenin eğitim yeteneğini yok eder. 2026-09-07'de
    tam olarak bu oldu: `start-server.ps1 -Install` sonrası `hektor train --run`
    "Eksik paketler: ['torch','transformers','peft']" ile düştü.
    """
    betik = (_SCRIPT.parent / "verify-install.ps1").read_text(encoding="utf-8", errors="replace")
    sync_satirlari = [s for s in betik.splitlines() if "sync" in s and "$UvPath" in s]
    assert sync_satirlari, "verify-install.ps1 içinde uv sync çağrısı bulunamadı"
    for satir in sync_satirlari:
        assert (
            "--inexact" in satir
        ), f"uv sync --inexact kullanmıyor → eğitim paketleri silinir: {satir.strip()}"
