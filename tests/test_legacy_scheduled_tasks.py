"""Achilles → Hektor: ESKİ Windows zamanlanmış görevleri temizlenmeli (çevrimdışı).

Bulgu (2026-09-07, çalışan makinede ölçüldü): yeniden adlandırma ortam
değişkenlerini ve SQLite dosya adını taşıdı (bkz. `test_legacy_env_migration`)
ama **Windows görevlerini taşımadı**. Makinede kalan üç görev
(`AchillesTrainingWatchdog` / `AchillesUpdate` / `AchillesWeb`) SİLİNMİŞ bir yolu
gösteriyordu ve sessizce başarısız oluyordu; `Hektor*` görevlerinin hiçbiri kayıtlı
değildi. Sonuç: eğitim nöbetçisi, günlük güncelleme ve web otomatik başlatma **üçü de
ölü** — ama dokümanlar çalıştıklarını söylemeye devam ediyordu (sessiz güvenlik ağı
kaybı; Kural 2'nin ruhu: doğrulanmadan "çalışıyor" sayma).

Bu testler betiği METİN olarak denetler; PowerShell çalıştırmaz (çevrimdışı, platform
bağımsız) — repoda `test_script_spawn_hardening` ile aynı yaklaşım.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start-server.ps1"
_LEGACY_TASKS = ("AchillesWeb", "AchillesUpdate", "AchillesTrainingWatchdog")


@pytest.fixture(scope="module")
def betik() -> str:
    assert _SCRIPT.is_file(), f"start-server.ps1 bulunamadı: {_SCRIPT}"
    return _SCRIPT.read_text(encoding="utf-8", errors="replace")


def test_eski_gorev_adlari_tanimli(betik: str) -> None:
    """Üç eski görev adı da betikte açıkça listelenmeli."""
    assert "$LegacyTaskNames" in betik
    for eski in _LEGACY_TASKS:
        assert eski in betik, f"eski görev adı betikte yok: {eski}"


def test_temizleyici_fonksiyon_var_ve_kaldiriyor(betik: str) -> None:
    """`Remove-LegacyAutostart` eski görevleri Unregister etmeli."""
    assert "function Remove-LegacyAutostart" in betik
    govde = betik.split("function Remove-LegacyAutostart", 1)[1].split("\nfunction ", 1)[0]
    assert "Unregister-ScheduledTask" in govde, "eski görev kaldırılmıyor"
    assert "$LegacyTaskNames" in govde, "eski görev listesi kullanılmıyor"
    assert "Remove-ItemProperty" in govde, "eski Registry Run anahtarı kaldırılmıyor"


def test_kurulumda_temizlik_cagriliyor(betik: str) -> None:
    """Sync-Autostart eski kayıtları temizlemeden yeni görevleri kurmamalı."""
    govde = betik.split("function Sync-Autostart", 1)[1].split("\nfunction ", 1)[0]
    assert "Remove-LegacyAutostart" in govde, "kurulumda eski kayıt temizliği çağrılmıyor"


def test_kaldirmada_da_temizleniyor(betik: str) -> None:
    """Uninstall eski kayıtları da almalı; aksi halde kaldırma yarım kalır."""
    govde = betik.split("function Uninstall-Autostart", 1)[1].split("\nfunction ", 1)[0]
    assert "Remove-LegacyAutostart" in govde


def test_sapma_tespiti_eski_gorevi_sayar(betik: str) -> None:
    """Repair-Autostart, duran bir eski görevi SAPMA saymalı — yoksa kırık görev
    'nöbetçi var' yanılgısı üretmeye devam eder."""
    govde = betik.split("function Repair-Autostart", 1)[1].split("\nfunction ", 1)[0]
    assert "$LegacyTaskNames" in govde, "onarım eski görevi sapma saymıyor"


def test_yeni_gorev_adlari_korundu(betik: str) -> None:
    """Regresyon: yeni (Hektor) görev adları kaybolmamalı."""
    for yeni in ("HektorWeb", "HektorUpdate", "HektorTrainingWatchdog"):
        assert f'"{yeni}"' in betik, f"yeni görev adı kayboldu: {yeni}"


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
        assert "--inexact" in satir, (
            f"uv sync --inexact kullanmıyor → eğitim paketleri silinir: {satir.strip()}"
        )


# --------------------------------------------------------------------------- #
# Gece güncellemesi eğitim yığınını YÖNETMELİ (2026-09-09 ve 2026-09-11'de yaşandı)
# --------------------------------------------------------------------------- #
_UPDATE = Path(__file__).resolve().parents[1] / "update.ps1"


@pytest.fixture(scope="module")
def update_betik() -> str:
    assert _UPDATE.is_file(), f"update.ps1 bulunamadı: {_UPDATE}"
    return _UPDATE.read_text(encoding="utf-8", errors="replace")


def test_update_sync_train_cpu_extrasini_kapsar(update_betik: str) -> None:
    """Gece 03:00 görevi eğitim yığınını yönetmeli — yoksa kilitle ayrışır.

    `train-cpu` senkron kümesinin DIŞINDA kalırsa torch/transformers/peft
    yönetilmeyen olur, ama ORTAK bağımlılıkları (tokenizers/safetensors) kilide
    çekilir → kurulu transformers ile çift bozulur. 2026-09-09 ve 2026-09-11
    03:00'te tam bu oldu: tokenizers 0.23.2 → 0.22.2 düştü, `import transformers`
    kırıldı. 09-11'de zarar eğitimin kendisinde değil, 39 saatlik koşuyu
    değerlendirecek `lora-eval` adımındaydı — eğitim koruması bitmiş koşuyu
    kapsamıyor (Kural 2: değerlendirilmeden "başarılı" sayma).
    """
    sync_satirlari = [s for s in update_betik.splitlines() if "sync" in s and "$UvPath" in s]
    assert sync_satirlari, "update.ps1 içinde uv sync çağrısı bulunamadı"
    for satir in sync_satirlari:
        assert "--extra train-cpu" in satir, (
            f"uv sync eğitim extra'sını kapsamıyor → kilit/venv ayrışır: {satir.strip()}"
        )


def test_update_egitim_korumasi_durmuyor(update_betik: str) -> None:
    """Eğitim koşarken güncelleme atlanmalı (3a551b0'de eklendi; regresyon kapısı)."""
    assert "peft_lora_train" in update_betik, "eğitim süreci yoklaması kayboldu"
