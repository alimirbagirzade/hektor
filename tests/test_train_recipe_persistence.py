"""Eğitim reçetesinin BÜTÜN hâlde taşınmasını kilitleyen regresyon testleri.

2026-09-08 bulgusu (v8 koşusu): `start-train.ps1 -Iterations 600` ile "600 örnek"
istendi ama betikte `-MaxExamples` parametresi YOKTU → `--max-examples` hiç geçmedi,
profildeki tavan (`discipline_safe_local: max_examples: 300`) yürürlükte kaldı ve
koşu 300 örnek × 2 epoch eğitti. Yani adım sayısı büyüdü, veri büyümedi: profilin
`epochs: 1` vaadi sessizce ihlal edildi (v5 disiplin-regresyonunun sınıfı).

Aynı boşluk `train_status.json`'da da vardı: nöbetçi (`training-watchdog.ps1`) çöken
eğitimi yalnız bu dosyadan diriltir; dosya profil/örnek tavanı taşımadığı için kurtarma
koşusu BAŞKA bir reçeteye kayıyordu.

Testler çevrimdışı ve salt-okuma: eğitim BAŞLATILMAZ (Kural 8).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.training.detached_launch import _status_payload

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_START_TRAIN = _SCRIPTS / "start-train.ps1"
_WATCHDOG = _SCRIPTS / "training-watchdog.ps1"


def test_status_payload_recetenin_tamamini_tasir() -> None:
    """Durum dosyası adapter/dtype/iterations ile SINIRLI kalmamalı."""
    payload = _status_payload(
        "hektor_lora_v9_4b",
        "bf16",
        1616,
        "Qwen/Qwen3-4B-Instruct-2507",
        "discipline_safe_local",
        1616,
        4242,
        "apr_deadbeef0000",
    )
    assert payload["adapter"] == "hektor_lora_v9_4b"
    assert payload["iterations"] == 1616
    assert payload["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert payload["profile"] == "discipline_safe_local"
    assert payload["max_examples"] == 1616
    assert payload["pid"] == 4242
    assert payload["approval_id"] == "apr_deadbeef0000"
    assert payload["started_at"]


def test_status_payload_bos_degerleri_normalize_eder() -> None:
    """None profil/temel model JSON'a "" olarak yazılır; negatif tavan 0'a çekilir."""
    payload = _status_payload("a", "bf16", 10, None, None, -5, 1)
    assert payload["base_model"] == ""
    assert payload["profile"] == ""
    assert payload["max_examples"] == 0
    assert payload["approval_id"] == ""


def test_status_payload_approval_id_bu_kosuyu_yetkilendiren_onayi_tasir() -> None:
    """K8-b: nöbetçi/kurtarma yolu bir ZAMAN PENCERESİ yerine bu kimliği doğrulamalı.

    2026-09-15 gözlemi (HANDOFF §4): durum dosyasının VARLIĞI nöbetçi için kalıcı
    yetki sayılıyordu — çöken bir koşu onaysız diriltildi. `approval_id` alanı,
    kurtarmanın "dosya var/taze" yerine "bu koşuyu GERÇEKTEN tüketilmiş bir onay
    başlattı mı" sorusuna cevap vermesini sağlar.
    """
    payload = _status_payload("a", "bf16", 10, None, None, 0, 1, approval_id="apr_abc123")
    assert payload["approval_id"] == "apr_abc123"


def test_start_train_max_examples_parametresi_ve_bayragi_var() -> None:
    """Betik örnek tavanını alabilmeli ve `--max-examples` olarak GEÇİRMELİ."""
    src = _START_TRAIN.read_text(encoding="utf-8")
    assert re.search(r"\[int\]\$MaxExamples\s*=\s*0", src), "-MaxExamples parametresi yok"
    assert "--max-examples" in src, "tavan CLI'ya geçirilmiyor (sessizce yok sayılır)"


def test_start_train_durum_dosyasina_tam_recete_yazar() -> None:
    """profile + max_examples + base_model durum dosyasında olmalı (nöbetçi bunu okur)."""
    src = _START_TRAIN.read_text(encoding="utf-8")
    for alan in ("adapter", "dtype", "iterations", "base_model", "profile", "max_examples"):
        assert re.search(rf"^\s*{alan}\s*=", src, re.MULTILINE), f"durum dosyasında {alan} yok"


def test_watchdog_receteyi_durum_dosyasindan_geri_okur() -> None:
    """Kurtarma koşusu profili ve örnek tavanını sabitlemek yerine reçeteden almalı."""
    src = _WATCHDOG.read_text(encoding="utf-8")
    assert "-MaxExamples $mx" in src, "nöbetçi örnek tavanını iletmiyor"
    assert "-Profile $prof" in src, "nöbetçi profili reçeteden okumuyor"
    assert '-Profile "discipline_safe_local"' not in src, "profil hâlâ sabit yazılmış"


# ---- K8-b: approval_id → train_status.json (zaman penceresi YERİNE) ----
# Önceki Kademe-2 seansı (K8-b, 51c634b) start-train.ps1'in KOŞULSUZ SUPERVISED=1
# vermesini kapattı; hemen ardından (5b367c0) train_guard.py'nin kurtarma doğrulaması
# eklendi — ama o modül onayı bir ZAMAN PENCERESİ (tüketimin koşu başlangıcına
# APPROVAL_WINDOW_MINUTES içinde olması) ile TAHMİN ediyordu, çünkü "onay kimliğini
# durum dosyasına yazan bir çağıran YOK"tu (train_guard.py'nin kendi docstring'i).
# Bu testler o boşluğu kilitler: approval_id artık GERÇEKTEN yazılıyor ve kullanılıyor.
def test_start_train_durum_dosyasina_approval_id_yazar() -> None:
    src = _START_TRAIN.read_text(encoding="utf-8")
    assert re.search(r"^\s*approval_id\s*=\s*\$ApprovalId", src, re.MULTILINE), (
        "durum dosyasında approval_id yazılmıyor"
    )


def test_start_train_taze_baslatmada_tuketilen_onayi_durum_dosyasina_isler() -> None:
    """Alt süreç kendi onayını tükettikten SONRA, kimliği log'dan alıp dosyaya İŞLENMELİ."""
    src = _START_TRAIN.read_text(encoding="utf-8")
    assert "consumedId" in src, "tüketilen approval_id log'dan okunmuyor"
    assert "approval_id" in src and "Add-Member" in src, (
        "okunan approval_id durum dosyasına yazılmıyor"
    )


def test_start_train_ApprovalId_parametresi_var() -> None:
    """Nöbetçi kurtarmasının (train-recovery-check ile doğrulanmış) kimliği geçirebilmesi için."""
    src = _START_TRAIN.read_text(encoding="utf-8")
    assert re.search(r"\[string\]\$ApprovalId\s*=\s*\"\"", src)


def test_watchdog_train_recovery_check_ile_gecer() -> None:
    """Nöbetçi dosya varlığına değil train-recovery-check'e (Kural 8 fail-closed) güvenmeli."""
    src = _WATCHDOG.read_text(encoding="utf-8")
    assert "train-recovery-check" in src, "kurtarma yetki kapısı yok"
    assert "recoveryOk" in src and "if (-not $recoveryOk) { exit 0 }" in src, (
        "kontrol çalışmazsa/reddederse nöbetçi yine de diriltebilir"
    )


def test_watchdog_dogrulanan_approval_id_yi_tasir() -> None:
    """K8-b: train-recovery-check'in bulduğu approval_id, ZAMAN PENCERESİ yerine

    bir sonraki teşhis/kurtarma için durum dosyasına doğrudan kimlik olarak taşınmalı.
    """
    src = _WATCHDOG.read_text(encoding="utf-8")
    assert "recoveryApprovalId" in src
    assert "-ApprovalId $recoveryApprovalId" in src
