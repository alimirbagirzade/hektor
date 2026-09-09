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
    )
    assert payload["adapter"] == "hektor_lora_v9_4b"
    assert payload["iterations"] == 1616
    assert payload["base_model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert payload["profile"] == "discipline_safe_local"
    assert payload["max_examples"] == 1616
    assert payload["pid"] == 4242
    assert payload["started_at"]


def test_status_payload_bos_degerleri_normalize_eder() -> None:
    """None profil/temel model JSON'a "" olarak yazılır; negatif tavan 0'a çekilir."""
    payload = _status_payload("a", "bf16", 10, None, None, -5, 1)
    assert payload["base_model"] == ""
    assert payload["profile"] == ""
    assert payload["max_examples"] == 0


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
