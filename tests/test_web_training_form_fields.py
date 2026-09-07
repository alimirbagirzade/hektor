"""Eğitim formu yalnız GERÇEKTEN ETKİLİ alanları göstermeli (çevrimdışı).

Bulgu (2026-09-07): form "Batch" ve "Katman" alanlarını sunuyordu ama bu değerler
Windows/Linux'ta hiçbir şey yapmıyordu:

* Gerçek eğitim ucu ``/api/training/run`` yalnız
  ``launch(adapter_name, iterations, base_model)`` çağırır — batch/layers HİÇ ulaşmaz.
* Dry-run ucunda bu ikisi YALNIZ MLX (macOS) dalında ``TrainConfig``e geçer; PEFT
  dalında geçmez.

Kullanıcı değer giriyor, hiçbir etkisi olmuyordu. Ölü kontrol, yanlış bir zihinsel
model üretir (Kural 2'nin ruhu: doğrulanmadan "çalışıyor" sayma). Alanlar kaldırıldı;
API şema varsayılanları yerinde kaldığı için macOS/MLX yolu bozulmadı.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_STATIC = Path(__file__).resolve().parents[1] / "app" / "web" / "static"


@pytest.fixture(scope="module")
def html() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def js() -> str:
    return (_STATIC / "assets" / "app.js").read_text(encoding="utf-8", errors="replace")


@pytest.mark.parametrize("alan_id", ["drBatch", "drLayers"])
def test_olu_alanlar_formda_yok(html: str, alan_id: str) -> None:
    """PEFT yolunda etkisiz olan alanlar kullanıcıya sunulmamalı."""
    assert f'id="{alan_id}"' not in html, f"{alan_id} hâlâ formda — etkisiz kontrol"


@pytest.mark.parametrize("anahtar", ["batch_size", "num_layers"])
def test_js_bu_alanlari_gondermiyor(js: str, anahtar: str) -> None:
    """Kaldırılan alanlar istek gövdesine de girmemeli."""
    assert f"{anahtar}: parseInt" not in js


@pytest.mark.parametrize("alan_id", ["drBaseModel", "trAdapterName", "drIterations"])
def test_etkili_alanlar_korundu(html: str, alan_id: str) -> None:
    """Regresyon: gerçekten kullanılan üç alan kaybolmamalı."""
    assert f'id="{alan_id}"' in html


def test_bellek_uyarisi_var(html: str) -> None:
    """4B modelin bu sınıf makinede sığmadığı açıkça yazmalı (ölçüldü: ~18-20 GB)."""
    assert "Bellek uyarısı" in html
    assert "Qwen2.5-1.5B-Instruct" in html, "dar bellek için küçük model önerisi yok"


def test_hiperparametrelerin_kaynagi_yaziyor(html: str) -> None:
    """Kullanıcı 'neden burada ayar yok' diye sormasın: reçete profili işaret edilmeli."""
    assert "lora_profiles.yaml" in html
    assert "discipline_safe_local" in html
