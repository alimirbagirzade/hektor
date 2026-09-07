"""Ayrık (CLI/detached) eğitim koşusu arayüzde GÖRÜNMELİ (çevrimdışı).

Bulgu (2026-09-07, canlı sistemde): `/api/training/progress` yalnız
``TrainingManager``ın KENDİ başlattığı (web butonu) koşuyu biliyordu. Ama projenin
ÖNERDİĞİ yol ayrık başlatmadır (`scripts/start-train.ps1` → `hektor train --run`).
Gerçek bir koşu %42'de ilerlerken uç ``state=idle, 0/0 iter`` dönüyordu; kullanıcı
kendi eğitimini arayüzden izleyemiyor, dahası hangi temel modelle eğittiğini
göremiyordu (Sistem sekmesindeki "LLM model" satırı Ollama çıkarım modelini gösterir,
eğitilen modeli değil — bu da ayrı bir karışıklık kaynağıydı).

Artık bellekte koşu yoksa DİSKTEN okunur: `storage/train_status.json` +
en son `checkpoint-*/trainer_state.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.web.server import _detached_training_progress


@pytest.fixture
def kurulum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """İzole bir kök: state_dir + adapters_dir tmp altında."""
    from app.config import get_settings

    s = get_settings()
    state = tmp_path / "storage"
    adapters = tmp_path / "adapters"
    state.mkdir()
    adapters.mkdir()
    monkeypatch.setattr(type(s), "state_dir", property(lambda self: state))
    monkeypatch.setattr(type(s), "adapters_dir", property(lambda self: adapters))
    return state, adapters


def _yaz_status(state: Path, **kw) -> None:
    veri = {
        "adapter": "hektor_lora_test",
        "dtype": "bf16",
        "iterations": 300,
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    veri.update(kw)
    (state / "train_status.json").write_text(json.dumps(veri), encoding="utf-8")


def _yaz_checkpoint(adapters: Path, adapter: str, step: int, max_steps: int, loss: float) -> None:
    ck = adapters / adapter / f"checkpoint-{step}"
    ck.mkdir(parents=True)
    (ck / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "max_steps": max_steps,
                "log_history": [{"step": step, "loss": loss}],
            }
        ),
        encoding="utf-8",
    )


def test_status_dosyasi_yoksa_none(kurulum) -> None:
    """Ayrık koşu yoksa uç bellekteki (idle) duruma düşmeli."""
    assert _detached_training_progress() is None


def test_checkpoint_yoksa_none(kurulum) -> None:
    """Reçete var ama henüz adım atılmadıysa ilerleme raporlanmaz."""
    state, _ = kurulum
    _yaz_status(state)
    assert _detached_training_progress() is None


def test_kosan_ayrik_egitim_raporlanir(kurulum) -> None:
    """ASIL BULGU: koşan ayrık eğitim artık görünür — temel modeliyle birlikte."""
    state, adapters = kurulum
    _yaz_status(state)
    _yaz_checkpoint(adapters, "hektor_lora_test", 150, 300, 1.69)

    r = _detached_training_progress()
    assert r is not None
    assert r["state"] == "running"
    assert r["current_iter"] == 150
    assert r["total_iters"] == 300
    assert r["pct"] == 50.0
    assert r["train_loss"] == pytest.approx(1.69)
    assert r["adapter_name"] == "hektor_lora_test"
    # Kullanıcının göremediği kritik bilgi: HANGİ modelle eğitiliyor.
    assert r["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert r["source"] == "detached"


def test_en_yeni_checkpoint_secilir(kurulum) -> None:
    """Sayısal sıralama: checkpoint-100, checkpoint-25'ten SONRA gelmeli (metin değil)."""
    state, adapters = kurulum
    _yaz_status(state)
    _yaz_checkpoint(adapters, "hektor_lora_test", 25, 300, 2.5)
    _yaz_checkpoint(adapters, "hektor_lora_test", 100, 300, 1.4)

    r = _detached_training_progress()
    assert r is not None and r["current_iter"] == 100


def test_biten_kosu_finished_doner(kurulum) -> None:
    """Adapter ağırlığı yazıldıysa koşu bitmiştir."""
    state, adapters = kurulum
    _yaz_status(state)
    _yaz_checkpoint(adapters, "hektor_lora_test", 300, 300, 1.1)
    (adapters / "hektor_lora_test" / "adapter_model.safetensors").write_bytes(b"x")

    r = _detached_training_progress()
    assert r is not None and r["state"] == "finished" and r["pct"] == 100.0


def test_bozuk_json_cokertmez(kurulum) -> None:
    """Savunmacı: bozuk dosya uç'u 500'e düşürmemeli."""
    state, _ = kurulum
    (state / "train_status.json").write_text("{bozuk", encoding="utf-8")
    assert _detached_training_progress() is None
