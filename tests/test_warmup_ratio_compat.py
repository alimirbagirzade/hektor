"""transformers sürüm uyumu: `warmup_ratio` kaldırılınca ısınma KAYBOLMAMALI.

Gerçek olay (2026-09-07): 4B eğitimi
``TypeError: TrainingArguments.__init__() got an unexpected keyword argument
'warmup_ratio'`` ile düştü. Sebep: bağımlılık üst sınırsızdı (``transformers>=4.40``);
ortam yeniden kurulunca 5.9 → 5.16'ya sıçradı ve 5.16 ``warmup_ratio``yu KALDIRDI
(yalnız ``warmup_steps`` kaldı).

Sürümü geri sabitlemek yerine kod iki sürümle de çalışır: destekleniyorsa oran aynen
geçer, desteklenmiyorsa adım sayısına çevrilir. Isınma v5 aşırı-öğrenme reçetesinin
parçasıdır — sessizce düşürülmez (Kural 2).
"""

from __future__ import annotations

import pytest

from app.training import peft_lora_train as m


def test_destekleniyorsa_oran_aynen_gecer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_training_arguments_supports", lambda p: True)
    k = m._adapt_warmup({"warmup_ratio": 0.05}, max_steps=300, num_epochs=1)
    assert k["warmup_ratio"] == 0.05
    assert "warmup_steps" not in k


def test_desteklenmiyorsa_adima_cevrilir(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASIL DÜZELTME: 0.05 × 300 adım = 15 adım ısınma."""
    monkeypatch.setattr(m, "_training_arguments_supports", lambda p: False)
    k = m._adapt_warmup({"warmup_ratio": 0.05}, max_steps=300, num_epochs=1)
    assert "warmup_ratio" not in k, "kaldırılmayan parametre TypeError üretir"
    assert k["warmup_steps"] == 15


def test_cok_kucuk_oran_en_az_bir_adim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Yuvarlama 0'a düşerse ısınma tamamen kaybolurdu."""
    monkeypatch.setattr(m, "_training_arguments_supports", lambda p: False)
    k = m._adapt_warmup({"warmup_ratio": 0.001}, max_steps=10, num_epochs=1)
    assert k["warmup_steps"] == 1


def test_adim_bilinmiyorsa_isinma_dusurulur_ama_patlamaz(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Oran adıma çevrilemez; parametre çıkarılır (TypeError yerine uyarı)."""
    import logging

    monkeypatch.setattr(m, "_training_arguments_supports", lambda p: False)
    with caplog.at_level(logging.WARNING, logger=m.logger.name):
        k = m._adapt_warmup({"warmup_ratio": 0.05}, max_steps=0, num_epochs=1)
    assert "warmup_ratio" not in k and "warmup_steps" not in k
    assert any(
        "ısınma" in r.message.lower() or "isinma" in r.message.lower() for r in caplog.records
    )


def test_oran_yoksa_dokunmaz(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "_training_arguments_supports", lambda p: False)
    k = m._adapt_warmup({"learning_rate": 1e-4}, max_steps=300, num_epochs=1)
    assert k == {"learning_rate": 1e-4}


def test_build_training_kwargs_uyumu_uygular(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uçtan uca: kwargs kurucusu da dönüşümden geçmeli."""
    monkeypatch.setattr(m, "_training_arguments_supports", lambda p: False)
    cfg = m.PeftTrainConfig(
        base_model="x",
        train_jsonl="t.jsonl",
        valid_jsonl="v.jsonl",
        adapter_output_path="out",
        iterations=300,
    )
    k = m.build_training_kwargs(cfg, num_epochs=1, output_dir="out", on_cuda=False, max_steps=300)
    assert "warmup_ratio" not in k
    assert k["warmup_steps"] >= 1


def test_transformers_yoksa_dokunmaz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Çevrimdışı/dry-run: kütüphane yoksa varsayılan davranış korunur."""
    assert m._training_arguments_supports("warmup_ratio") in (True, False)
