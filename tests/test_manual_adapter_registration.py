"""Manuel `train --run` sonrası adapter kayıt defterine (AdapterRegistry) eklenmeli.

hektor_lora_v9_4b tam 600 adım eğitildi ama kayıt defterine hiç girmedi — `train --run
--backend peft` eğitim BAŞARILI olsa bile hiçbir yerde `AdapterRegistry.register()`
çağırmıyordu. Web/auto_pipeline yolu (`launch()`) BU AYNI CLI komutunu subprocess
olarak çağırır ve eğitim bitince KENDİ kaydını (gerçek eval sonrası) açar; bu yüzden
kayıt yalnız SAF manuel çağrıda (HEKTOR_TRAIN_SUPERVISED yokken) yapılmalı — aksi
halde denetimli koşularda çift kayıt oluşur.

Çevrimdışı ve saf: gerçek eğitim/torch gerektirmez, yalnız `_register_manual_adapter`
ve `_count_jsonl_examples` doğrudan test edilir (bkz. tests/test_lora_adapter_registry.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.lora.adapter_registry import AdapterRegistry, AdapterStatus
from app.main import app


def test_count_jsonl_examples_bos_olmayan_satirlari_sayar(tmp_path: Path) -> None:
    from app.main import _count_jsonl_examples

    p = tmp_path / "train.jsonl"
    p.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert _count_jsonl_examples(p) == 2


def test_count_jsonl_examples_olmayan_dosya_sifir_doner(tmp_path: Path) -> None:
    from app.main import _count_jsonl_examples

    assert _count_jsonl_examples(tmp_path / "yok.jsonl") == 0


def test_register_manual_adapter_candidate_olarak_kaydeder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Başarılı manuel eğitim sonrası kayıt CANDIDATE olmalı — asla otomatik terfi."""
    from app.main import _register_manual_adapter

    monkeypatch.chdir(tmp_path)
    train_jsonl = tmp_path / "train.jsonl"
    valid_jsonl = tmp_path / "valid.jsonl"
    train_jsonl.write_text('{"messages": []}\n' * 5, encoding="utf-8")
    valid_jsonl.write_text('{"messages": []}\n', encoding="utf-8")

    adapter_id = _register_manual_adapter(
        adapter_name="hektor_lora_test",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        train_jsonl=train_jsonl,
        valid_jsonl=valid_jsonl,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj"],
        notes="manuel train --run --backend peft (iterations=600)",
    )

    records = AdapterRegistry().list_adapters()
    assert len(records) == 1
    record = records[0]
    assert record.adapter_id == adapter_id
    assert record.adapter_name == "hektor_lora_test"
    assert record.base_model == "Qwen/Qwen3-4B-Instruct-2507"
    assert record.status is AdapterStatus.CANDIDATE
    assert record.lora_r == 16
    assert record.lora_alpha == 32
    assert record.train_examples == 5
    assert record.valid_examples == 1
    assert record.target_modules == ["q_proj", "v_proj"]
    assert "backend peft" in record.notes
    # Kural 8: CANDIDATE production'a doğrudan terfi edemez.
    assert AdapterRegistry().promote(adapter_id, user_approved=True) is False
    assert AdapterRegistry().get_production() is None


def test_register_manual_adapter_id_uretir_ve_tekil(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.main import _register_manual_adapter

    monkeypatch.chdir(tmp_path)
    train_jsonl = tmp_path / "train.jsonl"
    valid_jsonl = tmp_path / "valid.jsonl"
    train_jsonl.write_text("", encoding="utf-8")
    valid_jsonl.write_text("", encoding="utf-8")

    id1 = _register_manual_adapter(
        adapter_name="a1",
        base_model="qwen",
        train_jsonl=train_jsonl,
        valid_jsonl=valid_jsonl,
        notes="test",
    )
    id2 = _register_manual_adapter(
        adapter_name="a2",
        base_model="qwen",
        train_jsonl=train_jsonl,
        valid_jsonl=valid_jsonl,
        notes="test",
    )
    assert id1 != id2
    assert {r.adapter_id for r in AdapterRegistry().list_adapters()} == {id1, id2}


def _install_cli_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`train --run --backend peft` çağrısının onay/veri/eğitim adımlarını sahteler."""
    from app.config import settings as settings_mod
    from app.training import unattended_policy

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    settings_mod.get_settings.cache_clear()

    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda root=None: False)
    monkeypatch.setattr(
        unattended_policy,
        "authorize_training_action",
        lambda *a, **k: unattended_policy.TrainingAuthorization(True, "test", "ok", "apr_test"),
    )

    def _fake_split(settings=None) -> tuple[int, int]:
        return 5, 1

    monkeypatch.setattr("app.training.detached_launch.ensure_train_split", _fake_split)
    monkeypatch.setattr(
        "app.training.peft_lora_train.train",
        lambda cfg: {"ok": True, "adapter_path": str(cfg.adapter_output_path), "device": "cpu"},
    )


def test_train_run_manual_registers_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SAF manuel `train --run` başarılı eğitim sonrası adapter'ı CANDIDATE kaydeder."""
    monkeypatch.delenv("HEKTOR_TRAIN_SUPERVISED", raising=False)
    _install_cli_mocks(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        app,
        ["train", "--run", "--backend", "peft", "--adapter-name", "manual_smoke"],
    )
    assert result.exit_code == 0, result.output

    records = AdapterRegistry().list_adapters()
    assert len(records) == 1
    assert records[0].adapter_name == "manual_smoke"
    assert records[0].status is AdapterStatus.CANDIDATE


def test_train_run_supervised_does_not_double_register(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Denetimli (web/auto_pipeline) koşu bu komuttan CANDIDATE kaydı AÇMAMALI.

    auto_pipeline kendi kaydını gerçek eval sonrası açar (SMOKE_PASSED/EVAL_PASSED);
    burada da eklenirse aynı koşu için çift kayıt oluşur.
    """
    monkeypatch.setenv("HEKTOR_TRAIN_SUPERVISED", "1")
    _install_cli_mocks(monkeypatch, tmp_path)

    result = CliRunner().invoke(
        app,
        ["train", "--run", "--backend", "peft", "--adapter-name", "supervised_smoke"],
    )
    assert result.exit_code == 0, result.output
    assert AdapterRegistry().list_adapters() == []
