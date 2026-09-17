"""train-doctor — rakip LLM/GPU yükü tespiti (offline testler).

Tümü çevrimdışı: Ollama `/api/ps` httpx.MockTransport ile sahte, nvidia-smi
parametre olarak enjekte edilir (gerçek subprocess yok). Kök neden: Ollama
`ollama_keep_alive` süresince modeli bellekte tutar; aynı anda gerçek LoRA
eğitimi başlarsa VRAM/RAM çakışır — bu modül eğitimden ÖNCE bunu tespit eder.
"""

from __future__ import annotations

import json

import httpx
from typer.testing import CliRunner

import app.main as m
import app.training.train_doctor as td
from app.main import app
from app.training.train_doctor import run_train_doctor

runner = CliRunner()
_ENV = {"COLUMNS": "200"}


def _transport(models: list[dict], *, status: int = 200) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(status, json={"models": models})

    return httpx.MockTransport(handler)


def _unreachable_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("bağlantı yok", request=request)

    return httpx.MockTransport(handler)


# --- run_train_doctor: çekirdek mantık --------------------------------------------


def test_no_models_no_gpu_signal_is_go() -> None:
    """Ollama boş + nvidia-smi yok → GO, uyarı yok."""
    report = run_train_doctor(transport=_transport([]), nvidia_smi=None)
    assert report.ollama_reachable is True
    assert report.loaded_models == []
    assert report.verdict == "GO"


def test_loaded_model_without_gpu_signal_is_warn() -> None:
    """Ollama'da model yüklü ama nvidia-smi yok (ör. Apple Silicon) → WARN, NO-GO değil."""
    models = [
        {"name": "qwen3:4b-instruct-2507-q4_K_M", "size": 4_000_000_000, "size_vram": 4_000_000_000}
    ]
    report = run_train_doctor(transport=_transport(models), nvidia_smi=None)
    assert report.verdict == "WARN"
    assert report.loaded_models[0].name == "qwen3:4b-instruct-2507-q4_K_M"
    assert any("nvidia-smi" in r for r in report.reasons)


def test_nvidia_smi_plenty_of_free_vram_is_go_even_with_loaded_model() -> None:
    """Model yüklü ama ölçülen boş VRAM eşiğin üstünde → yalnız WARN (NO-GO değil)."""
    models = [{"name": "qwen3:4b", "size": 2_000_000_000, "size_vram": 2_000_000_000}]
    report = run_train_doctor(
        transport=_transport(models), nvidia_smi=(2.0, 24.0), min_free_vram_gb=3.0
    )
    assert report.gpu_source == "nvidia-smi"
    assert report.free_vram_gb == 22.0
    assert report.verdict == "WARN"


def test_nvidia_smi_low_free_vram_is_no_go() -> None:
    """Ölçülen boş VRAM eşiğin altında → NO-GO + ollama stop önerisi."""
    models = [
        {"name": "qwen3:4b-instruct-2507-q4_K_M", "size": 5_000_000_000, "size_vram": 5_000_000_000}
    ]
    report = run_train_doctor(
        transport=_transport(models), nvidia_smi=(6.0, 8.0), min_free_vram_gb=3.0
    )
    assert report.free_vram_gb == 2.0
    assert report.verdict == "NO-GO"
    assert any("ollama stop qwen3:4b-instruct-2507-q4_K_M" in r for r in report.reasons)


def test_nvidia_smi_low_free_vram_without_ollama_model_still_no_go() -> None:
    """Ollama'da model yok ama GPU dolu → yine NO-GO (rakip Ollama olmayabilir)."""
    report = run_train_doctor(transport=_transport([]), nvidia_smi=(7.5, 8.0), min_free_vram_gb=3.0)
    assert report.verdict == "NO-GO"
    assert any("başka bir süreç" in r for r in report.reasons)


def test_ollama_unreachable_and_no_gpu_signal_defaults_to_go() -> None:
    """Ollama kapalı + nvidia-smi yok → ölçülemez ama bloklamaz (GO), bilgi notu düşer."""
    report = run_train_doctor(transport=_unreachable_transport(), nvidia_smi=None)
    assert report.ollama_reachable is False
    assert report.verdict == "GO"
    assert any("ölçülemedi" in r for r in report.reasons)


def test_min_free_vram_gb_override_wins_over_settings() -> None:
    models = [{"name": "m", "size": 1_000_000_000, "size_vram": 1_000_000_000}]
    report = run_train_doctor(
        transport=_transport(models), nvidia_smi=(1.0, 4.0), min_free_vram_gb=10.0
    )
    assert report.min_free_vram_gb == 10.0
    assert report.verdict == "NO-GO"


# --- CLI: `hektor train-doctor` -----------------------------------------------------


def test_cli_train_doctor_go_exit_zero(monkeypatch) -> None:
    monkeypatch.setattr(td, "run_train_doctor", lambda **kw: td.TrainDoctorReport(verdict="GO"))
    result = runner.invoke(app, ["train-doctor"], env=_ENV)
    assert result.exit_code == 0
    assert "GO" in result.stdout


def test_cli_train_doctor_no_go_exit_three(monkeypatch) -> None:
    monkeypatch.setattr(
        td,
        "run_train_doctor",
        lambda **kw: td.TrainDoctorReport(verdict="NO-GO", reasons=["boş VRAM yetersiz"]),
    )
    result = runner.invoke(app, ["train-doctor"], env=_ENV)
    assert result.exit_code == 3
    assert "NO-GO" in result.stdout


def test_cli_train_doctor_json_output(monkeypatch) -> None:
    monkeypatch.setattr(td, "run_train_doctor", lambda **kw: td.TrainDoctorReport(verdict="WARN"))
    result = runner.invoke(app, ["train-doctor", "--json"], env=_ENV)
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "WARN"


# --- `train --run` kapısı: NO-GO gerçek eğitimi engeller --------------------------


def test_train_run_blocks_on_no_go(monkeypatch) -> None:
    """train-doctor NO-GO derse `train --run` STOP_ALL/onay adımlarına hiç gelmeden çıkmalı."""
    monkeypatch.setattr(m, "get_settings", m.get_settings)
    from app.agents.runtime import supervisor

    monkeypatch.setattr(supervisor, "is_stop_all_active", lambda: False)
    monkeypatch.setattr(
        td,
        "run_train_doctor",
        lambda **kw: td.TrainDoctorReport(verdict="NO-GO", reasons=["boş VRAM yetersiz"]),
    )

    called = {"authorize": False}

    def _fake_authorize(*args, **kwargs):
        called["authorize"] = True
        raise AssertionError("NO-GO sonrası onay akışına gelinmemeli")

    from app.training import unattended_policy

    monkeypatch.setattr(unattended_policy, "authorize_training_action", _fake_authorize)

    result = runner.invoke(app, ["train", "--run"], env=_ENV)
    assert result.exit_code == 4
    assert called["authorize"] is False


def test_train_run_skip_load_check_bypasses_doctor(monkeypatch) -> None:
    """`--skip-load-check` verilirse train-doctor hiç çağrılmamalı."""
    from app.agents.runtime import supervisor

    monkeypatch.setattr(supervisor, "is_stop_all_active", lambda: False)

    called = {"doctor": False}

    def _fake_doctor(**kw):
        called["doctor"] = True
        return td.TrainDoctorReport(verdict="NO-GO")

    monkeypatch.setattr(td, "run_train_doctor", _fake_doctor)

    from app.training import unattended_policy

    def _fake_authorize(*args, **kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(authorized=False, approval_id="apr_test")

    monkeypatch.setattr(unattended_policy, "authorize_training_action", _fake_authorize)

    result = runner.invoke(app, ["train", "--run", "--skip-load-check"], env=_ENV)
    assert called["doctor"] is False
    # skip-load-check sonrası akış onay kapısına ulaşmalı (exit 3, doctor'dan (4) değil).
    assert result.exit_code == 3
