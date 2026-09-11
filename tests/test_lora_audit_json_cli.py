"""`hektor lora-audit --json` makine-okunabilir çıktı sözleşmesi (çevrimdışı).

Bu çıktı bir SÖZLEŞMEdir: `scripts/start-train.ps1` kalite kapısı eğitimi
başlatıp başlatmayacağına buradan karar verir (rapor `reports/agent-inventory/`,
bulgu B2). Rich tablosu ayrıştırılabilir arayüz değildir; alan adları değişirse
kapı sessizce "sonuç okunamadı"ya düşer ve eğitim hiç başlamaz. Bu yüzden
alanlar burada tek tek sabitlenir.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from app.lora.control_plane import PipelineReport
from app.lora.gates import GateResult
from app.main import app

runner = CliRunner()


def _fake_report(*, passed: bool) -> PipelineReport:
    """İki kapılı sentetik denetim raporu (DB'ye dokunmaz)."""
    return PipelineReport(
        stages=[
            GateResult(gate_id=0, name="source", passed=True, rejected_count=0, review_count=1),
            GateResult(gate_id=1, name="schema", passed=passed, rejected_count=3, review_count=2),
        ],
        total_input=10,
        total_approved=7,
        total_rejected=3,
        total_review_needed=2,
    )


@pytest.fixture
def _patch_plane(monkeypatch, tmp_path):
    """LoRAControlPlane + SqliteStore'u sahte ile değiştir; kök tmp_path olsun."""

    def _install(*, passed: bool) -> None:
        report = _fake_report(passed=passed)

        class _FakePlane:
            def __init__(self, *a, **kw) -> None:
                pass

            def run_audit(self) -> PipelineReport:
                return report

            def run_full(self) -> PipelineReport:
                return report

            def generate_report(self, _report, output_path) -> None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("sahte rapor", encoding="utf-8")

        monkeypatch.setattr("app.lora.control_plane.LoRAControlPlane", _FakePlane)
        monkeypatch.setattr("app.memory.sqlite_store.SqliteStore", lambda *a, **kw: object())

        settings = __import__("app.config.settings", fromlist=["get_settings"]).get_settings()
        monkeypatch.setattr(type(settings), "root", property(lambda _self: tmp_path))

    return _install


def _invoke_json(monkeypatch) -> dict:
    monkeypatch.setenv("COLUMNS", "300")
    result = runner.invoke(app, ["lora-audit", "--json"])
    assert result.exit_code == 0, result.stdout
    return json.loads(result.stdout)


def test_lora_audit_json_alanlari_sabit(monkeypatch, _patch_plane) -> None:
    """Kapının okuduğu alanlar eksiksiz ve doğru tipte."""
    _patch_plane(passed=True)
    data = _invoke_json(monkeypatch)

    assert data["passed"] is True
    assert data["total_input"] == 10
    assert data["total_approved"] == 7
    assert data["total_rejected"] == 3
    assert data["total_review_needed"] == 2
    assert data["report_path"].endswith("audit_report.md")

    assert len(data["stages"]) == 2
    stage = data["stages"][1]
    assert stage == {
        "gate_id": 1,
        "name": "schema",
        "passed": True,
        "rejected_count": 3,
        "review_count": 2,
    }


def test_lora_audit_json_basarisiz_kapiyi_bildirir(monkeypatch, _patch_plane) -> None:
    """Bir kapı düşerse `passed` False olur — betik eğitimi başlatmamalı."""
    _patch_plane(passed=False)
    data = _invoke_json(monkeypatch)

    assert data["passed"] is False
    dusukler = [s for s in data["stages"] if not s["passed"]]
    assert [s["gate_id"] for s in dusukler] == [1]


def test_lora_audit_json_ciktisi_saf_json(monkeypatch, _patch_plane) -> None:
    """stdout'ta JSON dışı satır olmamalı — rapor yolu tabloya değil JSON'a girer."""
    _patch_plane(passed=True)
    monkeypatch.setenv("COLUMNS", "300")
    result = runner.invoke(app, ["lora-audit", "--json"])

    assert result.exit_code == 0
    assert "Rapor:" not in result.stdout
    assert "LoRA Denetim" not in result.stdout
    json.loads(result.stdout)  # baştan sona ayrıştırılabilir


def test_lora_audit_json_olmadan_tablo_basar(monkeypatch, _patch_plane) -> None:
    """--json verilmezse eski davranış (tablo + rapor yolu) korunur."""
    _patch_plane(passed=True)
    monkeypatch.setenv("COLUMNS", "300")
    result = runner.invoke(app, ["lora-audit"])

    assert result.exit_code == 0
    assert "LoRA Denetim" in result.stdout
    assert "Rapor:" in result.stdout
