"""Kademe 2 kaydı TÜM eğitim yollarında zorunlu.

Yollar: web butonu / Auto-LoRA / kolay akış (``preflight_launch`` — onay tüketilmeden önce),
``hektor train --run`` (start-train.ps1, doğrudan CLI, nöbetçi kurtarması, web alt süreci),
``pretrain-gate`` (start-train.ps1 erken durur). Kayıt bu kod özeti + GÜNCEL eğitim verisi için
olmalı; veri ya da kod değişince eski kayıt geçmez. Kirli ağaç kabul edilmez.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from app.training import easy_train as et

CODE = {"ok": True, "code_sha": "c" * 64, "dirty": [], "head": "abc"}


def _row(i: int) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"Soru {i}: risk yönetimi nedir?"},
                {"role": "assistant", "content": f"Cevap {i}: hipotez ve test noktası."},
            ],
            "metadata": {"source_id": f"p{i % 7}"},
        },
        ensure_ascii=False,
    )


@pytest.fixture
def env(monkeypatch, tmp_path, real_kademe2_gate):
    from app.config import settings as settings_mod
    from app.lora.mix_common import load_mix_config, profile_weights
    from app.lora.weight_decision import WeightDecisionStore
    from app.training import detached_launch

    monkeypatch.setenv("HEKTOR_ROOT_PATH", str(tmp_path))
    monkeypatch.setenv("HEKTOR_SQLITE_PATH", str(tmp_path / "t.db"))
    settings_mod.get_settings.cache_clear()
    src = tmp_path / "data" / "lora_sft" / "lora_sft.jsonl"
    src.parent.mkdir(parents=True)
    src.write_text("\n".join(_row(i) for i in range(60)) + "\n", encoding="utf-8")
    monkeypatch.setattr(et, "code_state_provider", lambda: dict(CODE))
    monkeypatch.setattr(detached_launch, "_pretrain_gate_blockers", lambda s: [])
    monkeypatch.setattr("app.lora.mix_cli.leakage_eval_items", lambda: [])
    mix = sorted((load_mix_config().get("profiles") or {}).keys())[0]
    WeightDecisionStore().record(profile_weights(mix), mix, source="test")
    yield {"src": src, "mix": mix}
    settings_mod.get_settings.cache_clear()


def _record(data_sha: str) -> None:
    et.record_kademe2(
        scope={"data_sha256": data_sha},
        findings=[{"id": "F1", "status": "duzeltildi"}],
        closure_evidence="derin av bulguları kapandı; make ci yeşil (commit abc)",
        reviewer="insan",
    )


def test_web_and_auto_lora_preflight_requires_record(env) -> None:
    from app.training.detached_launch import preflight_launch

    pre = preflight_launch("hektor_lora_k2", run_load_doctor=False)
    assert not pre["ok"] and "Kademe 2 kaydı yok" in pre["message"]
    _lines, sha = et._lora_sft_lines()
    _record(sha)
    assert preflight_launch("hektor_lora_k2", run_load_doctor=False)["ok"]
    # Veri değişti → eski kayıt bu veriyi kapsamaz.
    env["src"].write_text(env["src"].read_text("utf-8") + _row(999) + "\n", encoding="utf-8")
    again = preflight_launch("hektor_lora_k2", run_load_doctor=False)
    assert not again["ok"] and "Kademe 2" in again["message"]


def test_dirty_tree_blocks_even_with_record(env, monkeypatch) -> None:
    _lines, sha = et._lora_sft_lines()
    _record(sha)
    monkeypatch.setattr(
        et, "code_state_provider", lambda: {"ok": False, "dirty": ["app/x.py"], "code_sha": ""}
    )
    msg = et.kademe2_check()
    assert msg and "kod durumu doğrulanamadı" in msg and "app/x.py" in msg


def test_web_training_run_endpoint_blocks_before_approval(env) -> None:
    from fastapi.testclient import TestClient

    from app.agents.runtime import approvals
    from app.web.server import app

    r = TestClient(app).post("/api/training/run", json={"adapter_name": "hektor_lora_k2"})
    body = r.json()
    assert body["ok"] is False and "Kademe 2" in body["message"]
    assert approvals.list_approvals() == []  # onay isteği açılmadı


def test_cli_train_run_exits_10_without_record(env) -> None:
    from app.agents.runtime import approvals
    from app.main import app
    from app.training import resource_lock

    res = CliRunner().invoke(
        app,
        [
            "train",
            "--run",
            "--backend",
            "peft",
            "--adapter-name",
            "hektor_lora_k2",
            "--mix-profile",
            env["mix"],
            "--skip-load-check",
        ],
    )
    assert res.exit_code == 10, res.output
    assert "Kademe 2" in res.output
    assert approvals.list_approvals() == []  # onay tüketilmedi/istenmedi
    assert not resource_lock.status()["held"]  # kilit alınmadı


def test_pretrain_gate_reports_missing_record(env) -> None:
    from app.main import app

    res = CliRunner().invoke(app, ["pretrain-gate", "--json"])
    data = json.loads(res.output[res.output.index("{") :])
    assert data["verdict"] == "NO-GO"
    assert any("Kademe 2" in b for b in data["blockers"])
