"""Kontrollü eğitim döngüsü: bitiş kanıtı (çıkış kodu), geliştirme değerlendirmesi kaydı ve
"sonraki eğitim" kapısı. Gerçek eğitim/Ollama/değerlendirme koşusu YOK; süreç tanığı yalnız
kısa ömürlü bir alt süreçle sınanır."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from app.orchestration import training_loop as tl
from app.procutil import NO_WINDOW
from app.training import exit_witness


@pytest.fixture
def clean_loop():
    import shutil

    shutil.rmtree(tl.loop_dir(), ignore_errors=True)
    yield
    shutil.rmtree(tl.loop_dir(), ignore_errors=True)


def _complete(monkeypatch, ok: bool = True, sha: str = "w" * 64):
    from app.training import candidate_checks

    monkeypatch.setattr(candidate_checks, "recipe_for_adapter", lambda a: {})
    monkeypatch.setattr(
        candidate_checks,
        "verify_run_completion",
        lambda a, r=None: {
            "ok": ok,
            "adapter_sha256": sha,
            "checks": [{"key": "cikis_kodu", "ok": ok, "detail": "test"}],
        },
    )


def _raw(path, *, condition, model, adapter, scores, critical=(), flags=()):
    """Puanlanmış llm30 raw.jsonl (soru başına 2 tekrar)."""
    rows = []
    for qi, score in enumerate(scores):
        for seed in (42, 43):
            dims = {"dogruluk": score, "alt_madde": 2, "tutarlilik": 2, "dogrulama": 2}
            rows.append(
                {
                    "condition": condition,
                    "question_id": f"q{qi}",
                    "split": "validation",
                    "model": model,
                    "adapter_id": adapter,
                    "rag_enabled": False,
                    "seed": seed,
                    "score_by_dimension": dims,
                    "critical_errors": ["look_ahead"] if qi in critical else [],
                    "flags": ["uzun_tekrar"] if qi in flags else [],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _manifest(run_dir, sha="w" * 64):
    (run_dir / "manifest.json").write_text(
        json.dumps({"adapter": {"adapter_sha256": sha}, "dataset_version": "validation@x"}),
        encoding="utf-8",
    )


def test_empty_loop_allows_new_data(clean_loop, real_training_loop_gate):
    assert tl.next_training_check(data_sha="a" * 64, adapter_name="yeni") == []


def test_unknown_data_sha_is_fail_closed(clean_loop, real_training_loop_gate):
    assert tl.next_training_check(data_sha="", adapter_name="yeni")


def test_same_data_never_retrained_from_scratch(clean_loop, real_training_loop_gate, monkeypatch):
    tl.register("aday1", source="test", data_sha256="a" * 64, started_at="2026-10-11")
    probs = tl.next_training_check(data_sha="a" * 64, adapter_name="aday2")
    assert any("zaten eğitildi" in p for p in probs)
    # Kendi kaydı (aynı adapter: alt süreç / kurtarma) kendini engellemez.
    assert not any(
        "zaten" in p for p in tl.next_training_check(data_sha="a" * 64, adapter_name="aday1")
    )


def test_previous_candidate_must_be_verified_and_pass(
    clean_loop, real_training_loop_gate, monkeypatch, tmp_path
):
    tl.register("aday1", source="test", data_sha256="a" * 64, started_at="2026-10-11")
    probs = tl.next_training_check(data_sha="b" * 64, adapter_name="aday2")
    assert any("bitiş doğrulaması yok" in p for p in probs)

    _complete(monkeypatch, ok=False)
    tl.record_completion("aday1")
    probs = tl.next_training_check(data_sha="b" * 64, adapter_name="aday2")
    assert any("doğrulanamadı" in p for p in probs)
    with pytest.raises(tl.TrainingLoopError, match="bitiş doğrulanmalı"):
        tl.record_dev_eval(
            "aday1",
            candidate_raw=tmp_path / "x",
            candidate_condition="B",
            reference_raw=tmp_path / "y",
            reference_condition="B",
        )

    _complete(monkeypatch, ok=True)
    tl.record_completion("aday1")
    probs = tl.next_training_check(data_sha="b" * 64, adapter_name="aday2")
    assert any("geliştirme değerlendirmesinden geçmedi" in p for p in probs)

    cand = _raw(
        tmp_path / "c" / "raw.jsonl",
        condition="B",
        model="hektor-aday1",
        adapter="aday1",
        scores=[4, 3, 4],
    )
    _manifest(cand.parent)
    ref = _raw(
        tmp_path / "r" / "raw.jsonl",
        condition="B",
        model="hektor-v12-30b",
        adapter="v12",
        scores=[4, 3, 4],
    )
    rec = tl.record_dev_eval(
        "aday1",
        candidate_raw=cand,
        candidate_condition="B",
        reference_raw=ref,
        reference_condition="B",
    )
    dev = rec["dev_eval"]
    assert dev["passed"] and dev["reference_model"] == "hektor-v12-30b"
    assert dev["decision"] == "insan_onayi_bekliyor" and dev["candidate_raw_sha256"]
    assert tl.next_training_check(data_sha="b" * 64, adapter_name="aday2") == []
    # Aynı veri yine yasak (geçmiş aday olsa da).
    assert tl.next_training_check(data_sha="a" * 64, adapter_name="aday2")


def test_critical_error_regression_or_flags_block(
    clean_loop, real_training_loop_gate, monkeypatch, tmp_path
):
    tl.register("aday1", source="test", data_sha256="a" * 64, started_at="2026-10-11")
    _complete(monkeypatch)
    tl.record_completion("aday1")
    ref = _raw(
        tmp_path / "r" / "raw.jsonl",
        condition="B",
        model="hektor-v12-30b",
        adapter="v12",
        scores=[4, 4, 4],
    )
    cases = {
        "kritik": {"scores": [4, 4, 4], "critical": (1,)},
        "gerileme": {"scores": [1, 1, 1]},
        "bayrak": {"scores": [4, 4, 4], "flags": (0,)},
    }
    for name, kw in cases.items():
        cand = _raw(
            tmp_path / name / "raw.jsonl",
            condition="B",
            model="hektor-aday1",
            adapter="aday1",
            **kw,
        )
        _manifest(cand.parent)
        rec = tl.record_dev_eval(
            "aday1",
            candidate_raw=cand,
            candidate_condition="B",
            reference_raw=ref,
            reference_condition="B",
        )
        assert not rec["dev_eval"]["passed"], name
        probs = tl.next_training_check(data_sha="b" * 64, adapter_name="aday2")
        assert any("GEÇMEDİ" in p for p in probs), name


def test_eval_must_come_from_verified_weights(
    clean_loop, real_training_loop_gate, monkeypatch, tmp_path
):
    tl.register("aday1", source="test", data_sha256="a" * 64)
    _complete(monkeypatch)
    tl.record_completion("aday1")
    cand = _raw(
        tmp_path / "c" / "raw.jsonl", condition="B", model="m", adapter="aday1", scores=[4, 4]
    )
    _manifest(cand.parent, sha="baska" * 10)
    ref = _raw(tmp_path / "r" / "raw.jsonl", condition="B", model="p", adapter="v", scores=[4, 4])
    rec = tl.record_dev_eval(
        "aday1",
        candidate_raw=cand,
        candidate_condition="B",
        reference_raw=ref,
        reference_condition="B",
    )
    assert not rec["dev_eval"]["passed"]
    assert any("ağırlık özeti" in p for p in rec["dev_eval"]["problems"])


def test_release_needs_fresh_human_approval(clean_loop, real_training_loop_gate, monkeypatch):
    from app.agents.runtime import approvals

    tl.register("aday1", source="test", data_sha256="a" * 64)
    _complete(monkeypatch, ok=False)
    tl.record_completion("aday1")
    with pytest.raises(tl.TrainingLoopError):
        tl.release("aday1", "kısa")
    first = tl.release("aday1", "Çöküş nedeni bulundu ve kodda düzeltildi; yeni veriyle devam.")
    assert first["status"] == "needs_approval"
    assert tl.next_training_check(data_sha="b" * 64, adapter_name="aday2")
    approvals.approve(first["approval_id"])
    second = tl.release("aday1", "Çöküş nedeni bulundu ve kodda düzeltildi; yeni veriyle devam.")
    assert second["status"] == "released"
    assert tl.next_training_check(data_sha="b" * 64, adapter_name="aday2") == []
    assert tl.next_training_check(data_sha="a" * 64, adapter_name="aday2")  # aynı veri istisnasız


def test_sync_from_status_registers_running_training(clean_loop, monkeypatch):
    from app.config import get_settings
    from app.training import detached_launch

    monkeypatch.setattr(detached_launch, "is_running", lambda: True)

    st = get_settings().root / "storage" / "train_status.json"
    st.parent.mkdir(parents=True, exist_ok=True)
    st.write_text(
        json.dumps(
            {
                "adapter": "hektor_lora_full_test",
                "data_sha256": "f" * 64,
                "pid": 1,
                "approval_id": "apr_x",
                "started_at": "2026-10-11T14:16:24",
            }
        ),
        encoding="utf-8",
    )
    try:
        rec = tl.sync_from_status()
        assert rec["adapter"] == "hektor_lora_full_test" and rec["data_sha256"] == "f" * 64
        o = tl.overview()
        assert o["running"] == "hektor_lora_full_test"
        assert o["candidates"][0]["next_step"].startswith("Eğitim sürüyor")
        # Süreç ölmüşse sonuçsuz durum dosyası "sürüyor" sayılmaz.
        monkeypatch.setattr(detached_launch, "is_running", lambda: False)
        o = tl.overview()
        assert o["running"] == "" and o["candidates"][0]["next_step"].startswith("Bitiş")
    finally:
        st.unlink()


# ── çıkış kodu tanığı ────────────────────────────────────────────────────────


def test_witness_records_os_exit_code(tmp_path):
    import psutil

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time, sys; time.sleep(1.5); sys.exit(3)"],
        creationflags=NO_WINDOW,
    )
    try:
        ct = psutil.Process(proc.pid).create_time()
        (tmp_path / "models" / "adapters" / "aday").mkdir(parents=True)
        rec = exit_witness.witness("aday", proc.pid, ct, root=tmp_path)
    finally:
        proc.wait(timeout=30)
    assert rec["status"] == "exited" and rec["exit_code"] == 3
    assert exit_witness.read_exit("aday", tmp_path)["exit_code"] == 3


def test_witness_cannot_observe_dead_or_reused_pid(tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=NO_WINDOW)
    proc.wait(timeout=30)
    rec = exit_witness.witness("aday", proc.pid, None, root=tmp_path)
    assert rec["status"] == "gozlenemedi" and rec["exit_code"] is None
    import os

    rec = exit_witness.witness("aday2", os.getpid(), 1.0, root=tmp_path)
    assert rec["status"] == "gozlenemedi"


def test_completion_requires_zero_exit_code(monkeypatch):
    """Final durum + ağırlık dosyası olsa da çıkış kodu kaydı yoksa / 0 değilse bitiş yok."""
    from app.config import get_settings
    from app.training.candidate_checks import verify_run_completion

    root = get_settings().root
    d = root / "models" / "adapters" / "hektor_exit_t"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "B"}), "utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"w")
    (d / "run_plan.json").write_text(json.dumps({"max_steps": 2}), "utf-8")
    (d / "run_complete.json").write_text(json.dumps({"global_step": 2, "max_steps": 2}), "utf-8")
    st = root / "storage" / "train_status.json"
    st.parent.mkdir(parents=True, exist_ok=True)
    st.write_text(json.dumps({"adapter": "hektor_exit_t", "finished_at": "x", "pid": 0}), "utf-8")
    try:
        keys = {c["key"]: c for c in verify_run_completion("hektor_exit_t")["checks"]}
        assert keys["cikis_kodu"]["ok"] is False and "YOK" in keys["cikis_kodu"]["detail"]
        (d / "train_exit.json").write_text(
            json.dumps({"adapter": "hektor_exit_t", "pid": 0, "status": "exited", "exit_code": 7}),
            "utf-8",
        )
        keys = {c["key"]: c for c in verify_run_completion("hektor_exit_t")["checks"]}
        assert keys["cikis_kodu"]["ok"] is False
        (d / "train_exit.json").write_text(
            json.dumps({"adapter": "hektor_exit_t", "pid": 0, "status": "exited", "exit_code": 0}),
            "utf-8",
        )
        keys = {c["key"]: c for c in verify_run_completion("hektor_exit_t")["checks"]}
        assert keys["cikis_kodu"]["ok"] is True
    finally:
        st.unlink()


def test_easy_readiness_shows_loop_and_knowledge_items(
    clean_loop, real_training_loop_gate, monkeypatch
):
    from app.training import easy_train

    monkeypatch.setattr(easy_train, "code_state_provider", lambda: {"ok": True, "code_sha": "c"})
    tl.register("aday1", source="test", data_sha256="a" * 64)
    items = {i["key"]: i for i in easy_train.readiness()["items"]}
    assert items["egitim_dongusu"]["ok"] is False
    assert "bitiş doğrulaması yok" in items["egitim_dongusu"]["detail"]
    assert items["bilgi_verisi"]["ok"] is True
