"""Paket eğitim engeli, kaynak kapısı ve süreç yalıtımı çevrimdışı sınanır."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.orchestration import research_package as package
from app.orchestration import research_worker as worker


def config() -> package.PackageConfig:
    return package.PackageConfig(
        wait_for_adapters=["v14"], interval_hours=dict.fromkeys(package.STAGES, 24)
    )


def finish(root: Path, name: str = "v14") -> None:
    path = root / "models" / "adapters" / name
    package.write_json(path / "run_plan.json", {"max_steps": 187})
    done = path / "run_complete.json"
    package.write_json(done, {"max_steps": 187, "global_step": 187, "finished_at": "2026-10-01"})
    os.utime(done, (time.time() - 3600, time.time() - 3600))


@pytest.mark.parametrize("model", ["hektor-v14", "custom-base", "qwen3:30b"])
def test_teacher_requires_explicit_base(model):
    with pytest.raises(ValidationError):
        package.PackageConfig(**{**config().model_dump(), "teacher_model": model})


def test_missing_failed_and_partial_training_all_block(tmp_path):
    cfg = config()
    assert package.blockers(tmp_path, cfg, check_processes=False)
    finish(tmp_path)
    assert not package.blockers(tmp_path, cfg, check_processes=False)
    package.write_json(tmp_path / "models/adapters/v15/run_plan.json", {"max_steps": 2})
    assert "v15" in " ".join(package.blockers(tmp_path, cfg, check_processes=False))
    package.write_json(
        tmp_path / "models/adapters/v15/run_complete.json",
        {
            "max_steps": 2,
            "global_step": 1,
            "finished_at": "2026-10-01",
        },
    )
    assert "v15" in " ".join(package.blockers(tmp_path, cfg, check_processes=False))


def test_corrupt_training_state_fails_closed(tmp_path):
    finish(tmp_path)
    path = tmp_path / "storage/train_status.json"
    path.parent.mkdir()
    path.write_text("broken", encoding="utf-8")
    assert "okunamadı" in " ".join(package.blockers(tmp_path, config(), check_processes=False))


def test_training_status_outside_plan_and_launch_lock_block(tmp_path):
    finish(tmp_path)
    package.write_json(tmp_path / "storage/train_status.json", {"adapter": "starting_v15"})
    assert "starting_v15" in " ".join(package.blockers(tmp_path, config(), check_processes=False))
    package.write_json(tmp_path / "storage/train_status.json", {})
    (tmp_path / "storage/.training_launching").touch()
    assert package.blockers(tmp_path, config(), check_processes=False)


def test_quiet_period_and_stop_are_observed(tmp_path):
    finish(tmp_path)
    (tmp_path / "models/adapters/v14/run_complete.json").touch()
    assert "bekleme" in " ".join(package.blockers(tmp_path, config(), check_processes=False))
    finish(tmp_path)
    (tmp_path / "storage").mkdir()
    (tmp_path / "storage/STOP_RESEARCH").touch()
    assert "STOP_RESEARCH" in " ".join(package.blockers(tmp_path, config(), check_processes=False))


def test_dry_run_and_blocked_run_never_spawn_or_write(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    package.write_json(cfg_path, config().model_dump())
    monkeypatch.setattr(package, "blockers", lambda *a, **kw: ["Eğitim devam ediyor"])
    monkeypatch.setattr(package, "run_stage", lambda *a: pytest.fail("İşçi doğmamalı"))
    for execute in (False, True):
        assert package.tick(tmp_path, cfg_path, execute=execute)["blocked"]
        assert not package.package_dir(tmp_path).exists()


def test_serial_lock_and_one_stage_per_tick(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    package.write_json(cfg_path, config().model_dump())
    monkeypatch.setattr(package, "blockers", lambda *a, **kw: [])
    calls = []

    def run(*args):
        calls.append(args[-1])
        with pytest.raises(FileExistsError), package.package_lock(tmp_path):
            pass
        return {"stage": args[-1], "queued": 0}

    monkeypatch.setattr(package, "run_stage", run)
    assert package.tick(tmp_path, cfg_path, execute=True)["executed"] == "discovery"
    assert package.tick(tmp_path, cfg_path, execute=True)["executed"] == "ingestion"
    assert calls == ["discovery", "ingestion"]
    assert not (package.package_dir(tmp_path) / "running.lock").exists()


def test_failed_stage_budget_does_not_claim_success(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    package.write_json(cfg_path, config().model_dump())
    monkeypatch.setattr(package, "blockers", lambda *a, **kw: [])

    def fail(*args):
        raise RuntimeError("Ollama kullanılamıyor")

    monkeypatch.setattr(package, "run_stage", fail)
    result = package.tick(tmp_path, cfg_path, execute=True)
    assert result["state"]["discovery"]["status"] == "failed"
    assert result["state"]["discovery"]["failures"] == 1
    assert "error" in result["outcome"]
    state = {stage: {"failures": 3} for stage in package.STAGES}
    assert package.next_stage(config(), state, time.time()) is None


def test_model_override_does_not_modify_parent_environment(monkeypatch):
    monkeypatch.setenv("HEKTOR_LLM_MODEL", "hektor-v12-30b")
    env = package.worker_env(config())
    assert env["HEKTOR_LLM_MODEL"] == config().teacher_model
    assert env["HEKTOR_UNATTENDED_TRAINING_ENABLED"] == "false"
    assert env["HEKTOR_ALLOW_FAKE_EMBEDDINGS"] == "false"
    assert os.environ["HEKTOR_LLM_MODEL"] == "hektor-v12-30b"


@pytest.mark.parametrize(
    "args, expected",
    [
        (["hektor.exe", "train", "--run"], True),
        (["python", "-m", "app.main", "lora-eval"], True),
        (["python", "scripts/merge_adapter.py"], True),
        (["hektor.exe", "train"], False),
        (["hektor.exe", "research-package", "--run"], False),
    ],
)
def test_heavy_process_detection(args, expected):
    assert package.process_is_heavy(args) is expected


def test_assessment_rejects_fabricated_evidence_and_string_boolean():
    source = "This research examines retrieval attribution under incomplete source evidence."
    row = {
        "relevant": True,
        "evidence": source,
        "reason": "Alakalı",
        "hypothesis": "Atıf kapsamı artabilir",
        "test": "Sabit sorularda A/B karşılaştır",
    }
    assert worker.verify_assessment(json.dumps(row), source)["relevant"]
    with pytest.raises(ValueError):
        worker.verify_assessment(json.dumps({**row, "evidence": "Uydurma " * 10}), source)
    with pytest.raises(ValueError):
        worker.verify_assessment(json.dumps({**row, "relevant": "true"}), source)


def test_pdf_outside_inbox_rejected_before_parse(tmp_path):
    with pytest.raises(ValueError, match="dışında"):
        worker.checked_pdf(tmp_path, {"pdf": str(tmp_path / "foreign.pdf")})


def test_pdf_hash_change_rejected(tmp_path):
    pdf = tmp_path / "data/research_package/inbox/paper.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF" + b"x" * 40_000)
    with pytest.raises(ValueError, match="hash"):
        worker.checked_pdf(tmp_path, {"pdf": str(pdf), "sha256": "old"})


def test_pause_resume_preserves_global_stop(tmp_path):
    package.control(tmp_path, "pause")
    (tmp_path / "storage/STOP_ALL").touch()
    package.control(tmp_path, "resume")
    assert not (tmp_path / "storage/STOP_RESEARCH").exists()
    assert (tmp_path / "storage/STOP_ALL").exists()


def test_running_training_interrupts_only_owned_worker(tmp_path, monkeypatch):
    package.package_dir(tmp_path).mkdir(parents=True)
    events = []

    class Child:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate-owned")
            self.returncode = 1

        def wait(self, timeout):
            return self.returncode

    monkeypatch.setattr(package.subprocess, "Popen", lambda *a, **kw: Child())
    monkeypatch.setattr(package, "blockers", lambda *a, **kw: ["Yeni eğitim başladı"])
    with pytest.raises(RuntimeError, match="duraklatıldı"):
        package.run_stage(tmp_path, tmp_path / "config.json", config(), "data")
    assert events == ["terminate-owned"]


def test_ingestion_gates_before_index_and_budget(tmp_path, monkeypatch):
    from app.memory import paper_indexer

    queue = {
        str(i): {"status": "pending", "attempts": 0, "sha256": str(i), "title": str(i)}
        for i in range(3)
    }
    package.write_json(worker.queue_path(tmp_path), queue)
    monkeypatch.setattr(worker, "guard", lambda *a: None)
    monkeypatch.setattr(worker, "checked_pdf", lambda *a: (tmp_path / "x.pdf", "text"))
    monkeypatch.setattr(worker, "assess", lambda text, row, cfg: {"relevant": row["title"] != "0"})
    indexed = []

    class Indexer:
        def ingest_one(self, disc, *, enrich):
            assert enrich is False
            indexed.append(disc.file_hash)
            return SimpleNamespace(skipped=False, n_chunks=5, paper_id="paper_1")

    monkeypatch.setattr(paper_indexer, "PaperIndexer", Indexer)
    counts = worker.ingestion(tmp_path, config())
    assert counts == {"ingested": 1, "already_present": 0, "rejected": 1, "errors": 0}
    assert indexed == ["1"]
    saved = package.read_json(worker.queue_path(tmp_path))
    assert saved["0"]["status"] == "rejected"
    assert saved["2"]["attempts"] == 0


def test_candidate_data_never_rewrites_training_source(tmp_path, monkeypatch):
    from app.brain import local_llm
    from app.memory import doc_purpose, reranking_retriever, sqlite_store
    from app.training import self_distill

    source = tmp_path / "data/lora_sft/lora_sft.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("SABİT EĞİTİM VERİSİ", encoding="utf-8")
    monkeypatch.setattr(worker, "guard", lambda *a: None)
    monkeypatch.setattr(doc_purpose, "excluded_paper_ids", lambda *a: frozenset())
    monkeypatch.setattr(
        sqlite_store, "SqliteStore", lambda: SimpleNamespace(list_papers=lambda: [])
    )
    monkeypatch.setattr(local_llm, "LocalLLM", lambda **kw: SimpleNamespace())
    monkeypatch.setattr(self_distill, "generate_questions", lambda *a, **kw: {})
    monkeypatch.setattr(self_distill, "load_generated_questions", lambda *a: [])
    monkeypatch.setattr(self_distill, "used_questions", lambda *a: set())
    job = self_distill.DistillJob("rag", "Kaynaklı örnek soru nasıl değerlendirilir?")
    monkeypatch.setattr(self_distill, "select_jobs", lambda *a, **kw: [job])
    monkeypatch.setattr(
        reranking_retriever,
        "RerankingRetriever",
        lambda: SimpleNamespace(retrieve=lambda q: [], last_search={}),
    )
    monkeypatch.setattr(
        self_distill,
        "distill_one",
        lambda *a, **kw: (
            "kabul",
            json.dumps({"messages": [], "metadata": {"teacher": config().teacher_model}}),
            "",
        ),
    )
    result = worker.data(tmp_path, config())
    assert result["accepted"] == 1
    assert Path(result["output"]).is_relative_to(tmp_path / "data/research_package/staging")
    assert source.read_text(encoding="utf-8") == "SABİT EĞİTİM VERİSİ"
    assert package.read_json(package.package_dir(tmp_path) / "questions_done.json")
