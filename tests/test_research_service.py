"""Tek tuş yöneticisi; gerçek abonelik, Ollama ve eğitim kullanmadan sınanır."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.orchestration import driver
from app.orchestration import research_package as package
from app.orchestration import research_service as service
from app.web import driver_scope
from app.web import research_package_routes as routes
from app.web.security import require_auth


@pytest.fixture
def manager(tmp_path, monkeypatch):
    obj = service.ResearchService(tmp_path)
    cfg = package.PackageConfig(
        wait_for_adapters=["v14"], interval_hours=dict.fromkeys(package.STAGES, 1)
    )
    package.write_json(obj.config_path, cfg.model_dump())
    monkeypatch.setattr(service.research_engines, "blocked_reason", lambda name: "")
    monkeypatch.setattr(service, "blockers", lambda *a, **kw: [])
    monkeypatch.setattr(package, "blockers", lambda *a, **kw: [])
    return obj


def decision(stage="discovery", proceed=True):
    return service.DECISION_MARKER + json.dumps(
        {"proceed": proceed, "stage": stage, "reason": "Sınırlı plan incelendi."}
    )


@pytest.mark.parametrize(
    "output", ["", "PASS", decision("cards"), decision().replace("true", '"true"')]
)
def test_missing_or_wrong_decision_is_rejected(output):
    with pytest.raises(ValueError):
        service.parse_decision(output, "discovery")


def test_status_never_creates_state(manager):
    assert not manager.status()["service"]["enabled"]
    assert not manager.path.exists()


def test_training_waits_without_motor_or_worker(manager, monkeypatch):
    monkeypatch.setattr(package, "blockers", lambda *a, **kw: ["Eğitim sürüyor"])
    monkeypatch.setattr(driver, "_default_runner", lambda *a, **kw: pytest.fail("Motor doğdu"))
    monkeypatch.setattr(package, "run_stage", lambda *a, **kw: pytest.fail("İşçi doğdu"))
    assert manager.start(["codex"])["service"]["status"] == "waiting"
    assert manager.reconcile_once()["blocked"] == ["Eğitim sürüyor"]
    run_id = manager.state()["run_id"]
    manager.start(["codex"])
    assert manager.state()["run_id"] == run_id
    assert service.ResearchService(manager.root).state()["enabled"]


def test_invalid_engines_cannot_enable(manager, monkeypatch):
    for names in ([], ["codex", "codex"], ["codex", "claude", "gemini"]):
        with pytest.raises(ValueError):
            manager.start(names)
    monkeypatch.setattr(service.research_engines, "blocked_reason", lambda name: "Kurulu değil")
    with pytest.raises(ValueError, match="Kurulu değil"):
        manager.start(["claude"])
    assert not manager.path.exists()


def test_two_reviews_then_one_stage_and_external_scheduler_blocked(manager, monkeypatch):
    events = []

    def runner(command, timeout, env, **kw):
        events.append(command[0])
        assert timeout == 300
        assert env["HEKTOR_API_TOKEN"] == ""
        return 0, decision()

    def worker(*args, **kwargs):
        events.append("worker:" + args[3])
        return {"stage": args[3]}

    monkeypatch.setattr(driver, "_default_runner", runner)
    monkeypatch.setattr(package, "run_stage", worker)
    manager.start(["codex", "claude"])
    external = package.tick(manager.root, manager.config_path, execute=True)
    assert external["blocked"] and not events
    result = manager.reconcile_once()
    assert events == ["codex", "claude", "worker:discovery"]
    assert result["state"]["discovery"]["status"] == "completed"
    assert len(manager.state()["last_reviews"]) == 2


def test_rejected_review_never_reaches_worker(manager, monkeypatch):
    monkeypatch.setattr(driver, "_default_runner", lambda *a, **kw: (0, decision(proceed=False)))
    monkeypatch.setattr(package, "run_stage", lambda *a, **kw: pytest.fail("İşçi doğdu"))
    manager.start(["codex"])
    result = manager.reconcile_once()
    assert "bekletti" in result["outcome"]["error"]
    assert manager.state()["failures"] == 1
    assert manager.reconcile_once()["status"] == "backoff"


def test_stop_preserves_training_and_other_stop_markers(manager, monkeypatch):
    manager.start(["codex"])
    marker = manager.root / "storage/STOP_ALL"
    marker.touch()
    terminated = []
    monkeypatch.setattr(service.engine_procs, "terminate_run", terminated.append)
    run_id = manager.state()["run_id"]
    assert manager.stop()["service"]["status"] == "disabled"
    assert terminated == [run_id]
    assert marker.exists()
    assert (manager.root / "storage/STOP_RESEARCH").exists()
    manager.start(["codex"])
    assert marker.exists()
    assert not (manager.root / "storage/STOP_RESEARCH").exists()


def test_loop_is_singleton_and_shutdown_preserves_enabled(manager, monkeypatch):
    monkeypatch.setattr(package, "blockers", lambda *a, **kw: ["Eğitim sürüyor"])
    manager.start(["codex"])

    async def run():
        manager.ensure_loop()
        task = manager._task
        manager.ensure_loop()
        assert task is manager._task
        await asyncio.sleep(0.05)
        await manager.shutdown()
        assert task.done()

    asyncio.run(run())
    assert manager.state()["enabled"]


def test_routes_only_human_can_start_and_stop(manager, monkeypatch):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_auth] = lambda: None
    monkeypatch.setattr(routes, "get_research_service", lambda: manager)
    monkeypatch.setattr(manager, "ensure_loop", lambda: None)
    with TestClient(app) as client:
        assert client.get("/api/research-package/status").status_code == 200
        assert not manager.path.exists()
        token = driver_scope.mint("test-research", ttl_s=60)
        headers = {
            driver_scope.DRIVER_TOKEN_HEADER: token,
            driver_scope.RUN_ID_HEADER: "test-research",
        }
        try:
            for action in ("start", "stop"):
                assert (
                    client.post(
                        "/api/research-package/" + action,
                        json={"engines": ["codex"]},
                        headers=headers,
                    ).status_code
                    == 403
                )
        finally:
            driver_scope.revoke_run("test-research")
        assert (
            client.post(
                "/api/research-package/start", json={"engines": ["codex"], "train": True}
            ).status_code
            == 422
        )
        assert (
            client.post("/api/research-package/start", json={"engines": ["codex"]}).status_code
            == 200
        )
        assert client.post("/api/research-package/stop").status_code == 200


def test_runner_stop_callback_terminates_real_child(monkeypatch):
    monkeypatch.setattr(driver, "STOP_POLL_S", 0.05)
    monkeypatch.setattr(driver, "_stop_all_active", lambda: False)
    code, _ = driver._default_runner(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        5,
        run_id="research-stop-test",
        stop_requested=lambda: True,
    )
    assert code == driver.STOPPED_RC
    assert driver.engine_procs.live_count() == 0


def test_unresolved_executable_is_not_spawned(monkeypatch):
    monkeypatch.setattr(driver, "_resolve_executable", lambda cmd: cmd)
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **kw: pytest.fail("Motor doğdu"))
    with pytest.raises(FileNotFoundError):
        driver._default_runner(["not-a-trusted-cli"], 1)


def test_stderr_prompt_cannot_override_final_decision(monkeypatch):
    monkeypatch.setattr(driver, "_stop_all_active", lambda: False)
    code, output = driver._default_runner(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1]); print(sys.argv[2], file=sys.stderr)",
            decision(proceed=False),
            decision(proceed=True),
        ],
        5,
        stdout_only=True,
    )
    assert code == 0
    assert not service.parse_decision(output, "discovery").proceed
