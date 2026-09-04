from __future__ import annotations

from types import SimpleNamespace

from app.training.unattended_policy import authorize_training_action


def test_passed_gates_use_unattended_policy(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(unattended_training_enabled=True),
    )
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda: False)
    monkeypatch.setattr(
        "app.agents.runtime.approvals.require_fresh_approval",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("approval path used")),
    )

    decision = authorize_training_action("train_run", "safe local run", gates_passed=True)

    assert decision.authorized is True
    assert decision.mode == "unattended_policy"


def test_stop_all_overrides_unattended_policy(monkeypatch) -> None:
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda: True)
    decision = authorize_training_action("train_run", "run", gates_passed=True)
    assert decision.authorized is False
    assert decision.mode == "stop_all"


def test_ungated_surface_uses_human_approval(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.config.get_settings",
        lambda: SimpleNamespace(unattended_training_enabled=True),
    )
    monkeypatch.setattr("app.agents.runtime.supervisor.is_stop_all_active", lambda: False)
    monkeypatch.setattr(
        "app.agents.runtime.approvals.require_fresh_approval",
        lambda *a, **k: SimpleNamespace(authorized=False, approval_id="apr_one"),
    )
    decision = authorize_training_action("train_run", "manual", gates_passed=False)
    assert decision.authorized is False
    assert decision.approval_id == "apr_one"
