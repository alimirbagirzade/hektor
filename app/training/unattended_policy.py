"""Single authority for training and promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingAuthorization:
    authorized: bool
    mode: str
    reason: str
    approval_id: str = ""


def authorize_training_action(
    action: str,
    summary: str,
    *,
    gates_passed: bool = False,
    agent_id: str = "auto-lora-pipeline",
) -> TrainingAuthorization:
    """Return the only authorization decision used by every training surface."""
    from app.agents.runtime import approvals, supervisor
    from app.config import get_settings

    if supervisor.is_stop_all_active():
        return TrainingAuthorization(False, "stop_all", "STOP_ALL active")
    if get_settings().unattended_training_enabled and gates_passed:
        return TrainingAuthorization(True, "unattended_policy", "quality gates passed")

    risk = "high" if "promote" in action else "critical"
    decision = approvals.require_fresh_approval(agent_id, action, risk, summary)
    return TrainingAuthorization(
        decision.authorized,
        "human_approval",
        "fresh approval consumed" if decision.authorized else "fresh approval required",
        decision.approval_id,
    )
