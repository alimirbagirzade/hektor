"""RLM tool allowlist testleri (talimat §12/§17, offline)."""

from __future__ import annotations

import pytest

from app.rlm.safe_tools import build_default_registry
from app.rlm.tool_registry import SafeToolRegistry, ToolNotAllowed


def test_register_denies_unallowed_name():
    reg = SafeToolRegistry()
    with pytest.raises(ToolNotAllowed):
        reg.register("os_system", lambda **k: None)  # allowlist dışı


def test_call_denies_unknown_tool():
    reg = build_default_registry()
    with pytest.raises(ToolNotAllowed):
        reg.call("shell_exec", cmd="rm -rf /")


def test_allowed_tool_calculator_works_and_is_safe():
    reg = build_default_registry()
    out = reg.call("calculator", expression="2 + 3 * 4")
    assert out["ok"] is True
    assert out["result"] == 14.0


def test_tool_exception_returns_structured_error_not_raise():
    reg = build_default_registry()
    out = reg.call("calculator", expression="")  # ValueError içeride → structured
    assert out["ok"] is False
    assert "error" in out


def test_default_registry_only_allowlisted_tools():
    reg = build_default_registry()
    from app.rlm.tool_registry import ALLOWED_TOOL_NAMES

    assert set(reg.available()).issubset(set(ALLOWED_TOOL_NAMES))
