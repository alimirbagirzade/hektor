"""Araştırma motorlarının çevrimdışı yetki ve profil testleri."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.orchestration import research_engines as engines


def test_missing_gemini_closed(monkeypatch):
    monkeypatch.setattr(engines, "resolve_cli", lambda name: None)
    assert engines.blocked_reason("gemini")
    assert engines.blocked_reason("unknown")
    assert engines.blocked_reason("ollama") == ""


def test_gemini_profile(tmp_path, monkeypatch):
    entry = tmp_path / "index.js"
    monkeypatch.setattr(engines, "gemini_entry", lambda: entry)
    monkeypatch.setattr(engines, "resolve_cli", lambda name: "node")
    cmd, env, cwd = engines.prepare(
        "gemini",
        "plan",
        "base",
        42,
        {"GEMINI_API_KEY": "secret", "NODE_OPTIONS": "bad", "PATH": "path"},
        tmp_path,
    )
    assert cwd == tmp_path
    assert cmd[:2] == ["node", str(entry)]
    assert "secret" not in str(env)
    assert "NODE_OPTIONS" not in env
    settings = json.loads(Path(env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"]).read_text())
    assert not settings["hooksConfig"]["enabled"]
    assert not settings["admin"]["mcp"]["enabled"]
    assert settings["security"]["auth"]["enforcedType"] == "oauth-personal"
    policy = Path(settings["adminPolicyPaths"][0]).read_text()
    assert 'decision = "deny"' in policy and 'toolName = "*"' in policy


def test_gemini_unverified_version_closed(tmp_path, monkeypatch):
    binary = tmp_path / "gemini.cmd"
    pkg = tmp_path / "node_modules" / "@google" / "gemini-cli"
    (pkg / "dist").mkdir(parents=True)
    (pkg / "dist" / "index.js").write_text("")
    monkeypatch.setattr(engines, "resolve_cli", lambda name: str(binary))
    for version, expected in [("unknown", False), (engines.GEMINI_VERSION, True)]:
        (pkg / "package.json").write_text(
            json.dumps({"name": "@google/gemini-cli", "version": version})
        )
        assert bool(engines.gemini_entry()) is expected


def test_ollama_separate_process(tmp_path):
    cmd, _env, cwd = engines.prepare("ollama", "plan", "base", 7, {}, tmp_path)
    assert cmd[1:3] == ["-m", "app.orchestration.research_review"]
    assert cmd[-3:] == ["base", "7", "plan"]
    assert cwd is None


def test_reviewer_rejects_unapproved_model(monkeypatch):
    from app.orchestration import research_review

    monkeypatch.setattr(research_review.sys, "argv", ["review", "hektor-v14", "42", "plan"])
    with pytest.raises(ValueError):
        research_review.main()
