"""Phase 4A — GitHub workflow ŞABLONLARI statik güvenlik testleri (offline).

Workflow GERÇEK çalıştırılmaz; yalnız dosya içeriği doğrulanır: auto-merge yok,
tehlikeli aksiyon yasak metni var, protected-path guard (pre+post) var, doğru CI
komutları var, secret literal değer yok, nightly kod değiştirmiyor, doküman güvenlik
modelini içeriyor.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_WF = _ROOT / ".github" / "workflows"
_TASK = _WF / "claude-code-task.yml"
_NIGHTLY = _WF / "nightly-automation-audit.yml"
_DOC = _ROOT / "docs" / "PHASE4_GITHUB_AUTOMATION.md"
_GUARD = _ROOT / "scripts" / "check_protected_paths.py"


def _r(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_workflow_and_doc_files_exist() -> None:
    assert _TASK.exists()
    assert _NIGHTLY.exists()
    assert _DOC.exists()
    assert _GUARD.exists()


def test_no_auto_merge_mechanism() -> None:
    for p in (_TASK, _NIGHTLY):
        c = _r(p).lower()
        assert "gh pr merge" not in c
        assert "--auto" not in c
        assert "automerge" not in c
        assert "auto_merge" not in c
        assert "merge_method" not in c


def test_task_workflow_forbids_dangerous_actions() -> None:
    c = _r(_TASK)
    assert "train --run" in c  # SERT kurallarda yasak
    assert "Do not push to main" in c
    assert "Do not promote adapters" in c
    assert "Do not modify data/, storage/, vector_db/, models/" in c


def test_task_workflow_has_protected_guard_pre_and_post() -> None:
    c = _r(_TASK)
    assert c.count("check_protected_paths.py") >= 2  # pre + post guard


def test_task_workflow_ci_commands() -> None:
    c = _r(_TASK)
    assert "uv sync --extra dev" in c
    assert "uv run ruff check app tests" in c
    assert "uv run ruff format --check app tests" in c
    assert "uv run mypy app" in c
    assert 'pytest -m "not ollama"' in c


def test_task_workflow_minimal_perms_no_push_main() -> None:
    c = _r(_TASK)
    assert "permissions:" in c
    assert "git push origin main" not in c
    assert "push --force" not in c
    assert "force-push" not in c.lower()


def test_no_literal_secret_values() -> None:
    for p in (_TASK, _NIGHTLY, _DOC):
        c = _r(p)
        assert "sk-ant" not in c  # gerçek Anthropic anahtar deseni yok
    # Secret YALNIZ ${{ secrets.* }} referansıyla geçer (literal değer değil)
    assert "secrets.ANTHROPIC_API_KEY" in _r(_TASK)


def test_task_only_triggers_on_claude_task_label() -> None:
    c = _r(_TASK)
    assert "claude-task" in c
    assert "github.event.label.name == 'claude-task'" in c


def test_nightly_is_report_only_no_code_change() -> None:
    c = _r(_NIGHTLY)
    assert "train --run" not in c
    assert "git commit" not in c
    assert "git push" not in c
    assert "create-pull-request" not in c
    assert "upload-artifact" in c  # rapor repoya commit edilmez → artifact


def test_nightly_cron_default_off() -> None:
    c = _r(_NIGHTLY)
    assert "schedule:" in c
    assert "ENABLE_NIGHTLY_AUDIT" in c  # zamanlanmış koşu varsayılan KAPALI


def test_doc_has_safety_model_sections() -> None:
    d = _r(_DOC)
    assert "Strict Safety Model" in d
    assert "Protected Paths" in d
    assert "ANTHROPIC_API_KEY" in d
    assert "auto-merge" in d.lower() or "merge" in d.lower()
    assert "TEMPLATE" in d or "ŞABLON" in d  # aktivasyon kullanıcıda


def test_guard_referenced_by_task_workflow() -> None:
    assert "scripts/check_protected_paths.py" in _r(_TASK)


# --- Phase 4B: doğrulanmış action syntax + label/dependency guard'ları ---
def test_task_skips_dangerous_and_human_only_labels() -> None:
    c = _r(_TASK)
    for lbl in ("human-only", "no-claude", "dangerous-change", "needs-approval"):
        assert lbl in c, f"bloklayıcı label eksik: {lbl}"
    # skip-notice işi bu görevleri atlayıp açıklama yazar
    assert "skip-notice" in c
    assert "skipped" in c.lower()


def test_task_uses_verified_claude_action() -> None:
    c = _r(_TASK)
    # Resmi README ile doğrulandı. Referans @v1 VEYA 40-hex commit SHA (supply-chain pin).
    assert re.search(r"anthropics/claude-code-action@([0-9a-f]{40}|v1)\b", c), (
        "claude-code-action referansı yok / pin formatı geçersiz"
    )
    assert "anthropic_api_key:" in c
    assert "claude_args:" in c or "allowed_tools:" in c
    assert "prompt:" in c


def test_task_activation_gated_inert_by_default() -> None:
    c = _r(_TASK)
    # Claude adımı repo değişkeni olmadan SKIP → varsayılan inert (aktivasyon kullanıcıda)
    assert "vars.ENABLE_CLAUDE_TASK == 'true'" in c
    assert "vars.ENABLE_CLAUDE_TASK != 'true'" in c  # inert placeholder


def test_task_dependency_change_detector() -> None:
    c = _r(_TASK)
    assert "needs-approval" in c
    assert "pyproject" in c  # bağımlılık/iş-akışı değişimi tespiti


# --- Phase 4C: doğrulanmış tool-kısıtlama + PR-tetikli needs-approval label ---
def test_task_tool_restriction_verified_format() -> None:
    c = _r(_TASK)
    # docs/configuration.md ile doğrulandı: claude_args içinde --allowedTools (camelCase)
    assert "--allowedTools" in c
    assert "--disallowedTools" in c
    # allow-list yalnız güvenli düzenleme + offline komut içerir
    assert "Bash(uv run pytest" in c
    assert "Bash(uv run mypy app)" in c
    # tehlikeli komutlar açıkça yasak
    assert "Bash(achilles train" in c
    assert "Bash(rm" in c


def test_task_pull_request_needs_approval_job() -> None:
    c = _r(_TASK)
    assert "pull_request:" in c  # PR olayında label işi koşar
    assert "dependency-approval-label" in c
    assert "gh pr edit" in c  # mekanik label-add (PR numarası bilinir)
    assert "github.event.pull_request.number" in c
