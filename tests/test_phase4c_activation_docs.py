"""Phase 4C — aktivasyon runbook'u (docs/PHASE4C_ACTIVATION.md) statik testleri (offline).

Runbook'un: asistanın YAPMAYACAKLARINI net belirttiğini, secret'ı yalnız referans
olarak anlattığını, aktivasyon/issue/rollback/geri-bildirim bölümlerini içerdiğini ve
auto-merge olmadığını doğrular. Gerçek aktivasyon yapılmaz.
"""

from __future__ import annotations

from pathlib import Path

_DOC = Path(__file__).resolve().parents[1] / "docs" / "PHASE4C_ACTIVATION.md"


def _r() -> str:
    return _DOC.read_text(encoding="utf-8")


def test_activation_doc_exists() -> None:
    assert _DOC.exists()


def test_assistant_will_not_do_section() -> None:
    d = _r()
    # ASCII-güvenli kontroller (Türkçe İ lower() sorunundan kaçın)
    assert "What the assistant will NOT do" in d
    assert "secret" in d  # "Asistan secret EKLEMEZ"
    assert "MERGE ETMEZ" in d  # "Asistan PR MERGE ETMEZ"
    assert "workflow" in d.lower()  # workflow tetiklemez


def test_secret_described_not_literal() -> None:
    d = _r()
    assert "ANTHROPIC_API_KEY" in d
    assert "sk-ant" not in d  # literal secret değeri yok


def test_enable_step_present() -> None:
    d = _r()
    assert "ENABLE_CLAUDE_TASK" in d
    assert "gh variable set ENABLE_CLAUDE_TASK" in d


def test_dryrun_issue_docs_only_scope() -> None:
    d = _r()
    assert "docs/PHASE4_GITHUB_AUTOMATION.md" in d
    assert "Allowed files:" in d
    forbidden_paths = (
        "app/**",
        "tests/**",
        "data/**",
        "storage/**",
        "vector_db/**",
        "models/**",
        ".env",
    )
    for forbidden in forbidden_paths:
        assert forbidden in d, f"yasak yol eksik: {forbidden}"


def test_rollback_present() -> None:
    d = _r().lower()
    assert "rollback" in d
    assert "gh variable delete enable_claude_task" in d
    assert "gh pr close" in d


def test_send_back_section() -> None:
    d = _r()
    assert "send back to the assistant" in d.lower()
    assert "guard" in d.lower()  # guard sonucu paylaşılacak


def test_no_auto_merge_and_manual_review() -> None:
    d = _r().lower()
    assert "no auto-merge" in d
    assert "manual review required" in d


def test_sha_pin_instructions() -> None:
    d = _r()
    assert "git/ref/tags/v1" in d  # SHA-pin için resmi komut
    assert "SHA" in d


def test_labels_listed() -> None:
    d = _r()
    assert "claude-task" in d
    assert "safe-refactor" in d
    assert "needs-approval" in d
