"""Phase 4B — manuel dry-run dokümanı statik testleri (offline).

Dry-run planının zararsız (yalnız-docs), guard/CI'lı, auto-merge'siz ve geri-alınabilir
olduğunu doğrular. Gerçek workflow tetiklenmez.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DRYRUN = _ROOT / "docs" / "PHASE4B_DRYRUN.md"


def _r() -> str:
    return _DRYRUN.read_text(encoding="utf-8")


def test_dryrun_doc_exists() -> None:
    assert _DRYRUN.exists()


def test_dryrun_is_docs_only_scope() -> None:
    d = _r()
    assert "docs/PHASE4_GITHUB_AUTOMATION.md" in d  # izinli tek dosya
    assert "Allowed files:" in d
    assert "Forbidden:" in d
    # app/ ve veri/model yolları yasak listesinde
    for forbidden in ("app/", "data/", "storage/", "models/", ".env"):
        assert forbidden in d, f"yasak yol eksik: {forbidden}"


def test_dryrun_requires_no_auto_merge_and_human_review() -> None:
    d = _r().lower()
    assert "no auto-merge" in d
    assert "manual review required" in d or "insan inceleme" in d


def test_dryrun_has_before_enabling_checklist() -> None:
    d = _r()
    assert "ANTHROPIC_API_KEY" in d
    assert "branch protection" in d.lower()
    assert "ENABLE_CLAUDE_TASK" in d  # aktivasyon anahtarı


def test_dryrun_has_rollback() -> None:
    d = _r().lower()
    assert "rollback" in d
    assert "close" in d or "kapat" in d
    assert "delete" in d or "sil" in d


def test_dryrun_labels() -> None:
    d = _r()
    assert "claude-task" in d
    assert "safe-refactor" in d
