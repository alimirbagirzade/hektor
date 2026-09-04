"""Çakışma dedektörü — çevrimdışı testler (enjekte edilen fake git runner).

Gerçek depo YOK: tüm git yoklamaları enjekte edilen fake ile sürülür. Verdict semantiği
(skip/pass/warn/fail), her sinyal (index.lock/worktree/head_drift/dirty) ve delege'nin
verdict→StageStatus eşlemesi (fail→blocked, warn→completed) doğrulanır.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import app.orchestration.collision as collision_mod
from app.orchestration import delegates
from app.orchestration.collision import CollisionDetector, CollisionFinding, CollisionResult
from app.orchestration.orchestrator import RunContext
from app.orchestration.pipeline import StageStatus


class FakeGit:
    """Enjekte edilebilir git runner: argv → (rc, stdout) sözlüğünden yanıtlar."""

    def __init__(self, responses: dict[str, tuple[int, str]]) -> None:
        # anahtar = ilk iki argv token'ı boşlukla ("rev-parse HEAD"), ya da ilk token.
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> tuple[int, str]:
        self.calls.append(args)
        key2 = " ".join(args[:2])
        if key2 in self.responses:
            return self.responses[key2]
        if args and args[0] in self.responses:
            return self.responses[args[0]]
        return 0, ""


def _git(
    *,
    is_worktree: bool = True,
    head: str = "abc123",
    git_dir: str = ".git",
    worktrees: str = "",
    dirty: str = "",
) -> FakeGit:
    return FakeGit(
        {
            "rev-parse --is-inside-work-tree": (0 if is_worktree else 128, "true\n"),
            "rev-parse --git-dir": (0, git_dir + "\n"),
            "rev-parse HEAD": (0, head + "\n"),
            "worktree list": (0, worktrees),
            "status --porcelain": (0, dirty),
        }
    )


def _find(result: CollisionResult, name: str) -> CollisionFinding | None:
    return next((f for f in result.findings if f.name == name), None)


# ── verdict semantiği ─────────────────────────────────────────────────────────


def test_skip_when_not_git(tmp_path: Path) -> None:
    det = CollisionDetector(_git(is_worktree=False), repo_root=tmp_path)
    r = det.run()
    assert r.verdict == "skip"
    assert r.findings == []


def test_pass_when_clean(tmp_path: Path) -> None:
    # .git/index.lock yok (tmp_path boş), tek worktree, temiz ağaç.
    det = CollisionDetector(_git(git_dir=str(tmp_path / ".git")), repo_root=tmp_path)
    r = det.run()
    assert r.verdict == "pass"


def test_warn_on_dirty_tree_does_not_block(tmp_path: Path) -> None:
    det = CollisionDetector(
        _git(git_dir=str(tmp_path / ".git"), dirty=" M app/foo.py\n M app/bar.py\n"),
        repo_root=tmp_path,
    )
    r = det.run()
    assert r.verdict == "warn"
    f = _find(r, "dirty")
    assert f is not None and f.severity == "warn" and "2" in f.detail


def test_fail_on_index_lock(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index.lock").write_text("", encoding="utf-8")
    det = CollisionDetector(_git(git_dir=str(git_dir)), repo_root=tmp_path)
    r = det.run()
    assert r.verdict == "fail"
    assert _find(r, "index_lock").severity == "block"


def test_fail_on_head_drift(tmp_path: Path) -> None:
    det = CollisionDetector(
        _git(git_dir=str(tmp_path / ".git"), head="newsha999"),
        repo_root=tmp_path,
        baseline_head="oldsha000",
    )
    r = det.run()
    assert r.verdict == "fail"
    f = _find(r, "head_drift")
    assert f is not None and f.severity == "block"


def test_no_head_drift_when_matches(tmp_path: Path) -> None:
    det = CollisionDetector(
        _git(git_dir=str(tmp_path / ".git"), head="samesha"),
        repo_root=tmp_path,
        baseline_head="samesha",
    )
    r = det.run()
    assert r.verdict == "pass"
    assert _find(r, "head_drift") is None


def test_fail_on_same_branch_multiple_worktrees(tmp_path: Path) -> None:
    porcelain = (
        "worktree /repo/main\nHEAD aaa\nbranch refs/heads/feature\n\n"
        "worktree /repo/wt2\nHEAD bbb\nbranch refs/heads/feature\n\n"
    )
    det = CollisionDetector(
        _git(git_dir=str(tmp_path / ".git"), worktrees=porcelain), repo_root=tmp_path
    )
    r = det.run()
    assert r.verdict == "fail"
    f = _find(r, "worktree")
    assert f is not None and f.severity == "block" and "feature" in f.detail


def test_info_on_distinct_branch_worktrees(tmp_path: Path) -> None:
    porcelain = (
        "worktree /repo/main\nHEAD aaa\nbranch refs/heads/main\n\n"
        "worktree /repo/wt2\nHEAD bbb\nbranch refs/heads/feature\n\n"
    )
    det = CollisionDetector(
        _git(git_dir=str(tmp_path / ".git"), worktrees=porcelain), repo_root=tmp_path
    )
    r = det.run()
    assert r.verdict == "pass"  # farklı branch'ler bilgi amaçlı (info), durdurmaz
    f = _find(r, "worktree")
    assert f is not None and f.severity == "info"


def test_block_dominates_warn(tmp_path: Path) -> None:
    """Hem kirli ağaç (warn) hem worktree çakışması (block) → fail (block baskın)."""
    porcelain = (
        "worktree /repo/main\nHEAD aaa\nbranch refs/heads/x\n\n"
        "worktree /repo/wt2\nHEAD bbb\nbranch refs/heads/x\n\n"
    )
    det = CollisionDetector(
        _git(git_dir=str(tmp_path / ".git"), worktrees=porcelain, dirty=" M a.py\n"),
        repo_root=tmp_path,
    )
    r = det.run()
    assert r.verdict == "fail"


def test_to_dict_shape(tmp_path: Path) -> None:
    d = CollisionDetector(_git(git_dir=str(tmp_path / ".git")), repo_root=tmp_path).run().to_dict()
    assert set(d) == {"verdict", "summary", "findings"}
    assert all(set(f) == {"name", "severity", "detail"} for f in d["findings"])


# ── delege verdict → StageStatus eşlemesi ──────────────────────────────────────


def _ctx() -> RunContext:
    return RunContext(run_id="r", stage="collision", run={}, params={}, store=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("pass", StageStatus.completed),
        ("warn", StageStatus.completed),  # uyarı durdurmaz
        ("skip", StageStatus.skipped),
        ("fail", StageStatus.blocked),  # çakışma insan eylemi bekler
    ],
)
def test_delegate_maps_verdict_to_status(
    monkeypatch: pytest.MonkeyPatch, verdict: str, expected: StageStatus
) -> None:
    monkeypatch.setattr(
        collision_mod.CollisionDetector,
        "run",
        lambda self: CollisionResult(verdict, "özet", [CollisionFinding("x", "info")]),
    )
    res = delegates.collision(_ctx())
    assert res.status == expected
    assert res.output["verdict"] == verdict
    assert res.message == "özet"


def test_delegate_passes_baseline_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delege ctx.params['baseline_head']'i CollisionDetector'a iletmeli."""
    captured: dict[str, str | None] = {}

    def _fake_init(self: object, *a: object, baseline_head: str | None = None, **k: object) -> None:
        captured["baseline_head"] = baseline_head

    monkeypatch.setattr(collision_mod.CollisionDetector, "__init__", _fake_init)
    monkeypatch.setattr(
        collision_mod.CollisionDetector, "run", lambda self: CollisionResult("pass", "ok", [])
    )
    ctx = RunContext(
        run_id="r",
        stage="collision",
        run={},
        params={"baseline_head": "deadbeef"},
        store=None,  # type: ignore[arg-type]
    )
    delegates.collision(ctx)
    assert captured["baseline_head"] == "deadbeef"
