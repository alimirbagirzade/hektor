"""collision.py — eşzamanlı oturum/worktree çakışması dedektörü.

Paylaşılan working tree'de başka bir oturum `git add -A && commit` yapınca BENİM
uncommitted fix'lerim kendi commit'ine süpürülebilir ([[concurrent-session-worktree-collision]]);
ya da HEAD/branch altımdan kayar. Bu modül, eğitim orkestrasyonu ilerlemeden ÖNCE git
durumunu salt-okuma yoklayıp böyle çakışma sinyallerini yüzeye çıkarır:

  1. is_git      — git deposu mu (değilse 'skip'; kusur değil, "burada test edilemez").
  2. index_lock  — `.git/index.lock` var mı (aktif eşzamanlı git işlemi — yüksek risk).
  3. worktree    — aynı branch birden çok worktree'de checkout'lu mu (worktree çakışması).
  4. head_drift  — başlangıçta kaydedilen HEAD'den kaymış mı (ağaç altımdan kaydı).
  5. dirty       — uncommitted izlenen değişiklik var mı (başka oturum `git add -A` ile yutabilir).

Severity → verdict:
  - git yok                      → 'skip' (hat DURMAZ; sonraki insan kapısına geçer).
  - 'block' bulgu (lock/worktree → 'fail': eğitim ilerlemeden çözülmeli (commit/stash/çakışmayı
     /head_drift)                  gider). Delege bunu BLOCKED'a çevirir (insan eylemi — Kural 8).
  - yalnız 'warn' bulgu (dirty)  → 'warn': gürültülü uyarı ama hattı durdurmaz (completed).
  - bulgu yok                    → 'pass'.

git ENJEKTE edilebilir (git_runner) → testler gerçek depo olmadan çalışır (offline, Kural).
Tüm yoklamalar SALT-OKUMA (hiçbir git mutasyonu yapılmaz).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# git_runner sözleşmesi: argv (git'siz) -> (returncode, stdout). stderr yutulur (yoklama).
GitRunner = Callable[[list[str]], tuple[int, str]]

# Bulgu ağırlıkları.
_BLOCK = "block"
_WARN = "warn"
_INFO = "info"


@dataclass
class CollisionFinding:
    """Tek bir çakışma sinyali."""

    name: str
    severity: str  # "block" | "warn" | "info"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "severity": self.severity, "detail": self.detail}


@dataclass
class CollisionResult:
    """Çakışma taramasının bütünsel sonucu (delege StageStatus'a çevirir)."""

    verdict: str  # "pass" | "skip" | "warn" | "fail"
    summary: str
    findings: list[CollisionFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
        }


class CollisionDetector:
    """git durumunu salt-okuma yoklayarak eşzamanlı oturum/worktree çakışmasını tespit eder."""

    def __init__(
        self,
        git_runner: GitRunner | None = None,
        *,
        repo_root: str | Path | None = None,
        baseline_head: str | None = None,
    ) -> None:
        self._git = git_runner
        self._repo_root = Path(repo_root) if repo_root is not None else None
        self.baseline_head = (baseline_head or "").strip()

    # ── bağımlılık çözümleme ───────────────────────────────────────────────────

    def _run_git(self, args: list[str]) -> tuple[int, str]:
        if self._git is not None:
            return self._git(args)
        return self._default_git(args)

    def _default_git(self, args: list[str]) -> tuple[int, str]:
        import subprocess  # lazy: enjekte edilen testler subprocess'e dokunmaz

        try:
            proc = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                cwd=str(self._repo_root) if self._repo_root else None,
            )
        except Exception as exc:  # git yok/patladı → çağıran skip'e düşsün
            log.debug("Collision: git çalıştırılamadı (%s): %s", args, exc)
            return 1, ""
        return proc.returncode, proc.stdout or ""

    def _git_dir(self) -> Path | None:
        """`.git` dizini (lock dosyası kontrolü için). Çözülemezse None."""
        rc, out = self._run_git(["rev-parse", "--git-dir"])
        if rc != 0:
            return None
        gd = out.strip()
        if not gd:
            return None
        p = Path(gd)
        if not p.is_absolute() and self._repo_root is not None:
            p = self._repo_root / p
        return p

    # ── ana akış ─────────────────────────────────────────────────────────────────

    def run(self) -> CollisionResult:
        findings: list[CollisionFinding] = []

        # 0) git deposu mu — değilse skip (kusur değil, "burada test edilemez").
        rc, _ = self._run_git(["rev-parse", "--is-inside-work-tree"])
        if rc != 0:
            return CollisionResult(
                "skip",
                "git deposu değil / git yok — çakışma taraması atlandı.",
                findings,
            )

        self._probe_index_lock(findings)
        head = self._probe_head_drift(findings)
        self._probe_worktree_collision(findings, head)
        self._probe_dirty(findings)

        return self._verdict(findings)

    def _verdict(self, findings: list[CollisionFinding]) -> CollisionResult:
        if any(f.severity == _BLOCK for f in findings):
            blockers = "; ".join(f.detail for f in findings if f.severity == _BLOCK)
            return CollisionResult("fail", f"Çakışma tespit edildi: {blockers}", findings)
        if any(f.severity == _WARN for f in findings):
            warns = "; ".join(f.detail for f in findings if f.severity == _WARN)
            return CollisionResult("warn", f"Olası çakışma riski: {warns}", findings)
        return CollisionResult("pass", "Çakışma sinyali yok — git durumu temiz.", findings)

    # ── tekil yoklamalar ──────────────────────────────────────────────────────────

    def _probe_index_lock(self, findings: list[CollisionFinding]) -> None:
        git_dir = self._git_dir()
        if git_dir is None:
            return
        try:
            locked = (git_dir / "index.lock").exists()
        except OSError:
            return
        if locked:
            findings.append(
                CollisionFinding(
                    "index_lock",
                    _BLOCK,
                    "`.git/index.lock` mevcut — başka bir git işlemi aktif "
                    "(eşzamanlı oturum çakışması riski).",
                )
            )

    def _probe_head_drift(self, findings: list[CollisionFinding]) -> str:
        rc, out = self._run_git(["rev-parse", "HEAD"])
        head = out.strip() if rc == 0 else ""
        if self.baseline_head and head and head != self.baseline_head:
            findings.append(
                CollisionFinding(
                    "head_drift",
                    _BLOCK,
                    f"HEAD kaydedilen koşu başlangıcından kaydı "
                    f"({self.baseline_head[:8]} → {head[:8]}) — ağaç altımdan değişti.",
                )
            )
        return head

    def _probe_worktree_collision(self, findings: list[CollisionFinding], head: str) -> None:
        """Aynı branch birden çok worktree'de checkout'lu mu (literal worktree çakışması)."""
        rc, out = self._run_git(["worktree", "list", "--porcelain"])
        if rc != 0 or not out.strip():
            return
        # Porcelain: her worktree bloğu "worktree <path>" + (ops.) "branch refs/heads/<b>".
        branch_counts: dict[str, int] = {}
        n_worktrees = 0
        for line in out.splitlines():
            if line.startswith("worktree "):
                n_worktrees += 1
            elif line.startswith("branch "):
                ref = line[len("branch ") :].strip()
                branch_counts[ref] = branch_counts.get(ref, 0) + 1
        dup = sorted(b for b, c in branch_counts.items() if c > 1)
        if dup:
            short = ", ".join(b.replace("refs/heads/", "") for b in dup)
            findings.append(
                CollisionFinding(
                    "worktree",
                    _BLOCK,
                    f"Aynı branch birden çok worktree'de checkout'lu: {short} "
                    "(eşzamanlı worktree çakışması).",
                )
            )
        elif n_worktrees > 1:
            findings.append(
                CollisionFinding(
                    "worktree",
                    _INFO,
                    f"{n_worktrees} worktree aktif (farklı branch'ler) — bilgi amaçlı.",
                )
            )

    def _probe_dirty(self, findings: list[CollisionFinding]) -> None:
        """Uncommitted izlenen değişiklikler — başka oturumun `git add -A`'i süpürebilir."""
        rc, out = self._run_git(["status", "--porcelain", "--untracked-files=no"])
        if rc != 0:
            return
        changed = [ln for ln in out.splitlines() if ln.strip()]
        if changed:
            findings.append(
                CollisionFinding(
                    "dirty",
                    _WARN,
                    f"{len(changed)} uncommitted izlenen değişiklik — başka bir oturumun "
                    "`git add -A`'i süpürebilir; commit/stash önerilir.",
                )
            )
