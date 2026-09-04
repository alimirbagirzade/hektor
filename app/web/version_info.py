"""Sürüm/sapma bilgisi — web header rozeti için (SALT-OKUMA, offline).

`achilles doctor`'ın web karşılığı: bu makinenin `origin/main`'e göre ne kadar
geride/ileride olduğunu, `main` dalında olup olmadığını ve son güncelleme log
satırını döndürür. Ağ YOK (origin/main yerel ref'ten okunur), git mutasyonu YOK.

Amaç: bir makinenin sessizce çok geride kalmasını (ör. 333 commit) önlemek —
drift artık kullanıcının sürekli baktığı web üst şeridinde görünür olur.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

# app/web/version_info.py → parents[2] = repo kökü
REPO = Path(__file__).resolve().parents[2]

# origin/main yerel ref'ini en fazla bu sıklıkta tazele (ağ). Drift'in bayat ref
# yüzünden olduğundan az görünmesini (sessiz "333 commit geride") önler; ama her
# 30 sn'lik poll'da GitHub'ı dövmez.
_FETCH_THROTTLE_S = 1800  # 30 dk


def _git(args: list[str], timeout: int = 8) -> tuple[int, str]:
    """Salt-okuma git (offline; ağ yok). (returncode, stdout.strip())."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, (proc.stdout or "").strip()


def _maybe_refresh_remote() -> None:
    """En fazla 30 dk'da bir `git fetch origin main` (offline-güvenli, throttle'lı).

    Yalnız remote-tracking ref'i (origin/main) günceller — çalışma ağacına,
    yerel dallara DOKUNMAZ. Offline ise sessizce başarısız. Böylece rozet,
    nightly güncelleme görevi bozuk olsa bile gerçek drift'i gösterir.
    """
    marker = REPO / "logs" / ".version_lastfetch"
    try:
        age = time.time() - marker.stat().st_mtime
    except OSError:
        age = float("inf")  # hiç tazelenmemiş
    if age < _FETCH_THROTTLE_S:
        return
    _git(["fetch", "--quiet", "origin", "main"], timeout=12)
    # Başarılı/başarısız fark etmez: marker'ı dokun ki offline'da da dövmeyelim.
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass


def _last_update_log() -> str | None:
    """logs/update.log son satırı — nightly güncellemenin sessiz başarısızlığını yüzeye çıkarır."""
    log = REPO / "logs" / "update.log"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    return lines[-1] if lines else None


def get_version_info() -> dict:
    """origin/main'e göre sapma özeti (salt-okuma, offline)."""
    info: dict = {
        "git": False,
        "branch": None,
        "head": None,
        "origin_main": None,
        "behind": 0,
        "ahead": 0,
        "on_main": False,
        "converged": False,
        "last_update": None,
    }

    rc, _ = _git(["rev-parse", "--is-inside-work-tree"])
    if rc != 0:
        return info
    info["git"] = True

    _maybe_refresh_remote()  # throttle'lı, offline-güvenli — origin/main'i taze tut

    _, branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    _, head = _git(["rev-parse", "--short", "HEAD"])
    _, head_full = _git(["rev-parse", "HEAD"])
    rc_om, origin_main = _git(["rev-parse", "--short", "origin/main"])
    _, om_full = _git(["rev-parse", "origin/main"])

    info["branch"] = branch or None
    info["head"] = head or None
    info["on_main"] = branch == "main"

    if rc_om == 0 and origin_main:
        info["origin_main"] = origin_main
        _, ab = _git(["rev-list", "--left-right", "--count", "origin/main...HEAD"])
        parts = ab.split()
        if len(parts) == 2:
            info["behind"] = int(parts[0]) if parts[0].isdigit() else 0
            info["ahead"] = int(parts[1]) if parts[1].isdigit() else 0
        info["converged"] = (
            info["on_main"] and bool(head_full) and bool(om_full) and head_full == om_full
        )

    info["last_update"] = _last_update_log()
    return info
