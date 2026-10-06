"""Abonelik CLI'sini çalışma klasöründeki taklitçiye düşmeden çöz.

PATH taraması önce gelir. ``claude`` PATH'te yoksa (masaüstü uygulamasıyla kurulan makinelerde
olağan durum) bilinen kurulum konumlarına düşülür — aksi hâlde web arayüzü motoru "kurulu
değil" gösterir ve ikinci görüş/⚡ RUN bağlanamaz (bkz. docs/MOTOR_BAGLAMA.md).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Açık geçersiz kılma: `claude` ikilisinin MUTLAK yolu (ör. özel kurulum konumu).
CLAUDE_BIN_ENV = "HEKTOR_CLAUDE_BIN"


def _version_key(name: str) -> tuple[int, ...] | None:
    parts = name.split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def _desktop_bundled_claude(root: Path, exe: str) -> list[Path]:
    """Masaüstü uygulamasının indirdiği CLI: ``<root>/<sürüm>/<hash>/claude[.exe]``.

    Yalnız uygulamanın bütünlük doğrulamasından geçmiş kopya (yanında ``.verified``) sayılır;
    en yeni sürüm önce gelir."""
    if not root.is_dir():
        return []
    versions = [(k, d) for d in root.iterdir() if d.is_dir() and (k := _version_key(d.name))]
    found: list[Path] = []
    for _, vdir in sorted(versions, reverse=True):
        for build in sorted(vdir.iterdir()):
            candidate = build / exe
            if candidate.is_file() and (build / ".verified").exists():
                found.append(candidate)
    return found


def _known_install_paths(binary: str) -> list[Path]:
    """PATH dışındaki bilinen ``claude`` konumları (öncelik sırasıyla)."""
    if binary != "claude":
        return []
    exe = "claude.exe" if sys.platform == "win32" else "claude"
    home = Path.home()
    paths: list[Path] = []
    override = os.environ.get(CLAUDE_BIN_ENV, "").strip().strip('"')
    if override and Path(override).is_absolute():
        paths.append(Path(override))
    # Yerel (native) kurulum betiği ve eski `claude migrate-installer` konumu.
    paths += [home / ".local" / "bin" / exe, home / ".claude" / "local" / exe]
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths += _desktop_bundled_claude(Path(appdata) / "Claude" / "claude-code", exe)
        # MSIX (Microsoft Store) paketi %APPDATA% yazılarını sanallaştırır: uygulama dışından
        # başlatılan `hektor-web` kopyayı yalnız paketin LocalCache'inde görür.
        local = os.environ.get("LOCALAPPDATA")
        if local:
            for pkg in sorted((Path(local) / "Packages").glob("Claude_*")):
                roaming = pkg / "LocalCache" / "Roaming" / "Claude" / "claude-code"
                paths += _desktop_bundled_claude(roaming, exe)
    elif sys.platform == "darwin":
        support = home / "Library" / "Application Support" / "Claude" / "claude-code"
        paths += _desktop_bundled_claude(support, exe)
    return paths


def resolve_cli(binary: str) -> str | None:
    """PATH dizinlerini açık mutlak adaylarla tara; Windows'un CWD önceliğini engelle."""
    if Path(binary).is_absolute():
        found = shutil.which(binary)
        return str(Path(found).resolve()) if found else None
    if Path(binary).name != binary:
        return None
    cwd = Path.cwd().resolve()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry.strip():
            continue
        directory = Path(entry.strip('"')).resolve()
        if directory == cwd:
            continue
        found = shutil.which(str(directory / binary))
        if found:
            return str(Path(found).resolve())
    for candidate in _known_install_paths(binary):
        found = shutil.which(str(candidate))
        if found and Path(found).resolve().parent != cwd:
            return str(Path(found).resolve())
    return None
