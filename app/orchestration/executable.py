"""Abonelik CLI'sini çalışma klasöründeki taklitçiye düşmeden çöz."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


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
    return None
