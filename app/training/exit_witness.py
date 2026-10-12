"""exit_witness.py — detached eğitim sürecinin ÇIKIŞ KODUNU kaydeden pasif tanık.

Neden: detached başlatıcı (``detached_launch.launch``) alt süreci başlatıp bırakır; web sunucusu
çıkış kodunu hiç görmez. ``train_status.json``'daki ``finished_at`` alt sürecin KENDİ yazdığı
bir beyandır — süreç kaydı yazdıktan sonra çökebilir ya da hiç yazamadan ölebilir. Bitişi
doğrulamak için bağımsız bir kanıt gerekir: işletim sisteminin bildirdiği çıkış kodu.

Tanık ayrı, gizli, düşük öncelikli bir süreçtir: hedef süreci (pid + oluşturulma zamanı ile
kimliği doğrulayarak) bekler, bitince çıkış kodunu ``models/adapters/<ad>/train_exit.json``'a
yazar. Hiçbir şeyi durdurmaz, başlatmaz, CPU harcamaz (işletim sistemi beklemesi).

Windows'ta ``psutil.Process.wait()`` çocuk olmayan süreçlerin de çıkış kodunu döndürür; bunun
için tanığın hedef süreç YAŞARKEN tutamaç açmış olması gerekir. Süreç tanık başlamadan ölmüşse
çıkış kodu artık öğrenilemez → kayıt ``gozlenemedi`` olur (başarı SAYILMAZ). POSIX'te çocuk
olmayan sürecin çıkış kodu alınamaz → yine ``gozlenemedi``.

    python -m app.training.exit_witness <adapter> <pid> <pid_create_time>
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXIT_FILE = "train_exit.json"
_ADAPTER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CREATE_TOLERANCE_S = 2.0


def exit_path(adapter: str, root: Path | None = None) -> Path:
    from app.config import get_settings

    r = root or get_settings().root
    return r / "models" / "adapters" / Path(adapter).name / EXIT_FILE


def read_exit(adapter: str, root: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads(exit_path(adapter, root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("adapter") == adapter else {}


def _write(adapter: str, rec: dict[str, Any], root: Path | None) -> None:
    dest = exit_path(adapter, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)


def witness(
    adapter: str, pid: int, create_time: float | None, *, root: Path | None = None
) -> dict[str, Any]:
    """Hedef süreci bekle, çıkış kodunu yaz. Kaydı döndürür."""
    import psutil

    if not _ADAPTER_RE.match(adapter):
        raise ValueError(f"Geçersiz adapter adı: {adapter!r}")
    base: dict[str, Any] = {
        "adapter": adapter,
        "pid": pid,
        "pid_create_time": create_time,
        "witness_started_at": datetime.now(UTC).isoformat(),
    }
    prev = read_exit(adapter, root)
    if (
        prev.get("status") == "exited"
        and prev.get("pid") == pid
        and prev.get("pid_create_time") == create_time
    ):
        return prev  # aynı koşu zaten kaydedildi
    try:
        proc = psutil.Process(pid)
        have = proc.create_time()
    except psutil.NoSuchProcess:
        rec = {
            **base,
            "status": "gozlenemedi",
            "exit_code": None,
            "detail": "Süreç tanık başlamadan bitmiş; çıkış kodu öğrenilemez.",
        }
        _write(adapter, rec, root)
        return rec
    if create_time is not None and abs(have - float(create_time)) > _CREATE_TOLERANCE_S:
        rec = {
            **base,
            "status": "gozlenemedi",
            "exit_code": None,
            "detail": f"pid {pid} başka bir süreç (oluşturulma zamanı tutmuyor).",
        }
        _write(adapter, rec, root)
        return rec
    _write(adapter, {**base, "status": "watching", "exit_code": None}, root)
    try:
        code = proc.wait()
    except psutil.NoSuchProcess:
        code = None
    rec = {
        **base,
        "status": "exited" if code is not None else "gozlenemedi",
        "exit_code": code,
        "exited_at": datetime.now(UTC).isoformat(),
        "detail": "İşletim sisteminin bildirdiği çıkış kodu."
        if code is not None
        else "Çıkış kodu alınamadı (POSIX'te çocuk olmayan süreç ya da tutamaç kaybı).",
    }
    _write(adapter, rec, root)
    return rec


def spawn(adapter: str, pid: int, create_time: float | None, root: Path) -> bool:
    """Tanığı ayrık, gizli süreç olarak başlat (başarısızlık eğitimi ETKİLEMEZ)."""
    import os
    import subprocess

    from app.procutil import DETACHED_HIDDEN

    cmd = [sys.executable, "-m", "app.training.exit_witness", adapter, str(pid)]
    if create_time is not None:
        cmd.append(repr(float(create_time)))
    from app.config import get_settings

    # `-m app.…` KOD kökünden çözülmeli (veri kökü ayrı olabilir: HEKTOR_ROOT_PATH).
    code_root = str(get_settings().source_root)
    kwargs: dict[str, Any] = {
        "cwd": code_root,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "env": {**os.environ, "HEKTOR_ROOT_PATH": str(root), "PYTHONPATH": code_root},
    }
    if os.name == "nt":
        kwargs["creationflags"] = DETACHED_HIDDEN
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(cmd, **kwargs)
    except (OSError, ValueError):
        return False
    return True


def main() -> None:
    adapter, pid = sys.argv[1], int(sys.argv[2])
    create_time = float(sys.argv[3]) if len(sys.argv) > 3 else None
    witness(adapter, pid, create_time)


if __name__ == "__main__":
    main()
