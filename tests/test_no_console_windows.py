"""Alt süreçler Windows'ta konsol penceresi AÇMAZ — 2026-10-06 masaüstü kilitlenmesi.

Olay: aday-işi çalıştırıcısı ``DETACHED_PROCESS`` ile (konsolsuz) başlıyor, asıl komutu
bayraksız açıyordu → Windows her komut için GÖRÜNÜR yeni konsol penceresi açtı. Paralel
stres testinde 45 dakikada yüzlerce pencere açıldı, explorer yanıt vermez oldu.

İki katman:
1. Statik (her platform): ``app/`` içindeki her ``subprocess`` çağrısı ``creationflags``
   verir ve kimse ``DETACHED_PROCESS`` (0x8) kullanmaz.
2. Davranış (yalnız Windows): ayrık başlatılan sürecin VE onun bayraksız açtığı torunun
   konsolu görünmez.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
FUNCS = {"run", "Popen", "check_output", "check_call", "call"}
# Ön planda koşan, çıktısı terminale AKMASI gereken (yakalanmayan) çağrılar. Yalnız macOS.
ALLOW = {"training/mlx_lora_train.py"}


def _subprocess_calls() -> list[tuple[str, int, ast.Call]]:
    found = []
    for p in sorted(APP.rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in FUNCS
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "subprocess"
            ):
                found.append((p.relative_to(APP).as_posix(), n.lineno, n))
    return found


def test_every_subprocess_call_sets_creationflags() -> None:
    calls = _subprocess_calls()
    assert len(calls) > 20  # tarayıcı gerçekten çağrı buluyor
    missing = [
        f"app/{rel}:{line}"
        for rel, line, n in calls
        if rel not in ALLOW
        and not any(k.arg == "creationflags" or k.arg is None for k in n.keywords)
    ]
    assert not missing, (
        "Bu alt süreç çağrıları Windows'ta konsol penceresi açabilir; "
        "creationflags=NO_WINDOW ekleyin (app/procutil.py): " + ", ".join(missing)
    )


def test_no_detached_process_flag_in_app() -> None:
    """``DETACHED_PROCESS`` + ``CREATE_NO_WINDOW`` birlikte verilince ikincisi yok sayılır."""
    pat = re.compile(r"\b0x0*8\b|\bDETACHED_PROCESS\b")
    bad = []
    for p in sorted(APP.rglob("*.py")):
        if p.name == "procutil.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line.split("#", 1)[0]):
                bad.append(f"{p.relative_to(APP).as_posix()}:{i}")
    assert not bad, "DETACHED_PROCESS yerine procutil.DETACHED_HIDDEN kullanın: " + ", ".join(bad)


def test_procutil_flags() -> None:
    from app import procutil

    if os.name == "nt":
        assert procutil.NO_WINDOW == 0x08000000
        assert procutil.DETACHED_HIDDEN & 0x08000000 and not procutil.DETACHED_HIDDEN & 0x8
    else:
        assert procutil.NO_WINDOW == 0 and procutil.DETACHED_HIDDEN == 0


_PROBE = r"""
import ctypes, subprocess, sys
k = ctypes.windll.kernel32; u = ctypes.windll.user32
def vis():
    h = k.GetConsoleWindow()
    return int(bool(h and u.IsWindowVisible(h)))
out = sys.argv[1]
if len(sys.argv) > 2:  # torun
    open(out, "a").write(f"torun={vis()}\n")
else:
    open(out, "a").write(f"surec={vis()}\n")
    # Çalıştırıcı gibi: torunu BAYRAKSIZ aç (üçüncü taraf betiklerin yaptığı gibi).
    subprocess.Popen([sys.executable, __file__, out, "torun"]).wait()
"""


@pytest.mark.skipif(os.name != "nt", reason="konsol penceresi yalnız Windows kavramı")
def test_detached_job_and_its_grandchild_have_no_visible_console(tmp_path) -> None:
    from app.training import candidate_jobs as cj

    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE, encoding="utf-8")
    out = tmp_path / "out.txt"
    proc = cj._popen_detached([sys.executable, str(probe), str(out)], tmp_path / "err.log")
    proc.wait(timeout=60)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and "torun=" not in (out.read_text() if out.exists() else ""):
        time.sleep(0.1)
    text = out.read_text() if out.exists() else ""
    assert "surec=0" in text and "torun=0" in text, text or (tmp_path / "err.log").read_text()


@pytest.mark.skipif(os.name != "nt", reason="konsol penceresi yalnız Windows kavramı")
def test_no_window_flag_hides_console_of_captured_call(tmp_path) -> None:
    from app.procutil import NO_WINDOW

    code = (
        "import ctypes;k=ctypes.windll.kernel32;u=ctypes.windll.user32;h=k.GetConsoleWindow();"
        "print(int(bool(h and u.IsWindowVisible(h))))"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, creationflags=NO_WINDOW
    )
    assert r.stdout.strip() == "0"
