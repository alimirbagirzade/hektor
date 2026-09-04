from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

# Güvenli komut whitelist — sadece bunlar çalıştırılabilir
_ALLOWED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^ollama\s+(--version|list|ps|version)$"),
    re.compile(r"^ollama\s+pull\s+[\w./:-]+$"),
    re.compile(r"^ollama\s+rm\s+[\w./:-]+$"),
    re.compile(r"^ollama\s+run\s+[\w./:-]+(\s+--nowordwrap)?$"),
]

_DENIED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+-rf"),
    re.compile(r"sudo"),
    re.compile(r"curl.*\|\s*sh"),
    re.compile(r"chmod\s+777"),
    re.compile(r"format"),
    re.compile(r">\s*/dev/"),
]


@dataclass
class CommandResult:
    command: str
    status: Literal["ok", "denied", "error", "not_found"]
    output: str = ""
    error: str = ""


def _is_allowed(cmd: str) -> bool:
    for pattern in _DENIED_PATTERNS:
        if pattern.search(cmd):
            return False
    return any(pattern.match(cmd.strip()) for pattern in _ALLOWED_PATTERNS)


def _run(cmd: str, timeout: int = 60) -> CommandResult:
    if not _is_allowed(cmd):
        return CommandResult(
            command=cmd, status="denied", error=f"Güvenlik: '{cmd}' izin listesinde yok"
        )
    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            command=cmd,
            status="ok" if result.returncode == 0 else "error",
            output=result.stdout.strip(),
            error=result.stderr.strip(),
        )
    except FileNotFoundError:
        return CommandResult(command=cmd, status="not_found", error="Komut bulunamadı")
    except subprocess.TimeoutExpired:
        return CommandResult(command=cmd, status="error", error=f"Timeout ({timeout}s)")
    except Exception as e:
        return CommandResult(command=cmd, status="error", error=str(e))


def is_ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def get_ollama_version() -> str:
    r = _run("ollama --version")
    return r.output if r.status == "ok" else "unknown"


def list_models() -> list[str]:
    r = _run("ollama list")
    if r.status != "ok":
        return []
    lines = r.output.splitlines()
    models = []
    for line in lines[1:]:  # başlık satırını atla
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models


def pull_model(ollama_name: str) -> CommandResult:
    """Modeli Ollama ile indir."""
    return _run(f"ollama pull {ollama_name}", timeout=600)


def smoke_test(ollama_name: str) -> CommandResult:
    """Modelin çalışıp çalışmadığını test et — kısa prompt."""
    cmd = f"ollama run {ollama_name} --nowordwrap"
    if not _is_allowed(cmd):
        return CommandResult(command=cmd, status="denied", error="İzin verilmedi")
    try:
        result = subprocess.run(
            ["ollama", "run", ollama_name],
            input="Say: OK",
            capture_output=True,
            text=True,
            timeout=60,
        )
        ok = result.returncode == 0 and len(result.stdout.strip()) > 0
        return CommandResult(
            command=cmd,
            status="ok" if ok else "error",
            output=result.stdout.strip()[:200],
            error=result.stderr.strip()[:200],
        )
    except Exception as e:
        return CommandResult(command=cmd, status="error", error=str(e))


def install_guide_text() -> str:
    """Ollama kurulum talimatlarını metin olarak döndür."""
    return (
        "Ollama kurulu değil.\n"
        "Kurmak için:\n"
        "  macOS/Linux: https://ollama.com/download\n"
        "  Windows    : https://ollama.com/download/windows\n"
        "Sonra tekrar: achilles install --model <model>"
    )
