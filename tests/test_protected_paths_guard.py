"""Phase 4A — korumalı yol bekçisi (scripts/check_protected_paths.py) testleri (offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_protected_paths.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_protected_paths", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


guard = _load()

ALLOWED = [
    "app/main.py",
    "app/web/static/assets/app.js",
    "docs/file.md",
    "tests/test_x.py",
    "scripts/foo.py",
    "automation_manifest.yaml",
    ".github/workflows/ci.yml",
    "README.md",
]
BLOCKED = [
    "data/papers/x.pdf",
    "storage/state.json",
    "models/adapters/a.bin",
    "models/x.gguf",
    "vector_db/chroma/y.bin",
    ".env",
    ".env.local",
    "secret.key",
    "cert.pem",
    "keystore.p12",
    "foo.sqlite",
    "storage/sqlite/achilles.db",
]


def test_allowed_paths_not_protected() -> None:
    for p in ALLOWED:
        assert not guard.is_protected(p), f"yanlışlıkla bloklandı: {p}"


def test_blocked_paths_protected() -> None:
    for p in BLOCKED:
        assert guard.is_protected(p), f"bloklanmadı: {p}"


def test_protected_changes_preserves_order_and_filters() -> None:
    mixed = ["app/main.py", "data/x.pdf", "docs/a.md", ".env", "tests/t.py"]
    assert guard.protected_changes(mixed) == ["data/x.pdf", ".env"]


def test_windows_backslash_normalized() -> None:
    assert guard.is_protected("data\\papers\\x.pdf")
    assert guard.is_protected("models\\adapters\\a.bin")
    assert not guard.is_protected("app\\main.py")


def test_dot_slash_prefix_normalized() -> None:
    assert guard.is_protected("./storage/state.json")
    assert not guard.is_protected("./app/main.py")


def test_main_files_mode_blocks() -> None:
    rc = guard.main(["--files", "app/ok.py", "storage/bad.json"])
    assert rc == 2


def test_main_files_mode_clean() -> None:
    rc = guard.main(["--files", "app/ok.py", "docs/ok.md", "tests/t.py"])
    assert rc == 0


def test_main_empty_is_clean() -> None:
    assert guard.main(["--files"]) == 0
