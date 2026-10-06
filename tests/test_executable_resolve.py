"""`resolve_cli`: PATH dışındaki bilinen `claude` konumları (masaüstü uygulaması kurulumu).

Çevrimdışı; gerçek `claude` doğurulmaz, PATH ve ev dizini geçici klasöre yönlendirilir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.orchestration import engines, executable

EXE = "claude.exe" if sys.platform == "win32" else "claude"


def _make_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def _bundle(root: Path, version: str, build: str, *, verified: bool = True) -> Path:
    exe = _make_exe(root / version / build / EXE)
    if verified:
        (exe.parent / ".verified").write_text("", encoding="utf-8")
    return exe


@pytest.fixture
def isolated(monkeypatch, tmp_path) -> Path:
    """Boş PATH + sahte ev/APPDATA/LOCALAPPDATA → yalnız testin kurduğu konumlar görünür."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "bos_path"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.delenv(executable.CLAUDE_BIN_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _desktop_root(tmp: Path) -> Path:
    if sys.platform == "darwin":
        return tmp / "home" / "Library" / "Application Support" / "Claude" / "claude-code"
    return tmp / "appdata" / "Claude" / "claude-code"


def test_missing_everywhere_returns_none(isolated) -> None:
    assert executable.resolve_cli("claude") is None


@pytest.mark.skipif(sys.platform not in ("win32", "darwin"), reason="masaüstü uygulaması yok")
def test_desktop_bundled_cli_newest_verified_wins(isolated) -> None:
    root = _desktop_root(isolated)
    _bundle(root, "2.1.9", "aaa")
    newest = _bundle(root, "2.1.10", "bbb")  # sayısal sıra: 10 > 9
    _bundle(root, "2.1.11", "ccc", verified=False)  # bütünlük işareti yok → sayılmaz
    found = executable.resolve_cli("claude")
    assert found is not None and Path(found) == newest.resolve()


@pytest.mark.skipif(sys.platform != "win32", reason="MSIX sanallaştırması yalnız Windows")
def test_msix_packaged_appdata_is_scanned(isolated) -> None:
    pkg = isolated / "localappdata" / "Packages" / "Claude_abc123"
    exe = _bundle(pkg / "LocalCache" / "Roaming" / "Claude" / "claude-code", "2.1.288", "x")
    found = executable.resolve_cli("claude")
    assert found is not None and Path(found) == exe.resolve()


def test_native_install_and_env_override(isolated, monkeypatch) -> None:
    native = _make_exe(isolated / "home" / ".local" / "bin" / EXE)
    assert Path(executable.resolve_cli("claude") or "") == native.resolve()
    custom = _make_exe(isolated / "ozel" / EXE)
    monkeypatch.setenv(executable.CLAUDE_BIN_ENV, str(custom))
    assert Path(executable.resolve_cli("claude") or "") == custom.resolve()


def test_path_wins_over_known_locations(isolated, monkeypatch) -> None:
    _make_exe(isolated / "home" / ".local" / "bin" / EXE)
    on_path = _make_exe(isolated / "pathdir" / EXE)
    monkeypatch.setenv("PATH", str(on_path.parent))
    assert Path(executable.resolve_cli("claude") or "") == on_path.resolve()


def test_fallback_only_for_claude(isolated) -> None:
    _make_exe(
        isolated / "home" / ".local" / "bin" / ("codex.exe" if EXE.endswith("exe") else "codex")
    )
    assert executable.resolve_cli("codex") is None


def test_engine_probe_uses_known_locations(isolated) -> None:
    """Web motor seçicisi (engines.available) PATH dışındaki kurulumu "kurulu" görmeli."""
    _make_exe(isolated / "home" / ".local" / "bin" / EXE)
    engines.reset_probe_cache()
    try:
        assert engines.available("claude", use_cache=False) is True
        assert engines.run_blocked_reason("claude") == ""
    finally:
        engines.reset_probe_cache()
