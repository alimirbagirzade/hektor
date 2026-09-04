"""Sürüm/sapma rozeti — /api/version + version_info (offline, ağsız).

`_maybe_refresh_remote` (ağ fetch'i) tüm testlerde monkeypatch ile kapatılır,
böylece testler deterministik ve çevrimdışı çalışır.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.web import version_info as vi
from app.web.server import app

client = TestClient(app)


def _table_git(table: dict):
    return lambda args, timeout=8: table.get(tuple(args), (0, ""))


def test_version_endpoint_offline(monkeypatch) -> None:
    monkeypatch.setattr(vi, "_maybe_refresh_remote", lambda: None)
    r = client.get("/api/version")
    assert r.status_code == 200
    d = r.json()
    for k in ("git", "branch", "behind", "ahead", "on_main", "converged", "last_update"):
        assert k in d
    assert isinstance(d["behind"], int)
    assert isinstance(d["converged"], bool)


def test_version_info_drift(monkeypatch) -> None:
    monkeypatch.setattr(vi, "_maybe_refresh_remote", lambda: None)
    monkeypatch.setattr(
        vi,
        "_git",
        _table_git(
            {
                ("rev-parse", "--is-inside-work-tree"): (0, "true"),
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
                ("rev-parse", "--short", "HEAD"): (0, "aaa1111"),
                ("rev-parse", "HEAD"): (0, "aaa1111full"),
                ("rev-parse", "--short", "origin/main"): (0, "bbb2222"),
                ("rev-parse", "origin/main"): (0, "bbb2222full"),
                ("rev-list", "--left-right", "--count", "origin/main...HEAD"): (0, "42\t0"),
            }
        ),
    )
    info = vi.get_version_info()
    assert info["git"] and info["on_main"]
    assert info["behind"] == 42 and info["ahead"] == 0
    assert info["converged"] is False


def test_version_info_converged(monkeypatch) -> None:
    monkeypatch.setattr(vi, "_maybe_refresh_remote", lambda: None)
    monkeypatch.setattr(
        vi,
        "_git",
        _table_git(
            {
                ("rev-parse", "--is-inside-work-tree"): (0, "true"),
                ("rev-parse", "--abbrev-ref", "HEAD"): (0, "main"),
                ("rev-parse", "--short", "HEAD"): (0, "ccc3333"),
                ("rev-parse", "HEAD"): (0, "samehash"),
                ("rev-parse", "--short", "origin/main"): (0, "ccc3333"),
                ("rev-parse", "origin/main"): (0, "samehash"),
                ("rev-list", "--left-right", "--count", "origin/main...HEAD"): (0, "0\t0"),
            }
        ),
    )
    info = vi.get_version_info()
    assert info["converged"] is True
    assert info["behind"] == 0 and info["ahead"] == 0


def test_version_info_no_git(monkeypatch) -> None:
    monkeypatch.setattr(vi, "_maybe_refresh_remote", lambda: None)
    monkeypatch.setattr(vi, "_git", lambda args, timeout=8: (127, ""))
    info = vi.get_version_info()
    assert info["git"] is False
    assert info["converged"] is False
