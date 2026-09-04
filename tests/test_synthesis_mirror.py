"""Sentez aynalama (synthesis mirror) birim testleri — çevrimdışı.

`mirror_synthesis_report` üretilen makaleyi yapılandırılmış ayna dizinine
kopyalar; ayar boşsa no-op'tur ve hatalar sentez üretimini KIRMAZ.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import app.research.synthesis_paper as sp


def _fake_settings(mirror_dir: str) -> types.SimpleNamespace:
    """get_settings() yerine geçecek minimal sahte ayar nesnesi."""
    return types.SimpleNamespace(synthesis_mirror_dir=mirror_dir)


def test_mirror_disabled_when_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sp, "get_settings", lambda: _fake_settings(""))
    report = tmp_path / "sentez_x.md"
    report.write_text("içerik", encoding="utf-8")

    assert sp.synthesis_mirror_dir() is None
    assert sp.mirror_synthesis_report(report) is None


def test_mirror_copies_and_creates_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Ayna dizini henüz YOK — fonksiyon oluşturmalı.
    mirror = tmp_path / "Sentez Makelesi"
    monkeypatch.setattr(sp, "get_settings", lambda: _fake_settings(str(mirror)))

    report = tmp_path / "sentez_20260617_1200.md"
    report.write_text("# Sentez\nhipotez", encoding="utf-8")

    dest = sp.mirror_synthesis_report(report)

    assert dest is not None
    assert dest == mirror / report.name
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "# Sentez\nhipotez"


def test_mirror_never_raises_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sp, "get_settings", lambda: _fake_settings(str(tmp_path / "ayna")))

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk dolu")

    monkeypatch.setattr(sp.shutil, "copy2", _boom)

    report = tmp_path / "sentez_y.md"
    report.write_text("x", encoding="utf-8")

    # Hata yutulmalı → None döner, exception fırlatmaz.
    assert sp.mirror_synthesis_report(report) is None
