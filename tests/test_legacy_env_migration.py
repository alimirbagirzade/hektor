"""Achilles → Hektor yeniden adlandırma geriye dönük uyum testleri (çevrimdışı).

Yeniden adlandırma mevcut kurulumları sessizce bozmamalı: eski ``ACHILLES_*``
ortam değişkenleri ve eski SQLite dosyası çalışmaya devam etmeli.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.config.settings import (
    _DEFAULT_SQLITE_PATH,
    ENV_PREFIX,
    LEGACY_ENV_PREFIX,
    Settings,
    _promote_legacy_env,
)


@pytest.fixture(autouse=True)
def _restore_environ():
    """_promote_legacy_env os.environ'a YAZAR; sızıntı diğer testleri kirletmesin."""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


def _clean_settings(**kwargs) -> Settings:
    """conftest HEKTOR_SQLITE_PATH'i ayarlar; varsayılan yol davranışını test etmek
    için açıkça varsayılana döneriz."""
    kwargs.setdefault("sqlite_path", _DEFAULT_SQLITE_PATH)
    return Settings(**kwargs)


def test_legacy_env_var_is_promoted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(f"{ENV_PREFIX}LLM_MODEL", raising=False)
    monkeypatch.setenv(f"{LEGACY_ENV_PREFIX}LLM_MODEL", "qwen3:8b")

    promoted = _promote_legacy_env(tmp_path / ".env")

    assert f"{LEGACY_ENV_PREFIX}LLM_MODEL" in promoted
    assert Settings().llm_model == "qwen3:8b"


def test_new_prefix_wins_over_legacy(monkeypatch, tmp_path: Path) -> None:
    """Açık HEKTOR_* ayarı ASLA eski önek tarafından ezilmemeli."""
    monkeypatch.setenv(f"{ENV_PREFIX}LLM_MODEL", "yeni")
    monkeypatch.setenv(f"{LEGACY_ENV_PREFIX}LLM_MODEL", "eski")

    assert _promote_legacy_env(tmp_path / ".env") == []
    assert Settings().llm_model == "yeni"


def test_legacy_env_file_entry_is_promoted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(f"{ENV_PREFIX}LOG_LEVEL", raising=False)
    monkeypatch.delenv(f"{LEGACY_ENV_PREFIX}LOG_LEVEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"# yorum\n{LEGACY_ENV_PREFIX}LOG_LEVEL=DEBUG\n", encoding="utf-8")

    assert _promote_legacy_env(env_file) == [f"{LEGACY_ENV_PREFIX}LOG_LEVEL"]
    assert Settings().log_level.upper() == "DEBUG"


def test_missing_env_file_is_not_an_error(tmp_path: Path) -> None:
    assert _promote_legacy_env(tmp_path / "yok.env") == []


def test_sqlite_falls_back_to_legacy_database(tmp_path: Path) -> None:
    """Yalnız eski veritabanı varsa ona düşülmeli (geçmiş öksüz kalmasın)."""
    legacy = tmp_path / "storage" / "sqlite" / "achilles_trader_ai.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"")

    assert _clean_settings(root_path=tmp_path).sqlite_file == legacy


def test_sqlite_prefers_new_database_when_present(tmp_path: Path) -> None:
    base = tmp_path / "storage" / "sqlite"
    base.mkdir(parents=True)
    (base / "achilles_trader_ai.db").write_bytes(b"")
    (base / "hektor_trader_ai.db").write_bytes(b"")

    assert _clean_settings(root_path=tmp_path).sqlite_file == base / "hektor_trader_ai.db"


def test_sqlite_uses_new_name_on_clean_install(tmp_path: Path) -> None:
    assert (
        _clean_settings(root_path=tmp_path).sqlite_file
        == tmp_path / "storage" / "sqlite" / "hektor_trader_ai.db"
    )


def test_explicit_sqlite_path_is_never_overridden(tmp_path: Path) -> None:
    """Açık ayar varsa eski dosya VARSA BİLE ona düşülmemeli."""
    legacy = tmp_path / "storage" / "sqlite" / "achilles_trader_ai.db"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"")

    settings = Settings(root_path=tmp_path, sqlite_path=Path("storage/sqlite/ozel.db"))
    assert settings.sqlite_file == tmp_path / "storage" / "sqlite" / "ozel.db"
