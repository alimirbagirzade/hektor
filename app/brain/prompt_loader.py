"""Load prompt templates from app/prompts/*.md."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache
def load_prompt(name: str) -> str:
    settings = get_settings()
    path = settings.prompts_dir / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt bulunamadı: {path}")
    return path.read_text(encoding="utf-8")
