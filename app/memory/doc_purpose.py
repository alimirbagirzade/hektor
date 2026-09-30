"""Korpus dokümanı amaç etiketi (``configs/rag/doc_purposes.yaml``).

Retrieval'ın proje kılavuzu gibi iç dokümanları teknik sorulara "kaynak" diye getirmesini
engellemek için amaç bazlı dışlama. Etiket paper_id'ye bağlıdır (içerik hash'i → kalıcı).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


def purposes_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "rag" / "doc_purposes.yaml"


@lru_cache(maxsize=4)
def _load(path: Path) -> tuple[str, dict[str, str]]:
    if not path.exists():
        return "kaynak", {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    default = str(data.get("default") or "kaynak")
    mapping: dict[str, str] = {}
    for purpose, spec in (data.get("purposes") or {}).items():
        for pid in (spec or {}).get("paper_ids") or []:
            if pid in mapping and mapping[pid] != purpose:
                raise ValueError(f"{pid} iki amaçta birden: {mapping[pid]}, {purpose}")
            mapping[str(pid)] = str(purpose)
    return default, mapping


def purpose_of(paper_id: str, path: Path | None = None) -> str:
    default, mapping = _load(path or purposes_path())
    return mapping.get(paper_id, default)


def parse_purposes(raw: str) -> frozenset[str]:
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def excluded_paper_ids(purposes: frozenset[str], path: Path | None = None) -> frozenset[str]:
    """Dışlanan amaçlara ait paper_id'ler. Varsayılan amaç dışlanamaz (tüm korpus giderdi)."""
    if not purposes:
        return frozenset()
    default, mapping = _load(path or purposes_path())
    if default in purposes:
        raise ValueError(f"varsayılan amaç '{default}' dışlanamaz — korpusun tamamı düşer")
    return frozenset(pid for pid, p in mapping.items() if p in purposes)
