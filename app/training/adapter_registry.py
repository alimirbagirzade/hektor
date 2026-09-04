"""Track LoRA adapters and their metadata.

Each adapter is versioned with: version, base_model, training_data_hash,
created_at, notes. Stored in SQLite (Adapter table) and mirrored as a JSON
sidecar next to the adapter weights for portability.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.memory.sqlite_store import Adapter, SqliteStore


class AdapterRegistry:
    def __init__(self, store: SqliteStore | None = None) -> None:
        self.store = store or SqliteStore()

    def register(
        self,
        version: str,
        base_model: str,
        adapter_path: str | Path,
        training_data_hash: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        meta = {
            "version": version,
            "base_model": base_model,
            "adapter_path": str(adapter_path),
            "training_data_hash": training_data_hash,
            "notes": notes,
            "created_at": dt.datetime.now(dt.UTC).isoformat(),
        }
        with self.store.session() as s:
            s.merge(Adapter(**meta))
        sidecar = Path(adapter_path).parent / f"{version}.meta.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def get(self, version: str) -> dict[str, Any] | None:
        with self.store.session() as s:
            a = s.get(Adapter, version)
            if not a:
                return None
            return {
                "version": a.version,
                "base_model": a.base_model,
                "adapter_path": a.adapter_path,
                "training_data_hash": a.training_data_hash,
                "notes": a.notes,
                "created_at": a.created_at,
            }

    def list_all(self) -> list[dict[str, Any]]:
        with self.store.session() as s:
            rows = s.scalars(select(Adapter).order_by(Adapter.created_at.desc()))
            return [
                {
                    "version": r.version,
                    "base_model": r.base_model,
                    "adapter_path": r.adapter_path,
                    "created_at": r.created_at,
                    "notes": r.notes,
                }
                for r in rows
            ]
