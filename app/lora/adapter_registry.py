"""Adapter kayıt defteri — LoRA adapter sürümlerini ve durumlarını izle.

Durum akışı: candidate → smoke_passed → eval_passed → approved → production.
PRODUCTION'a geçiş yalnızca kullanıcı onayıyla (`user_approved=True`)
mümkündür. Kayıtlar JSONL dosyasında saklanır (append-only + yeniden yazma).
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_REGISTRY_PATH = Path("registry/adapters/registry.jsonl")


class AdapterStatus(StrEnum):
    """Adapter yaşam döngüsü durumları."""

    CANDIDATE = "candidate"
    REJECTED = "rejected"
    SMOKE_PASSED = "smoke_passed"
    EVAL_PASSED = "eval_passed"
    APPROVED = "approved"
    PRODUCTION = "production"


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


@dataclass
class AdapterRecord:
    """Tek bir LoRA adapter'ının tam metadata kaydı."""

    adapter_id: str = ""
    adapter_name: str = ""
    base_model: str = ""
    dataset_version: str = ""
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)
    learning_rate: float = 2e-4
    epochs: int = 1
    train_examples: int = 0
    valid_examples: int = 0
    test_examples: int = 0
    eval_score: float | None = None
    status: AdapterStatus = AdapterStatus.CANDIDATE
    created_at: str = field(default_factory=_utcnow)
    notes: str = ""
    approved_by_user: bool = False

    def to_dict(self) -> dict:
        """JSON-serileştirilebilir dict (enum -> değer)."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> AdapterRecord:
        """Dict'ten AdapterRecord kur (status enum'a çevrilir)."""
        payload = dict(data)
        status = payload.get("status", AdapterStatus.CANDIDATE.value)
        payload["status"] = AdapterStatus(status)
        # Yalnızca bilinen alanları al (ileri uyumluluk).
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(**filtered)


class AdapterRegistry:
    """JSONL tabanlı adapter kayıt defteri."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self.registry_path = registry_path or DEFAULT_REGISTRY_PATH

    # --- okuma ---------------------------------------------------------
    def list_adapters(self) -> list[AdapterRecord]:
        """Tüm kayıtları oku (eski→yeni dosya sırasıyla)."""
        if not self.registry_path.exists():
            return []
        records: list[AdapterRecord] = []
        for line in self.registry_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(AdapterRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError):
                continue
        return records

    def get(self, adapter_id: str) -> AdapterRecord | None:
        """Tek adapter'ı id ile döndür."""
        for record in self.list_adapters():
            if record.adapter_id == adapter_id:
                return record
        return None

    def get_production(self) -> AdapterRecord | None:
        """PRODUCTION durumundaki adapter'ı döndür (yoksa None)."""
        for record in self.list_adapters():
            if record.status is AdapterStatus.PRODUCTION:
                return record
        return None

    # --- yazma ---------------------------------------------------------
    def register(self, record: AdapterRecord) -> str:
        """Yeni kayıt ekle; adapter_id yoksa üret ve döndür."""
        if not record.adapter_id:
            record.adapter_id = "adapter_" + uuid.uuid4().hex[:12]
        records = self.list_adapters()
        records.append(record)
        self._write_all(records)
        return record.adapter_id

    def promote(self, adapter_id: str, user_approved: bool) -> bool:
        """Adapter'ı PRODUCTION'a yükselt — kullanıcı onayı zorunlu.

        `user_approved=False` ise yükseltme yapılmaz. Mevcut production
        adapter'ı APPROVED'a (arşiv) düşürülür; tek production garantilenir.
        """
        if not user_approved:
            return False
        records = self.list_adapters()
        target = next((r for r in records if r.adapter_id == adapter_id), None)
        if target is None:
            return False

        for record in records:
            if record.status is AdapterStatus.PRODUCTION:
                record.status = AdapterStatus.APPROVED

        target.status = AdapterStatus.PRODUCTION
        target.approved_by_user = True
        self._write_all(records)
        return True

    def reject(self, adapter_id: str, reason: str) -> bool:
        """Adapter'ı REJECTED olarak işaretle ve sebebini nota ekle."""
        records = self.list_adapters()
        target = next((r for r in records if r.adapter_id == adapter_id), None)
        if target is None:
            return False
        target.status = AdapterStatus.REJECTED
        suffix = f"reddedildi: {reason}"
        target.notes = f"{target.notes} | {suffix}".strip(" |") if target.notes else suffix
        self._write_all(records)
        return True

    def _write_all(self, records: list[AdapterRecord]) -> None:
        """Tüm kayıtları JSONL'e yeniden yaz (atomik tam yazma)."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in records]
        self.registry_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
