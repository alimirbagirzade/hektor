"""Domain adapter kayıt defteri — bağımsız eğitilmiş domain LoRA'ları.

Domain'ler: math / statistics / reasoning / trading / coding.

Mevcut ``app.lora.adapter_registry`` (tek-adapter yaşam döngüsü) DEĞİŞMEZ; bu defter
karışım (mixing) için gereken tam izlenebilirlik alanlarını tutar.

Değişmezler:
- Her adapter AYNI base checkpoint'ten bağımsız eğitilir: ``parent_adapter`` dolu kayıt
  (adapter üstüne adapter eğitimi) REDDEDİLİR.
- ``production_status=production`` yalnız ``user_approved=True`` ile verilir.
- Adapter dosyaları birbirinden bağımsız dizinlerde durur (aynı ``adapter_path`` iki kayıtta
  olamaz).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.lora.mix_common import DOMAINS, read_jsonl, registry_dir, write_jsonl


class EvalStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class ProductionStatus(StrEnum):
    NONE = "none"
    CANDIDATE = "candidate"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class AdapterRegistryError(ValueError):
    """Kayıt değişmezi ihlal edildi."""


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


@dataclass
class DomainAdapterRecord:
    adapter_id: str
    adapter_version: str
    domain: str
    adapter_path: str
    base_model: str
    base_model_revision: str
    base_model_hash: str
    tokenizer_hash: str
    dataset_version: str
    dataset_hash: str
    training_config_hash: str
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    learning_rate: float
    epochs: float
    max_seq_length: int
    training_seed: int
    git_commit: str
    created_at: str = field(default_factory=_utcnow)
    eval_status: EvalStatus = EvalStatus.PENDING
    production_status: ProductionStatus = ProductionStatus.NONE
    # Bağımsızlık kanıtı: adapter başka bir adapter'ın üstünden eğitildiyse onun id'si.
    # Dolu ise kayıt REDDEDİLİR (bkz. validate()).
    parent_adapter: str | None = None
    # Tekil adapter'ın servis (Ollama) model adı — eval matrisi "Base + tekil LoRA" satırı
    # için. Boşsa o satır "koşulamadı" raporlanır.
    serving_model: str = ""
    notes: str = ""

    def validate(self) -> None:
        if self.domain not in DOMAINS:
            raise AdapterRegistryError(f"geçersiz domain: {self.domain} (geçerli: {DOMAINS})")
        if self.parent_adapter:
            raise AdapterRegistryError(
                f"{self.adapter_id}: adapter başka bir adapter'ın ({self.parent_adapter}) "
                "üstünden eğitilmiş — domain adapter'ları yalnız base checkpoint'ten eğitilir"
            )
        for name in ("adapter_id", "adapter_version", "base_model", "base_model_hash"):
            if not str(getattr(self, name)).strip():
                raise AdapterRegistryError(f"{name} boş olamaz")
        for name in ("tokenizer_hash", "dataset_hash", "training_config_hash"):
            if not str(getattr(self, name)).strip():
                raise AdapterRegistryError(f"{name} boş olamaz (tekrarlanabilirlik)")
        if self.r <= 0 or not self.target_modules:
            raise AdapterRegistryError("r > 0 ve target_modules dolu olmalı")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eval_status"] = self.eval_status.value
        d["production_status"] = self.production_status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainAdapterRecord:
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in data.items() if k in known}
        payload["eval_status"] = EvalStatus(payload.get("eval_status", "pending"))
        payload["production_status"] = ProductionStatus(payload.get("production_status", "none"))
        return cls(**payload)


def read_peft_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    """PEFT ``adapter_config.json``'dan r/alpha/dropout/target_modules/base oku."""
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json yok: {adapter_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tm = cfg.get("target_modules") or []
    if isinstance(tm, str):  # PEFT regex biçimi
        tm = [tm]
    return {
        "r": int(cfg.get("r", 0)),
        "lora_alpha": int(cfg.get("lora_alpha", 0)),
        "lora_dropout": float(cfg.get("lora_dropout", 0.0)),
        "target_modules": sorted(str(t) for t in tm),
        "base_model": str(cfg.get("base_model_name_or_path") or ""),
        "base_model_revision": str(cfg.get("revision") or ""),
    }


class DomainAdapterRegistry:
    """JSONL tabanlı domain adapter kayıt defteri."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or registry_dir() / "domain_adapters.jsonl"

    def records(self) -> list[DomainAdapterRecord]:
        out: list[DomainAdapterRecord] = []
        for row in read_jsonl(self.path):
            try:
                out.append(DomainAdapterRecord.from_dict(row))
            except (TypeError, ValueError):
                continue
        return out

    def get(
        self, adapter_id: str, adapter_version: str | None = None
    ) -> DomainAdapterRecord | None:
        matches = [r for r in self.records() if r.adapter_id == adapter_id]
        if adapter_version is not None:
            matches = [r for r in matches if r.adapter_version == adapter_version]
        return matches[-1] if matches else None

    def latest_for_domain(self, domain: str) -> DomainAdapterRecord | None:
        """Domain için kullanılabilir en son kayıt (deprecated ve eval=failed hariç)."""
        usable = [
            r
            for r in self.records()
            if r.domain == domain
            and r.production_status is not ProductionStatus.DEPRECATED
            and r.eval_status is not EvalStatus.FAILED
        ]
        # production varsa onu tercih et; yoksa en son kayıt.
        prod = [r for r in usable if r.production_status is ProductionStatus.PRODUCTION]
        pick = prod or usable
        return pick[-1] if pick else None

    def register(self, record: DomainAdapterRecord) -> DomainAdapterRecord:
        record.validate()
        rows = self.records()
        for r in rows:
            if r.adapter_id == record.adapter_id and r.adapter_version == record.adapter_version:
                raise AdapterRegistryError(
                    f"zaten kayıtlı: {record.adapter_id}@{record.adapter_version}"
                )
            if r.adapter_path and r.adapter_path == record.adapter_path:
                raise AdapterRegistryError(
                    f"adapter dosyaları bağımsız olmalı: {record.adapter_path} zaten "
                    f"{r.adapter_id}@{r.adapter_version} tarafından kullanılıyor"
                )
        # Yeni kayıt production olarak AÇILAMAZ (onay kapısı atlanmasın).
        record.production_status = ProductionStatus.NONE
        rows.append(record)
        write_jsonl(self.path, (r.to_dict() for r in rows))
        return record

    def set_eval_status(self, adapter_id: str, adapter_version: str, status: EvalStatus) -> bool:
        rows = self.records()
        hit = False
        for r in rows:
            if r.adapter_id == adapter_id and r.adapter_version == adapter_version:
                r.eval_status = status
                hit = True
        if hit:
            write_jsonl(self.path, (r.to_dict() for r in rows))
        return hit

    def set_production_status(
        self,
        adapter_id: str,
        adapter_version: str,
        status: ProductionStatus,
        *,
        user_approved: bool = False,
    ) -> bool:
        """Production durumu değiştir. PRODUCTION yalnız kullanıcı onayı + eval=passed ile."""
        rows = self.records()
        target = next(
            (
                r
                for r in rows
                if r.adapter_id == adapter_id and r.adapter_version == adapter_version
            ),
            None,
        )
        if target is None:
            return False
        if status is ProductionStatus.PRODUCTION:
            if not user_approved or target.eval_status is not EvalStatus.PASSED:
                return False
            for r in rows:  # domain başına tek production
                if r.domain == target.domain and r.production_status is ProductionStatus.PRODUCTION:
                    r.production_status = ProductionStatus.DEPRECATED
        target.production_status = status
        write_jsonl(self.path, (r.to_dict() for r in rows))
        return True
