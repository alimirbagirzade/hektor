"""Profil kayıt defteri — kurulmuş (merged) karışım profillerinin sürümü + eval sonucu + durumu.

Durum akışı::

    experimental → evaluating → validated ─(regression gate + kullanıcı onayı)→ production
                             ↘ rejected                                        ↘ deprecated

- Router YALNIZ ``validated`` / ``production`` profilleri seçebilir.
- ``production`` yalnız: status=validated + regression gate geçti + ``user_approved=True``.
- Aynı anda tek ``production`` profil; eskisi ``deprecated`` olur.
- LLM judge sonucu tek başına terfi gerekçesi DEĞİLDİR (gate deterministik metrikleri ister).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.lora.mix_common import read_jsonl, registry_dir, write_jsonl


class ProfileStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    EVALUATING = "evaluating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


ROUTABLE_STATUSES: frozenset[ProfileStatus] = frozenset(
    {ProfileStatus.VALIDATED, ProfileStatus.PRODUCTION}
)

_ALLOWED: dict[ProfileStatus, frozenset[ProfileStatus]] = {
    ProfileStatus.EXPERIMENTAL: frozenset({ProfileStatus.EVALUATING, ProfileStatus.REJECTED}),
    ProfileStatus.EVALUATING: frozenset(
        {ProfileStatus.VALIDATED, ProfileStatus.REJECTED, ProfileStatus.EXPERIMENTAL}
    ),
    ProfileStatus.VALIDATED: frozenset(
        {ProfileStatus.PRODUCTION, ProfileStatus.DEPRECATED, ProfileStatus.EVALUATING}
    ),
    ProfileStatus.REJECTED: frozenset({ProfileStatus.EXPERIMENTAL}),
    ProfileStatus.PRODUCTION: frozenset({ProfileStatus.DEPRECATED}),
    ProfileStatus.DEPRECATED: frozenset(),
}


class ProfileRegistryError(ValueError):
    """Geçersiz durum geçişi veya kayıt."""


def _utcnow() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


@dataclass
class ProfileRecord:
    profile_id: str  # ör. balanced_v1_svd (profil adı + merge yöntemi)
    profile_name: str  # ör. balanced_v1 (mix_profiles.yaml anahtarı)
    profile_version: str
    base_model_version: str  # base_model@revision#hash[:12]
    rag_version: str
    adapter_versions: dict[str, str]  # domain → adapter_id@adapter_version
    adapter_weights: dict[str, float]
    merge_method: str
    merge_parameters: dict[str, Any]
    profile_hash: str
    merged_adapter_path: str = ""
    # Ollama/servis tarafındaki model adı (GGUF dönüşümü sonrası elle girilir). Boşsa
    # eval C/D sistemleri bu profil için "koşulamadı" olarak raporlanır (sessiz 0 değil).
    serving_model: str = ""
    eval_dataset_version: str = ""
    eval_score: float | None = None
    domain_scores: dict[str, float] = field(default_factory=dict)
    grounding_score: float | None = None
    hallucination_score: float | None = None
    regression_score: float | None = None
    last_eval_run_id: str = ""
    status: ProfileStatus = ProfileStatus.EXPERIMENTAL
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileRecord:
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in data.items() if k in known}
        payload["status"] = ProfileStatus(payload.get("status", "experimental"))
        return cls(**payload)


class ProfileRegistry:
    """JSONL tabanlı profil kayıt defteri (profile_id benzersiz)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or registry_dir() / "profiles.jsonl"

    def records(self) -> list[ProfileRecord]:
        out: list[ProfileRecord] = []
        for row in read_jsonl(self.path):
            try:
                out.append(ProfileRecord.from_dict(row))
            except (TypeError, ValueError):
                continue
        return out

    def get(self, profile_id: str) -> ProfileRecord | None:
        return next((r for r in self.records() if r.profile_id == profile_id), None)

    def production(self) -> ProfileRecord | None:
        return next((r for r in self.records() if r.status is ProfileStatus.PRODUCTION), None)

    def routable(self) -> list[ProfileRecord]:
        return [r for r in self.records() if r.status in ROUTABLE_STATUSES]

    def upsert(self, record: ProfileRecord) -> ProfileRecord:
        """Kaydı ekle ya da (aynı profile_id + aynı profile_hash ise) güncelle.

        Aynı profile_id farklı hash ile gelirse (ağırlık/adapter değişti) reddedilir —
        yeni sürüm yeni profile_version/profile_id ister; eval geçmişi sessizce ezilmez.
        """
        rows = self.records()
        existing = next((r for r in rows if r.profile_id == record.profile_id), None)
        if existing is not None:
            if existing.profile_hash != record.profile_hash:
                raise ProfileRegistryError(
                    f"{record.profile_id} farklı içerikle zaten kayıtlı "
                    f"(hash {existing.profile_hash[:12]} ≠ {record.profile_hash[:12]}); "
                    "profil sürümünü artırın"
                )
            record.status = existing.status  # durum, upsert ile atlanamaz
            record.created_at = existing.created_at
            rows = [record if r.profile_id == record.profile_id else r for r in rows]
        else:
            if record.status not in (ProfileStatus.EXPERIMENTAL,):
                record.status = ProfileStatus.EXPERIMENTAL
            rows.append(record)
        record.updated_at = _utcnow()
        write_jsonl(self.path, (r.to_dict() for r in rows))
        return record

    def record_eval(
        self,
        profile_id: str,
        *,
        run_id: str,
        eval_dataset_version: str,
        eval_score: float | None,
        domain_scores: dict[str, float],
        grounding_score: float | None,
        hallucination_score: float | None,
    ) -> ProfileRecord:
        rows = self.records()
        rec = next((r for r in rows if r.profile_id == profile_id), None)
        if rec is None:
            raise ProfileRegistryError(f"profil yok: {profile_id}")
        rec.last_eval_run_id = run_id
        rec.eval_dataset_version = eval_dataset_version
        rec.eval_score = eval_score
        rec.domain_scores = dict(domain_scores)
        rec.grounding_score = grounding_score
        rec.hallucination_score = hallucination_score
        rec.updated_at = _utcnow()
        write_jsonl(self.path, (r.to_dict() for r in rows))
        return rec

    def transition(
        self,
        profile_id: str,
        new_status: ProfileStatus,
        *,
        gate_passed: bool = False,
        user_approved: bool = False,
        note: str = "",
    ) -> ProfileRecord:
        """Durum geçişi. PRODUCTION: gate_passed + user_approved zorunlu."""
        rows = self.records()
        rec = next((r for r in rows if r.profile_id == profile_id), None)
        if rec is None:
            raise ProfileRegistryError(f"profil yok: {profile_id}")
        if new_status not in _ALLOWED[rec.status]:
            raise ProfileRegistryError(f"geçersiz geçiş: {rec.status.value} → {new_status.value}")
        if new_status is ProfileStatus.VALIDATED and not rec.last_eval_run_id:
            raise ProfileRegistryError("eval koşusu olmadan validated olamaz")
        if new_status is ProfileStatus.PRODUCTION:
            if not gate_passed:
                raise ProfileRegistryError("regression gate geçilmeden production'a geçilemez")
            if not user_approved:
                raise ProfileRegistryError("production terfisi açık kullanıcı onayı ister")
            for r in rows:
                if r.status is ProfileStatus.PRODUCTION:
                    r.status = ProfileStatus.DEPRECATED
                    r.updated_at = _utcnow()
        rec.status = new_status
        rec.updated_at = _utcnow()
        if note:
            rec.notes = f"{rec.notes} | {note}".strip(" |")
        write_jsonl(self.path, (r.to_dict() for r in rows))
        return rec
