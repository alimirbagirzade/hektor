"""Terfi kapıları — bir varlık "production"a / "approved"a geçmeden ÖNCEKİ kontroller.

CLAUDE.md ve ek-modül talimatındaki kurallar:
- Adapter production'a alınmadan önce eval zorunlu (bu ZATEN ``app/lora/adapter_registry``
  + ``app/training/adapter_eval`` ile var → burada tekrarlanmaz).
- RAG indeks production'a alınmadan önce retrieval eval geçmeli (``ReleaseGate`` yeniden
  kullanılır: recall@10 ≥ 0.70 vb.).
- Dataset onaylanmadan LoRA eğitimine giremez (Kural 8).
- Ödül seti içinde private key / API key / kişisel veri / finansal gizli veri olmamalı.

Her kapı kararı ``promotion_decisions`` tablosuna append-only yazılır (denetim izi).

NOT: Sır/PII taraması burada KENDİ KENDİNE YETEN hafif bir regex kapısıdır. Ağır
tarayıcı ``app/lora/safety_scanner.py``'dedir; o dosya eş zamanlı bir oturumun aktif
WIP'i olduğundan bu modül onu içe AKTARMAZ (çakışma önlemi). İkisi de salt-regex,
``eval``/``exec`` yok (Kural 5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.registry.version_store import RegistryStore
from app.reliability.release_gate import MINIMUM_THRESHOLDS, ReleaseGate

# --- sır / PII desenleri (salt-regex; finansal veride yanlış-pozitifi azaltacak şekilde) --
_SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |)PRIVATE KEY-----"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    ),
}
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # Telefon: yalnız ULUSLARARASI biçim (+ önekli) — çıplak 11-haneli sayıların
    # (finansal veride sık) yanlış eşleşmesini önler.
    "phone_intl": re.compile(r"\+\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{3}[\s\-]?\d{2,4}\b"),
}


@dataclass
class ScanResult:
    """Sır/PII tarama sonucu (kanıt önizlemesi maskelenmiştir)."""

    clean: bool
    secret_findings: list[dict[str, str]] = field(default_factory=list)
    pii_findings: list[dict[str, str]] = field(default_factory=list)

    @property
    def has_secret(self) -> bool:
        return bool(self.secret_findings)

    @property
    def has_pii(self) -> bool:
        return bool(self.pii_findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "secret_findings": self.secret_findings,
            "pii_findings": self.pii_findings,
        }


def _mask(match: str) -> str:
    """Gizli dizgeyi maskele: 4 karakterden uzunsa ilk 4'ü göster, değilse tamamen gizle.

    4 ve daha kısa dizgelerde ilk-4'ü göstermek tüm sırrı açığa çıkarırdı; bu yüzden
    ``len <= 4`` durumunda yalnız ``***`` döndürülür (güvenli taraf — sır loglanmaz).
    """
    match = match.strip()
    if len(match) <= 4:
        return "***"
    return match[:4] + "***"


def scan_secret_pii(texts: str | list[str]) -> ScanResult:
    """Bir metni / metin listesini sır ve PII için tara (salt-regex, çevrimdışı).

    Ödül/dataset terfisinden önce çağrılır. Bulgular maskelenir → log'a sır sızmaz.
    """
    if isinstance(texts, str):
        texts = [texts]
    secrets: list[dict[str, str]] = []
    pii: list[dict[str, str]] = []
    for text in texts:
        if not text:
            continue
        for name, pat in _SECRET_PATTERNS.items():
            for m in pat.findall(text):
                hit = m if isinstance(m, str) else (m[0] if m else "")
                secrets.append({"type": name, "preview": _mask(hit or "")})
        for name, pat in _PII_PATTERNS.items():
            for m in pat.findall(text):
                hit = m if isinstance(m, str) else (m[0] if m else "")
                pii.append({"type": name, "preview": _mask(hit or "")})
    return ScanResult(clean=not secrets and not pii, secret_findings=secrets, pii_findings=pii)


# --- dataset onayı ---------------------------------------------------------
def approve_dataset(
    registry: RegistryStore,
    dataset_version_id: str,
    approver_id: str,
    reason: str = "",
) -> dict[str, Any]:
    """Dataset sürümünü ``approved`` yap (Kural 8 kapısı) + karar logla.

    Durum makinesi: yalnız ``pending`` → ``approved``. Zaten onaylıysa idempotent;
    ``rejected`` ise terminal olduğundan ValueError (sessiz çapraz-geçiş yok).
    Bilinmeyen sürüm → ValueError.
    """
    ds = registry.get_dataset(dataset_version_id)
    if ds is None:
        raise ValueError(f"Bilinmeyen dataset sürümü: {dataset_version_id}")
    prev = ds["approval_status"]
    if prev == "approved":
        return {"ok": True, "already": True, "dataset": ds}
    if prev != "pending":
        raise ValueError(
            f"'{prev}' terminal durumdaki dataset onaylanamaz (yalnız pending → approved)"
        )
    # ATOMİK geçiş (TOCTOU önlemi): pending→approved; eşzamanlı iki onaydan yalnız BİRİ kazanır.
    won = registry.cas_dataset_status(dataset_version_id, expected="pending", new_status="approved")
    if not won:
        # araya giren çağrı pending'i değiştirdi → mevcut durumu döndür (idempotent)
        return {"ok": True, "already": True, "dataset": registry.get_dataset(dataset_version_id)}
    decision = registry.log_decision(
        target_type="dataset",
        target_id=dataset_version_id,
        from_status="pending",  # CAS pending'den kazandı → kesin önceki durum
        to_status="approved",
        decision="approved",
        reason=reason or "kullanıcı onayı",
        approved_by=approver_id,
    )
    return {
        "ok": True,
        "already": False,
        "dataset": registry.get_dataset(dataset_version_id),
        "decision": decision,
    }


def reject_dataset(
    registry: RegistryStore,
    dataset_version_id: str,
    approver_id: str,
    reason: str,
) -> dict[str, Any]:
    """Dataset'i ``rejected`` yap + karar logla (atomik; yalnız pending → rejected)."""
    ds = registry.get_dataset(dataset_version_id)
    if ds is None:
        raise ValueError(f"Bilinmeyen dataset sürümü: {dataset_version_id}")
    prev = ds["approval_status"]
    if prev == "rejected":
        return {"ok": True, "already": True, "dataset": ds}
    if prev != "pending":
        raise ValueError(
            f"'{prev}' terminal durumdaki dataset reddedilemez (yalnız pending → rejected)"
        )
    won = registry.cas_dataset_status(dataset_version_id, expected="pending", new_status="rejected")
    if not won:
        return {"ok": True, "already": True, "dataset": registry.get_dataset(dataset_version_id)}
    decision = registry.log_decision(
        target_type="dataset",
        target_id=dataset_version_id,
        from_status="pending",
        to_status="rejected",
        decision="rejected",
        reason=reason,
        approved_by=approver_id,
    )
    return {
        "ok": True,
        "already": False,
        "dataset": registry.get_dataset(dataset_version_id),
        "decision": decision,
    }


# --- RAG indeks retrieval-eval kapısı --------------------------------------
def check_rag_index_eval(
    registry: RegistryStore,
    rag_index_version_id: str,
    metrics: dict[str, float],
    *,
    thresholds: dict[str, float] | None = None,
    approved_by: str | None = None,
) -> dict[str, Any]:
    """RAG indeksini ``ReleaseGate`` ile değerlendir; geçer/blok kararını logla.

    ``metrics``: {recall_at_10, citation_accuracy, grounding_score, abstention_correct}.
    Geçerse ``eval_passed`` kararı (approved), geçmezse ``blocked`` (gerekçe = eksikler).
    """
    gate = ReleaseGate(thresholds or MINIMUM_THRESHOLDS.copy())
    result = gate.check(metrics)
    if result.passed:
        decision = registry.log_decision(
            target_type="rag_index",
            target_id=rag_index_version_id,
            to_status="eval_passed",
            decision="approved",
            reason="retrieval eval eşikleri karşılandı",
            approved_by=approved_by,
        )
    else:
        decision = registry.log_decision(
            target_type="rag_index",
            target_id=rag_index_version_id,
            to_status="blocked",
            decision="blocked",
            reason="; ".join(result.failures),
            approved_by=approved_by,
        )
    return {"passed": result.passed, "failures": result.failures, "decision": decision}


# --- ödül seti sır/PII kapısı ----------------------------------------------
def gate_reward_dataset(
    registry: RegistryStore,
    reward_version_id: str,
    texts: str | list[str],
    *,
    approved_by: str | None = None,
) -> dict[str, Any]:
    """Ödül setini sır/PII için tara, bayrakları güncelle ve kararı logla.

    Temizse ``approved`` (secret/pii bayrağı=1), bulgu varsa ``blocked`` (bayrak=2).
    """
    scan = scan_secret_pii(texts)
    registry.set_reward_scan_flags(
        reward_version_id,
        secret_scanned=2 if scan.has_secret else 1,
        pii_scanned=2 if scan.has_pii else 1,
    )
    if scan.clean:
        decision = registry.log_decision(
            target_type="reward",
            target_id=reward_version_id,
            to_status="scan_passed",
            decision="approved",
            reason="sır/PII bulunamadı",
            approved_by=approved_by,
        )
    else:
        reason = (
            f"sır={len(scan.secret_findings)} pii={len(scan.pii_findings)} "
            "(ödül seti gizli/kişisel veri içeremez)"
        )
        decision = registry.log_decision(
            target_type="reward",
            target_id=reward_version_id,
            to_status="blocked",
            decision="blocked",
            reason=reason,
            approved_by=approved_by,
        )
    return {"clean": scan.clean, "scan": scan.to_dict(), "decision": decision}
