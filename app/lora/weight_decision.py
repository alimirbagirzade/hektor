"""Eğitim başı ağırlık kararı — her LoRA eğitiminden ÖNCE kullanıcıya karışım ağırlıkları sorulur.

Kullanıcı isteği (2026-09-27): "her lora eğitiminin başında ağırlık ortalamaları sor".
``hektor train --run`` bir ağırlık kararı olmadan BAŞLAMAZ. Karar şu yollardan biriyle gelir:

1. ``--mix-profile <ad>`` veya ``--mix-weights "math=0.3,..."`` bayrağı,
2. ``hektor mix weights`` ile önceden kaydedilmiş, TÜKETİLMEMİŞ ve taze (≤24 sa) karar
   (web butonu / start-train.ps1 gibi etkileşimsiz başlatmalar için),
3. etkileşimli terminalde doğrudan soru.

Karar tek kullanımlıktır (onay kapısı gibi): bir eğitim tüketir, sonraki eğitim yeniden sorar.
Aynı anda en fazla BİR açık karar vardır: yeni karar kaydı ya da herhangi bir kararın tüketimi,
diğer açık kararları ``superseded_at/superseded_by`` ile geçersiz kılar (satır silinmez). Oku-
değiştir-yaz döngüsü ``file_lock`` ile korunur (eşzamanlı kayıt/tüketimde kayıp güncelleme yok).
Ağırlıklar semantik yüzde DEĞİLDİR; adapter ölçek katsayısıdır — eğitimi DEĞİŞTİRMEZ, eğitim
kaydına (hangi karışım hedefiyle eğitildiği) iliştirilir ve profil kurulumunda kullanılır.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.lora.mix_common import (
    DOMAINS,
    MixConfigError,
    file_lock,
    format_weights,
    load_mix_config,
    parse_weights,
    profile_weights,
    read_jsonl,
    registry_dir,
    validate_weights,
    write_jsonl,
)

FRESH_HOURS = 24


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


@dataclass
class WeightDecision:
    decision_id: str
    weights: dict[str, float]
    profile_name: str  # seçilen hazır profil ya da "custom"
    source: str  # flag | recorded | interactive
    created_at: str = field(default_factory=lambda: _utcnow().isoformat())
    consumed_at: str | None = None
    consumed_by: str | None = None
    # Daha yeni bir karar kaydedildiğinde ya da başka bir karar tüketildiğinde eski bekleyen
    # karar SİLİNMEZ, "geçersiz kılındı" olarak işaretlenir (denetlenebilir kayıt).
    superseded_at: str | None = None
    superseded_by: str | None = None

    @property
    def is_open(self) -> bool:
        """Tüketilmemiş ve geçersiz kılınmamış."""
        return not self.consumed_at and not self.superseded_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightDecision:
        """Eski biçimli (superseded_* alanı olmayan) satırları da okur; bilinmeyen alanı atar."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def new_decision(weights: dict[str, float], profile_name: str, source: str) -> WeightDecision:
    """Kaydedilmemiş (decision_id boş) karar — eğitim başlarken ``finalize_decision`` yazar."""
    return WeightDecision(
        decision_id="",
        weights=validate_weights(weights),
        profile_name=profile_name,
        source=source,
    )


class WeightDecisionStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or registry_dir() / "weight_decisions.jsonl"

    def records(self) -> list[WeightDecision]:
        out = []
        for row in read_jsonl(self.path):
            try:
                out.append(WeightDecision.from_dict(row))
            except TypeError:
                continue
        return out

    @staticmethod
    def _supersede_others(rows: list[WeightDecision], keep_id: str, when: str) -> None:
        for r in rows:
            if r.decision_id != keep_id and r.is_open:
                r.superseded_at = when
                r.superseded_by = keep_id

    def record(self, weights: dict[str, float], profile_name: str, source: str) -> WeightDecision:
        """Yeni kararı kaydet; bekleyen eski kararlar ``superseded`` işaretlenir (silinmez)."""
        dec = new_decision(weights, profile_name, source)
        dec.decision_id = "wd_" + uuid.uuid4().hex[:10]
        with file_lock(self.path):
            rows = self.records()
            self._supersede_others(rows, dec.decision_id, dec.created_at)
            rows.append(dec)
            write_jsonl(self.path, (r.to_dict() for r in rows))
        return dec

    def pending(self, now: dt.datetime | None = None) -> WeightDecision | None:
        """En yeni açık (tüketilmemiş + geçersiz kılınmamış) karar — taze ise.

        Yalnız EN YENİ açık karar dikkate alınır: o bayatsa daha eski bir karara geri
        düşülmez (eski biçimli dosyada birden çok açık satır kalmış olsa bile).
        """
        now = now or _utcnow()
        limit = now - dt.timedelta(hours=FRESH_HOURS)
        newest = next((d for d in reversed(self.records()) if d.is_open), None)
        if newest is None:
            return None
        try:
            created = dt.datetime.fromisoformat(newest.created_at)
        except ValueError:
            return None
        return newest if created >= limit else None

    def revoke(self, decision_id: str, reason: str) -> None:
        """Açık kararı geri çek (başlamayan koşu); silinmez, ``superseded`` işaretlenir."""
        with file_lock(self.path):
            rows = self.records()
            for r in rows:
                if r.decision_id == decision_id and r.is_open:
                    r.superseded_at = _utcnow().isoformat()
                    r.superseded_by = f"geri_cekildi: {reason}"[:120]
            write_jsonl(self.path, (r.to_dict() for r in rows))

    def consume(self, decision_id: str, consumed_by: str) -> WeightDecision:
        """Kararı tek kullanımlık tüket; diğer açık kararlar ``superseded`` işaretlenir."""
        with file_lock(self.path):
            rows = self.records()
            target = next((r for r in rows if r.decision_id == decision_id), None)
            if target is None:
                raise KeyError(decision_id)
            if target.consumed_at:
                raise ValueError(f"{decision_id} zaten tüketilmiş ({target.consumed_by})")
            if target.superseded_at:
                raise ValueError(
                    f"{decision_id} daha yeni bir kararla geçersiz kılınmış "
                    f"({target.superseded_by}); `hektor mix weights --show` ile kontrol edin"
                )
            now = _utcnow().isoformat()
            target.consumed_at = now
            target.consumed_by = consumed_by
            self._supersede_others(rows, decision_id, now)
            write_jsonl(self.path, (r.to_dict() for r in rows))
        return target


class WeightDecisionRequired(RuntimeError):
    """Etkileşimsiz başlatmada ağırlık kararı yok → eğitim başlamaz."""


def ask_weights_interactively(
    prompt: Callable[[str], str],
    echo: Callable[[str], None],
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, float], str]:
    """Kullanıcıya profilleri göster; hazır profil adı ya da özel ağırlık iste."""
    cfg = config or load_mix_config()
    echo("LoRA karışım ağırlıkları (semantik yüzde DEĞİL — adapter ölçek katsayısı):")
    for name, w in cfg["profiles"].items():
        echo(f"  {name:<22} {format_weights(w)}")
    echo(f"Domain'ler: {', '.join(DOMAINS)}")
    while True:
        answer = prompt("Profil adı ya da özel ağırlık (ör. math=0.3,statistics=0.2,...)").strip()
        if not answer:
            continue
        try:
            if "=" in answer or ":" in answer:
                return parse_weights(answer), "custom"
            return profile_weights(answer, cfg), answer
        except MixConfigError as exc:
            echo(f"Geçersiz: {exc}")


def resolve_training_weights(
    *,
    mix_profile: str | None,
    mix_weights: str | None,
    interactive: bool,
    store: WeightDecisionStore | None = None,
    prompt: Callable[[str], str] | None = None,
    echo: Callable[[str], None] = print,
    config: dict[str, Any] | None = None,
) -> WeightDecision:
    """Eğitim için ağırlık kararını çöz. Hiçbir şey YAZMAZ/TÜKETMEZ.

    Bayrak/etkileşimli karar kaydedilmemiş döner (decision_id boş): eğitim onay kapısında
    bloklanırsa geride "bekleyen" karar kalmaz. Kaydedilmiş bekleyen karar yalnız
    ``hektor mix weights`` ile oluşur. Eğitim gerçekten başlarken ``finalize_decision``.
    """
    st = store or WeightDecisionStore()
    if mix_profile and mix_weights:
        raise MixConfigError("--mix-profile ve --mix-weights birlikte verilemez")
    if mix_weights:
        return new_decision(parse_weights(mix_weights), "custom", "flag")
    if mix_profile:
        return new_decision(profile_weights(mix_profile, config), mix_profile, "flag")
    pending = st.pending()
    if pending is not None:
        return pending
    if interactive and prompt is not None:
        weights, name = ask_weights_interactively(prompt, echo, config)
        return new_decision(weights, name, "interactive")
    raise WeightDecisionRequired(
        "Eğitim öncesi karışım ağırlığı kararı yok. Önce: `uv run hektor mix weights` "
        "(ya da `train --run --mix-profile balanced_v1` / `--mix-weights math=0.3,...`)."
    )


def finalize_decision(
    decision: WeightDecision, consumed_by: str, store: WeightDecisionStore | None = None
) -> WeightDecision:
    """Eğitim başlarken kararı (gerekirse kaydedip) tek kullanımlık olarak tüket."""
    st = store or WeightDecisionStore()
    if not decision.decision_id:
        decision = st.record(decision.weights, decision.profile_name, decision.source)
    return st.consume(decision.decision_id, consumed_by)
