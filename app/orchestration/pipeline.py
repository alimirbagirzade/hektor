"""pipeline.py — eğitim orkestrasyon hattının SAF aşama tanımları.

Yan etkisi olmayan veri (frozen dataclass) → birim test için ideal. Orchestrator
bu sırayı yürütür; her aşamanın bir delege fonksiyonu vardır (orchestrator.py).

Aşama sırası KANONİKTİR — CLAUDE.md eğitim yaşam döngüsünü yansıtır:
  preflight → collision → smoke → deep-hunt → data-gate → curriculum → dry-run
  → regression → approval → train → eval → registry

`autonomous` bayrağı: True ise orchestrator aşamayı gözetimsiz yürütebilir (salt-okuma
güvenli). False ise insan eylemi/onayı gerekir → orchestrator burada DURUR (Kural 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StageKind(StrEnum):
    """Bir aşamanın türü — delege seçimini ve güvenlik sınıfını belirler."""

    preflight = "preflight"  # sistem durumu (STOP_ALL, veri, bağımlılık) — salt-okuma
    collision = "collision"  # eşzamanlı oturum/worktree çakışması (git durumu) — salt-okuma
    smoke = "smoke"  # gerçek runtime uçtan-uca duman testi (Ollama+RAG+LLM) — salt-okuma
    hunt = "hunt"  # Kademe-2 zorunlu derin av — denetimli tetik
    data_gate = "data_gate"  # pretrain-gate GO/NO-GO + readiness — salt-okuma
    curriculum = "curriculum"  # kart L0-L4 sınıflama — salt-okuma
    dry_run = "dry_run"  # eğitim komutu önizleme — yürütme yok
    regression = "regression"  # baseline'a göre gerileme (v5 dersi) — salt-okuma
    approval = "approval"  # TAZE insan onayı sınırı (Kural 8)
    train = "train"  # gerçek LoRA eğitimi — delege, onaylıysa
    evaluate = "evaluate"  # adapter eval (base ile kıyas)
    registry = "registry"  # adapter'ı aday olarak kaydet (terfi ayrı onay)


class StageStatus(StrEnum):
    """Bir aşamanın yaşam döngüsü durumu."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    blocked = "blocked"  # onay/STOP_ALL/NO-GO → ilerlemez (insan eylemi bekler)
    skipped = "skipped"


class RunStatus(StrEnum):
    """Bir orkestrasyon koşusunun bütünsel durumu."""

    pending = "pending"
    running = "running"
    blocked = "blocked"  # bir aşama insan eylemi/onayı bekliyor
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# Terminal (artık ilerlemeyen) aşama durumları.
TERMINAL_STAGE_STATUSES: frozenset[StageStatus] = frozenset(
    {StageStatus.completed, StageStatus.skipped}
)
# Koşuyu durduran (ama henüz bitirmeyen) aşama durumları.
HALTING_STAGE_STATUSES: frozenset[StageStatus] = frozenset(
    {StageStatus.failed, StageStatus.blocked}
)


@dataclass(frozen=True)
class StageDef:
    """Tek bir aşamanın değişmez tanımı."""

    name: str
    kind: StageKind
    title: str
    autonomous: bool
    description: str


# Kanonik hat. Sıra önemlidir; `name` benzersizdir ve DB anahtarı olarak kullanılır.
PIPELINE: tuple[StageDef, ...] = (
    StageDef(
        name="preflight",
        kind=StageKind.preflight,
        title="Ön kontrol",
        autonomous=True,
        description=("STOP_ALL, veri sayımı, eğitim bağımlılıkları ve sistem durumu — salt-okuma."),
    ),
    StageDef(
        name="collision",
        kind=StageKind.collision,
        title="Çakışma denetimi",
        autonomous=True,
        description=(
            "Eşzamanlı oturum/worktree çakışması taraması (git durumu) — başka bir oturumun "
            "`git add -A`'i uncommitted fix'leri süpürebilir ya da HEAD altımdan kayabilir. "
            "git yoksa atlanır (skip); kirli ağaç UYARI; aktif lock / aynı-branch worktree / "
            "HEAD kayması BLOK (insan çözmeli). Salt-okuma."
        ),
    ),
    StageDef(
        name="smoke",
        kind=StageKind.smoke,
        title="Duman testi",
        autonomous=True,
        description=(
            "Gerçek runtime uçtan-uca duman testi (Ollama+RAG+LLM) — 'stub≠runtime' dersi: "
            "birim testleri stub'la geçse de canlı hat bozuk olabilir. Runtime çevrimdışıysa "
            "atlanır (skip, hat durmaz); canlı ama üretim boş/degenere ise BAŞARISIZ."
        ),
    ),
    StageDef(
        name="deep-hunt",
        kind=StageKind.hunt,
        title="Kademe-2 derin av",
        autonomous=False,
        description=(
            "Eğitim öncesi ZORUNLU adversarial bug-avı (CLAUDE.md). v5 regresyonu bu "
            "yüzden olmuştu. Denetimli olarak çalıştırılıp onaylanması gerekir."
        ),
    ),
    StageDef(
        name="data-gate",
        kind=StageKind.data_gate,
        title="Veri kalite kapısı",
        autonomous=True,
        description="pretrain-gate GO/NO-GO + lora-readiness eşiği — salt-okuma.",
    ),
    StageDef(
        name="curriculum",
        kind=StageKind.curriculum,
        title="Müfredat sınıflama",
        autonomous=True,
        description="Onaylı kartları L0-L4 zorluk seviyelerine ayır — salt-okuma.",
    ),
    StageDef(
        name="dry-run",
        kind=StageKind.dry_run,
        title="Kuru çalıştırma",
        autonomous=True,
        description="Eğitim komutu + örnek sayısı önizleme — gerçek yürütme YOK.",
    ),
    StageDef(
        name="regression",
        kind=StageKind.regression,
        title="Gerileme denetimi",
        autonomous=True,
        description=(
            "Eğitim/onay öncesi: mevcut aday veri setinin v5-ilgili kalite sinyallerini "
            "(açılış ezberi, zehir, sızıntı, disiplin kapsamı) son GEÇEN baseline ile kıyasla. "
            "Baseline yoksa atlanır (skip, ilk koşu); gerileme varsa BLOK (v5 dersi — eğitim "
            "ilerlemez). Baseline yalnız explicit --commit ile güncellenir. Salt-okuma."
        ),
    ),
    StageDef(
        name="approval",
        kind=StageKind.approval,
        title="Eğitim yetki politikası",
        autonomous=True,
        description=(
            "Tek UnattendedTrainingPolicy karar verir: gate zinciri geçtiyse otomatik, "
            "aksi durumda taze insan onayı gerekir."
        ),
    ),
    StageDef(
        name="train",
        kind=StageKind.train,
        title="LoRA eğitimi",
        autonomous=True,
        description=(
            "Gerçek eğitim yürütmesi merkezi politika kararıyla Auto-LoRA sahibine delege edilir."
        ),
    ),
    StageDef(
        name="evaluate",
        kind=StageKind.evaluate,
        title="Değerlendirme",
        autonomous=True,
        description="Adapter'ı base ile karşılaştır (accept/reject/inconclusive).",
    ),
    StageDef(
        name="registry",
        kind=StageKind.registry,
        title="Kayıt",
        autonomous=True,
        description="Adapter'ı registry'ye ADAY olarak kaydet (production terfisi ayrı onay).",
    ),
)

# Hızlı erişim için ad→tanım eşlemesi.
PIPELINE_BY_NAME: dict[str, StageDef] = {s.name: s for s in PIPELINE}


def stage_order(name: str) -> int:
    """Aşamanın hattaki 0-tabanlı sırası; bilinmeyen ad için -1."""
    for i, s in enumerate(PIPELINE):
        if s.name == name:
            return i
    return -1
