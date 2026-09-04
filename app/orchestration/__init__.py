"""app/orchestration — dayanıklı eğitim orkestrasyon katmanı.

Mevcut parçaları (detached_launch, auto_pipeline, control_plane, adapter_eval,
adapter_registry) YENİDEN YAZMAZ; üstlerine **per-run, checkpoint'li, event-loglu,
resume + panic-recovery** edilebilir bir koordinasyon katmanı kurar.

Tasarım sözleşmeleri:
  - Her koşu (run) DB'de kalıcıdır → web/CLI yeniden başlasa da timeline kaybolmaz.
  - Her aşama (stage) çıktısı checkpoint olarak yazılır → resume tamamlananı atlar
    (session-limit'e dayanıklı; v5/0-oy yanıltıcılığına karşı).
  - Gerçek eğitim ASLA gözetimsiz başlamaz: orchestrator `approval` sınırında durur
    (CLAUDE.md Kural 8); tehlikeli train/eval/registry adımları enjekte edilen
    delege'lere bırakılır (varsayılan: salt-devir / handoff).
  - Tüm okuma-aşamaları (preflight/data-gate/curriculum/dry-run) çevrimdışı çalışır.
"""

from __future__ import annotations

from app.orchestration.pipeline import (
    PIPELINE,
    StageDef,
    StageKind,
    StageStatus,
)
from app.orchestration.store import OrchestrationStore

__all__ = [
    "PIPELINE",
    "OrchestrationStore",
    "StageDef",
    "StageKind",
    "StageStatus",
]
