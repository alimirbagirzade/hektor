---
name: lora-training-control-plane
description: Controlled LoRA training lifecycle for Achilles. Dataset audit, curriculum classification, validation gates, smoke tests, LoRA config preparation, eval, adapter registry, and safe promotion. Use for any LoRA pipeline task.
when_to_use: Use when preparing, validating, testing, evaluating, registering, or promoting any LoRA adapter or dataset in Achilles.
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# LoRA Training Control Plane

## RAG vs LoRA Sınırı
- **RAG** = bilgi deposu, kaynak, belge, güncel veri
- **LoRA** = davranış, muhakeme formatı, hesaplama disiplini, belirsizlik yönetimi

## Pipeline (Gate 0-8)
- Gate 0: Source (approved RAG verisi mi?)
- Gate 1: Schema (JSONL format doğru mu?)
- Gate 2: Curriculum (level atanmış mı?)
- Gate 3: Domain (en az bir domain var mı?)
- Gate 4: Quality (kısa/tekrarlı/duplicate?)
- Gate 5: Math/Physics/Statistics (hesap doğru mu?)
- Gate 6: Logic/Philosophy (mantık tutarlı mı?)
- Gate 7: Safety/Secret (gizli veri yok mu?) — BLOCKER
- Gate 8: Dataset Split (train/valid/test sızıntı yok mu?)

## Komutlar
- `uv run achilles lora-audit` — Gate 0-7
- `uv run achilles lora-dataset` — JSONL üret
- `uv run achilles lora-registry` — adapter listesi
- `uv run achilles lora-status` — genel durum

## Kullanıcı Onayı Gerektiren İşler
- Smoke test başlatma (200+ örnek)
- Adapter promote
- GGUF/Ollama export
- Production adapter değiştirme
