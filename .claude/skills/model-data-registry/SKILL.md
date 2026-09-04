# /model-data-registry — Model/veri sürüm kaydı + terfi kapısı (Kural 8)

Dataset / RAG-indeks / embedding / RLM-ödül **sürümlerini** kaydeder ve terfi kapılarını uygular
(talimat §11, Modül 8). Eksik olan yatay katman: adapter yaşam döngüsü AYRI kalır
(`app/lora` + `app/training` adapter_registry). **app/rlm İÇE AKTARILMAZ.**

## Ne zaman kullan
- Eğitim öncesi: "bu dataset sürümü onaylı mı?" (Kural 8 — onaysız eğitime giremez).
- "Neyle eğittik / hangi RAG indeks vardı / neden production'a alındı?" denetimi.
- RAG indeks production'a alınmadan retrieval-eval kapısı.
- Ödül seti terfisinden önce sır/PII taraması.

## Komutlar
```bash
# Sürümleri listele
uv run achilles registry-list --kind datasets        # dataset_versions (+approval_status)
uv run achilles registry-list --kind indices         # rag_index_versions
uv run achilles registry-list --kind embeddings      # embedding_model_versions
uv run achilles registry-list --kind rewards         # rlm_reward_versions (sır/PII bayrağı)
uv run achilles registry-list --kind decisions       # promotion_decisions (denetim izi)

# Mevcut RAG indeks + embedding anlık görüntüsü (SQLite sayımından; ChromaDB/ağ gerekmez)
uv run achilles registry-snapshot

# Dataset TERFİSİ — İNSAN ONAYI (Kural 8)
uv run achilles registry-promote-dataset --version ds_abc123 --approver ali
uv run achilles registry-promote-dataset --version ds_abc123 --approver ali --reject --reason "kalite düşük"
```

Programatik:
```python
from app.registry import RegistryStore, approve_dataset, check_rag_index_eval, gate_reward_dataset
reg = RegistryStore()
ds = reg.register_dataset(name="lora_sft", content_hash=h, n_records=669)   # content_hash → idempotent
approve_dataset(reg, ds["dataset_version_id"], approver_id="ali")           # pending→approved (atomik)
check_rag_index_eval(reg, idx_id, {"recall_at_10": 0.8, "citation_accuracy": 0.9, ...})  # ReleaseGate
gate_reward_dataset(reg, rew_id, texts)                                     # sır/PII tarama → blocked/approved
```

## Kapılar (CLAUDE.md)
- **Kural 8:** dataset `approved` olmadan eğitime giremez; terfi İNSAN ONAYI (`requires_approval`).
- **Terminal durum makinesi:** yalnız `pending → approved|rejected`; atomik CAS (TOCTOU/çift-karar yok);
  terminal durumda yeniden terfi → `ValueError`.
- **RAG indeks:** ReleaseGate (recall@10≥0.70, citation≥0.85, grounding≥0.80, abstention≥0.90).
- **Ödül seti:** sır (sk-…/AKIA…/private key) veya PII (e-posta/+telefon) → `blocked` (bayrak=2).

Her karar `promotion_decisions` tablosuna append-only yazılır → tam denetim izi. Gözetimsiz
GERÇEK eğitimi ASLA önerme; yalnız sürüm/onay durumunu raporla.
