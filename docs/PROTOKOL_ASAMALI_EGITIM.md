# Hektor — Aşamalı Eğitim Protokolü (master)

> Tek cümle: **Önce lokal veri üret (Stage 1), eşik dolunca bulut-GPU'da gerçek
> LoRA eğit (Stage 2).** CPU sürekli-eğitimi yok (haftalar sürer + overfit).
> Kanıt ve gerekçe: [RAG_EGITIM_YENIDEN_TASARIM.md](RAG_EGITIM_YENIDEN_TASARIM.md).

```
┌─────────────────────────┐   GATE (≥1000 örnek    ┌──────────────────────────┐
│ STAGE 1 — VERİ ÜRET     │   + kalite denetimi    │ STAGE 2 — BULUT-GPU LoRA │
│ (lokal CPU, sürekli)    │ ─────────────────────▶ │ (Kaggle/Colab, dakikalar)│
│ synth-qa + zenginleştir │                        │ unsloth → GGUF → Ollama  │
└─────────────────────────┘                        └──────────────────────────┘
   PROTOKOL_VERI_URETIM.md                            PROTOKOL_BULUT_EGITIM.md
```

## Neden aşamalı?
| | Stage 1 (şimdi) | Stage 2 (eşikte) |
|---|---|---|
| Nerede | Lokal CPU (i7, GPU yok) | Bulut-GPU (ücretsiz T4) |
| Ne yapar | Makale → grounded sentetik QA üretir | Gerçek LoRA fine-tune |
| Süre | Sürekli (gece döngüsü) | ~30-60 dk/koşu |
| Maliyet | $0 | $0 (Kaggle 30sa/hafta / Colab free) |
| Çıktı | `data/lora_sft/synthetic_qa.jsonl` (birikir) | GGUF Q4_K_M adapter → Ollama |

## GATE — Stage 1'den Stage 2'ye geçiş koşulu (3 koşul birden)
1. **Nicelik:** ≥ **1000** sentetik örnek (anlamlı LoRA eşiği).
2. **Kalite:** dataset denetimi geçti (grounding + dedup + OOS bölme; `lora-audit`).
3. **Onay:** kullanıcı açıkça "Stage 2 / bulut-eğitim" dedi (CLAUDE.md kural 8:
   gerçek eğitim yalnız açık komutla).

Eşik durumu: `uv run hektor lora-readiness` (nicelik) + `uv run hektor rag-mastery`.

## Hızlı komut haritası
| Aşama | Komut |
|---|---|
| 1 | `uv run hektor synth-qa` — sentetik QA üret (birikir) |
| 1 | `bash scripts/continuous-learning.sh 72` — sürekli üretim döngüsü |
| 1 | `uv run hektor lora-readiness` — Stage 2 eşik durumu |
| GATE | `uv run hektor lora-audit` — Gate 0-7 kalite denetimi |
| 2 | `uv run hektor lora-cloud-prep` — notebook + veri paketi üret |
| 2 | (bulutta) notebook'u çalıştır → GGUF indir |
| 2 | `ollama create hektor-lora -f Modelfile` → eval → promote |

İlgili skiller: `/veri-uretim-protokolu`, `/bulut-egitim-protokolu`,
`/lora-training-control-plane`.
