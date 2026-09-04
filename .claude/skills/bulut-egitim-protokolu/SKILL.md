---
name: bulut-egitim-protokolu
description: Stage 2 — ücretsiz bulut-GPU'da (Kaggle/Colab T4) Qwen3-4B LoRA eğitimi. Veri paketle, doğrulanmış unsloth notebook üret, GGUF→Ollama, eval gate + güvenli promote. CPU eğitimi YAPMAZ.
when_to_use: Stage 1 eşiği (≥1000 örnek) dolup kullanıcı bulut-GPU LoRA eğitimine geçmek istediğinde; notebook hazırlama, HF yükleme, GGUF→Ollama kurulumu, eval veya promote işlemlerinde.
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Stage 2 — Bulut-GPU LoRA Eğitim Protokolü

Amaç: ≥1000 örnekle ücretsiz T4'te Qwen3-4B LoRA → GGUF Q4_K_M → Ollama. CPU'da
haftalar süren iş burada ~30-90 dk. Detay: `docs/PROTOKOL_BULUT_EGITIM.md`.

## ÖN KOŞUL — geçişten önce DOĞRULA (hepsi)
1. `uv run achilles lora-readiness` → ≥1000 örnek.
2. `uv run achilles lora-audit` → Gate 0-7 geçti.
3. **Kullanıcı açık onayı** (gerçek eğitim yalnız açık komutla — CLAUDE.md kural 8).
Eksikse Stage 2'ye GEÇME; `/veri-uretim-protokolu` ile üretime devam et.

## Hazırlık (lokal)
```bash
uv run achilles lora-cloud-prep --hf-repo KULLANICI/achilles-lora-sft --lora-r 16 --epochs 2
```
Üretir: `data/lora_sft/lora_sft.jsonl` + `notebooks/achilles_lora_stage2.ipynb` +
`notebooks/Modelfile`. Notebook doğrulanmış unsloth şablonudur (5 hata düzeltilmiş).

## Akış
| # | İş | Nerede |
|---|-----|--------|
| 1 | Veri → HF private dataset (`huggingface-cli upload … --repo-type dataset`) | lokal |
| 2 | HF READ token → Kaggle Secrets / Colab userdata (ad=HF_TOKEN, GÖMME YOK) | lokal |
| 3 | Kaggle T4×2 (Internet ON) / Colab T4 → Run All | bulut |
| 4 | GGUF Q4_K_M + Modelfile üret (HÜCRE 12; bozuksa 12B fallback) | bulut |
| 5 | İndir → `ollama create achilles -f Modelfile` | lokal |
| 6 | Eval gate: `evaluate evals/discipline_core.jsonl` (score=1.0, flags=0) | lokal |
| 7 | Registry promote (yalnız kullanıcı onayıyla) | lokal |

## Kritik kurallar
- **Base:** `Qwen/Qwen3-4B-Instruct-2507` (= Ollama `qwen3:4b-instruct-2507`). Çıplak
  `qwen3:4b` (2504 thinking) ile DEĞİŞTİRME — adapter oturmaz.
- **T4 = fp16** (bf16 değil). 4-bit QLoRA + gradient checkpointing.
- **`<|im_end|>` stop** Modelfile'da şart.
- Eğitimi ASLA otomatik başlatma; kullanıcı bulutta kendi çalıştırır.

## Kullanıcı onayı gerektiren
- Stage 2'ye geçiş · production adapter promote (`lora-status` ile doğrula).
