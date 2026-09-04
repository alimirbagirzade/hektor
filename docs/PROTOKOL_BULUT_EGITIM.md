# Stage 2 — Bulut-GPU LoRA Eğitim Protokolü

> Amaç: Stage 1'de üretilen ≥1000 örnekle, **ücretsiz bulut-GPU'da (Kaggle/Colab T4)**
> Qwen3-4B LoRA fine-tune → **GGUF Q4_K_M** → **Ollama**. CPU'da haftalar süren iş
> burada ~30-90 dk. Üst protokol: [PROTOKOL_ASAMALI_EGITIM.md](PROTOKOL_ASAMALI_EGITIM.md).
> Notebook 4-ajan araştırmasıyla doğrulandı (5 bilinen hata düzeltildi).

## Ön koşul (GATE)
- ≥1000 örnek (`uv run achilles lora-readiness`) **VE** `lora-audit` geçti **VE**
  kullanıcı onayı. Eğitim yalnız açıkça başlatılır (CLAUDE.md kural 8).

## Hazırlık (lokal, tek komut)
```bash
uv run achilles lora-cloud-prep --hf-repo KULLANICI/achilles-lora-sft \
  --adapter-name achilles_lora_cloud --lora-r 16 --epochs 2 --max-seq-len 2048
```
Üretir: `data/lora_sft/lora_sft.jsonl` (sentetik + kart, dedup'lı) +
`notebooks/achilles_lora_stage2.ipynb` + `notebooks/Modelfile`.

## Adım adım
1. **Veri eşiği:** `lora-readiness` ≥1000 + `lora-audit` (Gate 0-7). Az veri overfit eder.
2. **HF private dataset:** `huggingface-cli login` (write) →
   `huggingface-cli repo create achilles-lora-sft --repo-type dataset --private` →
   `huggingface-cli upload KULLANICI/achilles-lora-sft data/lora_sft/lora_sft.jsonl lora_sft.jsonl --repo-type dataset`.
   **Private** olmalı (veri sızmasın).
3. **READ token:** huggingface.co → Settings → Access Tokens → type=Read. Platform
   secret'ına ekle — Kaggle: Add-ons→Secrets, ad=`HF_TOKEN`; Colab: anahtar ikonu,
   ad=`HF_TOKEN`. **Token'ı asla notebook'a yazma.**
4. **Platform + GPU:** Kaggle (önerilen) Settings→Accelerator=`GPU T4 x2`, Internet=ON.
   Colab→Runtime→T4 GPU. Kaggle daha güvenilir kota (30sa/hafta) + kalıcı çıktı.
5. **Placeholder'lar zaten dolu** (`lora-cloud-prep` enjekte etti); yalnız
   `HF_DATASET_REPO`'yu kendi kullanıcı adınla doğrula. **Run All**.
6. **Süre/kota:** 4B / r16 / 2-3 epoch / ~1000-2000 örnek @ T4 ≈ 30-90 dk. İlk sefer
   küçük smoke (200 örnek, 1 epoch) ile süre ölç.
7. **GGUF üret + doğrula:** HÜCRE 12 (unsloth q4_k_m) + otomatik Modelfile. Çıktı
   bozuk (`GGGG`/NaN) ise HÜCRE 12B `RUN_FALLBACK=True` (16-bit merge → llama.cpp).
   İndirmeden önce render örneği (HÜCRE 8) + maske token sayısı (HÜCRE 10) kontrol.
8. **İndir:** `achilles-Q4_K_M.gguf` + `Modelfile` + `*_loss.json`. Kaggle: Output
   sekmesi. Aynı klasöre koy (örn `models/adapters/achilles_lora_cloud/`).
9. **Ollama'ya yükle:** `ollama create achilles -f Modelfile` →
   `ollama run achilles "Sharpe oranını ve sınırlamalarını açıkla."` →
   `ollama show achilles --modelfile`. Cevap durmuyorsa `<|im_end|>` stop eksik.
10. **EVAL GATE (kural 2):**
    `$env:ACHILLES_LLM_MODEL='achilles'; uv run achilles evaluate evals/discipline_core.jsonl`.
    Hedef: score=1.0, total_flags=0 (guaranteed_profit / ignores_costs düşmemeli).
11. **Registry + PROMOTE (yalnız onayla):** `uv run achilles lora-registry` /
    `lora-status`. CANDIDATE → eval geçince EVAL_PASSED → promote (user_approved).

## Notebook'taki 5 düzeltme (eski → yeni)
| Hata | Eski | Düzeltme |
|------|------|----------|
| target_modules | yok (PEFT default q/v) | 7 modül açık; lm_head/embed YOK (Qwen3 tied) |
| veri formatı | `{"text"}` + boş TRAIN_DATA | `{"messages":[…]}` okur |
| chat formatı | uydurma `<\|user\|>` | `apply_chat_template("qwen3-instruct")` |
| padding | `max_length` | dinamik + `train_on_responses_only` |
| export | yok | `save_pretrained_gguf("q4_k_m")` + Modelfile + fallback |

## Kritik notlar
- **Base eşleşmesi:** `Qwen/Qwen3-4B-Instruct-2507` = Ollama `qwen3:4b-instruct-2507`
  (çıplak `qwen3:4b` = 2504 thinking, adapter oturmaz). Notebook'ta sabit.
- **T4 = fp16** (bf16 NaN riski). 4-bit QLoRA + gradient checkpointing ile 16GB'a sığar.
- **`<|im_end|>` stop** Modelfile'da şart, yoksa model durmaz.

İlgili: `/bulut-egitim-protokolu`, `/lora-training-control-plane`.
