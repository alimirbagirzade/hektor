<!-- Achilles: RAG+LoRA entegrasyon plani — 3 paralel web arastirmasinin sentezi (2026-06-13). -->

# LoRA + RAG Entegrasyon Planı — Achilles (CPU-only Windows, Ollama qwen3:4b + PEFT)

Repo gerçekleriyle doğrulanmış sentez. Kritik repo bulguları: `app/config/settings.py:63` → `peft_base_model = "Qwen/Qwen3-4B"`; `app/training/peft_lora_train.py` → `max_seq_length=512`, `LoraConfig`'te `target_modules` YOK, `_row_to_text` Qwen3'ün gerçek ChatML şablonu yerine uydurma `<|system|>...<|end|>` formatı kullanıyor, loss tüm tokenlara uygulanıyor. Bunların hepsi aşağıdaki planda düzeltme maddesi.

---

## 1. Ollama'ya adapter bağlama

**Tek çalışan yol: PEFT safetensors → GGUF dönüşümü.** Ollama'nın `ADAPTER` komutu ham safetensors adaptörlerini yalnızca Llama/Mistral/Gemma için otomatik dönüştürüyor; Qwen3 listede yok ([docs.ollama.com/modelfile](https://docs.ollama.com/modelfile), [ollama#11084](https://github.com/ollama/ollama/issues/11084), [ollama#11376](https://github.com/ollama/ollama/issues/11376); Qwen3 safetensors import PR'ı [#14195](https://github.com/ollama/ollama/pull/14195) Şubat 2026 itibarıyla hala açıktı — durumu değişmiş olabilir, kontrol et). GGUF adaptörlerde mimari kısıtı yok; llama.cpp `qwen3` arch'ı destekliyor.

**Adımlar:**

```powershell
# 1) llama.cpp dönüşüm scripti
git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt   # torch, transformers, gguf, safetensors

# 2) PEFT adapter dizinini GGUF'a çevir (f16 tut — r=8'de çıktı onlarca MB)
python llama.cpp/convert_lora_to_gguf.py <peft_adapter_dir> `
    --base <lokal_Qwen3-4B_HF_checkpoint> --outfile qwen3-4b-achilles-lora.gguf --outtype f16

# 3) Modelfile
# FROM <eğitimde kullanılan base'in BİREBİR aynısı>
# ADAPTER ./qwen3-4b-achilles-lora.gguf

# 4) ollama create achilles-qwen3-lora -f Modelfile
```

Kaynak: [convert_lora_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py) (flag'ler: `--base`, `--base-model-id`, `--outtype {f32,f16,bf16,q8_0}` — varsayılan f32, f16'yı açıkça ver).

**Qwen3'e özgü uyarılar:**

1. **BASE EŞLEŞMESİ (en kritik):** Ollama'daki `qwen3:4b` tag'i 2025 sonunda sessizce **Qwen3-4B-Instruct-2507**'ye yönlendirildi; `settings.py`'daki `Qwen/Qwen3-4B` (orijinal, hybrid-thinking) ise farklı bir checkpoint. Base uyuşmazsa "behaviour will be erratic" ([docs.ollama.com/import](https://docs.ollama.com/import)). **İlk iş:** `ollama show qwen3:4b --modelfile` ile makinedeki gerçek tag'i doğrula; sonra ya `peft_base_model`'i `Qwen/Qwen3-4B-Instruct-2507`'ye çek (önerilen — instruct-2507 thinking'siz, CPU'da da hızlı) ya da `FROM`'a orijinal checkpoint'in kendi GGUF'unu koy.
2. **Tied embeddings:** Qwen3-4B `tie_word_embeddings=true`; dönüştürücü `lm_head`/`embed_tokens` deltalarını sessizce atlar ve adaptör kalitesi bozulur (Gemma2'de aynı sınıf bug: [llama.cpp#9065](https://github.com/ggml-org/llama.cpp/issues/9065)). → PEFT'te `target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` ile sınırla; `modules_to_save` yok, yeni token ekleme yok. (`peft_lora_train.py:128`'de `target_modules` şu an hiç belirtilmemiş — açıkça yaz.)
3. **Q4_K_M base + f16 adapter çalışır** (adaptör runtime'da ayrı matmul olarak uygulanır), ama bf16'da eğitilen adaptörün q4 base üstünde küçük kalite kayması beklenir.
4. **Flash attention:** Adapter yüklenmezse `OLLAMA_FLASH_ATTENTION` kapat ([ollama#8418](https://github.com/ollama/ollama/issues/8418)).
5. **Tek adapter / hot-swap yok** ([#7627](https://github.com/ollama/ollama/issues/7627), [#9548](https://github.com/ollama/ollama/issues/9548)) — her adapter varyantı ayrı `ollama create` modeli; base bir kez saklanır, varyantlar ~20-40 MB.
6. **QLoRA kullanma** — bnb-4bit base ile eğitilen adaptör GGUF'ta bozulur; mevcut trainer zaten fp32/fp16, sorun yok.

**Terfi eden adapter için (üretim yolu):** `merge_and_unload()` → `convert_hf_to_gguf.py --outtype f16` → `llama-quantize ... q4_k_m` → Modelfile'da `FROM ./merged-q4_k_m.gguf` + `ollama show qwen3:4b --modelfile`'dan TEMPLATE/PARAMETER bloklarını kopyala (ham GGUF import şablonu güvenilir devralmaz). En sağlam yol bu ([Chainstack pipeline örneği](https://docs.chainstack.com/docs/ai-trading-agent-fusing-llm-adapters-and-converting-to-ollama)); dezavantajı her iterasyonda ~8 GB ara dosya + yeniden quantize. **Öneri:** iterasyonda GGUF-adapter yolu, `app/lora/gates.py` terfi kapısını geçen adapter için merge.

---

## 2. RAG-uyumlu LoRA eğitim reçetesi (RAFT)

Kaynak reçete: [RAFT, arXiv 2403.10131](https://arxiv.org/abs/2403.10131) + [Gorilla raft üreteci](https://github.com/ShishirPatil/gorilla/tree/main/raft) + [Trust-Align, arXiv 2409.11242](https://arxiv.org/abs/2409.11242).

**Örnek formatı** (inference'taki RAG prompt'uyla BİREBİR aynı şablon):
- Soru + **1 golden chunk + 3 distractor chunk** (aynı korpustan embedding-benzer ama cevapsız — `app/memory/retrieval_service.py` ile seçilebilir).
- **P=0.8:** örneklerin %80'inde golden bağlamda var; %20'sinde yalnız distractor (model "bağlamda cevap her zaman var" varsayımını öğrenmesin diye; P=1.0 ile eğitme).
- **Cevap formatı (CoT + alıntı, Türkçe):** `##Reason: ##begin_quote## [kaynaktan birebir alıntı] ##end_quote## ... ##Answer: ...` — RAFT'ta düz cevaba göre büyük kazanç (HotpotQA 25.6→35.3).
- **Refusal örnekleri (toplamın %15-20'si):** cevaplanamayan soru + sadece distractor → TEK sabit kalıp: *"Sağlanan kaynaklarda bu sorunun cevabı bulunamadı."* Tek kalıp = regex ile ölçülebilir (CLAUDE.md determinizm kuralına uygun). Daha fazlası aşırı red üretir (Trust-Align dengesi).
- **Replay %10-15:** genel Türkçe instruction verisi — yoksa catastrophic forgetting ile model "yalnız RAFT formatında konuşan" şeye döner ([CURLoRA](https://arxiv.org/abs/2408.14572)).

**Boyut hedefi — dürüst olalım:** mevcut 5-50 örnek LoRA için **kesinlikle yetersizdir**; ezberler, genellemez. Minimum ~1.000, pratik hedef 2.000-5.000 ([Unsloth guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [arXiv 2506.14681](https://arxiv.org/pdf/2506.14681)). Çözüm: chunk (~512 token) başına 3-5 sentetik QA'yı lokal LLM ile üret (Gorilla `raft.py` deseni, OpenAI yerine Ollama). ~70-100 paper × ~15-20 chunk × 3 QA ≈ 3-5k erişilebilir.

**Repo'ya özgü zorunlu değişiklikler:**
1. **Chat template (en sinsi tuzak):** `peft_lora_train.py:73-93` `_row_to_text` uydurma `<|system|>...<|end|>` formatı kullanıyor — Qwen3'ün gerçek şablonu değil. `tokenizer.apply_chat_template` ile değiştir; eğitim tokenizasyonu Ollama Modelfile TEMPLATE'iyle birebir eşleşmeli, yoksa adaptör sessizce bozulur. Test: bir eğitim örneğini her iki yoldan tokenize edip dizileri karşılaştır.
2. **max_seq_length 512 → 2048+:** golden+3 distractor 512 token'a sığmaz. (CPU eğitim süresi orantılı artar — aşağıda dürüst tahmin.)
3. **Loss maskeleme:** şu an `DataCollatorForLanguageModeling` tüm tokenlara loss veriyor; loss'u yalnız assistant tokenlarına uygula (prompt masking).
4. **Hiperparametreler:** r=8-16, alpha=2×r (mevcut 8/16 uygun), LR 1e-4 (küçük sette 2e-4 yüksek), 1-2 epoch, %10 validation, eval loss yükselince dur. Loss 0.5-1.0 bandı sağlıklı; loss→0 = ezber.
5. **Müfredat:** `app/lora/curriculum.py` zaten var — kolay (golden, distractor'sız) → orta (golden+distractor) → zor (refusal + multi-hop) → karışık son epoch sırasına eşle.
6. **k uyumu:** inference'ta retriever top-k kaçsa eğitimde de o civarda doküman kullan.

---

## 3. Anlama % ölçümü (paper başına, lokal 4B judge)

**ragas/deepeval kütüphanelerini KULLANMA:** 4B modellerle JSON parse hataları → NaN skorlar belgeli ([ragas#1228](https://github.com/explodinggradients/ragas/issues/1228), [#1099](https://github.com/explodinggradients/ragas/issues/1099), [#805](https://github.com/explodinggradients/ragas/issues/805)). Bunun yerine RAGAS metrik *tanımlarını* tek-çağrılık, kategorik-etiketli el yapımı judge promptlarıyla uygula (her biri <30 satır). 4B modele asla 1-10 sayısal puan sordurma — yalnız `CORRECT/PARTIAL/WRONG` gibi kategorik etiket ([arXiv 2406.12624](https://arxiv.org/pdf/2406.12624)).

**Reçete (paper başına):**
1. K=10 chunk örnekle (bölümlere stratifiye, kaynakça hariç) → her birinden 1 soru + chunk'tan neredeyse birebir alıntı altın cevap (temp 0.3, `format: json`). Mevcut `app/learning/question_generator.py` (şu an şablon-tabanlı, LLM'siz) bunun doğal genişleme noktası.
2. Roundtrip filtre: "Altın cevap gerçekten bu chunk'ta mı? EVET/HAYIR" — 4B üreteçle %20-40 eleme bekle.
3. 2-3 abstention tuzağı ekle (korpusta cevabı olmayan soru).
4. Soruları tam RAG hattından geçir (`app/learning/rag_exam_runner.py` zaten yapıyor).
5. İki katman skor: **retrieval hit** (deterministik — `source_chunk_id` getirilen chunk'lar arasında mı / altın cevap substring mi) + **judge correctness** (tek kelimelik CORRECT/PARTIAL/WRONG, temp 0) → `ExamAnswer.passed`'e bağla.

**Toplama:** `comprehension = 100 × (0.40·correctness + 0.20·retrieval_hit + 0.20·faithfulness + 0.10·abstention + 0.10·relevancy)` — ama Achilles'te **paralel skor icat etme**: `app/learning/mastery_scorer.py` zaten 0-100 Paper Mastery Score üretiyor ve retrieval/citation/grounding/abstention 55 puanını kapsıyor; judge-correctness'i o ağırlığa entegre et. N=10-12 soruda **Wilson %95 aralığı** raporla (N=10'da ham %80 aslında "%49-94" demek — nokta tahminine güvenme).

**CPU bütçesi:** paper başına ~55 kısa çağrı ≈ 30-60 dk → gece batch'i; her çağrıyı `(paper_id, question_hash, model_tag)` ile cache'le. Thinking modunu kapat (`/no_think` veya instruct-2507 varyantı) — thinking tokenları CPU'da gecikmeyi 5-10× artırır. `OLLAMA_NUM_PARALLEL=1`, `num_ctx 8192`.

**LoRA kazancı ölçümü:** soru setini `app/evals/golden_dataset.py`'de **dondur**; aynı set üzerinde base model vs adapter modeli A/B çalıştır — comprehension % farkı = adapter kazancı. Cevaplanabilir ve cevaplanamaz soruları ayrı raporla (doğru-cevap oranı + doğru-red oranı, Trust-Score mantığı). CLAUDE.md kuralı: bu eval geçmeden "çalışıyor" yok.

**Kalibrasyon şartı:** 4B judge'da hoşgörü/aşırı güven yanlılığı belgeli ([arXiv 2508.06225](https://arxiv.org/html/2508.06225v2)) — 30-50 elle etiketli örnekle bir kez kalibre et; ara sıra 20 örneklik alt kümeyi daha güçlü modelle çapraz doğrula.

---

## 4. Uygulama sırası (Achilles repo)

1. **Base doğrulama (bloklayıcı, 10 dk):** `ollama show qwen3:4b --modelfile` + digest kontrolü — tag'in orijinal Qwen3-4B mi Instruct-2507 mi olduğunu makinede doğrula. Karar: `app/config/settings.py:63` `peft_base_model`'i eşleşen checkpoint'e sabitle (öneri: Instruct-2507, thinking'siz).
2. **Ölçüm altyapısı ÖNCE (baseline olmadan kazanç ölçülemez):** `question_generator.py`'ye LLM-backed QA üretimi + roundtrip filtre; `rag_exam_runner.py`'ye CORRECT/PARTIAL/WRONG judge çağrısı; `mastery_scorer.py` entegrasyonu; soru setini `app/evals/golden_dataset.py`'de dondur; LoRA'sız baseline comprehension % al ve kaydet.
3. **Trainer düzeltmeleri** (`app/training/peft_lora_train.py`): `apply_chat_template`, assistant-only loss masking, açık `target_modules` (attention+MLP, lm_head/embed yok), `max_seq_length` parametreleştir (2048).
4. **RAFT dataset üretimi:** `app/lora/dataset_builder.py`'yi genişlet — golden+3 distractor (distractor seçimi `retrieval_service` embedding-benzerliğiyle), P=0.8, `##Reason/##Answer` Türkçe CoT, %15-20 refusal, %10-15 replay; `app/lora/quality_filter.py` + `gates.py`'den geçir; hedef ≥1.000 (ideal 2-5k). (`lora-training-control-plane` skili bu yaşam döngüsünü zaten kapsıyor — onunla yürüt.)
5. **Eğitim:** önce dry-run (mevcut varsayılan, CLAUDE.md kuralı), sonra `--run`; %10 val split, eval loss izle, early stop.
6. **GGUF dönüşüm + Ollama'ya bağlama:** Bölüm 1 adımları; tekrarlanabilirlik için `scripts/convert_adapter_to_ollama.ps1` yaz. Smoke test: aynı prompt'u eğitim ve inference yolundan tokenize edip karşılaştır + 3-5 örnek soru elle.
7. **A/B eval + terfi kapısı:** dondurulmuş set üzerinde base vs adapter; doğru-cevap ve doğru-red oranları ayrı; geçerse `app/lora/adapter_registry.py`'ye kaydet, `gates.py` kapısından terfi.
8. **Terfi sonrası üretim artefaktı:** merge + q4_K_M quantize, TEMPLATE bloklu Modelfile, ayrı `ollama create`.

**Belirsizlikler (dürüst liste):**
- `qwen3:4b` tag'inin makinedeki gerçek checkpoint'i doğrulanmadan hiçbir şey kesin değil (Adım 1 bu yüzden bloklayıcı).
- [PR #14195](https://github.com/ollama/ollama/pull/14195) birleşmiş olabilir — birleştiyse safetensors doğrudan `ADAPTER` çalışır ve GGUF dönüşümü gereksizleşir; başlamadan kontrol et.
- **CPU eğitim süresi:** 2-5k örnek × 2048 token × CPU = muhtemelen günler mertebesi. Gerçekçi başlangıç: 1-1.5k örnek, max_seq 1536-2048, 1 epoch; süre kabul edilemezse örnek/uzunluk kırp — ama 5-50 örnekle "LoRA çalışıyor" iddiası test edilemez, bu net.
- P=0.8 ve %15-20 refusal oranları veri setine göre değişiyor (RAFT ablasyonlarında P %40-100 arası optimal) — küçük bir ablasyon (P=0.6/0.8/1.0) değer.
- 4B judge skorları kalibrasyonsuz güvenilmez; Wilson aralığı raporlamadan tek sayı sunma.

İlgili dosyalar: `C:\Users\sevinc\Development\achilles\app\config\settings.py`, `app\training\peft_lora_train.py`, `app\lora\dataset_builder.py`, `app\lora\curriculum.py`, `app\lora\gates.py`, `app\lora\adapter_registry.py`, `app\learning\question_generator.py`, `app\learning\rag_exam_runner.py`, `app\learning\mastery_scorer.py`, `app\evals\golden_dataset.py`.
