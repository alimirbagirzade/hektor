# Achilles — RAG + Eğitim Yeniden Tasarım Planı

> **Tek hedef:** Fine-tuning olmadan **şimdi** güvenilir "trader uzmanı"; ve doğru
> zamanda (≥1000 örnek) **bulut-GPU** ile LoRA. Sürekli CPU-eğitimi durdurulur.
> Tüm iddialar mevcut kodla ve dış kanıtla doğrulanmıştır; overclaim yok.
>
> Kaynak: 4 paralel araştırma ajanı (web + Achilles kodu okuması), 2026-06-13.

---

## 1. Dürüst teşhis — mevcut yaklaşım neden sağlam değil

Üç bağımsız kanıt, aynı sonuca işaret ediyor: **sürekli CPU-LoRA bu
konfigürasyonda yanlış araç.**

### (a) Veri eşiği — küçük-N fine-tune overfit eder, hatta zarar verir
Mevcut durum: 15-50 örnek (çoğu deterministik şablondan;
`auto_lora_state.json` → "4 kart onaylandı, 49 reddedildi"). Literatür pratik
minimumu **~1000 örnek/görev** koyar; 500 altında LoRA'nın tek "faydası"
overfit'i *azaltmasıdır* — iyi olması değil. Küçük set + domain-spesifik veri =
**catastrophic forgetting** riski: model yeni veriye overfit olup base
yeteneklerini kaybeder. Yani 15-50 örnekte net fayda **negatif** olabilir.
(Latitude; Inference.net; Towards AI; D²LoRA arXiv 2503.18089.)

### (b) Yanlış problem türü — asıl ihtiyaç bilgi enjeksiyonu, bu RAG işi
"Trader gibi düşünme" iki parçaya ayrılır: trading **faktları/formülleri**
(makalelerden) + akıl yürütme **üslubu**. Faktler için RAG, fine-tuning'i geniş
farkla yener — Ovadia et al. EMNLP 2024: *"RAG consistently outperforms it
[fine-tuning]… LLMs struggle to learn new factual information through
unsupervised fine-tuning"* (yeni-bilgi: RAG 0.875 vs FT 0.50; arXiv 2312.05934).
Üslup için LIMA tezi: bilgi pretraining'de zaten var; az örnek sadece **formatı**
öğretir, yeni bilgi koymaz — ve bu "az örnek mucizesi" 1000 *titizce elenmiş*
örnekle gerçekleşti, 15-50 ham örnekle değil (arXiv 2305.11206).

### (c) Donanım — CPU'da 4B LoRA pratikte çalışmıyor (kanıtlı)
`storage/auto_lora_state.json` → `"stage": "train_failed"`,
`"last_error": "Eğitim COMPLETED olmadı"`. CPU'da 4B fp32 ~76 sn/adım = haftalar.
Bu hem boşa kaynak, hem `CLAUDE.md` kural 8 ("otomatik ağır eğitim yok, train
varsayılan dry-run") ile doğrudan çelişiyor.

### (d) Acı ironi — güçlü RAG zaten yazılmış ama bağlı değil
**Doğrulandı (kodla):** `app/brain/rag_answerer.py` içinde
Reranker/HybridRetriever/SelfRefiningRAG **hiç geçmiyor**;
`retrieval_service.py:38` `retrieve()` tek embed + tek Chroma sorgusu (sıralama
yok); ingestion'da hiçbir yerde `bm25.add_document` çağrısı **yok** (BM25 indeksi
boş). Yani `HybridRetriever`, `Reranker`, `EnsembleReranker`,
`MultiQueryRetriever`, `QueryExpander`, `SelfRefiningRAG`, `HybridChunker`
**yazılı ama canlı yolda kullanılmıyor**.

> **Teşhis özeti:** Yanlış araç (FT), yanlış problemde (fakt enjeksiyonu),
> yanlış donanımda (CPU) — çalışan ama bağlanmamış doğru araç (RAG) varken.

---

## 2. Sağlam mimari

### 2.1 RAG-first — şimdi robust, eğitimsiz (en yüksek etki/en düşük efor)
OpenAI doğrulama sırası: **prompt → few-shot → RAG → (en son) fine-tuning.**
Achilles'te eksik olan teknoloji değil, **entegrasyon.**

| Sıra | İyileştirme | Achilles'te somut | Durum |
|------|-------------|-------------------|-------|
| **P0** | Over-fetch + rerank + truncate | `RerankingRetriever` → `RagAnswerer` varsayılanı | ✅ tamam |
| **P0** | BM25'i doldur → hybrid'i aç | `bm25_corpus.py` (Chroma'dan lazy/cache) + `RerankingRetriever` genişletir; `rag_hybrid=True` | ✅ tamam |
| **P1** | Cross-encoder reranker | `cross_encoder_reranker.py` (sentence-transformers, graceful fallback); OPT-IN `rag_cross_encoder` | ✅ tamam (opt-in) |
| **P2** | Contextual Retrieval (chunk prefix `başlık/bölüm: …`) | `paper_indexer.build_embed_text` + `rag_contextual_embed` flag + `reindex-contextual` CLI | ✅ kod hazır (re-index kullanıcı-tetikli) |
| **P3** | Prompt: tek format + zorunlu satır-içi atıf + 1 few-shot | `rag_answer.md` tek kaynak; `_BILINGUAL_FORMAT` kaldırıldı | ✅ tamam |
| **P4** | Eval'i gerçek yola yönelt | `retrieval_eval.py` `Retriever` protokolünü alır → `RerankingRetriever` geçilebilir | ✅ etkin |

> **Cross-encoder'ı açmak (P1, opsiyonel):** `uv pip install sentence-transformers` +
> `ACHILLES_RAG_CROSS_ENCODER=true`. Model (bge-reranker-base, ~280MB) ilk kullanımda
> iner; i7 CPU'da her sorguya latency ekler — bu yüzden varsayılan kapalı. Model
> yoksa otomatik heuristik reranker'a düşer (sistem her zaman çalışır).
>
> **Contextual Retrieval'ı açmak (P2, opsiyonel):** `uv run achilles reindex-contextual`
> (tüm korpusu "başlık/bölüm:" ön-ekiyle yeniden embed eder — AĞIR, Ollama) → sonra
> `.env`'e `ACHILLES_RAG_CONTEXTUAL_EMBED=true`. Varsayılan kapalı: yarı-prefix'li
> korpus tutarsız olurdu, bu yüzden hep-veya-hiç (re-index zorunlu).

**Korunan güçlü taraf:** Refusal-on-empty zaten DOĞRULANDI
(`rag_answerer.py` "kaynak bulunamadı") → `CLAUDE.md` kural 7 ile uyumlu. Bozma.

### 2.2 Veri inşası — 15 → 1000-2000 sentetik örnek (lokal üretim)
**Mevcut darboğaz:** üretim tamamen deterministik şablon
(`training_data_builder.py::examples_from_card()` kart başına ~4-6 sabit örnek;
`question_generator.py` LLM kullanmıyor). **Tek kritik eksik: LLM-tabanlı
chunk→N-QA üretici.** Altyapının ~%80'i (retrieval, embedding, LLM router, MLX,
eval, quality_filter iskeleti) zaten var.

Standart **RAFT/Self-Instruct** pipeline'ı (50 makale × ~10 chunk × 5 soru ≈
2500 ham → ~2000 net):
1. **Chunk → N QA/chunk** (yeni `app/brain/synthetic_qa_builder.py`). RAFT default 5 soru.
2. **Persona-çeşitliliği:** kantitatif araştırmacı / risk yöneticisi / backtester / şüpheci denetçi.
3. **RAFT formatı:** 1 oracle + 4 distractor; %80 oracle dahil, %20 çıkar (akıl yürütme, ezber değil).
4. **Kalite kapıları** (`quality_filter.py` genişlet): embedding-cosine ≥0.90 dedup + ROUGE-L <0.7 + roundtrip + LLM-judge grounding.
5. **Abstention dengesi:** sette ~%15-20 refusal/abstention tut.

**Üretim eforu (CPU-only, dürüst):** ~3M çıktı token ≈ **~6 gün kesintisiz** —
tek-thread pratik değil. **Çözüm:** gece batch'leri + `OLLAMA_NUM_PARALLEL`,
veya küçük üretici model (1.5-3B). Apple Silicon/MLX'te birkaç saat.

### 2.3 Eğitim — CPU sürekli-eğitimi DURDUR; "lokal-üret + bulut-GPU-eğit"
**CPU rolü yalnız:** dataset üretimi, Ollama çıkarım/backtest, eval. **Eğitim
değil.** `app/lora/auto_pipeline.py` sürekli CPU oto-eğitimi durdurulur.

Uçtan uca kadans (~30-60 dk, maliyet $0):

| Adım | Nerede | Süre |
|------|--------|------|
| 1. Dataset üret (train/valid JSONL) | Lokal CPU | rahat |
| 2. Yükle (HF **private** repo + token) | — | 1-2 dk |
| 3. Eğit: **unsloth**, `r=16/32, lr=2e-4, 2 epoch, batch 16` | **Kaggle (30 sa/hafta T4×2)** / Colab T4 | 15-40 dk |
| 4. Export: `save_pretrained_gguf("q4_k_m")` + Modelfile | Bulut | 3-8 dk |
| 5. İndir → `ollama create achilles` → eval gate → promote | Lokal | 2-5 dk |

**Ucuz GPU sırası:** Kaggle > Colab free T4 > Modal ($30/ay kredi) > RunPod 4090
($0.34/sa). 4B için A100 gereksiz; T4 yeterli.

**Mevcut Colab üreticisinde (`peft_lora_train.py::generate_colab_notebook`)
düzeltilecek 5 hata:** (1) unsloth değil düz transformers (~2× yavaş);
(2) `padding='max_length'` (~%40 yavaş); (3) `TRAIN_DATA=[]` boş; (4) GGUF/Ollama
export hücresi yok; (5) `target_modules` belirtilmemiş → GGUF'ta deltalar sessizce
düşer, **adapter bozulur.**

---

## 3. Geçiş planı (sıralı)

### Faz A — HEMEN (eğitimsiz değer; sıfır eğitim)
- **A1.** `auto_pipeline.py` sürekli CPU oto-eğitimini durdur; gece döngüsünü
  EĞİT yerine VERİ-ÜRET'e çevir. Smoke-test dalını koru.
- **A2. [P0]** `RerankingRetriever` → over-fetch + rerank + truncate. **✅ tamam.**
- **A3. [P0]** ingestion'da `bm25.add_document` + `HybridRetriever` enjekte.
- **A4. [P3]** `rag_answer.md` ↔ `_BILINGUAL_FORMAT` çatışmasını gider; satır-içi atıf + 1 few-shot.
- **A5. [P4]** `retrieval_eval.py` hybrid+rerank yolunu ölç.
- **A6.** `synthetic_qa_builder.py` (chunk→5 QA, persona havuzu); gece batch.
- **A7.** `quality_filter.py` genişlet: cosine ≥0.90 + ROUGE-L <0.7 + roundtrip + grounding.
- **A8. [P1]** `cross_encoder_reranker.py` (bge-reranker-v2-m3; fallback heuristik).
- **A9. [P2]** Contextual chunk-prefix.
- Her adımdan sonra: `make format && make lint && make typecheck && make test` (çevrimdışı).

### Faz B — dataset ≥1000 olunca (bulut-GPU)
- **B1.** `generate_colab_notebook` → unsloth şablonu (5 hatayı düzelt).
- **B2.** Dataset → HF private repo.
- **B3.** Kaggle/Colab'da eğit; GGUF Q4_K_M + Modelfile export.
- **B4.** Adapter indir → eval gate → smoke → güvenli promote. **Kural 8:** yalnız açık `--run`.

---

## 4. Karar tablosu — RAG-only vs RAG+LoRA

| Boyut | RAG-only (ŞİMDİ) | RAG + LoRA (SONRA) |
|-------|------------------|--------------------|
| Veri eşiği | Her boyut; few-shot için 15-50 örnek değerli | LoRA ≥**1000** kaliteli/çeşitli; altında erteleme |
| Çözdüğü problem | Faktel bilgi enjeksiyonu → **RAG'ın işi** | Üslup/format prompt+RAG ile çözülemiyorsa |
| Donanım | CPU'da rahat | Eğitim yalnız **bulut-GPU** (kanıt: `train_failed`) |
| Risk | Forgetting yok, güvenli | Küçük-N'de overfit + forgetting |
| Kural uyumu | `CLAUDE.md` 7,8 ile uyumlu | Yalnız açık `--run` (kural 8) |

**Promosyon kapısı (3 koşul birden):** (1) ≥1000 kaliteli/çeşitli örnek; **VE**
(2) tam RAG yolu üslup/format hatasını çözememiş; **VE** (3) eğitim bulut-GPU'da,
açık `--run`, eval gate + smoke geçerek.

**Nihai karar:** Achilles şimdi yazılı-ama-bağlanmamış RAG'ı canlıya almalı
(P0-P4), lokal sentetik veriyle 15→2000 örneğe çıkmalı, sürekli CPU-LoRA'yı
durdurmalı; LoRA'yı yalnız ≥1000 örnek + RAG ile çözülemeyen üslup hatası
koşulunda bulut-GPU kadansıyla yapmalı — hem 2024-2026 kanıtıyla hem projenin
"ağır otomatik eğitim yok" kuralıyla tutarlı.
