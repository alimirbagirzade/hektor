# Eğitim Yol Haritası — Evrensel + Achilles Durumu

> Kaynak: `İLGİLENECEĞİMİZ EĞİTİM TÜRLERİ.docx`  
> Sıralama: pratikte en kolay kurulandan en zora doğru.  
> Bu dosya **şablon** niteliğindedir — yeni bir AI projesine başlarken kopyala,
> "Proje Durumu" sütununu güncelle.

---

## Özet Tablo

| # | Yöntem | Zorluk | Achilles Durumu | Bir Sonraki Adım |
|---|--------|--------|-----------------|-----------------|
| 1 | **RAG** | ⭐ | ✅ Tamamlandı | Sorgu kütüphanesi otomasyonu (Görev D) |
| 2 | **Tool Use Training** | ⭐⭐ | ✅ Tamamlandı | `tool_use_trainer.py` + `tool_use_dataset_builder.py` |
| 3 | **SFT** | ⭐⭐⭐ | 🟡 Devam | Daha fazla onaylı kart biriktir |
| 4 | **LoRA** | ⭐⭐⭐ | 🟡 Devam | `achilles_lora_v2` hazır, faz 2–3 bekliyor |
| 5 | **DPO** | ⭐⭐⭐⭐ | 🔴 Planlandı | 500+ onaylı kart gerekiyor |
| 6 | **Verifiable Reward** | ⭐⭐⭐⭐ | ✅ Tamamlandı | `reward_signal.py` + `dpo_dataset_builder.py` |
| 7 | **Agentic Training** | ⭐⭐⭐⭐⭐ | 🟡 MVP | OSS Agent → tam agentic döngü |

---

## 1. RAG — Retrieval-Augmented Generation

**Ne yapar:** Modeli yeniden eğitmeden dış bilgi deposuna bağlar.  
**Avantaj:** Ucuz, güncellemesi kolay, kaynak gösterilebilir.  
**Dezavantaj:** Modelin "düşünme karakterini" değiştirmez.

### Achilles ✅
- Advanced RAG katmanı: 37 modül, ChromaDB + SQLite
- 7 PDF / 567 chunk indeksli, `rag_answerer.py` aktif

### Bekleyen
- [x] **Görev D** ✅ — arXiv sorgu kütüphanesi (`achilles arxiv-sync`, 2026-06-07)
- [ ] Scheduled RAG güncelleme (yeni makale → otomatik ingest)

---

## 2. Tool Use Training

**Ne yapar:** LLM'in araçları (backtest, grafik, hesaplama, API) kullanmayı öğrenmesi.  
**Finans için kritik:** Agent sadece konuşmamalı, test etmeli ve ölçmeli.

### Achilles 🟡
- OSS Agent MVP araç altyapısı var (profiler, installer, benchmark)
- CLI araçlar LLM tarafından çağrılabilir durumda

### Achilles ✅ (2026-06-07)
- `app/training/tool_use_trainer.py` — THINK→CALL→OBSERVE→CONCLUDE döngüsü
- `app/training/tool_use_dataset_builder.py` — DB→SFT JSONL dönüştürücü
- `tool_use_examples` SQLite tablosu
- CLI: `achilles tool-use-train` / `achilles tool-use-dataset`
- 11 birim testi

### Bekleyen
- [x] tool_use SFT → LoRA faz 2 besleme pipeline'ı ✅ (`achilles unified-dataset`, 2026-06-07)

---

## 3. SFT — Supervised Fine-Tuning

**Ne yapar:** `{instruction, input, output}` üçlüsüyle modeli "böyle cevap ver" diye eğitir.

### Achilles 🟡
- `DatasetBuilder`: curriculum-aware (faz 1–4), lora_eligible filtresi
- `achilles_lora_v2` eğitildi (300 iter, loss 0.028)

### Bekleyen
- [ ] Faz 2 veri seti (daha fazla onaylı kart)
- [ ] Finans-spesifik soru-cevap standart kütüphanesi

---

## 4. LoRA — Low-Rank Adaptation

**Ne yapar:** Büyük modeli dondurup üstüne küçük adaptör ağırlıkları eğitir. Ucuz ve taşınabilir.

### Achilles 🟡
- `mlx_lora_train.py` (Apple Silicon, MLX-LM)
- `achilles_lora_v2` mevcut (2 GB, faz 1)

### Bekleyen
- [ ] Faz 2 adaptörü
- [ ] Faz 3: strateji değerlendirme reasoning
- [ ] Faz 4: DPO hazırlık

---

## 5. DPO — Direct Preference Optimization

**Ne yapar:** `{prompt, chosen, rejected}` üçlüsüyle tercih öğrenir. RLHF'den daha pratik.

### Achilles 🔴
- **Engel:** 500+ onaylı bilgi kartı gerekiyor (şu an ~0 onaylı)

### Bekleyen
- [ ] Onay ekranından yeterli kart biriktir
- [ ] `DPODatasetBuilder` sınıfı
- [ ] `chosen`/`rejected` çiftlerini backtest sonuçlarından türet

---

## 6. Verifiable Reward Training

**Ne yapar:** Modelin cevabı otomatik doğrulanabiliyorsa (backtest geçti mi? kod çalıştı mı?) ödül sinyali kullanır.

### Doğrulanabilir görev kapsamı (hedef liste)
Finans'ta doğrulanabilir görevler:
- [ ] Backtest kodu çalıştı mı? → ✅/❌
- [ ] Look-ahead bias var mı? → ✅/❌
- [ ] Sharpe, MaxDD, CAGR doğru hesaplandı mı? → ✅/❌
- [ ] Pozisyon sizing doğru mu? → ✅/❌

### Achilles ✅ (2026-06-07)
- `app/training/reward_signal.py` — 6 kriter: execution, trade_count, sharpe, drawdown, return, win_rate
- `app/training/dpo_dataset_builder.py` — reward → chosen/rejected çifti → DPO JSONL
- `reward_signals` SQLite tablosu
- `app/memory/sqlite_store.py`: `save_reward_signal`, `list_reward_signals`, `get_reward_signal`
- CLI: `achilles reward-analyze [--session] [--build-dpo]`
- 19 birim testi

### Paper Mastery — RAG kalite ölçümü ✅ (2026-06-07)
- Paper Mastery Agent — 0-100 RAG kalite skoru
- `achilles mastery-run/queue/score/report` CLI komutları
- `achilles mastery-to-sft` — mastery → SFT eğitim verisi

---

## 7. Agentic Training

**Ne yapar:** Modeli tek cevap vermekten çıkarır, adım adım görev yürütmeyi öğretir.

### Achilles 🟡 (MVP)
- OSS Agent MVP: profiler → advisor → installer → benchmark
- Research orchestrator: hipotez → backtest → reflection

### Bekleyen
- [ ] OSS Agent Phase 2 (llama.cpp + MLX backend)
- [ ] OSS Agent Phase 3 (RAG memory indexing)
- [ ] "Stratejiyi incele" end-to-end agent pipeline

---

## Achilles Uygulama Sırası

```
Şu an aktif:
  RAG ✅        → Görev D ✅ tamamlandı
  Tool Use 🟡   → backtest aracı LLM döngüsüne bağla
  SFT/LoRA ✅  → unified-dataset pipeline tamamlandı

Orta vadeli (1-3 ay):
  DPO 🔴          → önce 500+ kart biriktir
  Verifiable 🔴   → backtest otomatik ödül pipeline

Uzun vadeli (3+ ay):
  Agentic 🔴      → tam pipeline
  120B model      → donanım hazır olduğunda
```

---

## Diğer Projeler İçin Kullanım

Bu dosyayı başka bir AI projesine uygularken:

1. Dosyayı projenin kök dizinine kopyala
2. "Achilles" referanslarını proje adıyla değiştir
3. Her satırı `🔴 Planlandı` ile başlat, tamamlandıkça güncelle
4. "Bekleyen" listelerini projeye özel görevlerle doldur

**Evrensel öneri sırası:**
> RAG → Tool Use → SFT → LoRA → DPO → Verifiable Reward → Agentic Training

---

_Son güncelleme: 2026-09-02 (bölüm yapısı düzeltildi). Güncel durum için tek kaynak: `HANDOFF.md`._
