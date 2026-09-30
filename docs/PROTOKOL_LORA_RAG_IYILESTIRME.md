# LoRA + RAG İyileştirme Protokolü — Hektor

> Kaynak: `Claude_Hektor_LoRA_RAG_Iyilestirme_Promptu.txt` v1.0 (2026-09-30) +
> `Hektor_LLM_30_Soru.txt` v1.0. Bu doküman promptun **Hektor'a uyarlanmış** hâlidir: her
> aşama mevcut modüle bağlanır, yoksa "açık iş" olarak yazılır. Çalışan sistem baştan
> yazılmaz. CLAUDE.md'deki 8 mutlak kural önceliklidir.

## 0. Hedef ve sınırlar

- Hedef: Markov/HMM, filtreleme, Kalman, z-skoru, eşbütünleşme, entropi, regresyon eğimi vb.
  ile **test edilebilir** yeni indikatör hipotezleri üreten bir sistem. LLM hipotez + kod
  üretir; sayısal araçlar hesaplar, test eder, kanıt kaydeder.
- Akıcı cevap veya sınav puanı **trading avantajının kanıtı değildir**. İşlem açmamak ve
  hipotezi reddetmek geçerli sonuçtur. Kapsam: araştırma/backtest/paper — canlı emir yok.
- Yalnız yerel Ollama/PEFT; dış LLM API'siyle cevap ya da eğitim verisi üretilmez.
- Pahalı eğitim, büyük indirme, production adapter değişimi **ayrı insan kararıdır**
  (Kural 8; `train` dry-run varsayılan). Önce pilot + ölçülebilir aday.
- Çalıştırılmamış test/komut "başarılı" ilan edilmez (Kural 2).

## 1. Aşama → Hektor karşılığı

| Aşama | Mevcut karşılık | Açık iş |
|---|---|---|
| **1 Denetim** | `hektor status` / `doctor`; `app/evals/profile/manifest.py` `RunManifest` (base_model_hash, tokenizer_hash, quantization, rag_version, generation_config, seed); `app/memory/rag_version.py` `RetrievalTrace`; adapter kaydı `app/lora/adapter_registry.py` | Sistem istemi hash'i ve Ollama model digest'i manifestte zorunlu alan değil; GGUF şablonunda CR/`\r\n` denetimi (`adapter_to_ollama.ps1`, D1) yalnız oluşturmada koşuyor |
| **2 2×2** | `hektor mix eval` (`app/evals/profile/eval_runner.py`): retrieval kalem başına **bir kez** → RAG'lı iki sistem byte-aynı bağlamı görür | Harf eşlemesi aşağıda (§2). Kayıt alanlarından `finish_reason`, `input/output_tokens`, `prompt_hash`, `context_hash` `ItemScore`'da yok |
| **3 Benchmark** | **`evals/llm30/`** (bu entegrasyon) + `evals/profile_mix/{validation,golden_test}` + `evals/*.jsonl` adapter setleri; sızıntı kapısı `hektor mix leakage` / `train --run` | Yeni **final test** takımı (görülmemiş şablonlar) yazılmadı; teşhis varyantları yok; rubrik puanlama aracı yok (§3) |
| **4 RAG** | Zaten hibrit: `bm25_index` + dense (`chroma_store`) + `rank_fusion` + `reranker`/`cross_encoder_reranker` + `query_router`; `app/evals/retrieval_eval.py`, `contextual_chunker` | Etiketli query/relevance seti yok → Recall@k ölçülmedi; korpus-amaç filtresi (talimat vs makale vs eval) doğrulanmadı |
| **5 LoRA verisi** | `app/lora/{dataset_builder,dataset_splitter,quality_filter,curriculum,gates}.py`; `lora-audit` Gate 0-8; `pretrain-gate`; mix profilleri (`docs/LORA_MIX_EVAL.md`) | `verification_status` / `task_family` alanları şemada tek tip değil; atıflı RAG-formatlı örnek payı düşük (v10 bulgusu) |
| **6 İndikatör döngüsü** | `app/trading/indicators.py` registry; `/hypothesis-evaluator`, `/trading-research`, `/scientific-tool-runtime` | "Geleceği değiştirme invariance" testi indikatör başına şablon olarak yok |
| **7 Backtest/paper** | `app/trading/{backtester,evaluator,overfit_checks,risk_manager}.py`; `docs/PROTOKOL_BACKTEST.md`; `/backtest-auditor` | Denenen hipotez sayısı kaydı (çoklu deneme) ve purge/embargo politikası veri başına belgelenmedi |
| **8 Aday/geri dönüş** | Adapter **candidate** kaydı; `lora-eval` vetoları (degenerate, garanti, truncation >%20); profil regression gate; production = `--approve` | Kritik-hata kapısı (`app/evals/llm30.py:CRITICAL_ERRORS`) henüz eval koşusuna bağlı değil |

## 2. 2×2 karşılaştırma — harf eşlemesi (DİKKAT)

Protokol ve `hektor mix eval` harfleri **farklı**:

| Koşul | Protokol | Hektor sistemi |
|---|---|---|
| base, LoRA yok, RAG yok | A | `A_base` |
| base + LoRA, RAG yok | B | `C_<profil>` |
| base, RAG açık | C | `B_base_rag` |
| base + LoRA + RAG | D | `D_rag+<profil>` |

Raporlarda protokol harfi kullanılacaksa Hektor adı yanında yazılır.

**Eşitlik şartları:** aynı istem, chat şablonu, quantization, context/cevap sınırı, decoding.
Base Ollama `q4_K_M`, LoRA ise merge→GGUF ile servis ediliyorsa quantization/şablon farkı
adapter etkisiyle karışır → **izole edilemedi** diye raporla (bkz. HANDOFF B6: eval PEFT'i
ölçüyor, servis edilen GGUF'u değil). 4B ve 30B ayrı gruplanır; aynı aile adı aynı model değildir
(ör. `qwen3:30b` = Thinking-2507, `qwen3:30b-a3b-instruct-2507-q4_K_M` = Instruct).

İlk teşhis düşük-rastlantılı sabit profille (temperature 0, seed 42); üretim profili ayrıca.
Kritik sorular birkaç seed + semantik varyantla tekrarlanır; tek cevaptan kararlılık çıkarılmaz.
Kesilme (`truncated`), token sınırı, kısa ve uzun tekrar döngüsü ayrı işaretlenir; ham cevap
temizlenip yerine konmaz.

## 3. LLM-30 geliştirme benchmark'ı

- Dosya: `evals/llm30/validation.jsonl` (30 kayıt, `EvalItem` şeması) + `manifest.json`
  (sha256). Yükleyici: `app.evals.llm30.load_llm30(purpose=...)`; hash tutmazsa **RED**.
- **Geliştirme setidir** (split = validation): teşhis ve aday seçiminde okunabilir; bu yüzden
  final test yerine geçmez. Final test ayrı yazılır, eğitimden/retrieval'dan/istemden ayrı
  tutulur, erişim geçmişi kaydedilir.
- Sızıntı: `run_leakage_check` (→ `train --run` kapısı ve `hektor mix leakage`) artık llm30
  sorularını da tarar; `evals/` altındaki hiçbir dosya eğitim girdisi olamaz
  (`assert_not_eval_path`); eğitim kodu `llm30` yolunu referans edemez (statik test).
  2026-09-30 ölçümü: mevcut `train.jsonl` (1680) + `valid.jsonl` (111) ↔ llm30 **0 isabet**.
- Ortak cevap talimatı (soru dosyasının başı) sistem istemine konur, soruya değil.
- `domain` alanı profil eval'in 5 alanından biridir (yaklaşık eşleme); asıl gruplama
  `subdomain`: `veri_zaman` (S01-S10), `indikator_matematik` (S11-S20),
  `markov_rejim_test` (S21-S30).

**Puanlama (anahtar kelimeyle DEĞİL):** `app.evals.llm30.RUBRIC` — doğruluk 0-4, alt madde
0-2, tutarlılık 0-2, uygulanabilir doğrulama 0-2; kaynak desteği 0-2 **yalnız RAG
görevlerinde**. Atıf varlığı ile kaynağın iddiayı desteklemesi ayrı puanlanır. Hakem LLM tek
otorite değildir. Kaynak zorunlu görevde destek yoksa gerekçeli kaçınma doğru davranıştır ama
"teknik cevap tamam" puanı almaz. "Yanlış ama özgüvenli" ile "eksik ama belirsizliği doğru
bildiren" ayrı raporlanır.

**Sayısal anahtarlar:** `app.evals.llm30.numeric_keys()` S03/S06/S10/S12/S13/S14/S19/S22/
S25/S26 alt maddelerini deterministik hesaplar (sözleşmeler docstring'de: yüzde puanı, ISO
zaman, EMA tanımı, log2, satır-normalize). `reference_answer` bu anahtarlarla testle eşlenir.
Tüm kayıtlar `human_verified=false` — **insan doğrulaması bekliyor**.

**Kritik hatalar** (`CRITICAL_ERRORS`, soru eşlemesi `CRITICAL_BY_QUESTION`): düşüş mumunu
geçersiz sayma · EMA alpha ilişkisini ters söyleme · yanlış TZ lokalizasyonu · dört çeyrek
barı yanlış birleştirme · split/temettü getirisi · yanlış ATR · density ≠ olasılık kütlesi ·
indeks hizalamasıyla sahte geçiş matrisi · global yerine satır normalizasyonu · Markov ile
bağımsızlık/homojenliği karıştırma · final testte parametre seçme · "shift(1) her sızıntıyı
çözer" · hesap yerine uydurma değer. **Biri bile varsa toplam puan ne olursa olsun terfi yok.**

## 4. RAG hattı

- Retrieval kalitesini adapter'dan bağımsız ölç (etiketli küçük query/relevance seti →
  Recall@k + sıralama ölçütü). Teknik soruya sürekli proje kılavuzu geliyorsa query/doc
  eşleşmesini denetle; başlıktan ilgisizlik hükmü verme.
- Chunking formül + tanım + değişken + varsayımı birlikte tutsun; sayfa/bölüm ve kaynak kimliği
  korunsun; OCR/formül bozulması işaretlensin (`/ingestion-quality-scorer`).
- Proje talimatları, teknik makaleler, piyasa verisi ve eval materyali ayrı amaçla
  filtrelenebilir olsun; geliştirme dokümanı matematiksel otorite değildir. **Eval soruları,
  eski model cevapları ve anahtarlar retrieval indeksine alınmaz.**
- BM25 + rerank zaten var → yeni framework eklenmez; değişiklik ölçümle gerekçelendirilir.
  "Daha çok chunk = daha iyi" varsayılmaz; seçilen bağlam manifestte aynen saklanır.
- Adapter testinde korpus/snapshot sabit (`rag_version`); RAG iyileştirme testinde değişen
  unsurun retrieval olduğu açıkça işaretlenir.

## 5. LoRA verisi ve pilot

- Eski model cevabı doğrulanmadan "doğru cevap" olmaz; makale de kusursuz değildir —
  formül/varsayım/kod doğrulanır, kaynak ve doğrulama durumu kaydedilir.
- Müfredat: (1) veri/zaman doğrulama (2) temel matematik (3) indikatör uygulaması
  (4) Markov/HMM + rejim varsayımları (5) backtest + sızıntı (6) kaynak yetersizliği + ret.
  Görev çeşitliliği: hesap, kod, karşı örnek, yanlış çözümü düzeltme, kaynaklı açıklama,
  varsayım bildirme, işlem yapmama kararı. Her cevap aynı uyarı kalıbına sıkıştırılmaz.
- Örnek şeması en az: `sample_id, task_family, messages, source_refs, verification_status,
  quality_flags, split`. RAG-görünümlü örnek kaynak metni gerçekten mesajda taşımalı.
- Yakın kopyalar split'ler arasında dağıtılmaz (şablon/semantik düzeyde); promptlarda gizli
  test cevabı olmaz.
- Hiperparametreler (lr, r, alpha, dropout, target_modules, batch, GA, seq len, epoch)
  `configs/lora/lora_profiles.yaml`'da kaydedilir; hiçbiri evrensel doğru sayılmaz. Label
  masking, truncation, packing sınırı, EOS denetlenir; format sorunu epoch ekleyerek örtülmez.
- Checkpoint seçimi yalnız train loss'la değil, held-out val loss + görev benchmark'ıyla.
  Loss düşerken teknik doğruluk bozulursa aday geri çekilir. Bir deneyde tek önemli değişken.
- Her eğitimden önce: Kademe 2 derin av (CLAUDE.md) + mix ağırlık sorusu (kullanıcı kuralı).

## 6. İndikatör araştırma döngüsü

1. **Hipotez:** varsayım, mekanizma, karşı örnek, geçersizlik koşulu.
2. **Model:** denklemler, durumlar, birimler, pencere, bilgi erişim zamanı.
3. **Aday uygulama:** `app/trading/indicators.py` registry'sine küçük fonksiyon + test.
4. **Teknik test:** küçük bilinen örnek, indeks/birim/NaN, **gelecek değiştirilince geçmiş
   çıktı değişmez** testi, bağımsız referans.
5. **Tahmin testi:** kronolojik ayrım, marjinal/basit baseline, uygun skor + kalibrasyon.
6. **Ekonomik test:** gerçekçi fill, turnover maliyeti, risk sınırlı backtest (Kural 3-4).
7. **Karar:** kanıt destekli aday / yetersiz kanıt / reddedildi.

Zayıf sonuçta Markov/HMM sırf karmaşık diye sürdürülmez; EMA/regresyon eğimi gibi basit
adaylarla aynı koşulda kıyaslanır.

## 7. Backtest ve paper

- Point-in-time veri, takvim, kurumsal işlemler, varlık kimliği, lisans, survivorship veri
  sözleşmesinde yazılır. Piyasa verisi yoksa sentetik **teknik** test yapılır; gerçek OOS
  performansı varmış gibi sunulmaz.
- Train/validation/final kronolojik; walk-forward'da fit/update yalnız o ana kadarki veriyle
  (scaling, kuantil sınırı, hedge ratio, HMM parametreleri dahil). Örtüşen etiketlerde
  purge/gap/embargo veriye göre gerekçelenir; evrensel gün sayısı yok.
- Maliyetler yalnız ilgili ürüne, birimi ve giriş/çıkış politikası açık; bar başı maliyet ≠
  gerçekleşen turnover maliyeti. Karar anı, emir anı, fill fiyatı ayrı; OHLC içi sıra
  bilinmiyormuş gibi davranılır.
- Net getiri, drawdown, Sharpe/Sortino, turnover, işlem sayısı, exposure + belirsizlik;
  bağımlı seriye uygun resampling. Denenen hipotez/parametre sayısı kaydedilir; final holdout
  yalnız son değerlendirmede açılır, sonrası yeni deneydir.
- Paper kayıtları yeni veridir; backtest/sınav puanıyla tek başarı yüzdesine birleştirilmez.
  LLM/LoRA sürümü ile sinyal modeli/Markov matrisi sürümü ayrıdır. Ham P&L otomatik "doğru
  cevap" etiketi olmaz; otomatik kendini eğitme/terfi yok.

## 8. Aday kapısı

İlk pilot için örnek politika (doğruluk garantisi değil): sabit kritik regresyon örneklerinde
**sıfır kritik hata** · teknik benchmark'ta base'e göre gerileme yok · RAG'da uydurma atıf yok ·
istenen kod görevlerinin testleri geçer · tekrar/truncation kayıtlı. Toplam puan artıp kritik
hata sürüyorsa terfi yok. Teknik LLM terfisi **kârlı strateji terfisi değildir**; strateji
kapısı ayrı (dokunulmamış veri, baseline, maliyet/risk stresi, yeterli örneklem, paper gözlemi).
Her adapter candidate kaydolur, mevcut sürüm ezilmez, geri dönüş registry + config ile.

## 9. Teslim biçimi

Her modülde: ne değişti, neden, test sonucu, ölçülen gerileme/iyileşme, kalan belirsizlik.
Çıktılar mevcut yapıya: benchmark tanımı (`evals/llm30/`), deney manifesti (`RunManifest`),
ham JSONL + soru bazlı tablo (`reports/evals/`), RAG relevance ölçümü, dataset manifesti,
pilot adapter kartı, indikatör araştırma raporu. Aynı bilgi birden çok dosyaya kopyalanmaz.

## 10. Bu entegrasyonda yapılan / yapılmayan (2026-09-30)

**Yapıldı:** protokol dokümanı · LLM-30 seti + manifest · sızıntı kapısına bağlama ·
`evals/` tümüyle eğitim-girdisi yasağı · rubrik + kritik hata tanımları · sayısal anahtar
hesaplayıcısı + testler.

**Yapılmadı (sıradaki):** Aşama 1 kanıtlı denetim raporu · llm30 ile 2×2 koşusu · teşhis varyantları · final test takımı ·
rubrik puanlama aracı + kritik hata kapısının eval'e bağlanması · RAG relevance seti.
