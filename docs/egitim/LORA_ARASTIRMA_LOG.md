# Achilles LoRA — Araştırma Logu & Tarama Defteri

Bu dosya **iki işlevi** birden görür:

1. **İnsan-okur araştırma logu:** Her araştırma turunda ne tarandı, ne bulundu, ne
   entegre edildi.
2. **Dedup defteri (machine-readable):** Tekrarlı tarama (günlük/haftalık) bu dosyaya
   karşı diff alır — "Kapsanan teknikler" ve "Kapsanan kaynaklar" listelerinde geçen
   bir teknik/arXiv-ID **yeniden derin-araştırılmaz** (boş tur gürültüsünü önler).

> Tekrarlı tarama tasarımı (kullanıcı kararı, 2026-06-17):
> - **Günlük hafif tarama:** arXiv/HF/Unsloth/PEFT'te *bu defterde olmayan* yeni öğe var mı?
>   Yoksa no-op (commit yok). Varsa → bu loga aday satırı + (doküman-değişikliğiyse) auto-push.
> - **Haftalık derin tur:** tam çok-ajanlı sweep + adversarial doğrulama + sentez + entegrasyon.
> - **Kapı:** kod/reçete değişikliği doğrudan `main`'e push EDİLMEZ → PR / inceleme chip'i
>   (doğrulanmamış reçete riski; CLAUDE.md Kural 2/8). Doküman-yalnız güncelleme auto-push olabilir.
> - **Mekanizma:** bulut zamanlı routine (bkz. `.claude/` scheduled routine).

---

## Kapsanan teknikler (dedup anahtarı — yeniden derin-araştırma YOK)

`rsLoRA` · `DoRA` · `LoRA+` · `PiSSA` · `OLoRA` · `EVA` · `LoftQ` · `CorDA` ·
`orthogonal-init` · `NEFTune` · `gaussian-init` · `QLoRA` · `train_on_responses_only` ·
`cosine-warmup` · `weight-decay` · `gradient-clipping` · `replay/rehearsal` ·
`n-gram-repetition-detection` · `assistant_only_loss` · `KL-regularized-SFT` (`kl_reg_beta`,
entegre edildi Tur 3) · `LoRA-GA` (native durum doğrulandı Tur 3 — entegre EDİLMEDİ, aşağıda) ·
`MiCA` (PEFT-native init, incelendi Tur 3 — entegre edilmedi) ·
`intruder-dimension-reduction` (`reduce_intruder_dimension`, PEFT 0.19.0 native, doğrulandı
Günlük 2026-07-22 — **eğitim-SONRASI** forgetting-onarım aracı, entegrasyon weekly-deep'e
havale edildi, aşağıda)

## Elenen adaylar (dedup anahtarı — yeniden derin-araştırma YOK)

`VeRA` · `MiLoRA` · `LoRA-FA` (param-azaltma, v5-ilgisiz) ·
`O-LoRA/orthogonal-subspace-CL` · `CLoRA` · `EWCLoRA` · `FIP` (continual-learning, native değil) ·
`DFT loss_type=dft` · `OPLoRA` · `SC-LoRA` · `AILoRA` · `D²LoRA` · `all-linear` (zaten hizalı) ·
`OFT/BEFT/Lily` (image-gen odaklı, disiplin/v5-ilgisiz) · `aLoRA/alora_invocation_tokens` ·
`MonteCLoRA` · `QALoRA` · `BDLoRA` · `VeLoRA` · `Arrow-routing` (2026-07-02 günlük; v5-ilgisiz) ·
`LoRA-GA` (Tur 3: PEFT-native OLDU ama quantize-desteksiz + residual dönüşüm + ayrı gradient-tahmin
ön-adımı gerektiriyor → karmaşıklık/fayda dengesi düşük, entegre edilmedi) ·
`MiCA` (Tur 3: instruction-tuned model üzerinde ÖNERİLMİYOR — continued-pretraining/base-model
odaklı; Achilles zaten instruction-tuned Qwen3 kullanıyor, uygun değil) ·
`approximate replay/corpus-karışımı` (arXiv 2512.22337'nin ikinci bileşeni — KL kısmı entegre
edildi ama replay/corpus-karışım kısmı ayrı veri hattı gerektirdiğinden bu turda ERTELENDİ) ·
`Full-FT forgetting-azaltma` (arXiv 2506.09428 — full fine-tuning, 70B'de test, kod yok, LoRA-ilgisiz)

## Kapsanan kaynaklar (arXiv ID / URL)

- arXiv 2312.03732 (rsLoRA) · arXiv 2402.09353 (DoRA) · arXiv 2310.05914 (NEFTune) ·
  arXiv 1904.09751 (neural text degeneration)
- unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune
- huggingface.co/docs/peft/main/en/developer_guides/lora (rsLoRA/DoRA/init/LoRA+)
- huggingface.co/docs/trl/sft_trainer (NEFTune/response-only/packing/assistant_only_loss)
- github.com/NVlabs/DoRA · github.com/neelsjain/NEFTune · github.com/yxli2123/LoftQ
- huggingface.co/docs/peft/main/en/developer_guides/lora (orthogonal/eva/gaussian init listesi)
- arXiv 2407.05000 (LoRA-GA) · github.com/huggingface/peft/issues/2927 (eski durum — Tur 3'te
  güncellendi: artık PEFT 0.19.1'de `init_lora_weights="lora_ga"` + `LoraGAConfig` NATIVE var,
  bkz. github.com/huggingface/peft/pull/2926)
- huggingface.co/docs/peft/main/package_reference/lora (`ensure_weight_tying` + yeni init/config alanları) ·
  huggingface.co/blog/peft-beyond-lora (2026-06-18 "Beyond LoRA": OFT/BEFT/Lily) ·
  unsloth.ai/docs/models/qwen3.5/fine-tune (Qwen3.5 + alpha=r default ablasyonu)
- **arXiv 2512.22337** (Riemer ve ark., IBM Research — "The Effectiveness of Approximate
  Regularized Replay for Efficient SFT of LLMs") — KL-regularization Tur 3'te ENTEGRE EDİLDİ
- huggingface.co/docs/peft/main/en/developer_guides/lora (MiCA init stratejisi — Tur 3 incelendi)
- github.com/BY571/sft-kl-lora-trainer (community örnek: TRL SFTTrainer + LoRA-base KL loss —
  arXiv 2512.22337 yaklaşımının pratik uygulanabilirliğini doğrulayan bağımsız referans)
- **arXiv 2410.21228** (Shuttleworth ve ark. — "LoRA vs Full Fine-tuning: An Illusion of
  Equivalence") — "intruder dimension" kavramının kaynağı; Günlük 2026-07-22'de bulundu
- github.com/huggingface/peft/releases/tag/v0.19.0 (9 yeni PEFT metodu + `reduce_intruder_dimension`
  duyurusu) · huggingface.co/docs/peft/main/en/developer_guides/lora (`peft.tuners.lora.intruders.
  reduce_intruder_dimension` API dokümantasyonu, kullanım örneği) — Günlük 2026-07-22

---

## Günlük tarama — 2026-07-22 (daily-light)

**Tetikleyici:** Otonom `lora-arastirma` ajanı (daily-light modu). `docs/PROTOKOL_LORA_ARASTIRMA.md`
§7 "daily-light" akışı doğrudan uygulandı. Not: bir önceki günlük tarama girdisi 2026-07-02
tarihli (Tur 3 2026-07-03'ten sonra ~19 gün boyunca günlük tarama loglanmamış) — bu turda yalnız
bugünkü tarama yapıldı, geriye dönük telafi denenmedi. CLAUDE.md Kural 2/7/8'e uyuldu; eğitim
BAŞLATILMADI.

**Yöntem:** Hafif tarama (tam sweep DEĞİL) — WebSearch: (a) arXiv LoRA/SFT catastrophic-forgetting
son gelişmeler, (b) PEFT GitHub release notları/changelog, (c) Unsloth blog 2026 güncellemeleri.
Öne çıkan aday, adversarial doğrulama için WebFetch ile birincil kaynaklara (GitHub release
sayfası, PEFT docs, arXiv paper sayfası) gidildi + **yerelde kurulu PEFT 0.19.1 üzerinde canlı
`import` testiyle** doğrulandı (`.venv\Scripts\python.exe -c "from peft.tuners.lora.intruders
import reduce_intruder_dimension; ..."` — başarılı, fonksiyon imzası ve docstring doğrulandı).

**Sonuç: 1 YENİ, doğrulanmış, PEFT-native, v5'e DOĞRUDAN ilgili aday bulundu → bu turda kod
entegrasyonu YAPILMADI (kapsam-dışı; weekly-deep'e havale), yalnız doküman/log güncellendi →
`main`'e push (kod dokunulmadı).**

### Öne çıkan aday: `reduce_intruder_dimension` (PEFT 0.19.0, eğitim-SONRASI onarım aracı)

- **Kaynak:** arXiv 2410.21228 (Shuttleworth, Andreas, Torralba, Sordoni — "LoRA vs Full
  Fine-tuning: An Illusion of Equivalence"), PEFT native uygulaması `v0.19.0`'da (14 Nisan 2026)
  eklendi, `v0.19.1`'de (yerelde kurulu sürüm) mevcut. `github.com/huggingface/peft/releases/tag/v0.19.0`
  + `huggingface.co/docs/peft/main/en/developer_guides/lora` ile WebFetch doğrulandı; ayrıca
  yerel `.venv` üzerinde canlı import ile teyit edildi (varsayım/tahmin değil).
- **Makale bulgusu:** LoRA ince-ayarı, base modelin önceden-eğitilmiş singular vektörleriyle
  düşük kosinüs-benzerliğine sahip "intruder dimension" adlı yeni tekil-vektörler yaratıyor; bu
  boyutların sayısı ile pretraining-dağılımındaki forgetting arasında güçlü korelasyon var
  (Spearman ρ=0.971). Bulgular LLaMA2-7B/LLaMA-7B/RoBERTa-base üzerinde + AdaLoRA/LoRA+/PiSSA/VeRA
  gibi çoklu LoRA varyantında genelliyor (yalnız vanilya-LoRA'ya özgü değil).
  Sequential/continual LoRA senaryolarında intruder boyutları birikip performansı hızla bozuyor.
- **Fonksiyon:** `peft.tuners.lora.intruders.reduce_intruder_dimension(peft_model,
  old_adapter_name="default", new_adapter_name="intruder_reduced", top_k=10,
  threshold_epsilon=0.5, mitigation_lambda=0.75, logging_sink=print)` — zaten eğitilmiş bir LoRA
  adaptörünü post-process eder; orijinal adaptörün YANINA yeni bir adaptör oluşturur (orijinali
  bozmaz, `set_adapter(old_adapter_name)` ile geri dönülebilir). `mitigation_lambda` görev-doğruluğu
  ↔ forgetting-azaltma arasında ödünleşim sağlar. **Yalnız LoRA destekleniyor** (karışık
  adaptör tipleri desteklenmiyor).
- **v5 bağlantısı — DİĞER kayıttaki tekniklerden NİTELİK FARKI:** Şimdiye kadar defterdeki tüm
  entegre/aday teknikler (NEFTune, KL-reg, assistant_only_loss vb.) **eğitim-ZAMANI** hiperparametreleri;
  yeniden eğitim gerektirir. `reduce_intruder_dimension` ise **eğitim-SONRASI** bir onarım aracı —
  teorik olarak zaten var olan v5'in degenere/disiplin-bozuk adaptörüne DOĞRUDAN uygulanıp
  yeniden-eğitim YAPMADAN forgetting'i azaltıp azaltmadığı test edilebilir. Bu, v5 regresyonuna
  (`memory/v5-adapter-regression`) potansiyel en hızlı/ucuz müdahale yolu olabilir.
- **GGUF-güvenlik ön-değerlendirmesi:** Fonksiyon yalnız LoRA delta-ağırlıkları üzerinde çalışıyor
  (base model ağırlıklarını değiştirmiyor, embed/lm_head'e dokunmuyor); çıktısı yine standart bir
  LoRA adaptörü — mimari/GGUF-dönüştürme sözleşmesini bozmamalı. **Bu ön-değerlendirmedir,
  adversarial doğrulama YAPILMADI** (Kural 2/7 gereği "GGUF-güvenli" iddiası weekly-deep'te
  gerçek adaptör üzerinde teyit edilmeden kesinleşmiş sayılmaz).
- **Bu turda ENTEGRE EDİLMEDİ (kapsam-dışı, weekly-deep'e havale edildi) — gerekçe:** (1)
  daily-light protokolü yalnız tarama+dedup öngörüyor, tam entegrasyon weekly-deep işi; (2) bu
  bir eğitim-öncesi reçete parametresi değil, ayrı bir CLI/script akışı (mevcut
  `PeftTrainConfig`/`build_lora_kwargs` builder desenine uymuyor — `adapter_eval.py` veya yeni bir
  `app/training/adapter_repair.py` gibi post-hoc bir giriş noktası gerektirir) — tasarım kararı
  ister; (3) gerçek v5 adaptörü üzerinde denenip `adapter_eval` ile ÖLÇÜLMEDEN "işe yarıyor"
  denemez (Kural 2).
- **Önerilen sıradaki adım (weekly-deep veya insan-gözetimli seans için not):** `app/training/
  adapter_eval.py`'a opt-in bir CLI/fonksiyon eklenip mevcut v5 adaptörüne (offline, eğitim
  BAŞLATMADAN) uygulanabilir; `adapter_eval` gate'iyle öncesi/sonrası karşılaştırılabilir. Bu,
  Kural 8'i ihlal ETMEZ (eğitim değil, mevcut adaptörün post-processing'i) ama yine de "test
  edilmeden daha iyi deme" (Kural 2) geçerli — sonuç ölçülmeden terfi önerilmeyecek.

**Diğer PEFT v0.19.0 yeni metodları (hafif not, derin değerlendirme YAPILMADI → weekly-deep'e
ertelendi):**

| Yeni yöntem (PEFT 0.19.0) | Durum |
|---|---|
| `GraLoRA` (Granular LoRA, blok-altbölümleme) | ⏭️ ERTELENDİ — ifade edici güç artışı, forgetting'e özgü değil |
| `PSOFT` (Principal Subspace Orthogonal FT) | ⏭️ ERTELENDİ — OFT ailesi, önceki "Beyond LoRA" değerlendirmesiyle benzer olabilir, teyit gerek |
| `Cartridges` (bağlam sıkıştırma prefix) | ❌ İLGİSİZ — RAG/uzun-bağlam odaklı, LoRA fine-tuning kalitesiyle ilgisiz |
| `PEANuT`, `TinyLoRA`, `AdaMSS`, `PVeRA` | ⏭️ ERTELENDİ — ilk bakışta v5-forgetting'e doğrudan bağlantı görünmüyor |
| `BD-LoRA` | ✅ ZATEN ELENDİ (`BDLoRA` dedup'ta mevcut) |
| `Lily` | 🟡 DURUM-DEĞİŞİKLİĞİ OLASI — önceden "image-gen odaklı" elenmişti (HF blog); PEFT 0.19.0
resmi açıklaması "Low-Rank Interconnected Adaptation across Layers" (genel LoRA varyantı) diyor,
image-gen sınırlaması görünmüyor → weekly-deep'te yeniden teyit edilmeli, bu turda dedup
değiştirilmedi |
| `LoRA-GA` init iyileştirmesi (v0.19.0 changelog notu) | ✅ ZATEN KAPSANIYOR — Tur 3'te native-durum zaten doğrulanmış/elenmiş |

**NOT (Kural 2):** Hiçbir reçete/kod değişikliği yapılmadı. `reduce_intruder_dimension` güçlü bir
adaydır ama gerçek adaptör üzerinde ölçülmeden "v5'i düzeltir" DENMEDİ — yalnız hipotez + kaynak.
**NOT (Kural 7):** Yalnız WebFetch ile doğrulanmış + yerel canlı import ile teyit edilmiş kaynak
loglandı; ikincil arama sonuçlarında görülen bazı düşük-güvenilirlikli/muhtemelen-üretilmiş
blog kaynakları (ör. "Microsoft LoRA+ dinamik rank" iddiası — brics-econ.org, thecodeforge.io,
spheron.network gibi SEO-içerik siteleri) **KULLANILMADI** — tutarsız/doğrulanamayan iddialar
(bilinen LoRA+ Microsoft ürünü değildir, orijinal kaynağı farklıdır) içerdiğinden reddedildi.

### Kaynaklar (Günlük tarama 2026-07-22)

| Teknik / Konu | Kaynak |
|---|---|
| `reduce_intruder_dimension` (ENTEGRASYON BEKLİYOR) | arXiv 2410.21228 · github.com/huggingface/peft/releases/tag/v0.19.0 · huggingface.co/docs/peft/main/en/developer_guides/lora · yerel `.venv` canlı import testi |
| PEFT v0.19.0 diğer 8 yeni yöntem (hafif tarandı) | github.com/huggingface/peft/releases/tag/v0.19.0 |
| Unsloth 2026 güncellemeleri (MoE/embedding/RL — v5-ilgisiz, not amaçlı) | unslothai.substack.com/p/unsloth-2026-update-faster-moe |

---

## Tur 3 — 2026-07-03 (derin tur — haftalık zamanlı görev)

**Tetikleyici:** Zamanlı görev `lora-arastirma-haftalik-derin` (weekly-deep). `lora-arastirma`
alt-ajan tipi bu ortamda yok → protokol (`docs/PROTOKOL_LORA_ARASTIRMA.md`, weekly-deep) doğrudan
uygulandı. CLAUDE.md Kural 2/7/8'e uyuldu; eğitim BAŞLATILMADI (yalnız yerel offline test/ruff/mypy).

**Yöntem:** Çok-açılı paralel WebSearch sweep — (a) LoRA-GA'nın PEFT-native durum-değişikliği
(2026-07-02'den devir), (b) yeni LoRA varyantları/forgetting makaleleri, (c) PEFT release
notları/yeni `init_lora_weights` seçenekleri, (d) Unsloth/Qwen3 güncel rehberi, (e) TRL SFTConfig
yeni regularizasyon, (f) SFT veri kalitesi/degenerasyon. Her aday adversarial süzgeçten geçti:
*gerçek mi? PEFT 0.19+/Unsloth native mi? Achilles'e uygun mu? GGUF-güvenli mi? v5'e yardım eder mi?*
Öne çıkan adaylar için tam-metin doğrulama yapıldı (arXiv PDF indirilip okundu; kurulu PEFT 0.19.1
üzerinde gerçek import testiyle native-durum ölçüldü).

**Sonuç: 1 YENİ, doğrulanmış, GGUF-güvenli, opt-in teknik ENTEGRE EDİLDİ (kod + config + test) →
PR açıldı.** Ayrıca 1 durum-değişikliği (LoRA-GA artık native) doğrulandı ama pratik kısıtlar
nedeniyle entegre edilmedi.

**Entegre edilen (kod):**

- **KL-regularized SFT (`kl_reg_beta`)** — Kaynak: arXiv:2512.22337 (Riemer ve ark., IBM
  Research, Aralık 2025), tam metin okunarak doğrulandı. Qwen2.5-Instruct (1.5B/3B/7B/14B)
  üzerinde LoRA SFT'nin standart haliyle bile CİDDİ catastrophic forgetting yarattığını
  (β=0 satırı: ortalama ↓F=15.4, model boyutu arttıkça forgetting de ARTIYOR — küçük
  modellerde bile ciddi), β=0.01 KL cezasının forgetting'i neredeyse tamamen ortadan
  kaldırdığını (hafif plastisite kaybıyla), β=0.001'in ise plastisiteyi TAM koruyup
  forgetting'i approximate-replay ile birlikte ortalama >7× azalttığını gösteriyor.
  **v5 bağlantısı:** v5 regresyonu (disiplin/refusal davranışının bozulması + degenerate
  tekrar) tam olarak bu makalenin ölçtüğü "catastrophic forgetting during LoRA SFT"
  fenomenidir. **LoRA-sinerjisi (makalenin ana bulgusu):** base-model referans forward-pass'i
  `model.disable_adapter()` context manager'ıyla AYNI ağırlıklar üzerinden alınır — ikinci
  model kopyası belleğe YÜKLENMEZ (yalnız ~1.5-2× hesaplama overhead'i, sıfır ek bellek).
  **PEFT/TRL native değil** (yalnız RLHF/DPO trainer'larında KL var, SFT'de yok) →
  `_KLRegTrainer` (`Trainer.compute_loss` override) ile uygulandı. **GGUF-güvenli:** yalnız
  eğitim-zamanı ek loss terimi; mimari/ağırlık şekli değişmez. **OPT-IN:** varsayılan
  `kl_reg_beta=0.0` → düz `Trainer`, davranış değişmez (`_make_trainer_cls` doğrular).
  Replay/açık-corpus karışım kısmı (makalenin ikinci bileşeni) bu turda kapsam dışı
  bırakıldı — ayrı veri hattı gerektirir.
- **Dosyalar:** `app/training/peft_lora_train.py` (`PeftTrainConfig.kl_reg_beta`,
  `_KLRegTrainer`, `_make_trainer_cls`, `recipe_summary` genişletmesi, `load_lora_profile`
  field_map), `configs/lora/lora_profiles.yaml` (yeni deneysel profil `discipline_safe_kl`,
  β=0.01), `tests/test_peft_lora_recipe.py` (6 yeni test: default-off, recipe_summary,
  `_make_trainer_cls` beta=0/pozitif, profil yükleme).
- **Doğrulama:** `ruff format`/`ruff check` temiz; `mypy app` temiz (228 dosya); tüm
  30 test `test_peft_lora_recipe.py` içinde geçti; tam paket (`pytest -m "not ollama and
  not slow"`) çalıştırıldı — yalnız ÖNCEDEN VAR OLAN 1 ilgisiz başarısızlık
  (`test_training_dry_run_empty_base_uses_brain_not_1p5b`, benim değişikliklerimden
  bağımsız olduğu `git stash` ile doğrulandı). **NOT (Kural 2):** `discipline_safe_kl`
  reçetesi HİPOTEZDİR — bulut eğitim koşusu + `adapter_eval` gate'i ile doğrulanmadan
  "daha iyi" denmez. Eğitim BAŞLATILMADI.

**Durum-değişikliği doğrulandı ama entegre EDİLMEDİ:**

- **LoRA-GA** (arXiv 2407.05000) — 2026-07-02 günlük taramasının bıraktığı bayrak
  doğrulandı: kurulu PEFT 0.19.1'de `from peft.tuners.lora.config import LoraGAConfig` ve
  `from peft import preprocess_loraga` GERÇEKTEN import edilebiliyor (canlı `python -c`
  testiyle doğrulandı) — artık native (PEFT PR #2926 ile eklenmiş). **Yine de entegre
  EDİLMEDİ:** (1) quantized modelleri DESTEKLEMİYOR, (2) base ağırlıkları init'te
  değiştiriyor → PiSSA/OLoRA gibi GGUF-öncesi residual dönüşüm ister, (3) ayrı bir
  gradient-tahmin ön-adımı gerektiriyor (`preprocess_loraga(model, config, train_step)` —
  N adet forward+backward geçişi, ayrı dataloader) — mevcut tek-adımlı `train()` akışına
  tek config-satırı eklemekten çok daha büyük bir iş akışı değişikliği gerektiriyor.
  Karmaşıklık/fayda dengesi bu turda düşük görüldü; "Elenen adaylar"a güncellenmiş
  gerekçeyle işlendi (dedup — tekrar araştırılmayacak, ama durumu artık DOĞRU).

**Değerlendirilip elenen (bu turda yeni):**

- **MiCA** (PEFT-native init stratejisi, `init_lora_weights="mica"`) — PiSSA'nın
  tamamlayıcısı (asıl yerine küçük singular komponentleri kullanır); dokümantasyon açıkça
  **continued/domain-adaptive pretraining** için önerildiğini ve **instruction/chat-tuned
  olmayan** base model kullanılmasını tavsiye ettiğini belirtiyor. Achilles zaten
  instruction-tuned Qwen3 üzerinde çalıştığından uygun değil.
- **"Improved SFT to Mitigate Catastrophic Forgetting"** (arXiv 2506.09428) — tam metin
  incelendi: full fine-tuning (LoRA değil), yalnız Llama-3-70B-Instruct'ta test edilmiş,
  kod/repo paylaşılmamış. Achilles'e (LoRA + 1.5B-4B) uygun değil.
- **"Mitigating Forgetting in Low Rank Adaptation"** (arXiv 2512.17720) — metodoloji/model
  boyutu/kod durumu PDF metadata'sından net çıkarılamadı (adversarial doğrulama başarısız —
  Kural 7: emin olunmayan kaynak entegre edilmez). Ele alındı, dedup'a işlenmedi (net
  doğrulanamadığı için yeniden bakılabilir — "elenen" değil "belirsiz" statüsünde bırakıldı).

**NOT (Kural 2):** `kl_reg_beta` kod+config+test olarak bağlandı ve ÇEVRIMDIŞI doğrulandı;
ancak bir bulut eğitim koşusu + `adapter_eval` gate'i ile fiilen "daha iyi" olduğu henüz
KANITLANMADI. Eğitim başlatılmadı (Kural 8) — gerçek eğitim bulutta, kullanıcı tarafından.

### Kaynaklar (Tur 3 — 2026-07-03)

| Teknik / Konu | Kaynak |
|---------------|--------|
| KL-regularized SFT + approximate replay (ENTEGRE EDİLDİ) | arXiv:2512.22337 (tam metin okundu) |
| LoRA-GA native durum doğrulama (PEFT 0.19.1 canlı import testi) | github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py · github.com/huggingface/peft/pull/2926 |
| MiCA init stratejisi (elendi — continued-pretraining odaklı) | huggingface.co/docs/peft/main/en/developer_guides/lora |
| Full-FT forgetting makalesi (elendi — LoRA/kod yok) | arXiv 2506.09428 |
| Community KL-LoRA örneği (bağımsız doğrulama referansı) | github.com/BY571/sft-kl-lora-trainer |

---

## Günlük tarama — 2026-07-02 (daily-light — zamanlı görev)

**Tetikleyici:** Zamanlı görev `lora-arastirma-gunluk-tarama` (daily-light). `lora-arastirma`
alt-ajan tipi bu ortamda YOK → protokol (`docs/PROTOKOL_LORA_ARASTIRMA.md`, daily-light) doğrudan
uygulandı. Hafif tarama (tam sweep YOK): arXiv (LoRA/PEFT/forgetting), HF PEFT docs/blog, Unsloth
Qwen3. CLAUDE.md Kural 2/7/8'e uyuldu; eğitim BAŞLATILMADI.

**Sonuç: YENİ, doğrulanmış, PEFT-native, v5-ilgili ENTEGRE edilebilir teknik BULUNAMADI → PR YOK.**
Yalnız dedup defteri güncellendi (bu doküman) + 1 durum-değişikliği bayrağı (LoRA-GA) weekly-deep'e
devredildi. Doküman-yalnız değişiklik → `main`'e push (kod/reçete dokunulmadı).

**Taranan ve elenen/ertelenen adaylar:**

| Aday / gözlem | Karar | Gerekçe |
|---------------|-------|---------|
| Forgetting makaleleri: arXiv 2603.09684 (survey), 2606.06920 (sub-1B math), 2503.02659 (init) | ⏭️ İLGİSİZ | Survey/analiz veya PEFT-native-DEĞİL yöntemler (LoRETTA tensör-ayrışım, WeGeFT). Entegre edilebilir aday değil. |
| EWCLoRA / FIP / Hierarchical (arXiv 2501.13669) | ✅ ZATEN ELENDİ | Dedup defterinde mevcut; yeniden derin-araştırılmadı. |
| Instruction-data karışımı %5–20 (forgetting azaltma) | ✅ ZATEN VAR | `replay/rehearsal` olarak kapsanan teknikte. |
| **"Beyond LoRA" (HF blog, 2026-06-18): OFT / BEFT / Lily** | ❌ ELE | PEFT-native ama blog açıkça **image-generation** odaklı (OFT "strictly dominates on image metrics"); disiplin/forgetting için üstünlük gösterilmemiş. v5-ilgisiz. |
| **PEFT `ensure_weight_tying=True`** (LoraConfig, native) | 🟡 FARKINDALIK | GERÇEK + Achilles GGUF-güvenlik tasarımıyla doğrudan ilgili: bağlı `embed_tokens`/`lm_head` katmanlarında adapter'ların da bağlı kalmasını garanti eder. Achilles bu katmanları `target_modules`'e KOYMADIĞINDAN şu an **no-op** → entegrasyon gerekmez. Ancak ileride embed/lm_head eğitilirse native mekanizma budur. Log'a farkındalık notu. |
| **PEFT `lora_ga_config`** (docs'ta görüldü) | ⚠️ DURUM-DEĞİŞİKLİĞİ | Defter LoRA-GA'yı "PEFT-native değil" (issue #2927) kaydetmişti; artık docs/API'de gradient-tahminli init callback'i görünüyor. Kesin merge PR'ı bu turda doğrulanamadı (Kural 7 → overclaim yok) → **weekly-deep'te native-durum yeniden doğrulanacak**. |
| Yeni native config alanları: `alora_invocation_tokens`, `monteclora_config`, `use_qalora`, `use_bdlora`, `velora_config`, `arrow_config` | ⏭️ ERTELENDİ | Daily-light kapsamında derin-değerlendirme YOK; "Elenen adaylar"a işlendi (çoğu param-verimlilik/routing, v5-forgetting-ilgisiz). Gerekirse weekly-deep bakar. |
| Unsloth 2026 ablasyonu: `alpha=r` "temiz varsayılan" | 🟡 NOT | Reçete önerisi; Kural 2 gereği bulut eğitim + `adapter_eval` gate'i olmadan "daha iyi" DENMEZ. Yalnız not; reçete değişmedi. |

**NOT (Kural 2):** Hiçbir reçete/kod değişikliği yapılmadı; yukarıdakiler hipotez/gözlem.
Bulut eğitim koşusu + `adapter_eval` gate'i ile doğrulanmadan "daha iyi" denmez.

### Kaynaklar (Günlük tarama 2026-07-02)

| Teknik / Konu | Kaynak |
|---------------|--------|
| PEFT `ensure_weight_tying` + yeni config alanları | <https://huggingface.co/docs/peft/main/package_reference/lora> |
| "Beyond LoRA" (OFT/BEFT/Lily) | <https://huggingface.co/blog/peft-beyond-lora> |
| Forgetting survey / sub-1B / init | arXiv 2603.09684 · arXiv 2606.06920 · arXiv 2503.02659 |
| Unsloth Qwen3.5 (alpha=r default) | <https://unsloth.ai/docs/models/qwen3.5/fine-tune> |

---

## Tur 2 — 2026-06-22 (derin tur — haftalık zamanlı görev)

**Tetikleyici:** Zamanlı görev `lora-arastirma-haftalik-derin` (weekly-deep). `lora-arastirma`
alt-ajan tipi bu ortamda yoktu → protokol (`docs/PROTOKOL_LORA_ARASTIRMA.md`, weekly-deep) doğrudan
uygulandı. CLAUDE.md Kural 2/7/8'e uyuldu; eğitim BAŞLATILMADI.

**Yöntem:** 7 açılı paralel WebSearch sweep — (a) yeni LoRA varyantları, (b) init yöntemleri,
(c) küçük-LLM catastrophic forgetting/refusal koruma, (d) SFT regularizasyon, (e) Unsloth/Qwen3
güncel, (f) SFT veri kalitesi, (g) LoRA-GA özel doğrulama. Her aday adversarial süzgeçten geçti:
*gerçek mi? PEFT 0.19+/Unsloth native mi? Achilles'e uygun mu? GGUF-güvenli mi? v5'e yardım eder mi?*

**Sonuç: YENİ, doğrulanmış, PEFT-native, v5-ilgili teknik BULUNAMADI → kod PR'ı YOK.**

**Adaylar ve eleme gerekçeleri:**

| Aday | Karar | Gerekçe |
|------|-------|---------|
| **LoRA-GA** (arXiv 2407.05000) | ❌ ELE | PEFT'e EKLENMEDİ — issue #2927 native merge olmadan kapandı; harici `Outsider565/LoRA-GA` reposu gerekir. Kural 7: native-değil entegre edilmez. |
| `orthogonal` init | ✅ ZATEN VAR | `peft_lora_train.py:_INIT_STRATEGIES` içinde mevcut (kod kapsıyor) ama dedup defterinde eksikti → log'a eklendi (doküman düzeltmesi). |
| `assistant_only_loss` (2026-06-22 günlük adayı) | ⚠️ GEREKSİZ | Bulut notebook şablonu (Hücre 10) ZATEN `train_on_responses_only` ile asistan-dışı turları maskeliyor — TRL `assistant_only_loss` ile işlevsel olarak AYNI. Ayrı entegrasyon mükerrer; veri-format dönüşümü riski de cabası. Günlük aday bu mevcut entegrasyonla **karşılanmış** sayılır. |
| VeRA / MiLoRA / LoRA-FA | ❌ ELE | Parametre-azaltma odaklı; v5 disiplin-gerilemesiyle ilgisiz (sorun param sayısı değil, forgetting/degenerasyon). |
| O-LoRA / CLoRA / EWCLoRA / FIP | ❌ ELE | Continual-learning; PEFT-native değil + çok-görev orthogonality makinesi gerektirir (tek-adapter Achilles akışına uymaz). CLoRA/EWCLoRA zaten 2026-06-22 günlükte elenmişti. |
| DFT `loss_type="dft"` (ICLR 2026) | ❌ ELE | 2026-06-22 günlükte elendi; yalnız math/reasoning'de test, 4B/disiplin doğrulanmadı. Tekrar araştırılmadı. |
| `target_modules="all-linear"` (Unsloth 2026) | ✅ ZATEN HİZALI | Achilles `TARGET_MODULES` zaten tüm linear projeksiyonları (q/k/v/o + gate/up/down) hedefliyor; embed/lm_head bilinçli dışarıda (Qwen3 tied-embeddings / GGUF güvenliği). Değişiklik yok. |

**Doküman/sürüm:** Anlamlı (entegre edilebilir) yeni bulgu olmadığından `LORA_EGITIM_DETAYLI_ANLATIM.md`
sürümü ARTIRILMADI; PDF yeniden üretilmedi (protokol §6: yalnız anlamlı bulguda). Yalnız bu defter
güncellendi (dedup listesine `orthogonal` + elenen-adaylar bloğu + kaynaklar).

**Takip (gözetimli seansa not):** `configs/lora/lora_profiles.yaml` yorum satırı (init seçenekleri)
`orthogonal`/`corda`'yı listelemiyor — saf yorum düzeltmesi ama reçete dosyası olduğu için Kural 5
gereği PR ister; bu otonom turda dokunulmadı.

### Kaynaklar (Tur 2 — 2026-06-22)

| Teknik / Konu | Kaynak |
|---------------|--------|
| LoRA-GA (PEFT-native değil — issue) | <https://github.com/huggingface/peft/issues/2927> · arXiv 2407.05000 |
| PEFT init listesi (orthogonal/eva/gaussian) | <https://huggingface.co/docs/peft/main/en/developer_guides/lora> |
| Unsloth Qwen3 2026 (all-linear öneri) | <https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune> |
| TRL SFT (assistant_only_loss ↔ response-only) | <https://huggingface.co/docs/trl/sft_trainer> |

> Kural 7 notu: VeRA/MiLoRA/LoRA-FA/O-LoRA/CLoRA/AILoRA/D²LoRA için atıf gören arXiv'ler var ama
> Achilles'e uygun + PEFT-native + v5-ilgili kriterini geçmediklerinden entegre edilmedi; yeniden
> derin-araştırma yapılmaması için "Elenen adaylar" defterine işlendi.

---

## Günlük tarama — 2026-06-22 (yönlü tarama, kullanıcı isteği)

**Tetikleyici:** Kullanıcı isteği — "v5 disiplin gerilemesine yönelik güncel SFT/LoRA tekniği
ara; kodu değiştirme, yalnız öner."

**Yöntem:** Çok açılı WebSearch (catastrophic forgetting / küçük model / disiplin koruma /
eval-aware / TRL/PEFT yeni özellikler) + adversarial doğrulama.

**Taranan ve elenen adaylar:**

| Aday | Eleme gerekçesi |
|------|-----------------|
| OPLoRA (arXiv 2510.13003) | PEFT native desteği yok; custom trainer gerekir |
| SC-LoRA (arXiv 2505.23724) | PEFT native desteği yok; research prototype |
| EWCLoRA / Hierarchical Layer-wise (arXiv 2501.13669) | Kod henüz yayımlanmadı |
| DFT `loss_type="dft"` (arXiv 2508.05629) | Yalnız math/reasoning'de test edilmiş; disiplin etkisi belirsiz; 4B doğrulanmamış |

**Onaylanan aday: `assistant_only_loss=True` (TRL SFTConfig)**

- **Kaynak:** TRL v1.6.0 resmi docs + sft_config.py main branch
  (https://huggingface.co/docs/trl/sft_trainer)
- **Ne yapar:** SFTConfig'e `assistant_only_loss=True` eklendiğinde kayıp yalnız asistan
  yanıtı tokenlarından hesaplanır; sistem mesajı ve kullanıcı turları maskelenir. Qwen3
  için TRL otomatik chat template patch uygular (jinja `{% generation %}` ekleme gerektirmez).
- **v5 bağlantısı:** v5 gerilemesinin kök sebebi sentetik-QA reçetesinin disiplin
  refusal tokenlerini bozuk loss hesabıyla (sistem turu dahil) ezmesiydi. `assistant_only_loss`
  sistem/kullanıcı turlarını maskeler → refusal/abstain davranışını öğrenirken gereksiz
  sistem-token sinyali karışmaz.
- **GGUF-güvenli:** Evet — yalnız eğitim-zamanı loss maskeleme; mimariyi/ağırlıkları değiştirmez.
- **PEFT uyumlu:** Evet — TRL SFTTrainer + SFTConfig; bulut notebook zaten SFTConfig kullanıyor.
- **Kısıt:** Bulut notebook'ta veri formatı `"text"` alan kullanıyor (dil modelleme modu);
  `assistant_only_loss` yalnız conversational (mesaj listesi) formatında çalışır. Veri
  dönüşümü veya `dataset_text_field` kaldırılması gerekir. Bu yüzden **OPT-IN** olarak
  bırakılmalı; varsayılan bozulmamalı.

**Entegrasyon önerisi (kod değiştirilmedi — PR gerektirir):**

- `app/training/peft_lora_train.py` → `PeftTrainConfig`'e `assistant_only_loss: bool = False`
  alanı ekle; `build_training_kwargs` içinde `SFTConfig`'e (veya `TrainingArguments` yerine
  geçirilecek `SFTConfig`'e) bu bayrağı geç.
- `app/training/cloud_notebook.py` + şablon → `build_stage2_notebook` parametresine
  `assistant_only_loss: bool = False` ekle; notebook şablonunda SFTConfig'e enjekte et;
  veri formatını conversational moda çevir (messages listesi).
- `configs/lora/lora_profiles.yaml` → `discipline_safe` profiline `assistant_only_loss: true`
  satırını ekle (v5 reçetesiyle doğrudan uyumlu).

**NOT (Kural 2):** Reçete hipotezdir. Bulut eğitim koşusu + `adapter_eval` gate'i ile
doğrulanana kadar "daha iyi" denmez.

### Kaynaklar (Günlük tarama 2026-06-22)

| Teknik / Konu | Kaynak |
|---------------|--------|
| assistant_only_loss / TRL SFTConfig | <https://huggingface.co/docs/trl/sft_trainer> |
| TRL sft_config.py (parametre doğrulama) | <https://github.com/huggingface/trl/blob/main/trl/trainer/sft_config.py> |

---

## Tur 1 — 2026-06-17 (derin tur)

**Tetikleyici:** Kullanıcı isteği — "LoRA eğitimini iyileştirmek için makaleleri araştır,
dokümanı güncelle, entegre et, push et."

**Yöntem:** Çok-ajanlı workflow (`lora-research-sweep`): 8 paralel web-tarama açısı →
adversarial doğrulama (her teknik gerçek mi? PEFT/Unsloth destekli mi? Achilles'e uygun mu?
GGUF-güvenli mi? v5 regresyonuna yardım eder mi?) → önceliklendirilmiş sentez.

**Entegre edildi (kod):**
- **rsLoRA / DoRA / `init_lora_weights` (PiSSA/OLoRA/EVA/LoftQ/CorDA)** — `PeftTrainConfig`
  alanları + `build_lora_kwargs()` saf builder; `LoraConfig(**build_lora_kwargs(cfg))`.
- **LoRA+** — `loraplus_lr_ratio>0` ise `create_loraplus_optimizer` Trainer'a bağlanır.
- **NEFTune** — `neftune_noise_alpha` → `TrainingArguments`/`SFTConfig`.
- **Regularizasyon** — yerel trainer'a warmup·cosine·weight_decay·grad-clip·seed eklendi
  (`build_training_kwargs()`); bulut reçetesiyle hizalandı.
- **`discipline_safe` profili** — v5 catastrophic-forgetting reçetesi (düşük lr + az epoch +
  NEFTune + yüksek dropout + grad-clip).
- **Bulut notebook parametrik** — alpha/dropout/rsLoRA/NEFTune/weight_decay/warmup placeholder.
- **Degenerasyon tespiti güçlendirildi** — `_max_ngram_repeat` (token-düzeyi döngü) + satır tekrarı.

**Dosyalar:** `app/training/peft_lora_train.py`, `app/training/cloud_notebook.py`,
`app/training/templates/stage2_lora_colab.ipynb`, `app/training/adapter_eval.py`,
`configs/lora/lora_profiles.yaml`, `app/main.py` (`--profile`). Testler:
`tests/test_peft_lora_recipe.py`, `tests/test_adapter_eval_degenerate.py`.

**Doğrulama:** ruff + mypy temiz; 162 + 33 yeni test geçti; bulut notebook üretimi
uçtan uca doğrulandı (geçerli JSON, placeholder yok). **NOT (Kural 2):** Reçete bir
*hipotez*tir — bir bulut eğitim koşusu + `adapter_eval` gate'i ile doğrulanana kadar
"daha iyi" denmez.

### Kaynaklar (Tur 1)

| Teknik / Konu | Kaynak |
|---------------|--------|
| LoRA hiperparametre rehberi | <https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide> |
| Qwen3 fine-tune | <https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune> |
| rsLoRA/DoRA/init/LoRA+ API | <https://huggingface.co/docs/peft/main/en/developer_guides/lora> |
| NEFTune/SFT | <https://huggingface.co/docs/trl/sft_trainer> |
| rsLoRA | arXiv 2312.03732 |
| DoRA | arXiv 2402.09353 · <https://github.com/NVlabs/DoRA> |
| NEFTune | arXiv 2310.05914 · <https://github.com/neelsjain/NEFTune> |
| LoftQ | <https://github.com/yxli2123/LoftQ> |
| Nöral metin degenerasyonu | arXiv 1904.09751 |

> Not: Çok-ajanlı sweep bazı ek aday makaleler (örn. 2024-2025 catastrophic-forgetting
> çalışmaları) getirdi; bağlantısı/atfı kesin doğrulanmayanlar bu tabloya alınmadı (Kural 7).
