# Sürekli (gözetimsiz) eğitim döngüsü — tasarım (2026-10-15)

Kod: `app/orchestration/continuous_loop.py` · CLI `hektor surekli-dongu …` · zamanlayıcı
`scripts/install-continuous-loop-task.ps1` (30 dk'da bir `tik`) · hakem
`app/evals/llm30_judge.py` · web: Öğrenme sekmesi → "Bilgi denetimi ve eğitim döngüsü" kartı.

## 1 · Kullanıcı kararları (2026-10-15)

| Konu | Karar |
|---|---|
| Eğitim onayı | **İnsan onayı olmadan** başlar (CLAUDE.md Kural 8 istisnası). Onay kaydı yine yazılır, `decided_by=policy:unattended_training` (insan kararı gibi görünmez). |
| Ölçüt | "Sorular ve doğru cevaplar": LLM-30 geliştirme seti, **yerel hakem** rubriği (`dogruluk` 0–4 …, kritik kodlar). |
| İnsan ne zaman | **Tüm sorular doğru** olunca döngü durur → insan: kilitli final takımı + terfi. Ayrıca üst üste **2** aday kapıdan geçemezse durur. |
| Kademe 2 | **Yalnız kod değişince** (`HEKTOR_KADEME2_CODE_ONLY=true`); veri değişince otomatik kalite + sızıntı + Kural 1 kapıları. |
| RAG makaleleri | Doğrulamalı araştırma (araştırma paketi / makale-arastirma deseni). Zayıf alaka filtreli `kaynak-tamamla` döngüye GİRMEZ (2026-10-15 denemesi ilgisiz "heat transfer" makalesini geçirdi). |
| Terfi | **Asla otomatik değil** (`policy_authorize` `promote` aksiyonunu reddeder). |

## 2 · Açma

`.env` (canlı kurulum):

```
HEKTOR_CONTINUOUS_LOOP_ENABLED=true
HEKTOR_UNATTENDED_TRAINING_ENABLED=true
HEKTOR_KADEME2_CODE_ONLY=true
```

Sonra `powershell -ExecutionPolicy Bypass -File scripts\install-continuous-loop-task.ps1`.
`HEKTOR_UNATTENDED_TRAINING_ENABLED=false` iken döngü ölçer/puanlar ama eğitim noktasında
`insan_onayi_gerekli` der. `storage\STOP_ALL` / `STOP_LEARNING` her şeyi durdurur.
Eski Auto-LoRA arka plan hattı artık `HEKTOR_AUTO_LORA_AUTO_ENABLED` ile ayrı açılır (iki hat
aynı anda eğitim başlatmasın).

## 3 · Tik (her çağrıda en çok bir yan etkili adım)

```
kapalı / STOP / duraklatıldı                    → hiçbir şey
döngünün arka plan işi (pid+oluşturma zamanı)    → bekle
eğitim / aday işi / ağır iş kilidi               → bekle
son aday (eğitim döngüsü defteri):
  bitiş yok                → training_loop.record_completion
  Ollama etiketi yok       → candidate_jobs dönüşüm (<ad>-aday, şablon base'den)
  referans LLM-30 yok      → llm30_run (B koşulu, tohumlar 42 43) — referans özeti değişince yeniden
  aday LLM-30 yok          → llm30_run (B)
  puansız                  → llm30_judge (yerel hakem; önce referans, sonra aday)
  puanlı                   → record_dev_eval + all_correct
     hepsi doğru           → DURAKLAT (insan)
     kapı geçti            → ardışık = 0
     kapı geçmedi          → ardışık +1 → sınırda DURAKLAT, değilse release_by_policy
yeni eğitim: easy_train.readiness() tamamen hazır → prepare_snapshot(sabit profil) →
             easy_train.launch(unattended=True)  (precheck AYNEN: veri, sohbet seçimi, kalite,
             sızıntı, kilit, RAM/uzunluk, Kademe 2, döngü kuralları, yük doktoru)
```

Aynı-veri kuralı değişmedi: yeni eğitim için **yeni doğrulanmış veri** gerekir. Döngü veri
üretmez; veri araştırma paketi / sentetik QA / bilgi kalemleri hattından gelir
(`scripts/assemble_sft.py` ile kanonik birleştirme). Veri aynıysa tik `yeni_veri_bekleniyor` der.

## 4 · Sabit reçete

`moe30b_attn_long_ma` = `moe30b_attn_long` + **merge-aware bf16** (dropout 0, NEFTune 0).
Neden: `hektor_lora_full_20261011` (standart LoRA) Ollama dönüşümünde birleştirme sadakati
kapısından geçemedi (fark/etki 0,17–0,20 > 0,1; üç istemde top-1 değişti) → eğitilen adapter
servis edilemedi, döngü ölçemezdi. LR 2e-4 merge-aware ile ÖLÇÜLMEDİ (v15 5e-5) — ilk koşu ölçüm.
Karışım `balanced_v1` (ayar: `HEKTOR_CONTINUOUS_LOOP_MIX_PROFILE`).

## 5 · Ölçüm ve sınırlar (Kural 2)

- Hakem varsayılanı ham `qwen3:30b-a3b-instruct-2507-q4_K_M` — ne aday ne referans (kendi
  cevabını kayırmasın). Bulut etiketi reddedilir. Şemaya uymayan hakem cevabı → satır puansız →
  kapı geçmez (sessiz varsayılan yok).
- LLM-30'da 30 sorunun 12'sinde yazılı referans cevap, 10'unda deterministik sayısal anahtar
  var; kalanında hakem yalnız rubrik + alan bilgisiyle puanlar → "doğru" o sorularda hakem
  kanaatidir, kanıt değildir. Final takımı (60 soru, taslak, kilitsiz) döngüye GİRMEZ —
  seçim için kullanılırsa final olmaktan çıkar; terfi adımında insan koşar.
- "Tüm sorular doğru" geliştirme setinde aşırı uyum riskini taşır (aynı set her turda ölçülür,
  sorular eğitime girmese de seçim baskısı oluşur). Bu yüzden durma noktası insandadır.
- Süre: tam havuz eğitimi bu makinede ~58 sa (2026-10-11 koşusu). Bir tur (eğitim + dönüşüm +
  2 LLM-30 + hakem) ≈ 3–4 gün → "sürekli" pratikte haftada 1–2 aday.
- Masaüstü kısıtı: ağır işler sıralı (tek 30B); döngü eğitim/dönüşüm/ağır iş kilidi ve kendi
  arka plan işi sürerken bekler.
