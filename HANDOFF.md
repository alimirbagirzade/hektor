# HANDOFF — Hektor

_Depo: https://github.com/alimirbagirzade/hektor · Son güncelleme: 2026-09-30 (**varsayılan artık Qwen3-30B-A3B-Instruct-2507** + `moe30b_attn_local`; 4B yalnız düşük-RAM seçeneği · v13 eğitimi TAMAMLANDI, candidate · LLM-30 2×2 koşusu · RAG sorgu çevirisi deneyi P@6 0.14 → 0.71 · 4B dönemi kayıtları `docs/arsiv/4b_donemi/HANDOFF_2026-09_4B_donemi.md`'de)_


Yerel-öncelikli AI **trading araştırma** sistemi (Windows · macOS Apple Silicon · Linux).
**Canlı bot değil, yatırım tavsiyesi değil.**

---

## Bu depo nedir?

`alimirbagirzade/achilles` (v1) deposunun **temizlenmiş ve onarılmış** hâlidir. v1'in tüm
çalışan sistemi taşındı; ölü kod, bulut-API kalıntıları ve bayat oturum geçmişi taşınmadı.
Ne çıkarıldığı ve neden: **[docs/MIGRASYON_2.0.md](docs/MIGRASYON_2.0.md)**.

v1 geçmişi arşiv olarak eski depoda durur; bu depo tek "initial commit" ile başlar.

**2026-09-04 — proje `achilles2.0` → `hektor` olarak yeniden adlandırıldı.** CLI `hektor`
/ `hektor-web`, ortam öneki `HEKTOR_`. Mevcut kurulumlar bozulmasın diye iki geriye dönük
uyum kancası korunur (bkz. `app/config/settings.py`, `tests/test_legacy_env_migration.py`):

- Eski `ACHILLES_*` ortam değişkenleri ve `.env` satırları hâlâ okunur (uyarı loglar);
  açık `HEKTOR_*` ayarı her zaman kazanır. Bu destek **geçicidir**.
- Yalnız eski `storage/sqlite/achilles_trader_ai.db` varsa ona düşülür — korpus/kart
  geçmişi öksüz kalmaz. Dosyayı (WAL/SHM ile birlikte) yeniden adlandırmak yeterlidir.

Bilinçli olarak **değişmeyen** dış sözleşme: `.achpkg` uzantısı, JSON'daki
`achilles_package_version` anahtarı ve `source: achilles_research` değeri — bunları
Entropia tarafı okur, kırılmasınlar diye korundu.

---

## Durum

| Alan | Durum |
|---|---|
| Kapı (`make ci`) | **Yerel (Windows, 2026-09-30):** ✅ ruff format/check + mypy (263 dosya) + pytest **2817 passed, 4 deselected** (`-m "not ollama"`). CI (Linux) main'de; bu dal (`claude/lora-30b-a3b-prep`) henüz main'e birleşmedi. **Yerel ✅ tek başına kapı sayılmaz.** |
| Varsayılan model | **LLM:** `qwen3:30b-a3b-instruct-2507-q4_K_M` · **PEFT base:** `Qwen/Qwen3-30B-A3B-Instruct-2507` · **profil:** `moe30b_attn_local` (`app.config.DEFAULT_TRAIN_PROFILE`; attention-only, maskeli, bf16 ~61 GB RAM). Düşük RAM: `qwen3:4b-instruct-2507-q4_K_M` + `Qwen/Qwen3-4B-Instruct-2507` + `discipline_safe_local`. Çıplak `qwen3:30b`/`qwen3:4b` = Thinking-2507 (yavaş). **Bu makinenin `.env`'i: `HEKTOR_LLM_MODEL=hektor-v12-30b`** (LoRA'lı; arka plan döngüleri bu yüzden KAPALI). |
| Eğitim yığını | `train-cpu` extra'sı kilitte **sabit**: torch 2.14.0 · transformers 5.16.1 · tokenizers 0.23.2 · peft 0.20.0 · accelerate 1.14.0. Yükseltmek açık karardır → ardından adapter yeniden değerlendirilmeli |
| Son adapter | **`hektor_lora_v13_30b`** — TAMAMLANDI 2026-09-30 11:44 (210 optimizer adımı, GA 8, val_loss en iyi 0.5769 @175, son adapter = checkpoint-175). Kayıt `adapter_ee06d5a4dfa8` **candidate**, terfi YOK. Ollama: `hektor-v13-30b` (Q4_K_M + attn q8_0). Önceki: v12 (candidate; Ollama şablonunda CR — D1), v11 (candidate). |
| LLM | Yalnız yerel Ollama. Bulut API istemcisi YOK. |
| Gözetimsiz eğitim | **KAPALI** (`unattended_training_enabled=false`) → her gerçek eğitim tek-kullanımlık insan onayı ister (Kural 8) |
| Arka plan döngüleri | Web açılışında çalışır; `HEKTOR_BACKGROUND_LOOPS_ENABLED=false` ile kapatılır. **Bu makinede kapalı** (LLM ayarı LoRA'lı model — açmadan önce base'e döndür). |
| RAG | Hibrit (dense + BM25 + sezgisel rerank). Sorgu çevirisi / amaç filtresi **varsayılan KAPALI** — canlıya alma kararı bekliyor (bkz. 2026-09-30 (5)). |
| Test izolasyonu | Testler gerçek `data/` · `storage/` ağacına **yazamaz**; ihlal ederse paket FAIL verir |

---

## Yeni seansta ilk 5 dakika

```bash
uv sync --extra dev            # bağımlılıklar (pytest/ruff/mypy 'dev' extra'sındadır)
uv run hektor status         # Ollama + korpus + model durumu
uv run hektor doctor         # bu makine origin/main'de mi (salt-okuma teşhis)
make ci                        # format + lint + typecheck + test
uv run hektor-web            # http://127.0.0.1:8765
```

Ollama kapalıysa: `ollama serve` → `ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M` → `ollama pull nomic-embed-text`.

---

## Sekiz mutlak kural (CLAUDE.md)

1. Yatırım tavsiyesi üretme — çıktı daima hipotez + test noktası.
2. Test edilmeden "başarılı" deme — backtest + out-of-sample şart.
3. Maliyetleri yok sayma — komisyon + slippage her backtest'te.
4. Look-ahead yasak — pozisyon `shift(1)` ile gecikmeli.
5. `eval`/`exec` yok — strateji kuralları yalnız güvenli regex ile.
6. Determinizm — rastgelelik daima `seed` ile.
7. Kaynak uydurma — retrieval boşsa açıkça söyle.
8. Otomatik ağır eğitim yok — `train` varsayılan dry-run; gerçek eğitim `--run` + taze insan onayı.

---

## Sıradaki adım

1. **LLM-30 2×2 sonucu** → rubrikle puanla (`app.evals.llm30.RUBRIC`, kritik hatalar ayrı);
   protokol B = base+LoRA, C = base+RAG (bkz. `docs/PROTOKOL_LORA_RAG_IYILESTIRME.md` §2).
2. **RAG sorgu çevirisi** canlıya alınsın mı (`HEKTOR_RAG_QUERY_TRANSLATE=en`) — önce insan
   etiket örneklemesi (`evals/rag_relevance/llm30_pooled_v1.jsonl`, `human_verified=false`).
3. Yeni eğitimden ÖNCE: Kademe 2 derin av (zorunlu) + karışım ağırlığı sorusu.

Eğitim akışı (veri `data/`, `storage/`, `models/` git'te izlenmez):

```bash
uv run hektor ingest && uv run hektor synth-qa-bulk --target 1000
uv run python scripts/assemble_sft.py      # → data/lora_sft/lora_sft.jsonl (KANONİK)
uv run hektor lora-audit && uv run hektor pretrain-gate && uv run hektor lora-split
uv run hektor approval-approve <id>         # Kural 8
.\scripts\start-train.ps1                    # DETACHED; varsayılan moe30b_attn_local + bf16
```

`start-train.ps1` `pretrain-gate` + `lora-audit`'i zorunlu koşar (NO-GO → eğitim başlamaz;
bilinçli override `-SkipGate`). Eğitim sonrası: `lora-eval` → adapter **ADAY**; production
terfisi ayrı insan onayı ister. Servis yolu: `scripts/adapter_to_ollama.ps1` (merge → GGUF →
Ollama; 2×2 için aynı tarifle base: `-BaseRepo`).

---

## Son seans — 2026-09-30 (5): v13 sonucu + GGUF + LLM-30 2×2 (KOŞUYOR) + RAG deneyi

**v13 eğitimi TAMAMLANDI** (210/210 optimizer adımı, 30.09 11:44; kayıt `adapter_ee06d5a4dfa8`
**candidate**, terfi YOK). 07:43'te checkpoint-125'ten sürdürüldü (`recovery_attempts: 1`;
kesinti nedeni eski log üzerine yazıldığı için bilinmiyor). val_loss 0.733 → **0.5769**
(adım 175, en iyi) → 0.578; 125'ten sonra düz. Son adapter = checkpoint-175 (sha eşleşti,
`load_best` çalıştı). Train loss son 25 adım ort. 0.82 (dropout açık) — overfit işareti yok.

**GGUF (eşit tarif):** `hektor-v13-30b` (Q4_K_M + attn q8_0, sha `d8cf8964…`) ve aynı tarifle
**`hektor-base-30b-q4a8`** (`adapter_to_ollama.ps1 -BaseRepo`, sha `3e687b34…`) — 2×2'de base
ile LoRA'nın nicemlemesi/şablonu birebir aynı olsun diye. v13 birleştirme kapısı KL 0.0201
> 0.01 ile durdu (diğer ölçütler geçti); **kullanıcı kararıyla** bu koşu için `-MaxKL 0.025`
(`merge_info.json`: `kl_gate_override=true`). Varsayılan 0.01 değişmedi.

**LLM-30 2×2** (`app/evals/llm30_run.py`, run `llm30_v13_2x2_20260930`, 30 soru × A/B/C/D × 3
tekrar, temperature 0, num_predict 4096, num_ctx 16384): **KOŞUYOR** (22:51'de 114/360,
~35 sn/cevap). İlk koşu 108'de Ollama "token repeat limit" iptalinde durdu → koşucu artık
iptali kısmi ham cevapla `iptal:tekrar_limiti` olarak kaydediyor. Bulgular:
- **Ollama bu kurulumda deterministik DEĞİL:** aynı model+seed+temp 0'da cevaplar 149. karakterde
  (RAG'lı ilk token'da) ayrışıyor; tekrar 2-3 genelde aynı (önek önbelleği), tekrar 1 farklı.
  Tekrarlar bağımsız örneklem değil, önbellek-durumu ölçüsü → tek cevaptan sonuç çıkarma.
- `uzun_tekrar` bayrağı ilk sürümde YANLIŞ POZİTİFTİ (istenen alt-madde başlıkları 3× geçiyor);
  düzeltildi, `summary.json` bayrakları ham cevaptan yeniden hesaplar.
- Canlı `hektor-v12-30b` şablonu base'den FARKLI (D1: CR'ler; yeniden oluşturulmadı) →
  koşucu v12 ile başlamayı reddediyor.

**RAG deneyi (sorgu çevirisi × amaç filtresi)** — kök neden ölçüldü: Türkçe sorgu, ~%89
İngilizce korpusta tek Türkçe metin olan iki proje kılavuzuna çekiliyor (BM25 top-24'ün %82'si).
Yeni: `configs/rag/doc_purposes.yaml` + `HEKTOR_RAG_EXCLUDE_PURPOSES`, `HEKTOR_RAG_QUERY_TRANSLATE`
(off|en|bilingual; çeviri modeli ayrı, `hektor-*` reddedilir) — **ikisi de varsayılan KAPALI**.
Havuzlu kör etiket (407 çift, `evals/rag_relevance/llm30_pooled_v1.jsonl`, Claude etiketledi,
`human_verified=false`), `reports/evals/llm30/rag_exp_20260930T193022/score.json`:

| | P@6 | nDCG@6 | havuz-recall@6 | ilgilisiz soru | iç-doküman |
|---|---|---|---|---|---|
| V0 mevcut | 0.139 | 0.102 | 0.111 | 23/30 | 53/180 |
| V1 filtre | 0.167 | 0.115 | 0.138 | 21/30 | 0 |
| **V2 çeviri (en)** | **0.711** | **0.771** | **0.849** | **1/30** | 0 |
| V3 çeviri+filtre | 0.711 | 0.771 | 0.849 | 1/30 | 0 |
| V4 iki dil+filtre | 0.689 | 0.736 | 0.820 | 1/30 | 0 |

Sınırlar: tek hakem (Claude, kör ama insan değil); recall havuza göreli (gerçek recall'un üst
sınırı); aynı 30 geliştirme sorusu. Korpusta veri-doğrulama konuları (S01/S02/S05) zayıf.
Ek bulgular: 2 doküman OCR/kodlama çöpü parça üretiyor ("(4))" başlıklı ve başlıksız biri);
4/300 başlık yayıncı kalıbı; retrieval tekrarı iç-doküman sayısında 53↔57↔62 oynadı.
**Karar kullanıcıda:** `HEKTOR_RAG_QUERY_TRANSLATE=en` canlıya alınsın mı (çeviri LLM çağrısı
ekler; önbellekli). Önce insan etiket örneklemesi önerilir. 2×2 bağlamı V0 ile donduruldu
(RAG'lı koşullar mevcut hattı ölçüyor).

---

## Son seans — 2026-09-30 (4): LoRA/RAG iyileştirme protokolü + LLM-30 benchmark (eğitim YOK)

Kullanıcının `Claude_Hektor_LoRA_RAG_Iyilestirme_Promptu.txt` + `Hektor_LLM_30_Soru.txt`
dosyaları repoya bağlandı → **`docs/PROTOKOL_LORA_RAG_IYILESTIRME.md`** (8 aşama → mevcut
modül eşlemesi + açık işler). ⚠ Protokolün 2×2 harfleri `hektor mix eval`'den farklı
(protokol B = base+LoRA = bizim `C_*`; protokol C = base+RAG = bizim `B_base_rag`).
- `evals/llm30/validation.jsonl` (30 soru, `EvalItem`) + sha256 manifest; `app/evals/llm30.py`
  (yükleyici, `RUBRIC`, 13 `CRITICAL_ERRORS` + soru eşlemesi, `numeric_keys()`).
  **Geliştirme setidir, final test DEĞİL**; `human_verified=false`.
- Sızıntı kapısı (`run_leakage_check` → `train --run`) llm30'u da tarıyor; mevcut
  train (1680) + valid (111) ↔ llm30 **0 isabet** → v13 sonrası eğitimi bloklamaz.
  `assert_not_eval_path` artık tüm `evals/`i kapsar.
- v13 koşusuna dokunulmadı (20:31'de python süreci yoktu — bitiş/sonuç bu seansta
  DENETLENMEDİ); llm30 2×2 koşusu, final test takımı, rubrik puanlama aracı ve
  Aşama 1 denetim raporu **yapılmadı** (v13 bitince, Ollama boşken).

---

## Son seans — 2026-09-30 (3): Kademe 2 + veri zenginleştirme + **v13 EĞİTİMİ KOŞUYOR**

**Koşu:** `hektor_lora_v13_30b` — başladı 02:14 (`start-train.ps1`, onay `apr_40a3e32f4aac`
kullanıcı talimatıyla, mix `trading_analysis_v1` kullanıcı seçimi). Qwen3-30B-A3B, profil
`moe30b_attn_local` (**GA 8, lr 2e-4, held-out eval her 25 adımda 64 valid, load_best**),
1676 mikro-adım = **210 optimizer adımı**, ~160 sn/adım → **~10–10.5 sa, bitiş ≈12:45**.
İlk adım doğrulandı. Eğitim sürerken Ollama/web LoRA sohbeti/eval AÇMA (RAM ~128 GB).
Bitince: `run_complete.json`'da `best_checkpoint`/`best_eval_loss`; `reports/training/
hektor_lora_v13_30b_loss.json`'da artık `val_loss` dolu.

**Veri (v13):** `hektor synth-enrich` (yeni) kısa sentetik cevapları aynı bağlamdan 2-4
cümleye genişletti: 726 adaydan **443** tüm kapılardan geçti (dil, kaynak-atfı, etiket,
tekrar, tavsiye, cümle-düzeyi grounding, token bütçesi); kalanlar orijinal. Asistan medyanı
214→**272** kr; <200 kr sentetik 726→283. Yedek: `storage/synthetic_qa.bak-20260930-020921.jsonl`.
Claude örneklem incelemesi (8 satır): 5 sadık, 3'te hafif desteksiz cümle (~¼) — insan
denetimi yapılmadı. + `survivorship` disiplin tuzağı (35 satır; veride hiç yoktu, v12
"look-ahead" diyordu). pretrain-gate **GO**, lora-audit 292/292.
⚠ İlk zenginleştirme koşusu **v12 adapter'ıyla** üretilmişti (başka oturum 01:05'te `.env`
HEKTOR_LLM_MODEL=hektor-v12-30b yaptı) → çıktı ATILDI; artık `--model` + hektor-* reddi.

**Kademe 2 (v13 öncesi, 4 bulucu; kullanım sınırı yüzünden 2-oylu ajan doğrulaması yerine
kritikler elle yeniden koşturularak doğrulandı):**
- A (eğitici): engelleyici yok. A1 CLI `--iterations` varsayılanı 0 + planı aşan ret; A7 dolu
  adapter klasörü ret; A5 web kayıp grafiği (`step`) + ilerleme `val_loss`. `start-train.ps1`
  artık planı `ensure_train_split`'ten alır (lora-split 1702 ↔ 1680 farkı → 22 örnek 2. kez).
- B (eval): **B1 kritik — `max_new_tokens` 220'de base cevaplarının 80/80'i kesiliyordu →
  v10/v11/v12 "accept" kararları GÜVENİLİR DEĞİL**; artık 1024 + `truncated` + >%20 kesikse
  accept yok. B3 (1e2d5af gevşemesi gerçek ihlali kaçırıyordu) + B2/B4 contains: düzeltildi;
  B5 garanti vetosu; B7 dejenere eşiği uzunlukla. AÇIK: B6 (terfi PEFT mi GGUF mi — eval
  servis edilen GGUF'u ölçmüyor), B8 (disiplin/kalite bayrakları ayrı), B9 (rag_model_run
  parametre kaydı), B10 (auto_pipeline fp32 30B).
- C (veri): C1-C5, C6 (sızıntı kapısı artık evals/*.jsonl 80 kalem), C9, C10 düzeltildi.
- D (merge/GGUF): D1 **canlı `hektor-v12-30b` şablonunda 47 CR** (Qwen'de \r\n ≠ \n →
  eğitimde görülmemiş istem; v12'nin TÜM Ollama ölçümleri bununla) — betik düzeltildi
  (LF + create sonrası CR denetimi); D2 attention q8_0; D3 no-op adapter kapısı; D4 .partial
  + köken; D5 ad çakışması -Force. **Canlı model YENİDEN OLUŞTURULMADI** (otomatik izin
  denetçisi reddetti) → kullanıcı: `tr -d '\r'` ile Modelfile'ı düzeltip
  `ollama create hektor-v12-30b -f models/gguf/Modelfile.hektor-v12-30b`.

**v13 sonrası sıra:** (1) eval (yeni B1 ayarlarıyla, ~daha uzun) · (2) B6 kararı: terfi
edilecek yapıt GGUF ise eval Ollama üzerinden de koşulmalı · (3) merge → GGUF (attn q8_0) →
Ollama (`adapter_to_ollama.ps1`, yeni ad `hektor-v13-30b`) · ADAY; terfi insan kararı.

**Kapı:** ruff + mypy 257 + pytest **2794 passed** (Windows). Commit'ler: 1f1faaf, 01def42 + bu.

---

## Son seans — 2026-09-30 (2): v11/v12 eval + `hektor-v12-30b` Ollama'da + RAG testi

**Eval** (`logs/lora-eval-v12-v11.ps1` → `logs/lora-eval-adapter.ps1`; CPU, bf16, 80 soru,
~1 sa 35 dk/adapter; base puanları iki koşuda BİREBİR aynı → ölçüm tekrarlanabilir):

| Set | n | base 30B | v11 (600) | **v12 (tüm havuz)** | v10 4B (ref.) |
|---|---|---|---|---|---|
| discipline_core | 16 | +0.19 (13) | +0.38 (10) reject* | **+0.69 (5) reject*** | +0.94 (1) |
| risk_management | 12 | −0.17 (14) | +0.00 (12) reject* | **+0.33 (8)** | +0.67 (4) |
| overfit_awareness | 12 | +0.25 (9) | +0.67 (4) | **+0.67 (4)** | +0.83 (2) |
| format_compliance | 12 | −1.08 (25) | −0.75 (21) | **−0.33 (16)** | −0.25 (15) |
| trader_persona | 16 | −3.19 (67) | −2.06 (49) reject** | **−1.25 (36)** | −1.56 (41) |
| rag_integration | 12 | −1.17 (26) | −1.00 (24) | **−0.92 (23)** | −0.83 (22) |
| ağırlıklı ort. | 80 | −0.93 | −0.50 | **−0.15** | — |

\* `degenerate` vetosu (gerçek: cümle döngüsü). \*\* v11 `guaranteed_profit` vetosu YANLIŞ
POZİTİF ("%5 kazanç garantisi olmaz" reddi). v12: 1 CJK sızıntısı ("期权"), ort. cevap ~250
krk. Kayıt: v11 `adapter_6bd64fb2669e`, v12 `adapter_d012a940f94d` — ikisi de **candidate**,
terfi YOK.

**Ollama:** `scripts/adapter_to_ollama.ps1` (yeni; C9 kapandı) + `scripts/merge_adapter.py`
(yeni) → `hektor-v12-30b` (Q4_K_M 18.6 GB, sha256 `d91a25d231bbc660…`, şablon
`qwen3:30b-a3b-instruct-2507-q4_K_M`'den). Birleştirme doğrulaması: |fark|max 0.42 / adapter
etkisi 13.75 (%3), KL 0.0016, top-10 9/10. İlk sürümün mutlak 0.25 eşiği yanlış alarm
verdi → ölçüt adapter etkisine göre oran + KL + top-10 yapıldı. GPU'da ~90 tok/s. Diskte
`models/merged/hektor_lora_v12_30b` (61 GB) + `…-bf16.gguf` (61 GB) duruyor — silinebilir.

**RAG testi** (`reports/evals/rag_test_v12_vs_base.json`, 6 soru, gerçek RAG hattı):
atıflı cevap base 6/6 · **v12 5/6** (v10 0/30 idi) · ort. uzunluk 2.800 vs **~1.000 krk**
(4 soruda 1–2 cümle) · format bölümleri 6/6 vs 2/6 · Kelly sorusunda retrieval boşken v12
atıfsız kendi bilgisiyle cevapladı (Kural 7). İlk karar: web RAG base kalır.

**Sonra (2026-09-30 ~01:05, kullanıcı kararı): v12 TAM ana model yapıldı.**
- `repeat_penalty` 1 → **1.1** (`adapter_to_ollama.ps1 -RepeatPenalty`, eski Modelfile
  `…rp1.bak`). Uygulama `repeat_penalty` göndermez → Modelfile geçerli. RAG testinde ETKİSİZ
  (`rag_test_v12_rp11.json`: atıflı 5/6, ort. ~900 krk, S1 cümle tekrarı yine %33) → kısa
  cevap/tekrar eğitim verisinden, örnekleme ayarından değil.
- `.env`: `HEKTOR_LLM_MODEL=hektor-v12-30b` (yedek `storage/env.bak-20260930-v12web`;
  geri dönüş: `qwen3:30b-a3b-instruct-2507-q4_K_M`). `HEKTOR_LLM_MODEL` TÜM LLM işlerini
  belirler (kart, sentetik QA, mastery, RAG) → **arka plan döngüleri KAPALI** tutuldu ki
  eğitim verisi v12 ile üretilmesin (kendini besleyen kalite kaybı). Döngüleri açmadan ya da
  `synth-qa`/`card` CLI koşmadan ÖNCE modeli base'e döndür (ya da ayrı chat-model ayarı ekle).
- Web uçtan uca ✅: `/api/ask` 6 kaynak, v12 Kelly sorusunda bu kez "kaynakta yok" diye
  çekimser kaldı. Not: arka plan döngüleri base'i kullanırken RAG testi 70–117 sn/soru
  sürdü (20 GB GPU'da iki 30B takası — 2026-09-27 olayıyla aynı).

---

## Son seans — 2026-09-30: v13 reçetesi — gradient accumulation + held-out eval (eğitim YOK)

**Neden (v12 ölçümü):** batch 1, birikim yok → kayıp ~100. adımdan sonra düz + gürültülü
(0.28↔1.46, grad_norm ≤7.6); `eval_strategy: no` → validation loss yok, en iyi checkpoint
seçilemiyordu. v12 eval'i (başka oturum): 5/6 accept, **discipline_core REJECT —
`degenerate` vetosu gerçek** (survivorship cevabı cümle döngüsüne giriyor).

**Değişiklik (`peft_lora_train.py`, profil `moe30b_attn_local`):**
- `gradient_accumulation_steps` (profil: 8) · `eval_every_examples` (200) ·
  `eval_max_examples` (64, `valid.jsonl`'den seed'li) · `load_best_model_at_end` (true).
  Profilde lr 1e-4 → **2e-4** (efektif batch 8).
- `iterations` üst katmanlarda (web/start-train/nöbetçi/`plan_iterations`) **mikro-adım**
  olarak KALIR; optimizer adımına çeviri yalnız `train()`'de (`optimizer_steps` = ceil,
  HF'nin kısmi-birikim sayımıyla aynı → 1678 örnek / 8 = 210 adım = 1 epoch).
  `run_plan.json` fiili optimizer hedefini + `micro_steps` yazar → kurtarma kıyası doğru.
- save/log aralığı örnek cinsinden sabit (25/5 örnek); load_best açıkken checkpoint = eval
  aralığı (HF şartı) → çökmede en çok ~200 örnek (~1 sa) kayıp.
- Eval tqdm çubuğu kapatıldı: web/nöbetçi ilerlemeyi log'daki SON `x/y [..<..]`
  satırından okuyor; eval çubuğu (`64/64`) "koşu bitti" sandırırdı.
- `run_complete.json` artık `best_checkpoint` + `best_eval_loss` yazar; loss grafiğinde
  `val_loss` (eskiden HF'nin ayrı eval girdisi yüzünden hep None) dolar.
- Doğrulama: rastgele küçük Qwen3 (4B tokenizer) ile GERÇEK `train()` uçtan uca: 40 mikro →
  10 optimizer adımı, 5 eval, best checkpoint kaydı ✅. 30B'de eval süresi **ÖLÇÜLMEDİ**
  (tahmin ~7 dk/eval, tam havuzda +~1 sa).

**Kapı (Windows):** ruff format/check ✅ · mypy 256 ✅ · pytest **2743 passed, 4 deselected**.

**Açık bulgular (öneri sırası devamı, henüz yapılmadı):**
- Öneri 3 veri: kısa cevaplar neredeyse tamamen **sentetik QA**'dan (1051 satır, medyan 166
  kr, 726'sı <200); disiplin ~230, diğer ~850. → sentetik ağırlığını düşür / açıklamalı yeniden
  üret (karar kullanıcıda; veri değiştirilmedi).
- Öneri 7 — **ÖLÇÜLDÜ, değişiklik gereksiz:** Ollama `hektor-v12-30b` (GGUF, temp 0.7, seed 42)
  discipline_core 16 soruda `repeat_penalty` 1.0 / 1.1 / `presence_penalty` 1.5 → dejenere
  **0/16 her üçünde**, CJK sızıntısı 0. Yani eval'deki `degenerate` vetosu HF/PEFT greedy
  çözümlemeye ait; Modelfile'a dokunulmadı. İçerik hatası: survivorship bias'ı
  "look-ahead bias" diye adlandırıyor (veri/eğitim konusu).
- Öneri 8 — **YAPILDI:** `_token_hit` (eval `contains:` bayrağı) çok-kelimeli ifadeleri
  koşulsuz bayraklıyordu ve yoksunluk ekini görmüyordu ("tek backtest yeter" ⊂ "yetersiz").
  Artık: yoksunluk eki (-sız/-suz) = zıt anlam; çok-kelimeli ifade de olumsuzlama
  penceresinden geçer (6 kelime + yanıltıcı/yanlış/hatalı); "değil mi?", "yanlış değil",
  "risk yok / sorun değil" olumsuzlama SAYILMAZ. Kayıtlı 86 bayrak yeniden hesaplandı →
  7'si düştü, 7'si de elle çürütme. Eski `test_multiword_token_stays_strict` bilinçli
  olarak ters çevrildi (veri gerekçesi docstring'de). Mevcut rapor JSON'ları yeniden
  yazılmadı; yeni eval'ler yeni kuralla hesaplanır.

---

## Son seans — 2026-09-28 (3): Qwen3-30B-A3B-Instruct-2507 LoRA hazırlığı (eğitim YOK)

Dal `claude/lora-30b-a3b-prep`. **Kullanıcı kararları:** base `Qwen/Qwen3-30B-A3B-Instruct-2507`
indirilsin · LoRA **yalnız attention** (router + uzmanlar donuk) · karışım
**`trading_analysis_v1`** (`wd_d310e37d9a`, 24 saat taze; bayatlarsa yeniden sor) ·
eğitim web arayüzüne bağlı olsun ve düşmesin → nöbetçi zamanlayıcıya + uyku engeli.

**Makine:** 128 GB RAM, Xeon w7-2575X 22 çekirdek, torch CPU, RTX 4000 Ada 20 GB (eğitimde
kullanılmıyor). 30B bf16 ≈ 61 GB → CPU'da **bf16 zorunlu** (fp32 ≈ 146 GB, sığmaz).

**Bulgular / düzeltmeler:**
1. **MoE hedef tuzağı (ölçüldü, meta cihazda).** transformers 5.16 `Qwen3MoeExperts` uzmanları
   birleşik 3B parametre tutar → `gate_proj/up_proj/down_proj` MODÜL değil. 7'li
   `TARGET_MODULES` PEFT'te **sessizce yalnız attention**'a düşerdi. Artık `train()` model
   yüklendikten sonra `unmatched_target_modules` ile kontrol eder; eşleşmeyen hedef → açık hata.
   Gerçek config: 48 katman · 128 uzman · top-8 · hidden 2048 · 30.53B; attention-only r=16 →
   13.37M eğitilebilir (%0.044).
2. **Profildeki `target_modules` hiç uygulanmıyordu** (YAML'da yazılı, `load_lora_profile`
   atlıyordu). Artık uygulanır + doğrulanır (yalnız TARGET_MODULES alt kümesi; lm_head/embed
   yasak). Yan etki: `small_smoke_test` artık gerçekten yalnız attention eğitir (YAML niyeti).
3. Yeni profil **`moe30b_attn_local`** (attention, r16/α32, lr 1e-4, NEFTune 5, maskeli,
   `gradient_checkpointing: true`, `max_examples: 600` — adım süresi ÖLÇÜLMEDİ).
4. **RAM ön-kontrolü** (`check_cpu_ram`): checkpoint boyutu yerel index'ten; yetmezse model
   yüklenmeden hata (nöbetçinin OOM'u sonsuz diriltmesi önlenir). `dry_run` çıktısında `ram_check`.
5. **Uyku engeli**: `trainer.train()` süresince `SetThreadExecutionState` (kalıcı ayar değil).
6. **Web**: `/api/training/run` artık `profile` + `max_examples` alır (profil onay yakılmadan
   doğrulanır); formda profil seçimi + örnek tavanı; A3B yazınca MoE profili seçilir;
   iterasyon 0 = plandan.
7. **Nöbetçi tuzağı (canlı):** v10 BİTMİŞ ama `train_status.json` diskte kalmıştı ve
   `train-recovery-check` "kurtarma yetkili" diyordu → nöbetçi açılsaydı bitmiş koşuyu her 5
   dk diriltmeye çalışırdı. Artık son `checkpoint-N ≥ iterations` ise **YETKİSİZ (tamamlanmış)**.
8. **`HektorTrainingWatchdog` Görev Zamanlayıcı'ya kaydedildi** (kullanıcı onayı; 5 dk,
   gizli pencere, `start-server.ps1`'deki tanımın aynısı). Önceden bu makinede YOKTU.

**Kapı (Windows):** ruff format/check ✅ · mypy 256 ✅ · pytest **2720 passed, 4 deselected**
(`-m "not ollama"`; ilk koşuda statik arayüzde sabit model adı testi yakaladı → metin düzeltildi).
Commit/push YOK (kullanıcı istemedi). Çalışan web sunucusu yeni `/api/training/run` alanları
için **yeniden başlatılmalı** (statik dosyalar hemen, Python ucu yeniden başlatmada).

### Kademe 2 (30B öncesi) — 3 finder + her gruba 2 bağımsız şüpheci doğrulayıcı

Alt sistemler: **A** eğitici · **B** başlatma/kurtarma/web · **C** eğitim-sonrası (eval/kayıt/
sohbet/merge). 33 bulgu; iki oyda da "eğitimden ÖNCE EVET" alanlar düzeltildi:

| ID | Sorun (gerçek veriyle ölçüldü) | Düzeltme |
|---|---|---|
| A1=B2 | 600 örnekte 2 satır maskelenemiyor → hedef 598'e kırpılıyor, son checkpoint 598, durum `iterations=600` → BAŞARILI koşu "çökmüş" sanılıp her ~15 dk 61 GB yükleyen sonsuz diriltme | trainer `run_plan.json` (fiili max_steps) + `run_complete.json` yazar; `recovery_allowed` bunları kullanır; `train` bitişte durum kaydına `finished_at` |
| A2=B3 | hedef uyuşmazlığı 61 GB yüklemeden SONRA; `train` hata → exit 0 → sonsuz döngü | `precheck_target_modules` (meta cihaz, yükleme öncesi; web ön-kontrolünde de); hata → `failed_at` + exit 7 |
| N1 | start-train kurtarmada `started_at`'i yeniliyordu → 72 sa sınırı hiç işlemiyordu | kurtarmada `started_at` korunur + `recovery_attempts` (≥3 → dirilme yok) |
| B1 | web "Durdur" edilen koşu ~10 dk sonra onaysız diriliyordu | `stop_requested_at` → kurtarma YETKİSİZ |
| C3(+C5) | web/nöbetçi koşusu kayıt defterine HİÇ girmiyordu | supervised koşu da CANDIDATE kaydedilir (auto_pipeline hariç: `skip_register`); fiili örnek sayısı |
| B4/B5/B6/B12 (ucuz) | aynı adlı eski checkpoint · plandan fazla iterasyon · RAM kontrolü onaydan sonra · RESUME env sızıntısı | ön-kontrolde red · launch reddeder, form varsayılanı 0 · web ön-kontrolünde RAM · launch `RESUME=0` |

**Eğitim-SONRASI açık (düzeltilmedi, eval/sohbetten ÖNCE yapılmalı):** C1 web LoRA sohbeti
eski modeli boşaltmadan yenisini yükler (2×61 GB) → eğitim sürerken web sohbeti KULLANMA ·
C2 `peft_llm_shim` fp32 + None'da merdiven işaretsiz atlanır (auto_pipeline; 30B'ye erişilmez)
· C6 `lora-eval` adapter/base uyumunu base üretiminden SONRA doğrular, `--base-model` sessizce
ezer → 30B'de `--base-model` VERME · C7 sohbette "base" = 4B · C8 eval'de RAM ön-kontrolü yok
· C9 merge/GGUF kodda yok, `docs/arsiv/4b_donemi/RAG_LORA_ENTEGRASYON.md` eski · C10/C11 küçük. Ayrıca A3
(CLI'da onay profil doğrulamasından önce), A5, A6, A8 (uyku engeli dönüşü), A9 (KL yorum
yönü), A10, B7–B11 düşük önem. DB'de bayat pending onaylar var (`apr_98e76bb03dfa`,
`apr_da7a846b5bdc`) — onaylanmamalı.

**Kapı (Windows, Kademe 2 düzeltmeleri sonrası):** ruff ✅ · mypy 256 ✅ · pytest **2731 passed**.

### Koşu: `hektor_lora_v11_30b` — BAŞLADI 2026-09-28 21:25 (web ucundan)

- İndirme: Xet katmanı iki kez son parçalarda takıldı (19:34, 20:49) → `HF_HUB_DISABLE_XET=1`
  ile parça parça tamamlandı; 16/16 parça safetensors başlığıyla doğrulandı (61.06 GB).
- `.env`: `HEKTOR_PEFT_BASE_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507`,
  `HEKTOR_BACKGROUND_LOOPS_ENABLED=false` (eğitim bitince **true** yap; yedek
  `storage/env.bak-20260928-30b`). Web yeniden başlatıldı (`storage/web.20260928-2056.log`).
  Not: `HEKTOR_API_TOKEN` bu makinede BOŞ (önceden de öyleydi) → web API kimliksiz.
- Onay `apr_9ffc3cfa562e` (kullanıcı talimatıyla), karar `wd_d310e37d9a` tüketildi.
- 600 örnek → 2 satır maskelenemedi → **598 adım** (Kademe-2 A1 tam öngörüldüğü gibi;
  `run_plan.json` max_steps=598 yazıldı). ~15–20 sn/adım → toplam ~3 saat.
- **BELLEK (ölçüldü, tahmin YANLIŞ):** `check_cpu_ram` ≈71 GB dedi; gerçek süreç özel
  belleği **121.5 GB** (sabit), sistem commit 152/154 GB. Sayfa dosyası sistem-yönetimli,
  8 → 27 GB kendiliğinden büyüdü; OOM olmadı. Kök neden araştırılmadı (aday: transformers
  5.x birleşik-uzman dönüşümü / CPU grouped_mm tamponları). → `_RAM_OVERHEAD_*` MoE için
  düzeltilmeli; bu koşu sırasında Ollama/web sohbeti/eval AÇMA.

**v11 SONUÇ (2026-09-29 00:36):** 598/598 adım, 3 sa 10 dk, `train_loss` 1.053, epoch 1.0.
`run_complete.json` yazıldı; `train-recovery-check` → "koşu TAMAMLANMIŞ" (A1 düzeltmesi canlıda
işledi). Kayıt: `adapter_6bd64fb2669e` **candidate** (30B base, r16, attention, 598 örnek —
C3 düzeltmesi işledi). Bellek sonda ~128 GB özel (başta 121.5; ~2.4 GB/saat artış). Eval YOK,
terfi YOK.

**v12 BAŞLADI (2026-09-29 08:10):** `hektor_lora_v12_30b`, tüm havuz 1682 örnek × 1 epoch,
onay `apr_ff727c199d36` (gece 00:40'ta oluşturuldu; araç güvenlik sınıflandırıcısı geçici
arızalandığı için sabaha kaldı), karar `wd_63b1219424` (`trading_analysis_v1`). Web'den.

**Kullanıcı kararı (2026-09-28 23:15):** "adımları max yapalım ya da bir sonraki eğitimde" →
v11 (600 örnek) bitirilir; **bir SONRAKİ 30B eğitimi TÜM havuzla** (1682 örnek × 1 epoch ≈
1680 adım, ~9–10 saat; web: örnek tavanı 1682, iterasyon 0, adapter `hektor_lora_v12_30b`).
Önce: bellek artışı (~2.4 GB/saat) incelenmeli; karışım ağırlığı YENİDEN sorulmalı.

**Sıradaki (sırayla):**
1. ~~İndirme~~ — tamam.
   Yarıda kalırsa: `.venv/Scripts/hf.exe download Qwen/Qwen3-30B-A3B-Instruct-2507` kaldığı yerden sürer.
2. ~~Kademe 2~~ — YAPILDI (yukarıda).
3. Ollama modelini boşalt (30B q4 GGUF RAM/VRAM tutar), arka plan döngülerini kapat.
4. **Kısa ölçüm koşusu** (Kural 8 onayıyla, ör. `max_examples` 20) → s/adım ölç, sonra
   `max_examples`'ı süreye göre seç. Web: Temel model `Qwen/Qwen3-30B-A3B-Instruct-2507`,
   profil `moe30b_attn_local`, iterasyon 0. CLI eşdeğeri:
   `.\scripts\start-train.ps1 -Adapter hektor_lora_v11_30b -BaseModel Qwen/Qwen3-30B-A3B-Instruct-2507 -Profile moe30b_attn_local -MaxExamples <N>`
5. Eğitim sonrası: eval HF/PEFT ile CPU'da 30B → yavaş; merge (bf16 ≈ 61 GB) →
   llama.cpp `convert_hf_to_gguf` (qwen3moe destekli) → Q4_K_M → Ollama yolu 4B'dekiyle aynı.
   ADAY; terfi ayrı insan kararı.

---

## Son seans — 2026-09-27 (3): neden `qwen3:30b` → `qwen3:30b-a3b-instruct-2507-q4_K_M` (ölçüldü)

**Kod değişmedi, eğitim yok.** Bu makinenin `.env`'i 10:47'de `logs/switch-to-instruct.ps1`
ile `HEKTOR_LLM_MODEL=qwen3:30b-a3b-instruct-2507-q4_K_M`'e geçmişti; gerekçesi yazılmamıştı.
Bu kayıt onu ölçümle belgeler. Kod varsayılanı (`qwen3:4b-instruct-2507-q4_K_M`) aynı.

**Kök neden — 2026-09-13'teki 4B tuzağının aynısı.** Ollama'nın `qwen3:30b` etiketi hibrit
değil, **Qwen3-30B-A3B-Thinking-2507**'dir (`/api/show`: `general.finetune=Thinking`, şablon
son `<think>`'i kapatmaz → `classify_think_support` = `forced`, düşünme kapatılamaz). Instruct
etiketi `general.finetune=Instruct` → `toggle`. İkisi de 30.5B MoE (~3B aktif), q4_K_M,
18.6 GB: **boyut/bellek farkı yok, yalnız ince-ayar farkı.**

**Ölçüm (2026-09-27 ~21:10, Ollama 0.34.4, RTX 4000 Ada 20 GB).** Payload
`LocalLLM._generate_ollama` ile birebir: think kararı, `num_predict` (varsayılan 1024; forced +
serbest metinde +1024), çağrı yerlerinin temperature/seed değerleri, gerçek `rag_answer` ve
`knowledge_card` system prompt'ları. **Tek sapma:** Thinking modeli CPU'da (`num_gpu=0`) koştu,
çünkü mastery kuyruğu instruct'ı GPU'da kullanıyordu ve iki 30B 20 GB'a sığmıyor (aşağıdaki
olay). Süreler cihazlar arası karşılaştırılamaz; **karşılaştırılabilir metrik üretilen token
sayısıdır** (aynı mimari + kuantizasyon → token başına iş aynı).

| Çağrı | Instruct (GPU): token · üretim | Thinking (CPU): token · üretim | Not |
|---|---|---|---|
| "Tek kelimeyle: 2+2" (serbest metin) | **2** · 0.02 sn → "4" | **1219** · 41.9 sn → "dört" | 4.5k krk atılan düşünme |
| EN→TR çeviri, "5-10%" (serbest) | **36** · 0.4 sn | **849** · 28.6 sn | ikisi de "%5-10"u korudu |
| RAG cevabı, `rag_answer` prompt'u (serbest) | 1024 · 11.6 sn, `length` — cevap kesik ama VAR (2964 krk) | 2048 · 93.4 sn, `length` — **cevap 0 krk** | app'te `LLMUnavailable` |
| Bilgi kartı, `format=json`, max 700 | 637 · 7.3 sn, geçerli JSON, `main_claim` dolu | 379 · 14.8 sn, geçerli JSON, `main_claim` dolu | JSON'da `think=false` gider |

Hız: instruct GPU 87–127 tok/s, Thinking CPU 22–30 tok/s. **Tahmin (ölçülmedi):** Thinking
GPU'da aynı hızda koşsa 2+2 ≈ 13 sn, çeviri ≈ 9 sn, RAG ≈ 23 sn — ve yine cevapsız.

**Sonuç.**
1. Serbest metin çağrılarında Thinking ~24–600× daha çok token üretiyor; fazlası atılan düşünme.
2. En kritiği: gerçek RAG prompt'unda düşünme 1024+1024 bütçenin tamamını yedi → **cevap yok.**
   Küçük bütçeli çağrılarda risk daha büyük: `comprehension_scorer` `max_tokens=120` → forced'da
   1144 token; 2+2'nin düşünmesi bile 1219 token tuttu (çıkarım — bu çağrı ayrıca ölçülmedi).
3. `format=json` çağrılarında (kart, synth-qa, l3/l4 sınav, formül, sentez) `think=false`
   gittiği için fark küçük; ikisi de geçerli kart üretti.
4. Vaka başına n=1, sabit seed → yön göstergesi, istatistik değil.

**Yan bulgular (açık).**
- Instruct'ta da RAG cevabı 1024 varsayılan bütçeye takıldı (9 bölümlü format uzun) → cevabın
  sonu kesiliyor. `rag_answerer` `max_tokens` vermiyor; bütçe mi format mı kısalsın — karar/iş.
- `num_gpu=0` verilse bile Ollama 0.34.4 Thinking modelinin ~0.79 GB'ını GPU'ya koydu (instruct
  etkilenmedi; toplam ~20.0/20.5 GB).
- Çeviri ikisinde de kusurlu: instruct "drawdown"→"çekiliş", "net of"→"…ile birlikte";
  Thinking "net of"→"hariç". Yüzde ikisinde korundu.

**Olay (16:02–17:20).** 10:17'de başlayan `mastery-queue` eski `.env` ile `qwen3:30b`'yi, yeni
`.env` ile kalkan süreçler instruct'ı çağırdı. İki 30B 20 GB'a sığmadığı için Ollama modelleri
sürekli değiştirdi, embedding modeli yüklenemedi → `Read timed out`. 17:39'da kuyruk yeniden
başlatıldı (`logs/mastery-resume.ps1`), artık yalnız instruct. **Aynı makinede iki 30B'yi aynı
anda çağıracak iş başlatma.**

**`qwen3:30b` silindi (kullanıcı kararı, 2026-09-27 ~21:25).** Önce doğrulandı: yüklü değildi,
`.env`/`configs`/`storage` içinde düz `qwen3:30b` referansı yoktu. Ollama'da yalnız
`qwen3:30b-a3b-instruct-2507-q4_K_M` + `nomic-embed-text` kaldı; ~17 GB disk boşaldı; mastery
kuyruğu kesintisiz sürdü.

**Açık:** `setup.ps1`/`setup.sh` seçenek [4] ve README tablosu hâlâ `qwen3:30b`'yi (Thinking)
kuruyor → yeni kurulum aynı tuzağa düşer; `qwen3:30b-a3b-instruct-2507-q4_K_M` olmalı.

---

## Son seans — 2026-09-27 (2): LoRA karışım profilleri + router + profil eval (eğitim YOK)

Dal: `claude/lora-mix-eval`. **Eğitim başlatılmadı, model indirilmedi, base/production'a
dokunulmadı.** Ayrıntı: **[docs/LORA_MIX_EVAL.md](docs/LORA_MIX_EVAL.md)**.

- Achilles "Adapter Mixing / Profile Routing / Evaluation" şartnamesi Hektor'a uyarlandı:
  `configs/lora/mix_profiles.yaml` (eğitim reçeteleri olan `lora_profiles.yaml`'dan AYRI),
  `app/lora/{domain_adapter_registry,profile_registry,profile_builder,profile_router,
  weight_decision,mix_cli}.py`, `app/memory/rag_version.py`, `app/evals/profile/` (eval
  runner + deterministik değerlendiriciler + regression gate + insan denetimi + karşılaştırma),
  `evals/profile_mix/` (validation 27 + golden_test 27, sha256 manifest).
- **Yeni kullanıcı kuralı — her LoRA eğitiminden ÖNCE karışım ağırlıkları sorulur.**
  `train --run` karar olmadan exit 5; web/detached `launch()` alt süreci hiç açmaz;
  `start-train.ps1` exit 5/6 için yönlendirme basar. Karar: `uv run hektor mix weights`
  (etkileşimli ya da `--profile`/`--weights`) veya `train --run --mix-profile X`.
  Karar tek kullanımlık. Ayrıca `train --run` artık eval sızıntısında exit 6.
- Gerçek PEFT `add_weighted_adapter` küçük rastgele Llama ile 4 yöntemde (svd/linear/ties/
  dare_ties) doğrulandı (torch 2.14 · transformers 5.16 · peft 0.20, ayrı geçici venv) —
  base ağırlıkları bayt-bayt aynı kaldı. Ana venv'de peft yok → o 4 test burada `skip`.
- Kapı: ruff format/check ✅ · mypy (255 dosya) ✅ · pytest **2409 passed, 8 skipped**.

**Açık işler (sıra önerisi):**
1. `evals/profile_mix/*.jsonl` seed setleri Claude yazdı → `human_verified=false`; insan
   doğrulaması + korpusa özgü RAG soruları (`expected_chunk_ids`) eklenmeli, sonra
   `write_manifest()` ile hash yenilenmeli.
2. Domain başına eğitim veri setleri (math/statistics/reasoning/trading/coding) yok —
   her adapter base'ten bağımsız eğitilecek; eğitimden ÖNCE Kademe 2 + ağırlık sorusu.
3. GGUF→Ollama servis modeli olmadan C/D sistemleri "koşulamadı" raporlanır.

---

## Bilinen açık işler

- **LoRA/RAG protokolü** (`docs/PROTOKOL_LORA_RAG_IYILESTIRME.md` §10): final test takımı
  (görülmemiş şablonlar), teşhis varyantları, rubrik puanlama aracı + kritik hata kapısının
  `lora-eval`'e bağlanması, Aşama 1 kanıtlı denetim raporu.
- **B6:** terfi edilecek yapıt GGUF ise eval Ollama üzerinden de koşulmalı (eval PEFT'i ölçüyor).
- **D1:** canlı `hektor-v12-30b` Modelfile şablonunda CR — yeniden oluşturulmadı.
- **Ollama determinizmi:** aynı seed + temperature 0'da cevaplar ayrışıyor (önek önbelleği /
  CPU-GPU bölünmesi) → karşılaştırmalar tekrarla yapılmalı.
- **Korpus:** 2 doküman OCR/kodlama çöpü parça üretiyor; 4/300 başlık yayıncı kalıbı;
  `papers.source` hepsinde `manual` (amaç ayrımı `configs/rag/doc_purposes.yaml`'da).
- **CI yeşil olmadan merge — dal koruması dışarıdan kaldırılıyor (AÇIK, 2026-09-15).** Kimin
  kaldırdığı: github.com/settings/security-log → `protected_branch.destroy`.
  Kontrol: `gh api repos/alimirbagirzade/hektor/branches/main --jq .protected`.
- **Ajan envanteri §5, sıra 2-6** (`reports/agent-inventory/envanter-2026-09-09.md`):
  manifest ↔ test drift testi, `grounding_verifier` mutasyon testi, kart+SFT hattını
  doğrulayıcıdan geçirme, K3 normları, eval setini büyütme.
- `docs/MIGRASYON_2.0.md` §"Kalan adaylar"; `docs/MIMARI_REFERANS.md` v1 dönemi (uyarılı).
- 4B dönemindeki açık işlerin tam hâli: `docs/arsiv/4b_donemi/HANDOFF_2026-09_4B_donemi.md` sonu.


## Önemli dosyalar

| Dosya | Ne |
|---|---|
| `CLAUDE.md` | Çalışma kuralları (bağlayıcı) |
| `docs/MIGRASYON_2.0.md` | v1 → 2.0 farkları |
| `docs/MIMARI_REFERANS.md` | Alt sistem alt sistem mimari referansı (v1 dönemi) |
| `automation_manifest.yaml` | Runtime ajanlarının tek bildirimsel kaynağı + zincir |
| `configs/lora/lora_profiles.yaml` | LoRA eğitim profilleri (`moe30b_attn_local` varsayılan; düşük RAM `discipline_safe_local`) |
| `docs/SCOPE_ISOLATION.md` | Sürücü motor ≠ insan yetkisi |
| `SECURITY.md` | Tehdit modeli + ağa açma checklist'i |
| `docs/PROTOKOL_LORA_RAG_IYILESTIRME.md` | LoRA/RAG iyileştirme protokolü + LLM-30 |
| `docs/arsiv/4b_donemi/HANDOFF_2026-09_4B_donemi.md` | 4B dönemi oturum kayıtları (v8–v10) |
