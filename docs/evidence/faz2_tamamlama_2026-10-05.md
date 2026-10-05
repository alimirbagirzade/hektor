# Faz 1–2 sonrası açıkların kapatılması — gerçek sistem kanıtları (2026-10-05)

Dal `claude/hektor-faz-1-2-completion-297c12` (main `890cb91` üstü). Makine: Windows 11,
127.6 GB RAM (başlangıçta 107 GB boş), C: 700 GB boş, RTX 4000 Ada. Ana kurulum `C:\HP\hektor`
(8765'teki web süreci yeniden başlatılmadı; aktif model `hektor-v12-30b` — **değişmedi**).

## 1 · Strateji çevirisi (gerçek yerel model)

İzole test kökünde (8771), önceki olaydaki metni taşıyan **etiketli test turu** ("TEST KAYDI";
cevap metni sabit) üzerinde "Taslak çıkar" gerçek Ollama modeliyle (`qwen3:30b-a3b-instruct-
2507-q4_K_M`) koştu. Model long için "Stop-loss: fiyat + 2 ATR" ifadesini yine `atr_initial 2.0`
(girişin altı) olarak "düzeltti"; yeni akış bunu **ÇELİŞKİ** olarak gösterdi, stop alanını boş
bıraktı ("karar gerekli"), Londra seansı kuralını ÖNEMLİ desteklenmeyen kural yaptı. Kaydetme
sonrası 6 fark listelendi; 5/6 işaretliyken onay düğmesi kapalı, gerekçesiz onay reddedildi
("Orijinal öneride çelişki var…"), gerekçeli onaydan sonra test düğmeleri açıldı.

## 2 · Gerçek piyasa verisi

`docs/evidence/gercek_veri_backtest_denetimi_2026-10-05.md` — BTCUSDT spot 1h, 2023-01→2024-12,
Binance kamu arşivi (24 dosya, SHA256 doğrulandı), 17 543 bar, 1 boşluk. Geliştirme + doğrulama
(final koşulmadı), long+short; 683 işlemin tamamı bağımsız hesapla yapı olarak aynı, azami göreli
fark 1.5e-15. Uç durumlar `tests/test_independent_backtest_crosscheck.py` kontrollü barlarla.

## 3 · Mevcut adaptörle model yaşam döngüsü

**Aday:** `hektor_lora_v14_30b` (tamamlanmış; yeni eğitim BAŞLATILMADI).

Çapraz doğrulama (`candidate-verify` mantığı, salt-okuma) — tümü ✅:

| Kontrol | Kanıt |
|---|---|
| süreç sonucu | `train_status.json` adapter=v14, finished 2026-10-03T17:35:22Z, failed yok, pid 28400 yaşamıyor |
| adım | run_complete 187/187 = run_plan 187 (GA 8, best ckpt 168, eval_loss 0.5329) |
| adapter dosyaları | `adapter_model.safetensors` sha `36e2c6d03fff…` |
| temel model | adapter_config = kayıt defteri `adapter_a85dc1dd5e40` = `Qwen/Qwen3-30B-A3B-Instruct-2507` |
| veri özeti | koşu `613700551afb…` = bugün yeniden özetlenen `lora_sft.jsonl` (1569 satır; 1489 eğitim örneği) |
| koşu kimliği / onay kaydı | `apr_9e27b2e343f3`: insan onayı 08:41:55Z, eylem `train_run`, tüketildi 08:43:40Z; ağırlık kararı `wd_db97219915` aynı anda `train:hektor_lora_v14_30b` tarafından tüketildi |

Bulunan ve düzeltilen hata: anlık görüntü klasörü yoksa kayıt tabanlı reçeteye düşülmüyordu
(`4433456`). Karşıt örnek: v13 dürüstçe "doğrulanmadı" (koşu kaydı artık v14'e ait).

**Dönüşüm + servis (ayrı aday etiketi):** `hektor candidate-prepare hektor_lora_v14_30b
--ollama-tag hektor-v14-30b-aday --template-from hektor-base-30b-q4a8` → iş `job_43a558abf3da`
(yeni ortak servis, aynı `adapter_to_ollama.ps1`, `-Force` yok). İlk deneme `job_4a9c4211a014`
çalıştırıcı çöktüğü için **"kesildi (tamamlanmadı)"** göründü (çalışma dizini hatası, `b6ced16`
ile düzeltildi). İkinci deneme 78 sn'de **done**: 1–3. adımlar köken anahtarı eşleştiği için
atlandı, Modelfile + `ollama create`; iş sonrası `verify_conversion` ✅ (birleştirme KL 0.01545 ≤
0.02 — not: v14 dönüşümünde KL kapısı `kl_gate_override=true` ile 0.02'ye yükseltilmişti; yarım
çıktı yok; GGUF köken anahtarları adapter sha'sını taşıyor; Modelfile FROM doğru). Ollama digest
`e9d7a4ae27fe…` — mevcut `hektor-v14-30b` ile **birebir aynı** (aynı GGUF + aynı şablon). Mevcut
etiketler ezilmedi; kilit iş sonunda bırakıldı.

**Karşılaştırma (ENTEGRASYON TESTİ — kalite kanıtı DEĞİL):** `job_5f0dc9fd2c00` →
`cmp_ab63a45247fe` (rol development). Aktif `hektor-v12-30b` (`acebe185…`) · aday
`hektor-v14-30b-aday` (`e9d7a4ae…`) · temel `hektor-base-30b-q4a8` (`6a92e8fb…`). Set
`evals/candidate_compare/smoke_v1.jsonl` (6 aile < 8 asgari → karar en fazla yetersiz kanıt),
temperature 0, seed 42, num_ctx 8192, num_predict 1024, aynı sistem istemi, dönüşümlü sıra.
18 cevap, cevap başına 11–23 sn; üç modelin digest'i koşu başında ve sonunda aynı. Matematik
(doğrulanmış anahtar): aday 3/3, temel 3/3, aktif 2/3 (aktif bir soruda soruyu tekrarlayıp hesap
vermedi → otomatik kritik hata). Temel iki açık soruda `num_predict` sınırına takıldı. **Açık
uçlu 3 sorunun 9 cevabı kör insan incelemesi bekliyor — puanlanmadı, karar yok.**

**Etkinleştirme / geri dönüş (izole kök, 8771):** TEST etiketli kararlar gerçek etiket+digest'e
bağlandı. `hektor-v14-30b-aday` (kabul) → ana model; `hektor-v13-30b` (yetersiz kanıt) → ana
model **reddedildi**, deneme yuvası kabul (etiketli banner); geri dönüş önceki modelin digest'ini
Ollama'da doğrulayarak yapıldı; kayıt defteri tutarlı. Ana kurulum: main `hektor-v12-30b`,
deneme boş, karar dosyası yok — **değişmedi**.

## 4 · Web aday hattı — tarayıcı (izole kök 8771, dönüşüm betiği TEST DUBLÖRÜ)

| Akış | Sonuç |
|---|---|
| Desteklenmeyen platform/araç | önbelleksiz adayda "Başlat" kapalı + açık gerekçe |
| Başlatma + çift tıklama | tek iş; aynı istek kimliğiyle eşzamanlı 2 istek + farklı kimlikle 3. istek → "Başka bir ağır iş sürüyor" |
| Yenileme | KOŞUYOR, adım 3/5, ilerleme çubuğu (gerçek durum) |
| Güvenle durdur | DURDURULDU (adım 3/5), 3 süreç sonlandı (doğrulandı) |
| Hata | kasıtlı hata → BAŞARISIZ "komut başarısız (çıkış 1) — kısmi çıktı geçerli değil"; çıkış 0 ama model yok → BAŞARISIZ "dönüşüm doğrulaması geçmedi" (tamamlandı DENMEDİ) |
| Tekrar dene | yeni iş kimliği |
| Sunucu yeniden açılışı | ilk sürümde çalıştırıcı sunucuyla öldü → ekran doğru biçimde "kesildi"; düzeltmeden (`c6fd93d`, çift başlatma) sonra çalıştırıcı sunucu kapalıyken yaşadı, açılışta KOŞUYOR 5/5, ardından gerçek sonuç |

Tarayıcıda bulunup düzeltilenler: aday başına platform yeteneği; yoklamanın düğmeleri yeniden
yaratması; durdurulan işte eski istek kimliğinin yeniden kullanılması; koşarken yanlış etiket;
günlükteki terminal denetim dizileri.

## 5 · Gerçek eğitim gerektiren son bağlantı — küçük pilot reçetesi (BAŞLATILMADI)

Amaç: kolay akış → reçete kapsamlı Kademe 2 → bağlı alt süreç → bütünlüğü doğrulanan veri
okuması zincirini GERÇEK eğitimde bir kez görmek (kalite değil, hat testi).

1. Ağır iş yok, Ollama boş, 8765 web'de sohbet yok (`hektor status`, `ollama ps`).
2. **Eğitim öncesi karışım ağırlıkları kullanıcıya sorulur** (kullanıcı kuralı).
3. Web → Eğitim hattı → Kolay eğitim: adapter `hektor_lora_pilot_k2_30b`, temel
   `Qwen/Qwen3-30B-A3B-Instruct-2507`, profil `moe30b_attn_local`, örnek tavanı **64**.
   "Anlık görüntü hazırla" → `recipe_sha` not edilir.
4. Bu kod durumu için **gerçek** Kademe 2 derin avı (finder + 2-oylu doğrulama) yapılır; kapanınca
   `uv run hektor kademe2-kayit --findings f.json --recipe-sha <sha> --evidence "..."`.
   (Bu görevde Kademe 2 kaydı ÜRETİLMEDİ.)
5. "Özeti onayla" → onay → `uv run hektor approval-approve <id>` → tekrar onayla.
6. Doğrula: `train-full.log`'da reçete bağlaması; çıkış 11/10 yok; `train_status.json` reçete
   verisi; bitince `hektor candidate-verify hektor_lora_pilot_k2_30b`.

Tahmin (v14 ölçümünden: 187 optimizer adımı 56 sa 52 dk → ~18 dk/adım, GA 8): 64 örnek → 8
adım ≈ **2.5 sa** + model yükleme/eval ~20 dk. Bellek: v14'te süreç private ~120 GB'a çıktı
(boş RAM 12.7 GB) → pilot sırasında başka hiçbir şey açılmamalı; sayfa dosyası büyütülmesi
önerilir. Dönüşüm (isteğe bağlı) +~1–1.5 sa ve ~116 GB.
