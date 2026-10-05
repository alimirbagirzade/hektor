# Aday doğrulama, karşılaştırma ve etkinleştirme — Faz 2A/2D protokolü

_Karşılaştırma kararı bir LLM soru-cevap ölçümüdür; trading performansının arttığı anlamına
gelmez. Hiçbir adım eğitim ya da etkinleştirmeyi kendiliğinden başlatmaz._

## 1 · Aday gerçekten tamamlandı mı? (`hektor candidate-verify`)

`run_complete.json` tek başına kanıt değildir. Hepsi gerekir (`app/training/candidate_checks.py`):
süreç sonucu (`train_status.json` bu adapter'a ait, `finished_at` var, `failed_at` yok, süreç
bitti) · adım (`global_step == max_steps == run_plan.max_steps`) · adapter dosyaları + ağırlık
özeti · temel model kökeni (`adapter_config.base_model_name_or_path` = reçete) · veri özeti
(koşu = reçete) · koşu kimliği (onay kimliği = kolay akış başlatma kaydı).

Dönüşüm: `merge_info.json` (adapter özeti eşleşir, KL ≤ kapı, ilk-token aynı) · GGUF köken
anahtarları adapter özetini taşır · `.partial` yarım çıktı yok · Modelfile başlığı + `FROM`
doğru dosya · Ollama etiketi var (digest kaydedilir). Yarım çıktı başarı sayılmaz.

## 2 · Ölçüt (adayın sonucu GÖRÜLMEDEN kilitlenir)

`lock_criteria` içerik özetli, değişmez dosya yazar (`reports/comparisons/criteria/`); karşılaştırma
yalnız kendisinden önce kilitlenmiş ölçütle açılır, değiştirilmiş ölçüt reddedilir.

| Öğe | Varsayılan |
|---|---|
| Rubrik | soru başına 0–4 (çapalar ölçüt dosyasında) |
| Toplama | soru → aile ortalaması → aileler üzerinden ortalama |
| Fark | aday − aktif, aile düzeyinde eşlenmiş |
| Tolerans (`margin`) | 0.25 puan (0–4 ölçeğinde) |
| Güven aralığı | %95, AİLE düzeyinde bootstrap (B=2000, seed 42) |
| Asgari kanıt | 8 aile |
| Kritik hata | Kural 1 dili · doğrulanmış matematik anahtarına aykırı sonuç · inceleyicinin işaretlediği kritik hata |

Karar: adaya özgü kritik hata → **kritik_ret** · aile < 8 → **yetersiz_kanit** · GA üst <
−0.25 → **ret** · GA alt > −0.25 ve nokta ≥ 0 → **kabul** · aksi → **yetersiz_kanit**.
Tamamlanma/dönüşüm doğrulaması geçmeyen aday en fazla **yetersiz_kanit** alır.

## 3 · Koşu (`hektor compare-run`)

Donmuş soru seti (özet kaydedilir) · aktif + aday + **temel model referansı** · aynı sistem
istemi ve decoding (temperature 0, seed 42, sabit num_ctx/num_predict) · ortak ağır iş kilidi
altında, model sırası soru başına dönüşümlü · digest koşu başında ve sonunda okunur, değişirse
koşu geçersiz.

## 4 · Puanlama

Matematik: doğrulanmış `answer_key` ile otomatik. Diğerleri: **kör inceleme** (Öğrenme Havuzu →
Kör inceleme) — cevaplar A/B/C, seed'li karışık; eşleme mühürlü dosyada, API'de sunulmaz;
kaynaklı sorularda kanıt metni inceleyene gösterilir. Tüm cevaplar puanlanmadan karar yok.

## 5 · Gizli final seti

`--role final` her kullanımda `storage/final_set_access.jsonl`'e yazılır (kullanım, sonuç
erişimi). Aynı set ikinci kez kullanılırsa koşu **geliştirme** sayılır ve gerekçesi kararda yazar.

## 6 · Etkinleştirme (2A, Öğrenme Havuzu → Modeller)

Karar `storage/candidate_decisions.jsonl`'e (aday etiketi + KARŞILAŞTIRILAN digest) yazılır;
kabulde kayıt defteri `eval_passed`, ret/kritik retta `rejected` olur (production yok).
Ana model: yalnız **kabul** + Ollama'daki digest eşleşmesi + gerekçe. Deneme sohbeti: kabul ya da
**yetersiz kanıt**. Ret/kritik ret hiçbir yuvada. Mevcut model "başlangıç kaydı —
değerlendirilmemiş". Geri dönüşte önceki modelin varlığı ve digest'i doğrulanır. Etkinleştirme
günlüğü + kayıt defteri iki aşamalı tutarlı güncellenir; devam eden cevap başladığı modelle biter.

## 7 · Aday hattı (web + CLI ortak servis, `app/training/candidate_jobs.py`)

Öğrenme Havuzu → **Aday hattı**: Doğrula → Ollama'ya hazırla → Karşılaştır → Sonuçları incele
(kör inceleme) → Kullanıma al / Geri dön (Modeller kartı). CLI: `candidate-prepare`,
`candidate-compare`, `candidate-jobs`, `candidate-job-stop`.

- Ağır işler mevcut komutları koşar (aynı `adapter_to_ollama.ps1`, `-Force` ASLA; aynı
  `hektor compare-run`); iş mantığı kopyalanmaz.
- Ayrık çalıştırıcı web sunucusunun süreç ağacı DIŞINDA başlar (çift başlatma), sonucu iş
  dosyasına yazar; sayfa yenileme / sunucu yeniden açılışı gerçek süreçle uzlaşır. Çalıştırıcı
  sonuç yazmadan ölürse iş **kesildi (tamamlanmadı)**.
- Aynı istek kimliği aynı işi döndürür; koşan iş, ortak ağır iş kilidi ya da sohbet kirası
  varken yeni iş başlamaz; mevcut ya da canlı Ollama etiketi ezilmez.
- Başarı = çıkış 0 + iş sonrası doğrulama (`verify_conversion` / manifest `generated`).
  Yarım dosya ya da durdurulmuş üretim geçerli aday değildir; durdurulan karşılaştırma `kesildi`.
- Desteklenmeyen platform / araç yoksa düğme açık gerekçeyle kapalıdır (aday başına).
- Tamamlanma doğrulaması kolay akış kaydı yoksa bağımsız kayıtlardan (kayıt defteri, onay
  deposu, ağırlık kararı, bugün yeniden özetlenen veri) yapılır; onay kaydı insan onaylı ve
  tüketilmiş olmalı.
- `evals/candidate_compare/smoke_v1.jsonl` yalnız ENTEGRASYON setidir (6 aile < 8 → en fazla
  yetersiz kanıt); sızıntı kapsamındadır, eğitime girmez.
