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

**Sınır (Kademe 2 F4-6, kabul edildi):** mühürlü eşleme `storage/comparisons/<id>/
sealed_mapping.json` dosyasıdır; aynı bilgisayarın sahibi onu AÇABİLİR — "erişilemez" değildir,
yalnız API/arayüz sunmaz. Çalışma kuralı: model kimlikleri karar verilene kadar açılmaz; AI
incelemesine yalnız kör cevap paketi verilir (`ai_review.json` ayrı dosya, insan puanı yerine
geçmez).

## 5 · Gizli final seti

`--role final` her kullanımda `storage/final_set_access.jsonl`'e yazılır (kullanım, sonuç
erişimi). Aynı set ikinci kez kullanılırsa koşu **geliştirme** sayılır ve gerekçesi kararda yazar.
Set kimliği içerikten hesaplanır (satır sonu / sıra / anahtar sırası değişikliği yeni set
yapmaz); her final kullanımında normalize soru parmak izleri de loglanır → daha önce final'de
GÖRÜLMÜŞ bir soruyu içeren set (kimliği değiştirilmiş olsa da) **geliştirme** sayılır. Final
soruları görüldükten sonra aynı sorularla yapılan yeni geliştirmeler bağımsız final başarısı
DEĞİLDİR (Kademe 2 F4-5).

## 6 · Etkinleştirme (2A, Öğrenme Havuzu → Modeller)

Karar `storage/candidate_decisions.jsonl`'e (aday etiketi + KARŞILAŞTIRILAN digest) yazılır;
kabulde kayıt defteri `eval_passed`, ret/kritik retta `rejected` olur (production yok).
Ana model: yalnız **kullanılmamış gizli final setiyle** verilmiş **kabul** + Ollama'daki digest
eşleşmesi + insan eylemi ve gerekçe (Kademe 2 F4-5, kullanıcı kararı). Geliştirme rollü kabul
(ya da final'den geliştirmeye düşmüş koşu) yalnız deneme sohbetine yeter. Deneme sohbeti: kabul ya
da **yetersiz kanıt**. Ret/kritik ret hiçbir yuvada. Mevcut model "başlangıç kaydı —
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

## 8 · Geniş boyutlu set, AI incelemesi, entegrasyon işareti (2026-10-06)

- **`broad_v1`** (27 aile, her aile tek soru): matematik 10 (doğrulanmış anahtar, otomatik),
  kaynak 7 (kanıt metni istemde; biri "kaynakta yok" çekimserliği), talimat 5, strateji kural
  eksiksizliği 5. Ölçüt `DIMENSION_CRITERIA` cevaplar görülmeden `hektor compare-lock-criteria
  --preset boyutlu` ile kilitlenir; `compare-run` ölçütü setin biçiminden seçer. Sonuç boyut
  bazında AYRI raporlanır (ortalama + kritik sayısı); karar kuralları v1 ile aynıdır.
- **Sızıntı:** `scripts/eval_set_leak_check.py` aile düzeyinde (bir soru sızarsa aile sızmış)
  eğitim/damıtma verisine VE geliştirme setlerine karşı. Sözcükseldir; anlamsal yakınlığı
  yakalamaz. Set eğitime girmez (`adapter_eval_items`).
- **Gizli final:** set ilk kullanımda `--role final` istenebilir; kod her kullanımı kaydeder,
  ikinci kullanımdan itibaren GELİŞTİRME sayar. Geliştirmede görülmüş set final diye sunulmaz.
- **AI incelemesi** (`hektor compare-ai-review`) ayrı dosyadır (`ai_review.json`): insan puanı
  DEĞİLDİR, `finalize` okumaz, durum değişmez, kayıt defterine/terfiye girmez. Web'den yazılamaz;
  insan incelemesi bitene kadar ekranda gizli ("Yine de göster" ile açılır — çapalanma riski).
- **Yalnız entegrasyon testi** (`hektor compare-mark-integration`): karar en fazla yetersiz
  kanıt, kayıt defteri güncellenmez, ana model terfisinde kullanılamaz. `cmp_ab63a45247fe`
  (smoke_v1, 6 aile) bu işaretle tutulur.
