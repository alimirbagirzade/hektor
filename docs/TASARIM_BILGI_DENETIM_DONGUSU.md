# Kaynak bilgi denetimi + kontrollü eğitim döngüsü

_2026-10-12 · kod: `app/research/knowledge_review.py`, `app/orchestration/training_loop.py`,
`app/training/exit_witness.py` · politika: `storage/training_loop_policy.md` (yerel)_

## 1. Neden

İnceleme (2026-10-12) şu boşlukları buldu:

| Boşluk | Kanıt (değişiklik öncesi) |
|---|---|
| Araştırma paketinin kaynaklı aday verisi kalem bazında denetlenmiyordu | `research_worker.data` satırları `data/research_package/staging/distill_qa.jsonl`'e yazıyordu; kaynak güvenilirliği / güncellik / tekrar / çelişki kaydı, karantina ya da insan kararı yoktu; staging'den eğitime giden yol da yoktu |
| GPT (Codex) yalnız aşama PLANINI inceliyordu | `ResearchService.review` tek bir devam/beklet kararı üretiyordu; bilgi kalemine yorum yoktu |
| Eğitim bitişinde çıkış kodu yoktu | detached başlatıcı süreci bırakıp gidiyor; `finished_at` alt sürecin kendi beyanı |
| Eğitim sonrası değerlendirme ↔ sonraki eğitim bağı yoktu | politika "regresyonda yeni eğitim başlatılmaz, aynı veri özetiyle sıfırdan eğitim yok" diyor ama hiçbir başlatma yolu bunu denetlemiyordu; `training_loop_checkpoint.json` elle yazılıyordu |
| Web'de bilgi durumu / döngü görünmüyordu | yalnız araştırma paketi aşama durumu vardı |

## 2. Bilgi kalemi yaşam döngüsü

```
staging satırı ──import──▶ kalem (orijinal satır DEĞİŞMEZ)
   │ deterministik inceleme (kr-1): kaynak güvenilirliği · doğruluk · güncellik · fayda ·
   │                                tekrar · çelişki + sızıntı · köken · Kural 1
   │ yerel hakem (local_judge istemi, temperature 0, seed 42)
   │ [isteğe bağlı] GPT yorumu — yalnız araştırma yöneticisinde Codex seçiliyse
   ▼
 durum: reddedildi | tekrar | karantina | onaylandi(otomatik|insan) | incelemede
   │ (insan) duzelt → AYRI sürüm (önceki inceleme + onay taşınmaz)
   │ (insan) onayla/reddet --kim --gerekce
   ▼
 surum (değişmez rv_…) ──sec──▶ data/lora_sft/research_selection.json
   ▼
 sft_assembly (revalidate_line + dedup + disiplin + kalite/sızıntı/tazelik kapıları)
```

* **Otomatik onay** yalnız: tüm deterministik kontroller TAM kapsadı (yalnız hesap birimleri
  "kapsanmış" sayılır — sözcük benzerliği doğrulama değildir) **ve** yerel hakem çalıştı ve
  şüphe bildirmedi. Uzun sözel cevaplar pratikte bu şartı sağlamaz → karantina → insan.
* **Otomatik çürütme** (sızıntı, Kural 1, hesap hatası, getirilmeyen atıf kimliği, içerik
  kapısı, yerel olmayan üretici) insan onayıyla geçersizleştirilemez.
* **GPT yorumu** doğruluk kanıtı değildir: `supheli` → karantina, `tutarli` → hiçbir şey.
  Metni hiçbir eğitim satırına girmez; gönderilen istemin özeti kayda yazılır.
* **Güncellik** otomatik ölçülemez; kaynak yayın yılı kaydedilir, kapı değildir.
* **Çelişki** sezgiseldir: neredeyse aynı soruda (3-gram Jaccard ≥ 0.6) farklı sayısal değer →
  temkinli karantina.

Kayıtlar: `storage/knowledge_review/items/kr_*.json` (kaynak, sürümler, her inceleme için
model/istem sürümü/tarih/gerekçe, insan kararları). Sürümler:
`data/learning/research_datasets/rv_*/`.

## 3. Eğitim döngüsü defteri

`storage/training_loop/<adapter>.json`:

1. **Kayıt** — detached başlatıcı (`register`) ya da durum dosyası (`sync_from_status`; bu
   sürümden önce başlatılan `hektor_lora_full_20261011` böyle girer).
2. **Bitiş** — `egitim-dongusu bitis`: `verify_run_completion` (+ yeni `cikis_kodu` kontrolü:
   `exit_witness`'in yazdığı `train_exit.json`, pid + oluşturulma zamanı eşleşmeli, kod 0).
3. **Geliştirme değerlendirmesi** — `egitim-dongusu degerlendirme`: rubrikle puanlanmış LLM-30
   `raw.jsonl` (aday) ↔ önceki modelin puanlı `raw.jsonl`'i, `rubric_gate` ile eşlenmiş;
   manifestteki ağırlık özeti doğrulanan ağırlıklarla aynı olmalı; yalnız validation seti;
   bayraklı cevap artmamalı. Geçti ≠ terfi (terfi insanda, final takımı ayrı).
4. **Kapı** — `next_training_blockers` şu yolların hepsinde: kolay akış `precheck`,
   `preflight_launch` (web/Auto-LoRA), `hektor train --run` (kurtarma/sürdürme hariç; çıkış 12),
   `direct_run_guard`. Engeller: önceki adayın bitişi doğrulanmamış / değerlendirme yok /
   geçmedi; aynı veri özeti daha önce eğitilmiş (istisnasız). Başarısızlık sonrası devam:
   `serbest-birak` = gerekçe + TAZE insan onayı.

## 4. Bilinçli olarak YAPILMAYANLAR

* Eğitim sonrası LLM-30 koşusu / dönüşüm / rubrik puanlaması **otomatik başlatılmaz** (ağır iş;
  rubrik insan/hakem puanıdır). Defter yalnız sonucu okur ve kapıya bağlar.
* Üretim modeline terfi yok; `.env` ve etkin model değişmez.
* İkinci zamanlayıcı yok: inceleme araştırma paketinin `data` aşamasının sonunda, GPT yorumu
  aynı yönetici turunda çalışır.
