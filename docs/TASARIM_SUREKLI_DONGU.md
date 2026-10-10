# Tasarım — Sürekli döngü + buluttan cevap kontrolü

_2026-10-10 · Karar: **Seçenek A (tek tıklamalı döngü)** · Durum: **uygulandı**, varsayılan
KAPALI (bulut kısmı `HEKTOR_CLOUD_SECOND_OPINION` ile açılır). Gerçek CLI ile henüz tur
DENENMEDİ — testler sahte sağlayıcıyla._

## 1 · İstek ve kararlar

İstek: eğitimden sonra cevapları GPT/Claude kontrol etsin, sistem sürekli döngüde dönsün.

| Soru | Karar (2026-10-10) |
|---|---|
| Ne kontrol edilsin? | **İkisi:** eğitim sonrası değerlendirme cevapları (K1) + öğrenme adayları (K2) |
| Bağlantı | **Abonelik CLI** (`claude -p`), **her istek ayrı insan tıklaması** |
| Bulutun yetkisi | **Yalnız rapor** — karar, terfi ve eğitime alma kullanıcıda |
| Eğitim | **A:** eğitim onayı insanda kalır (Kural 8 + Kademe 2 değişmedi) |
| K2'de tık başına aday | **1** (toplu gönderim yok — CLAUDE.md "toplu/otomatik bulut döngüsü yasak") |
| Bildirim | Yalnız web kutusu ("Bekleyen kararlar"); Windows bildirimi yok |
| K1 hakemi | Yalnız Claude Code CLI (Codex sağlayıcısı yok; eklemek ayrı iş) |

Reddedilen **B** (eğitim dahil tam otomatik) Kural 8'i, "eğitim öncesi Kademe 2" kuralını ve
"eğitim öncesi ağırlık sor" tercihini kaldırmayı gerektiriyordu; v5 regresyonunun yaşandığı
durumu yeniden açıyordu.

## 2 · Bağlayıcı kısıtlar

- Sağlayıcı şartları (`docs/TASARIM_FAZ3_BULUT.md` §1): abonelikle otomatik/toplu erişim ve
  bulut çıktısıyla eğitim yasak → her istek bir insan tıklaması; bulut metni eğitim verisine
  girmez (`app/cloud/policy.py · cloud_origin_lines` → NO-GO; eğitim/öğrenme kodu
  `app.cloud.judge`'ı içe aktarmaz — statik test).
- Bulut puanı `ai_review.json` / ayrı hakem kaydı olarak durur; kör insan incelemesinin,
  `finalize` kararının ve terfinin yerine geçmez.

## 3 · Döngü

```
araştırma → RAG → kart → aday veri ─┐   (otomatik, değişmedi)
                                    ▼
   [eğitime hazır mı?] kolay akış hazırlığı (veri · sohbet verisi · ağır iş · kod · Kademe 2)
                                    ▼
   ★ Bekleyen kararlar: "Kademe 2 derin av gerekli" ya da "Eğitim hazır — onay sizde"
   ★ İnsan: ağırlık onayı + tek tık eğitim onayı                         (Kural 8 aynen)
                                    ▼
   ★ "Eğitim bitti — hazırla + karşılaştır (tek tık)"
             dönüşüm → DOĞRULANINCA karşılaştırma kendiliğinden başlar (aynı kurallar)
                                    ▼
   ★ "Değerlendirme hazır — kör inceleme bekliyor"  → [K1] bulut hakem (parça başına tık)
             bulut puanı = ayrı AI incelemesi → İNSAN kör inceleme → karar (finalize)
                                    ▼
                               döngü başa döner
   ★ Öğrenme Havuzu: aday başına [K2] bulut kontrolü (tutarlı/şüpheli + gerekçe; rapor)
```

## 4 · Uygulama

| Parça | Yer | Not |
|---|---|---|
| K1/K2 hakem | `app/cloud/judge.py` | İkinci görüşle aynı önizleme → özet eşleşmesi → tık → `claude -p` altyapısı; ortak günlük kota |
| Bekleyen kararlar | `app/orchestration/loop_state.py` · `GET /api/loop/pending` | Salt-okuma; diskten türetilir (yeniden açılışta korunur); bölüm başına 10 sn süre sınırı |
| Hazırla → karşılaştır zinciri | `candidate_jobs.start_job(..., then_compare=True)` · `_chain_compare` | Dönüşüm doğrulanmadan başlamaz; kilit/kira/etiket kuralları aynı; başlatılamazsa sebep iş kaydında |
| Web | `/api/cloud/judge/*` (POST'lar `require_human`) · Öğrenme Havuzu "Bekleyen kararlar" kutusu · karşılaştırma ve aday kartlarında K1/K2 düğmeleri · aday hattında "sonra karşılaştır" kutusu | |
| Ayar | `HEKTOR_CLOUD_JUDGE_MAX_CHARS` (varsayılan 30000) | K1 parça boyutu |

**K1 ayrıntıları:** kör paket (rol/model kimliği ve mühürlü eşleme YOK; anahtarla otomatik
puanlanan sorular yok) + kilitli ölçüt çapaları ve kritik hata tanımları. Paket parça
sınırını aşarsa sorular bölünmeden parçalara ayrılır; **her parça ayrı önizleme + ayrı tık +
ayrı kota**. Cevap katı JSON olarak istenir; her soru/etiket 0–4 tam sayıyla doğrulanır,
ayrıştırılamazsa parça `failed` olur (ham metin kayıtta kalır), yeni tıkla tekrar gönderilir.
Tüm parçalar bitince birleşik puan **bir kez** `submit_ai_review` ile yazılır
(`method` "K1 …", `reviewer_model` "bulut:…"). Engeller: var olan (K1 olmayan) AI incelemesi
**ezilmez**; **gizli final setinin** (`role_requested=final`) cevapları buluta gitmez.

**K2 ayrıntıları:** soru + aday hedef metni + yerel deterministik kontrol özeti gider; kaynak
parçaları gitmez. Cevabın ilk satırı `KARAR: tutarli|supheli` → `tutarli | supheli | belirsiz`.
Sonuç `storage/cloud/judge/jd_*.json`'da durur; aday kaydı (durum, hedef, doğrulama)
**değişmez** (test). Eval sızıntısı işaretli ya da hariç tutulan aday gönderilmez.

## 5 · Tasarım taslağından sapmalar

- **Eğitim bitişi → dönüşüm kendiliğinden başlamaz.** Taslakta "eğitim → hazırla →
  karşılaştır (otomatik zincir)" vardı. Uygulamada eğitim bitince "Bekleyen kararlar" tek tık
  önerir; o tıktan sonrası (dönüşüm → karşılaştırma) zincirlidir. Sebep: eğitim ayrık bir
  süreçte biter ve dönüşüm de ağır iştir; masaüstü yük kuralı ve tamamlanma doğrulaması
  (ağırlık özeti) insanın gördüğü aday hattında kalsın.
- K2, "faydalı" işaretli sohbet cevaplarına değil **öğrenme adaylarına** (Öğrensin/Düzelt)
  uygulanır — "Faydalı" aday üretmez (`app/feedback/learning.py`), eğitime giden metin adaydır.

## 6 · Bilinen açık

- 2026-10-10 UI denemesinde, arka plan döngüleri AÇIK ve gözetimsiz sürücü (`codex exec`)
  koşarken `GET /api/loop/pending` bir kez 2 dk+ yanıtsız kaldı; döngüler kapalıyken 0,3 sn.
  Kök neden bulunamadı (aynı bölümler ayrı süreçte <0,5 sn). Önlem: bölümler paralel ve
  süre sınırlı okunur, takılan bölüm "zaman aşımı" olarak görünür. Aynı `readiness()` kolay
  akış ekranında da çağrılır — orada süre sınırı yok.

## 7 · Test

`tests/test_cloud_judge.py` (çevrimdışı, sahte sağlayıcı): tıklamasız/özetsiz gönderim yok ·
K1 parça başına tık · AI incelemesi karar açmaz · var olan AI incelemesi ezilmez · final set
gitmez · ayrıştırılamayan cevap → failed + tekrar · K2 tek aday, aday kaydı değişmez ·
sızıntılı aday gitmez · ortak kota · eğitim kodu `judge`'ı içe aktarmaz + bulut satırı NO-GO ·
web POST'ları `require_human` · bekleyen kararlar diskten türetilir, iş başlatmaz, takılan
bölüm diğerlerini gizlemez · zincir aynı kurallarla başlar, başlayamazsa sebep kaydedilir.

**Sıradaki elle deneme:** gerçek `claude -p` ile tek K1 parçası + tek K2 adayı.
