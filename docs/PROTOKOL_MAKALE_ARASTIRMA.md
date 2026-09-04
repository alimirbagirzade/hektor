# Protokol — Makale Araştırma Ajanı (periyodik kaynak besleme)

_Son güncelleme: 2026-06-17 (v1.0)._

> **Tek cümle:** Belirli aralıklarla (varsayılan **haftada 1**) devreye girip Achilles'in
> RAG/LoRA gelişimine yarayacak **YENİ, GERÇEK** arXiv makalelerini bulur, doğrular, yerel
> klasöre indirir ve "neden önemli" indeksini günceller. **Eğitim başlatmaz** — yalnız besler.

> **🧠 Bağlam:** Bu ajan, [`PROTOKOL_RAG_LORA_ZINCIR.md`](PROTOKOL_RAG_LORA_ZINCIR.md)
> zincirinin **giriş ucudur**: RAG = bilgi, makaleden gelir. Sistem kendi kendine arxiv
> ÇEKMEZ (loop'ta kapalı); makale akışı ya kullanıcının elle yüklemesiyle ya da **bu ajanın
> seçip indirdiği** kaynaklarla gelir. Kullanıcı indirilenleri inceleyip web arayüzünden RAG'a
> yükler.

---

## 1. Sıklık (frekans)

- **Varsayılan: haftada 1 — Pazartesi ~03:08** (cron `0 3 * * 1`; sistem çakışma-önleme için
  birkaç dk kaydırabilir).
- **Gerekçe (neden günlük değil):** Bu niş alanlarda gerçekten yeni/değerli arXiv makalesi
  günlük değil ~aylık birkaç tane çıkar; günlük tur çoğu zaman boş döner veya niteliksiz
  makale toplama riski yaratır. Darboğaz makale arzı değil, kullanıcının inceleyip yüklemesi +
  loop'un işlemesidir. Her tur tam bir Claude araştırma oturumu (maliyet). Ajan zaten "yeni
  değerli yoksa indirme" ilkesiyle çalışır → boş turlar ucuzdur.
- **Alternatifler:** Daha taze isteniyorsa `0 3 */3 * *` (her 3 günde 1). Eğitim görünür
  biçimde tıkanırsa veya yeni bir eğitim döngüsü başlıyorsa **elle de tetiklenebilir** (§5).

## 2. Hedef klasör ve dosyalama

```
C:\Users\sevinc\Desktop\RAG Kaynak\Gerekli kaynaklar\
  ├─ 00_NEDEN_ONEMLI_oku_once.md   ← indeks (her makale: neden önemli + Achilles'e ne katar)
  ├─ <id>_<Kisa_Baslik>.pdf         ← YENİ indirilenler (RAG'a eklenmeyi bekler)
  └─ RAG Eklendi\                   ← kullanıcının zaten RAG'a yüklediği makaleler (DOKUNMA)
```
- Yeni PDF'ler **köke** iner. Kullanıcı RAG'a yükledikçe `RAG Eklendi\`'ye taşır.
- Dosya adı: `<arxiv_id>_<Kisa_Ingilizce_Baslik>.pdf`.

## 3. Mutlak kurallar

- **Kural 7 — kaynak uydurma YASAK.** Bir makaleyi indirmeden önce arXiv ID'si GERÇEK olmalı.
  İki kademeli doğrulama:
  1. WebFetch ile `https://arxiv.org/abs/<id>` → başlık/yıl sayfadan teyit.
  2. **İndirme = kesin kapı:** sahte ID `https://arxiv.org/pdf/<id>` için 404/HTML döner;
     gerçek PDF `%PDF` ile başlar. **Yalnız ilk 5 baytı `%PDF` ve boyut >40KB olanı tut.**
- **Tekrar yok.** Klasördeki (kök + `RAG Eklendi\`) mevcut PDF adlarından arXiv ID'lerini
  çıkar; bunları ve çekirdek seti (`2403.10131, 2006.08307, 2507.09554, 1807.09423`) atla.
- **Sınır.** Tur başına **3–6 yeni** makale hedefle; kaliteli az > niteliksiz çok. Yeni değerli
  yoksa **hiçbir şey indirme**, indekse "<tarih>: yeni uygun kaynak bulunamadı" notu düş.
- **Eğitim başlatma** (CLAUDE.md Kural 8). Türkçe yaz.

## 4. Odak alanları (eksik açıkları doldur)

| Tema | Konu | Achilles bağı |
|------|------|---------------|
| A | LoRA/PEFT doğru reçetesi + küçük-veri tuzakları (forgetting/overfit/veri kalitesi) | **v5 gerileme onarımı — en öncelikli** |
| B | RAG sadakat / atıf / halüsinasyon / abstention ("bilmiyorum") | v5-fix; Kural 7 |
| C | Markov/HMM/rejim-değişimi (change-point, Markov-switching), look-ahead'siz | L5 Markov vizyonu |
| D | Wasserstein / optimal-transport rejim kümeleme | rejimi veriden öğren |
| E | Permütasyon entropisi / ordinal & yasak örüntüler | `PERMENTROPY` temeli + kriz tespiti |
| F | Belirsizlik-farkında / olasılıksal tahmin (conformal prediction, aralık) | risk-pozisyon boyutlandırma |
| G | Davranışsal yanlılık & belirsizlik (sentiment, EPU, aşırı-güven) | bağlam zenginleştirme |
| H | Sınav/benchmark-temelli LLM anlama & akıl-yürütme değerlendirmesi | L3/L4/L5 + UnderstandingScore |

## 5. Yöntem (akış)

1. Klasördeki mevcut ID'leri oku → tekrar-önleme seti.
2. Eksik temalardan başla; WebSearch ile aday bul.
3. Her aday için WebFetch `https://arxiv.org/abs/<id>` → başlık/yıl doğrula. arXiv'de yoksa atla.
4. İndir (Windows): `powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://arxiv.org/pdf/<id>' -OutFile '<path>' -UserAgent 'Mozilla/5.0 (Achilles research)' -TimeoutSec 90"` (veya `curl -L`). arXiv'e kibar ol: indirme arası ~1 sn.
5. **`%PDF` + >40KB** doğrula; geçersiz/yarım dosyaları **ayrı bir temizleme adımında** ele
   (önce indir+doğrula, sonra ayrı komutta temizle — daha güvenli ve okunur).
6. `00_NEDEN_ONEMLI_oku_once.md`'ye **EKLE** (mevcut içeriği bozma): bugünün tarihiyle
   "🆕 YENİ PARTİ (<YYYY-AA-GG>)" başlığı; her makale: ID + tema harfi + "neden önemli +
   Achilles'e ne katar" (1-2 cümle Türkçe).
7. **Bonus — loop sağlık kontrolü:** `scripts/start-loop.ps1 -Status`; "durdu" ise
   `scripts/start-loop.ps1 -Hours 72` ile başlat. Logda `os error 32` görünürse `UV_NO_SYNC`
   fix'ini (continuous-learning.sh) kontrol et.

## 6. Tetikleme

- **Periyodik:** `mcp__scheduled-tasks` görevi `achilles-gunluk-makale-arastirma`
  (SKILL.md `~/.claude/scheduled-tasks/`). Yalnız Claude **uygulaması açıkken** çalışır;
  kapalıysa sonraki açılışta. İlk kez "Run now" ile araçları ön-onayla.
- **Elle / oturum içi:** `makale-arastirma` ajanını Agent aracıyla başlat (bu protokolü izler),
  ya da bu dosyayı doğrudan uygula.

## 7. Çıktı

Türkçe özet: indirilenler (ID + başlık + tema), atlananlar (sebep), indeks güncellendi mi,
loop durumu.

---
İlişkili: [`PROTOKOL_RAG_LORA_ZINCIR.md`](PROTOKOL_RAG_LORA_ZINCIR.md) ·
[`PROTOKOL_VERI_URETIM.md`](PROTOKOL_VERI_URETIM.md) ·
Ajan tanımı: `.claude/agents/makale-arastirma.md`
