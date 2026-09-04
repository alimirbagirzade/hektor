# 📚 İndirilen Makaleler — neden önemli ve Achilles'e ne katar

> Bu klasördeki PDF'ler **gerçek arXiv makaleleridir** (uydurma değil, ID'leri
> doğrulandı). Claude tarafından, Achilles'in geliştirme yönüne (Markov + entropi +
> doğru LoRA reçetesi + anlama doğrulama) uygun seçildi.
>
> **Senin yapacağın:** Bu özeti oku → işine yarayanları **web arayüzünden makalelere
> yükle** (PDF yükle) → RAG bunları kart + formül + sentez için kullansın.
>
> _Tarih: 2026-06-16 · Claude (otomatik araştırma)_
>
> 🗂️ **PDF'ler `ragagirdi/` alt klasöründe.** Bu özet ayrıca git'te kayıtlı:
> repo `docs/kaynaklar/00_NEDEN_ONEMLI_oku_once.md` (PDF'ler değil, yalnız bu kayıt push'lanır).

---

## 1. RAFT — Adapting Language Model to Domain-Specific RAG ⭐ (en kritik)
- **Dosya:** `ragagirdi/2403.10131_RAFT_Domain_Specific_RAG.pdf`
- **arXiv:** https://arxiv.org/abs/2403.10131 (Zhang, Patil ve ark., 2024)

**Ne anlatıyor (basitçe):** Bir modeli RAG ile birlikte çalışacak şekilde nasıl
eğiteceğini gösteriyor. Anahtar fikir: modele bir soru + getirilmiş belgeler verilir;
model **işe yaramayan "distractor" belgeleri görmezden gelmeyi** ve cevabı **ilgili
belgeden birebir alıntılayarak** vermeyi öğrenir.

**Neden önemli (BİZİM İÇİN kritik):** Bizim **v5 LoRA adaptörümüz tam burada battı.**
v5'i "pasaja göre cevapla" mantığıyla eğitmiştik → model olmayan pasaja atıf yapıp
uydurdu (REDDEDİLDİ). RAFT, **doğru reçeteyi** veriyor: bağlamı kullan, distractor'ı
reddet, alıntıla, uydurma.

**Achilles'e ne katar:**
- Bir sonraki LoRA eğitiminin **veri reçetesini düzeltir** (bizim `raft_discipline_seed.jsonl`
  seed'imizin akademik temeli).
- v5 regresyonunu önler → eğitim 47 saat boşa gitmez.
- Doğrudan: `docs/PROTOKOL_RAG_LORA_ZINCIR.md` §3c (RAFT veri reçetesi).

---

## 2. Hidden Markov Models Applied To Intraday Momentum Trading ⭐ (Markov vizyonu)
- **Dosya:** `ragagirdi/2006.08307_HMM_Intraday_Momentum_Trading.pdf`
- **arXiv:** https://arxiv.org/abs/2006.08307

**Ne anlatıyor:** Gizli bir "momentum durumu" (latent state) tanımlayan bir HMM ile
intraday momentum trading. Önemli özellik: **zaman gecikmesi (lag) olmadan** rejim
değişim noktalarında sinyal işaretini doğru yakalayabiliyor.

**Neden önemli:** Senin **"Markov zinciri üzerinde yeni indikatör"** vizyonunun tam
çekirdeği. "No time-lagging" kısmı bizim için altın değerinde — çünkü CLAUDE.md
Kural 4 (look-ahead yasak) ile çelişmeden rejim değişimi yakalamanın yolunu gösteriyor.

**Achilles'e ne katar:**
- **L5 kompozisyon sınavı** için somut Markov indikatör tasarımı (rejim olasılığı → filtre).
- `continuous-learning.sh` sentez konusu (HMM rejim değişimi) için kaynak materyal.
- "Rejim Geçiş Entropisi (RTE)" fikrimizin HMM tarafını besler.

---

## 3. Transfer Entropy & Kramers-Moyal — Crisis-Driven Market Dynamics ⭐ (entropi vizyonu)
- **Dosya:** `ragagirdi/2507.09554_Transfer_Entropy_Market_Dynamics.pdf`
- **arXiv:** https://arxiv.org/abs/2507.09554 (2025)

**Ne anlatıyor:** Endeksler (Nasdaq, WTI petrol, altın, dolar) arası bilgi akışını
**Transfer Entropi** ile haritalıyor. Kriz dönemlerinde (COVID, Rusya-Ukrayna) ortalama
transfer entropi %35 ve %28 arttığını ölçüyor — yani entropi **rejim değişiminin erken
sinyali**.

**Neden önemli:** Senin **"entropi"** vizyonunun pratik karşılığı. Entropi sadece bir
kavram değil, **ölçülebilir bir rejim/risk göstergesi** olarak kullanılabiliyor.

**Achilles'e ne katar:**
- **Entropi-temelli indikatör** hipotezi (L5): yüksek transfer entropi → rejim belirsiz →
  pozisyonu kıs.
- Markov (rejim) + Entropi (belirsizlik) **birleşimi** = senin "olasılıksal istatistik
  zinciri / vektörel çıktı" beklentine en yakın iki kavramın füzyonu.

---

## 4. Entropy Analysis of Financial Time Series (temel/araç kutusu)
- **Dosya:** `ragagirdi/1807.09423_Entropy_Analysis_Financial_Time_Series.pdf`
- **arXiv:** https://arxiv.org/abs/1807.09423

**Ne anlatıyor:** Finansal zaman serilerinde farklı entropi ölçülerinin (Shannon, örneklem,
permütasyon entropisi) sistematik analizi — temel/referans niteliğinde.

**Neden önemli:** Yukarıdaki uygulamalı makalelerin **matematiksel temeli**. Hangi entropi
ölçüsü neyi yakalar, nasıl hesaplanır — bunu doğru kurmazsak indikatör yanlış olur.

**Achilles'e ne katar:**
- L3/L4 sınavlarımıza **entropi referans hesapları** eklemek için formül kaynağı
  (`app/verification/exams/registry.py`'ye yeni gösterge olarak).
- "Rejim Geçiş Entropisi" indikatörünün doğru entropi tanımı.

---

## 📌 Öncelik sırası (önerim)
1. **RAFT (2403.10131)** — eğitim reçetesini düzeltir, en acil.
2. **HMM Intraday (2006.08307)** + **Transfer Entropy (2507.09554)** — Markov+entropi
   indikatör sentezinin çekirdeği.
3. **Entropy Analysis (1807.09423)** — matematiksel temel/araç kutusu.

## Sıradaki indirmeler (loop — gelecek turlarda eklenecek)
Permütasyon entropi & yasak örüntüler (kriz tespiti), Wasserstein rejim kümeleme,
RAG değerlendirme (RAGalyst), olasılıksal zaman-serisi tahmini. İşe yarayanlar buraya
eklenip bu dosyaya not düşülecek.

> ✅ **Kodda ilerleme (2026-06-16):** Permütasyon entropisi (Bandt-Pompe) **gösterge olarak
> uygulandı** — `app/trading/indicators.py` (`PERMENTROPY`) + L3/L4 sınav registry'si. Yani
> #4 makalenin (Entropy Analysis) permütasyon-entropi kısmı koda dönüştü; makale indirilince
> akademik temeli sağlamlaşır. ENTROPY (yönsel Shannon) + PERMENTROPY (ordinal örüntü) ikilisi
> entropi vizyonunun ilk iki yapı taşı.
