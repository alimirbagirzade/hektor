# Protokol — RAG + LoRA Zinciri (uçtan uca)

_Son güncelleme: 2026-06-16. Teknik detay: [`RAG_LORA_ENTEGRASYON.md`](RAG_LORA_ENTEGRASYON.md)._

> **Tek cümle:** RAG ve LoRA **ayrı inşa edilir**, **zincir olarak birlikte kullanılır.**
> RAG **bilgiyi** (makalelerden) sağlar, LoRA **üslubu/disiplini** sağlar.

> **🧠 ANA FİKİR (anlama):** Achilles bir bilgiyi "anladı" demek = onu **doğru kullanıp**
> ondan **test edilebilir yeni bir şey üretebildi** demektir. Anlama bir yüzdeyle değil bir
> **sınavla** kanıtlanır: uygula (L3) → karşıolgu (L4) → yeni formül üret (L5) → matematik
> geçerli + maliyet dahil backtest/OOS geçir. Geçmeyen "yeni formül" halüsinasyondur (Kural 2).
> Okunabilir özet: README → "Achilles okuduğunu *anladı* mı?".

---

## 1. Mimari — ayrı inşa, zincir kullanım

```
İNŞA (ayrı yollar)
  Makaleler ──indeksle (eğitim YOK)──►  RAG indeks   (BİLGİ)
  Sentetik veri ──eğit (offline)─────►  LoRA adapter (ÜSLUP)

KULLANIM (zincir / "tek beyin")
  Soru → RAG getir (makale parçaları) → [base + LoRA] cevaplar → Cevap
```

- **RAG = BİLGİ.** Vektör DB; eğitilmez. Makale eklenince **anında** güncel. Cevabın
  *gerçekleri* buradan gelir. Kaynak yoksa "bulunamadı" der (uydurmaz — Kural 7).
- **LoRA = ÜSLUP/DİSİPLİN.** Offline eğitilen küçük adapter; **bilgi tutmaz**. Modelin
  *nasıl* akıl yürütüp cevaplayacağını şekillendirir.
- **Karıştırma!** LoRA'yı "bilgi ezberlesin" diye eğitmek (v5'in hatası) RAG'ın işini
  taklit ettirir ve disiplini bozar → reddedilen adapter.

---

## 2. Uçtan uca akış (pipeline)

```
1) Makale yükle ──► 2) %100 ANLA (kart + anlama skoru)
                       │
                       ▼
3) SENTEZLE (Markov-zinciri indikatör fikri) ──► backtest et ──► sentez makalesi (RAG'a geri besle)
                       │
                       ▼
4) RAFT VERİ REÇETESİ (getirilen parçayı kullan / yoksa reddet / rakam uydurma / disiplin)
                       │
                       ▼
5) LoRA EĞİT (lokal CPU detached → ileride remote)
                       │
                       ▼
6) EVAL (base vs adapter, dürüst gate) ──► geçerse ──► 7) RAG+LoRA ZİNCİR devrede
```

---

## 3. Aşamalar

### 3a · %100 Anlama (RAG tarafı)
Hedef: her makale **içerikli bilgi kartına** dönüşsün + anlama skoru alsın.
- Öğrenme döngüsü (`continuous-learning.sh`) kartsız makaleleri işler: `card` → içerikli
  onay → `mastery-*` (anlama skoru). Dürüst metrik: `achilles rag-mastery` (kapsam/anlama %).
- "İçeriksiz kabuk" kart eğitime/sentetik-veriye **girmez**.

### 3b · Sentez — Markov-zinciri indikatör fikirleri
- Her 3 turda research orchestrator bir **hipotez** üretir, **backtest** eder (maliyet dahil,
  look-ahead yok), geçemezse FAIL raporlar, sentez makalesi olarak RAG'a + insana (web) döner.
- **Odak: Markov zinciri / gizli Markov modeli (HMM) ile rejim değişimi** → yeni indikatör/filtre.
  Konu `continuous-learning.sh` adım 3'te; `scripts/enrichment-topics.txt`'te de Markov var.

### 3c · RAFT veri reçetesi (KRİTİK — v5'in eksiği)
LoRA, RAG ile **birlikte** çalışacak şekilde eğitilmeli (Retrieval-Augmented Fine-Tuning):
- **Getirilen bağlamı kullan**, dışındakini uydurma.
- **Kaynak yoksa REDDET** ("yeterli kaynak yok").
- **Yönlendirici/garanti-kâr sorularını REDDET**; rakam uydurma; maliyet/look-ahead disiplinini koru.
- **Adversarial disiplin örnekleri** veriye dahil (garanti-kâr → reddet, backtest şart, maliyet dahil).
- ❌ "Pasaja göre cevapla" reflexini sızdırma (v5 olmayan pasaja atıf yapıp uydurdu).

**📄 Somut SAF örnek:** [`examples/raft_discipline_seed.jsonl`](examples/raft_discipline_seed.jsonl)
— 6 elle yazılmış RAFT/disiplin örneği (eğitim formatında: `{"messages":[system,user,assistant]}`):
bağlam-VAR→kullan+belirsizlik, bağlam-YOK→reddet (uydurma), garanti-kâr→reddet, backtest-şart,
maliyetsiz-rakam→reddet, alakasız-bağlam→"yetersiz". v5'te bu tür örnekler **yoktu** —
sonraki dataset'e bu reçete karıştırılmalı.

### 3d · LoRA eğitim
- Lokal CPU (detached): web/terminal kapansa da sürer. `keep_alive=0` (OOM önle).
- Eğitim BAŞLATMA: web "▶ EĞİTİME HAZIR" tek-tık / `.\scripts\start-train.ps1` / `achilles train --run`.
- Detay + ölçülen süreler: [`EGITIM_PROTOKOLU.md`](EGITIM_PROTOKOLU.md).

### 3e · Eval — dürüst gate (Kural 2)
- `achilles lora-eval <adapter> --eval-set evals/discipline_core.jsonl` → **base vs adapter**
  (adapter'ı gerçekten yükler). Adapter base'den **iyiyse** kabul, **kötüyse REDDET** (terfi etme).
- Not: red-flag sezgisi negasyon-kör → ideal judge **LLM-judge** (yol haritası).

### 3f · RAG+LoRA zincir (devreye alma)
- Adapter onaylanınca: GGUF'a çevir → Ollama'ya `ADAPTER` ile bağla (`RAG_LORA_ENTEGRASYON.md` §1),
  ya da transformers/PEFT ile servis et. Web "01·ARAŞTIRMA" → model menüsünden seçilir.
- Production'a terfi **yalnız kullanıcı onayıyla** + eval geçmişse.

---

## 4. Kurallar (CLAUDE.md)
- Çıktı = **hipotez + test noktası** (tavsiye değil). Eval geçmeden "hazır" deme (Kural 2).
- Maliyet dahil, look-ahead yok, determinizm seed ile, eval/exec yok, kaynak uydurma yok.
- Otomatik ağır eğitim yok — yalnız açık kullanıcı eylemiyle.

## 5. Komutlar (özet)
```bash
uv run achilles rag-mastery            # anlama/kapsam % (dürüst)
bash scripts/continuous-learning.sh 72 # döngü: anla → sentez(Markov) → synth-qa
uv run achilles lora-cloud-prep        # sentetik + kart → lora_sft.jsonl
uv run achilles lora-split             # → train/valid
uv run achilles train --run --backend peft --adapter-name <ad> --iterations <n>
uv run achilles lora-eval <ad> --eval-set evals/discipline_core.jsonl  # base vs adapter
```

İlgili: [`RAG_LORA_ENTEGRASYON.md`](RAG_LORA_ENTEGRASYON.md) · [`EGITIM_PROTOKOLU.md`](EGITIM_PROTOKOLU.md) · [`PROTOKOL_VERI_URETIM.md`](PROTOKOL_VERI_URETIM.md)
