# RAG Retrieval A/B Bulguları — 2026-06-20

Ölçüm araçları: `scripts/rag_retrieval_ab.py`, `scripts/rag_ab_multi.py`,
`scripts/rag_ab_crossenc.py`. Metrik: **makale-düzeyi self-retrieval** (bilgi
kartından türetilen sorgu o kartın makalesini geri getiriyor mu, hangi sırada) —
recall@1/3/5/10 + MRR + sorgu başına gecikme. Korpus: 153 makale / ~91.705 chunk.
Donanım: i7-1165G7, **GPU yok**. Çevrimdışı, deterministik (Kural 2/6).

## 1. KRİTİK BUG (düzeltildi) — canlı RAG ~18s/sorgu + BM25 hibrit ölü

- `ChromaStore.get_all()` tüm 91.705 chunk id'sini TEK SQL `get()`'ine koyuyordu →
  SQLite **"too many SQL variables"** → get_all KOMPLE BAŞARISIZ → BM25 korpusu hiç
  kurulamıyordu (hibrit sessizce dense-only'e düşüyordu).
- `get_corpus_bm25()` her çağrıda YENİ `ChromaStore()` yaratıyordu → her sorgu SOĞUK
  koleksiyon yükleme + `count()` (~10-18s) ödüyordu; get_all hatası cache'i
  doldurmadığından her sorgu yeniden deniyordu.
- **Fix:** `get_all()` sayfalama (limit/offset, page=5000) + modül-düzeyi paylaşılan
  ChromaStore. Ölçülen: `retrieve()` **~18s → ~150-650ms (~30-100×)**, BM25 hibrit
  artık çalışıyor.

## 2. Config A/B (BM25 fix sonrası, 70 sorgu)

| config | recall@1 | recall@5 | MRR | gecikme p50 |
|---|---|---|---|---|
| **dense_only** | **68.6%** | 72.9% | **0.702** | **234 ms** |
| hybrid+rerank | 64.3% | 71.4% | 0.665 | 2216 ms |
| rrf | 67.1% | 72.9% | 0.689 | 2286 ms |
| dense + cross-encoder (bge-reranker-base) | — | — | — | **>15.000 ms** (CPU'da kullanılamaz) |
| dense + FlashRank (ms-marco-MiniLM-L-12, ONNX-int8) | 67.5% | 70.0% | 0.688 | **12.388 ms** |

> **FlashRank (Zincir 3, derin araştırma sonrası ölçüldü):** Web-araştırma FlashRank'i CPU'da
> ~30-100ms olarak veriyordu (M-serisi Mac / kısa pasaj). Bu GPU'suz i7-1165G7'de 40 GERÇEK
> akademik aday (≤1200 char) ile **TEMİZ koşulda bile ~12,4 s/sorgu** VE recall@1 dense'den
> DÜŞÜK (67.5 < 70.0). Sonuç KESİN: **bu donanımda hiçbir cross-encoder reranking (bge >15s,
> FlashRank ~12s) kullanılamaz** — dense-only kazanır. FlashRank opt-in olarak KODDA (GPU
> gelirse / kısa chunk'larda işe yarayabilir) ama KAPALI. `HEKTOR_RAG_FLASHRANK`.

## 3. Sonuç ve karar

Bu korpus + GPU'suz donanımda **dense-only hem en hızlı hem en doğru**:
- hibrit/rerank/RRF kaliteyi düşürüp ~2.2s ekliyor (uzun sorgularda BM25 araması yavaş).
- cross-encoder CPU'da sorgu başına >15s → kullanılamaz (kalite kazancı olsa bile hız ihlali).

**Karar:** canlı sistemde `HEKTOR_RAG_RERANK=false` + `HEKTOR_RAG_HYBRID=false`
(dense-only). Her sorgu ~234ms, BM25 soğuk-başlatma yok. **Geri alınabilir** (.env satırlarını kaldır).
BM25 sayfalama fix'i kodda kalıcı (hibrit ileride açılırsa veya başka kullananlar için doğru).

**Uyarı:** metrik semantik (kart-türevi) sorguları kayırır; keyword-ağırlıklı kullanıcı
sorularında BM25 hibrit teorik olarak yardımcı olabilir → ileride keyword golden-set ile
yeniden değerlendirilebilir. Cevap-üretimi (LLM, qwen3:4b) gecikmesi retrieval'dan AYRI ve
CPU/model-bağlı (mimari kısıt; bu çalışmanın kapsamı dışı).

## 4. Keyword golden-eval — ROUTER VALIDATED (2026-06-21, SQLite-BM25 ile ölçüldü)

`scripts/rag_keyword_eval.py` (korpustan ayırt edici nadir terim → kısa keyword sorgu;
golden = makale). 40 sorgu, BM25 korpusu SQLite'tan (94.104 chunk), öğrenme döngüsü dönerken.

| config | recall@1 | recall@5 | recall@10 | MRR | gecikme p50 |
|---|---|---|---|---|---|
| dense_only | 30.0 | 50.0 | 57.5 | 0.386 | **368 ms** |
| **router (lexical→konveks-hibrit)** | **37.5** | 50.0 | **77.5** | **0.454** | **12.577 ms** |

**Sonuç:** Keyword/exact-term sorgularda router hibridi BELIRGIN kazanıyor (recall@10 **+20 puan**,
recall@1 +7.5, MRR +18%). "Sorgu-tipine göre yönlendir" (BEIR/Bruch) gerçek veriyle DOĞRULANDI —
bu, semantik metrikte dense'in kazandığı bulgunun TERSİ DEĞİL, tamamlayıcısı (rejim farkı).

**AMA router ~34× yavaş** (özel BM25Index.search 94k chunk'ta O(korpus)). Bu, araştırmanın işaret
ettiği yavaş-BM25 → **BM25S** (eager sparse scoring, ~ms) ile çözülür → router hem daha doğru hem
hızlı olur. KARAR: BM25S uygulanana kadar router opt-in/KAPALI; BM25S sonrası hız ölçülüp enable.
Ayrıca BU ÖLÇÜM mümkün oldu çünkü BM25 korpusu artık SQLite'tan kuruluyor (eşzamanlı döngüyle
çakışmadan; bkz commit "BM25 korpusu SQLite'tan kur").

### 4b. Gecikme DÜZELTMESİ — router 12.6s'i contention'dı, BM25 DEĞİL (izole ölçüm)

İlk verdict'te router'ın 12.6s gecikmesini "yavaş BM25 → BM25S gerek" diye yorumladım — YANLIŞ.
İzole ölçüm (öğrenme döngüsü DURDURULDU, dense/Ollama YOK, sadece bm25.search 94.104 chunk):
**BM25.search mean=5.2ms, p50=2.8ms** (nadir-2-kelime sorgu). BM25Index zaten ters-indeks
kullanıyor → yalnız sorgu-terimini İÇEREN doc'ları skorlar (nadir terim df≤2 → ~µs).

Yani 12.6s = **dense sorgu-embed'inin Ollama'da döngünün qwen3:4b kart-üretimine TAKILMASI**
(dense_only düşük-contention penceresinde, router yüksek-contention penceresinde ölçüldü).
TEMİZ ortamda router gecikmesi ≈ dense (~370ms) + BM25 (~3ms) + füzyon ≈ **~370ms (hızlı)**.

**DÜZELTİLMİŞ KARAR:** BM25S GEREKMEZ. Router keyword'de hem DAHA İYİ (recall@10 +20p) hem
HIZLI (warm) → **ENABLE** (.env HEKTOR_RAG_ROUTER=true). Tek operasyonel not: ilk lexical
sorgu BM25'i kurar (~171s, tek-seferlik/process) → startup warm-up follow-up önerilir.

## 5. YENİDEN ÖLÇÜM + KARAR DÜZELTMESİ — 2026-09-08 (router ENABLE, α 0.7→0.3; hibrit "kapat" kararı DÜŞTÜ)

§3 ve §4b'nin kararları uygulanmamıştı; uygulamadan önce yeniden ölçtüm ve **ikisi de
kısmen yanlış çıktı**. Eski bölümler bilerek duruyor (karar geçmişi denetlenebilir kalsın).

**Ölçüm koşulları (geçerlilik).** Korpus ARADA YENİDEN KURULMUŞ: **161 makale / 13.103
chunk** (§1-4'te 153 makale / ~91-94k chunk) → chunk/makale ~615'ten ~81'e düştü, eski
MUTLAK sayılarla kıyaslanamaz; bu yüzden tüm config'ler baştan ölçüldü. Temiz ortam
otomatik kapıyla doğrulandı (ağır süreç yok — lora-eval/peft/pytest/mypy/hektor CLI;
Ollama'da yüklü model yok; toplam CPU < %30, üst üste 2 yoklama + 30 s dinlenme) — §4b'nin
ders çıkardığı çekişme hatası tekrarlanmasın diye. `HEKTOR_ALLOW_FAKE_EMBEDDINGS=false`
ve her koşu `# embedder mode:` basıyor; iki koşu da **ollama** raporladı (sahte embedding
sessizce devreye girip "sonuç" üretmedi). Rastgelelik yok: sorgu kümeleri korpustan
sıralı/deterministik türer, limitler sabit (semantik 70, keyword 50). BM25 kurulumu artık
**3-4 s** (94k chunk'ta 171 s idi) → §4b'nin "startup warm-up şart" notu bu boyutta konu dışı.

### 5a. Semantik rejim (kart-türevi self-retrieval, 70 sorgu) — METRİK DOYDU

| config | recall@1 | recall@5 | recall@10 | MRR | gecikme p50 |
|---|---|---|---|---|---|
| dense_only | 100.0 | 100.0 | 100.0 | 1.0 | **193 ms** |
| hybrid+rerank (KOD VARSAYILANI = canlı) | 100.0 | 100.0 | 100.0 | 1.0 | **1074 ms** |
| rrf | 100.0 | 100.0 | 100.0 | 1.0 | 1011 ms |
| router (hybrid=false) | 100.0 | 100.0 | 100.0 | 1.0 | 212 ms |
| router (hybrid=true) | 100.0 | 100.0 | 100.0 | 1.0 | 214 ms |

**Bu metrik artık AYIRT ETMİYOR:** yeni chunk'lamada beş config de %100/MRR 1.0. §2'nin
"dense-only en doğru (68.6 > 64.3)" bulgusu bu korpusta ARTIK ÖLÇÜLEMİYOR — yani §3'ün
`HEKTOR_RAG_HYBRID=false` kararının **dayanağı çökmüştür** (kalite farkı yok; yalnız
gecikme farkı var). Doymuş metrikten "eşit kalite" sonucu çıkarılabilir, "hibrit zararsız"
sonucu ancak keyword rejimiyle birlikte okunabilir (5b). Daha zor bir semantik golden-set
gerekiyor — açık iş.

### 5b. Keyword rejimi (nadir-terim golden, 50 sorgu) — asıl karar burada

| config | recall@1 | recall@5 | recall@10 | MRR | gecikme p50 |
|---|---|---|---|---|---|
| dense_only | 42.0 | 64.0 | 72.0 | 0.5155 | 35-90 ms |
| hybrid+rerank (canlı) | 66.0 | 86.0 | 88.0 | 0.7475 | 477 ms |
| router α=0.7 (**kod varsayılanı**) | 46.0 | 72.0 | 84.0 | 0.5799 | 452 ms |
| router α=0.5 | 70.0 | 92.0 | 94.0 | 0.7863 | 459 ms |
| **router α=0.3 (SEÇİLEN)** | **84.0** | **94.0** | **94.0** | **0.8733** | **459 ms** |
| router α=0.15 | 84.0 | 94.0 | 96.0 | 0.8787 | 447 ms |
| router α=0.0 (saf BM25 sıralaması) | 84.0 | 94.0 | 96.0 | 0.8743 | 464 ms |

1. **§4b'nin "router'ı aç, hibridi kapat" kararı VARSAYILAN α ile YANLIŞ olurdu:** router
   α=0.7 keyword'de canlı hibrit+rerank'in ÇOK ALTINDA (46.0 vs 66.0; MRR 0.58 vs 0.75) ve
   gecikme aynı. §4 router'ı yalnız dense ile kıyasladığı için bunu göremedi.
2. **Sorun yönlendirmede değil, füzyon ağırlığında.** α dense ağırlığıdır; keyword
   rejiminde dense EN ZAYIF taraftır (42.0) → ona 0.7 vermek BM25'in doğru cevabını bastırır.
   Ağırlık tersine çevrilince router her şeyi geçiyor: **recall@1 66→84, MRR 0.7475→0.8733**,
   gecikme aynı (477→459 ms).
3. α=0.15/0.0 ile α=0.3 arasındaki fark gürültü içinde (MRR Δ0.005; recall@10 farkı = 1
   sorgu). **α=0.3 seçildi**: ölçülen kalite α=0.15 ile aynı sayılır ama dense'e biraz daha
   ağırlık bırakır. Gerekçe ölçümün DIŞINDA ve açıkça yazılıyor: `classify_query` 1-2
   kelimelik HER sorguyu lexical sayar (kavramsal kısa sorgular dahil), bu golden-set ise
   nadir-terim ağırlıklı — yani BM25'i kayırır. α=0.3 o sınıflandırma zaafına karşı yastık.
   α=0.15 bu sette ölçülen optimumdur; tek satırlık geri alınabilir değişiklik.

### 5c. Bayrak çakışması ÖLÇÜMLE çözüldü (rerank / hybrid / router)

Kod (`app/memory/reranking_retriever.py:90-126`):
- `rag_rerank` bir **ana kapıdır**: `false` ise `retrieve()` düz dense döner ve **router
  hiç çalışmaz**. Yani §3'ün `RERANK=false` kararı ile §4b'nin `ROUTER=true` kararı
  BİRLİKTE UYGULANAMAZ — biri diğerini iptal eder. Çelişkinin kaynağı buydu.
- Router açıkken `_convex_hybrid_retrieve` **`self.hybrid`'e hiç bakmaz** → `rag_hybrid`
  router altında ÖLÜ bayraktır. Kanıt varsayım değil ölçüm: `router(hybrid=false)` ve
  `router(hybrid=true)` her iki rejimde de rakamı rakamına aynı çıktı (keyword
  46.0/72.0/84.0/0.5799; semantik 100/1.0).
- Router açıkken heuristik `Reranker` iki kolda da uygulanmaz (semantik → saf dense;
  lexical → konveks füzyon). Yani §3'ün "heuristik rerank olmasın" AMACI, `rag_rerank=true`
  ile korunur; `rag_rerank` burada yalnız ana kapıdır.

Sonuç: `RERANK=true` (kapı) + `ROUTER=true` + `ALPHA=0.3`. `rag_hybrid` router altında
etkisiz olduğu için **açık bırakıldı** (kod varsayılanı `true`): router kapatılırsa sistem
keyword'de en kötü config'e (dense-only, recall@1 42.0) sessizce düşmesin.

### 5d. Karar (.env — kod varsayılanı DEĞİŞTİRİLMEDİ, geri alınabilir)

```
HEKTOR_RAG_ROUTER=true         # §5b: keyword recall@1 66→84, semantik gecikme 1074→212 ms
HEKTOR_RAG_ROUTER_ALPHA=0.3    # varsayılan 0.7 keyword'de ters tepiyor (46.0)
HEKTOR_RAG_RERANK=true         # ANA KAPI: false yapılırsa router hiç çalışmaz (§5c)
```

Canlı sisteme etkisi (ölçülen, tahmin değil):
- uzun/semantik sorgu: **1074 ms → 212 ms (~5×)**, kalite eşit (iki config de doymuş metrikte %100)
- kısa/keyword sorgu: **recall@1 66.0 → 84.0**, MRR 0.7475 → 0.8733, gecikme 477 → 459 ms

Yani her iki rejimde de **kayıp yok** — §3'ün dense-only kararı uygulansaydı keyword
recall@1 42.0'a düşerdi (-24 puan).

### 5e. `rag_graph` (SPRIG-lite PPR + RRF) — İLK KEZ ÖLÇÜLDÜ, ELENDİ

Bu bayrak hiçbir raporda ölçülmemişti (korpus 94k chunk iken pahalıydı; 13k'da ölçülebilir
hâle geldi). Keyword rejimi, aynı 50 sorgu:

| config | recall@1 | recall@5 | recall@10 | MRR | gecikme p50 |
|---|---|---|---|---|---|
| dense_only | 42.0 | 64.0 | 72.0 | 0.5155 | 36 ms |
| router α=0.3 | 84.0 | 94.0 | 94.0 | 0.8733 | 470 ms |
| graph (PPR+RRF) | **38.0** | 64.0 | 72.0 | **0.4981** | **8829 ms** |

Graf çok-hop recall vaat ediyordu ama bu korpusta dense-only'in bile ALTINDA (recall@1 38.0
< 42.0; MRR 0.4981 < 0.5155) ve sorgu başına **8,8 saniye** — router'dan ~19× yavaş.
KARAR: `HEKTOR_RAG_GRAPH` KAPALI kalır (kod varsayılanı zaten `false`; .env'e satır eklenmedi).

**Tekrarlanabilirlik:** α=0.3 bu ikinci, bağımsız koşuda birebir aynı çıktı
(84.0 / 94.0 / 94.0 / MRR 0.8733) → ölçüm deterministik, rastgelelik yok.

### 5f. Bu koşuda ölçülmeyenler

- `rag_cross_encoder` / `rag_flashrank`: §2-§3'te elenmişti (CPU'da >15 s / ~12 s) → KAPALI.
- `rrf` bu koşuda da ölçüldü: router'a göre ~5× yavaş (1011 ms) + semantikte fark yok → KAPALI.
- `rag_contextual_embed`: **ölçülmedi.** Önce `hektor reindex-contextual` ister (tüm korpusu
  Ollama ile yeniden embed eder, AĞIR) → eğitim/eval hattı bu makinede sıradayken yapılmadı.
  Açık iş olarak HANDOFF'ta.
