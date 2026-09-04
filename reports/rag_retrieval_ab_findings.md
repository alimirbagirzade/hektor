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
> gelirse / kısa chunk'larda işe yarayabilir) ama KAPALI. `ACHILLES_RAG_FLASHRANK`.

## 3. Sonuç ve karar

Bu korpus + GPU'suz donanımda **dense-only hem en hızlı hem en doğru**:
- hibrit/rerank/RRF kaliteyi düşürüp ~2.2s ekliyor (uzun sorgularda BM25 araması yavaş).
- cross-encoder CPU'da sorgu başına >15s → kullanılamaz (kalite kazancı olsa bile hız ihlali).

**Karar:** canlı sistemde `ACHILLES_RAG_RERANK=false` + `ACHILLES_RAG_HYBRID=false`
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
HIZLI (warm) → **ENABLE** (.env ACHILLES_RAG_ROUTER=true). Tek operasyonel not: ilk lexical
sorgu BM25'i kurar (~171s, tek-seferlik/process) → startup warm-up follow-up önerilir.
