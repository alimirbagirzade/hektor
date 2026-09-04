# RAG Derin Araştırma + Yol Haritası — 2026-06-20

5 paralel web-araştırma ajanı (kaynaklı, adversarial) → bu sistemin (yerel, **GPU yok**,
i7-1165G7 CPU, qwen3:4b LLM + nomic-embed-text, ChromaDB 153 makale / ~91.705 chunk)
"hızlı + kaliteli oku/anla" hedefine göre güncel en iyi pratikler. Her madde **ölçümle**
doğrulanacak (CLAUDE.md Kural 2). Ölçüm metriği ve config kararı: `reports/rag_retrieval_ab_findings.md`.

---

## Zincir 1 — Embedding modeli + Değerlendirme

**Karar: nomic'te KAL.** Daha güçlü küçük modeller (bge-base, gte-v1.5, arctic-embed-m,
mxbai) yalnız **+1–3 nDCG@10** verir (mütevazı, korpusta doğrulanmamış) ve 91k chunk'ı
yeniden-embed maliyeti bunu karşılamaz. Chunk'lar ~300 token → 512 ctx zaten yeter; nomic'in
8192 ctx'i kullanılmıyor. (MTEB: nomic ~62.3, bge-base 63.55/retrieval 53.25 — başa baş.)

**ASIL İŞ: değerlendirmeyi düzelt.** Mevcut "makale-düzeyi self-retrieval" metriği **zayıf/
yanlı**: (1) sorgu karttan, kart makaleden türediği için **sızıntı** (dense'i şişirir),
(2) makale-düzeyi (chunk değil) — kaba, (3) doygun, (4) dense'i kayırır. **Yapılacak:**
chunk'lardan LLM ile Q üret (bir kez) → n-gram kopyalarını ele (decontaminate) → ~50-100'ünü
insan-incele → **dondur** (`data/eval/`) → chunk-düzeyi recall@{1,5,10}+MRR+nDCG@10 + çok-hop
+ "cevap yok" (abstain) örnekleri. Eval anında LLM'siz (deterministik kapı).
Kaynak: RAGAS testset-gen, Evidently RAG-eval, BEIR (2104.08663), sızıntı kritiği (2504.14175).

## Zincir 2 — Hibrit arama + füzyon

**Bulgu:** "dense > hibrit" ÖLÇÜMÜ uzun/semantik sorgular için DOĞRU ama global sonuç DEĞİL.
Hibrit **kısa keyword/entity sorgularda** kazanır (ticker, "Sharpe ratio", kısaltma, sayı,
tırnaklı ifade); **uzun semantik sorgularda** kaybeder (BM25 ortak-kelime gürültüsü ekler).
Kaynak: BEIR (BM25 OOD'de daha sağlam), Bruch et al. füzyon (2210.11934).

**Yapılacak (öncelik):**
1. **BM25S** (2407.03618, saf NumPy/SciPy) → BM25 ~2.2s → **on-ms** (~500× hızlı). Mevcut
   `rank_bm25`-tarzı per-doc döngü darboğazı; eager sparse scoring + scipy sparse matris.
   → BM25 hızı artık hibridi engellemez.
2. **Füzyonu düzelt:** "union + sezgisel rerank" bir **ANTİ-DESEN** (doğru dense isabetini
   gürültülü BM25 adayları DÜŞÜRÜYOR → ölçülen 68.6→64.3 recall@1 düşüşü). Yerine **min-max
   normalize konveks kombinasyon:** `α·dense + (1−α)·bm25`, **α≈0.7–0.9 (dense-ağırlıklı)**,
   kart→makale etiketleriyle bir kez tune. Bruch et al.: konveks > RRF, OOD'de bile; tek param.
3. **Sorgu yönlendirme (router):** ucuz regex özellikleri (uzunluk, BÜYÜK-harf kısaltma,
   rakam, tırnak) → uzun-semantik → **dense-only**; kısa-keyword/entity → **hibrit**. En yüksek
   kaldıraç; iki rejim temiz ayrılır.

**Global varsayılan şimdilik dense-only KALSIN** (mevcut metrik için doğru); hibrit yalnız
router + BM25S + konveks-füzyon + keyword-eval-dilimi kazanınca AÇILSIN. Eski union+sezgisel
hibridi geri açma (ölçüm onu daha kötü gösterdi).

## Zincir 3 — Reranker (CPU)

**Bulgu (ÖNEMLİ — önceki kararı çürütür):** ">15s cross-encoder = kullanılamaz" bir **araç**
sorunuydu, reranking-imkansız değil. **FlashRank** (ONNX-int8 cross-encoder, torch'suz) ~24
adayı CPU'da **~30–100ms** rerank eder (150–500× hızlı). Önerilen: `ms-marco-MiniLM-L-12-v2`
(~23MB) → bu i7'de tahmini **~50–120ms**, hedef <500ms altında.

**Yapılacak:** FlashRank'i reranker arayüzüne entegre et (graceful fallback); aday havuzunu
~50'ye genişlet (reranker uzun listede yardımcı); **dense-only'e karşı A/B** (recall@6/cevap
doğruluğu). Reranking yüksek-recall/düşük-precision'da yardımcı; dense zaten %69 recall@1 →
kazanç küçük olabilir → ölç, faydalıysa tut. Naif bge-base ONNX (hâlâ 8-15s) ve qwen3:4b
LLM-reranker (saniyeler) ATLA. Kaynak: FlashRank repo, bağımsız M-serisi CPU bench (~31ms/16).

## Zincir 4 — Contextual Retrieval

**Bulgu:** Repodaki "title/section ön-eki" Anthropic'in GERÇEK yöntemi DEĞİL (o, her chunk
için LLM ile 50-100 token bağlam üretip embed VE BM25 öncesi ekler; başarısızlık 5.7%→2.9%).
Statik ön-ek faydanın AZINLIĞINI yakalar, hatta exact-match'i biraz düşürebilir. 91k chunk'a
gerçek yöntem CPU'da **infeasible** (~500+ CPU-saat).

**Yapılacak (ucuz→pahalı):**
1. **Contextual BM25** (en yüksek ROI, ~bedava): ön-eki BM25 indeksine de uygula (şu an yalnız
   embedding'e; BM25 yarısı eksik — Anthropic'te −35%→−49% farkı bu). Hibrit A/B arkasında.
2. **Ön-eki zenginleştir:** title yerine **title + ABSTRACT (veya bilgi-kartı tek-cümle)** —
   makale başına 1 özet (153 LLM çağrısı, 91k değil) → gerçek doküman bağlamı.
3. **Hedefli per-chunk LLM bağlamı:** yalnız `contextual_chunker` bayraklı yüksek-değer
   chunk'lar (formül/teorem) için Anthropic-tarzı bağlam üret (birkaç bin, 91k değil; gece).
4. **Late chunking (Jina) pilotu:** Ollama yalnız pooled vektör döndürüyor (#5907) → token
   embedding için nomic'i sentence-transformers/HF ile çalıştır; 5-10 makalede ölç. Deneysel.

**Süren title/section re-embed:** TUT (ucuz, net-pozitif, çoğu ödendi) ama "küçük kazanç",
"contextual bitti" değil. Bitince `.env ACHILLES_RAG_CONTEXTUAL_EMBED=true` + restart + ölç.

## Okuma/Anlama kalitesi (çapraz, Kural 7 ile hizalı)

**En yüksek ROI hepsi DETERMİNİSTİK / $0 ekstra-LLM:**
1. **CRAG-lite abstain kapısı** — eldeki mesafelerle: top-1 benzerlik < kalibre eşik (~0.6)
   VEYA top-1↔top-2 marjı küçük → ZAYIF/abstain. "Bilmiyorum"u zorlar (Kural 7). $0 LLM.
2. **"Lost in the middle" yeniden-sırala** — en güçlü chunk başa/sona, az chunk. Bedava.
3. **Yapı-farkında chunking** — cümle-penceresi + üretim öncesi bölüme genişlet (1200/200 zayıf).
4. Citation-forcing prompt + deterministik cited-id doğrulama (ucuz).
**ATLA:** HyDE, multi-query, Self-RAG — 4B'de net-zararlı/çok pahalı (4B-özel çalışma HyDE'yi
zararlı buldu; Self-RAG yalnız 7B/13B). Kaynak: 2506.21568, ARAGOG (2404.01037), Lost-in-the-
middle (2307.03172), CRAG.

---

## Uygulama sırası (bu çalışma)

| # | İş | Zincir | Maliyet | Ölçüm |
|---|----|--------|---------|-------|
| 1 | FlashRank reranker entegrasyonu | 3 | düşük | re-embed sonrası A/B |
| 2 | CRAG-lite abstain kapısı | okuma/Kural7 | düşük, det. | birim test |
| 3 | BM25S hızlı BM25 | 2 | düşük | re-embed sonrası |
| 4 | Konveks füzyon + sorgu router | 2 | orta | re-embed sonrası A/B |
| 5 | Lost-in-the-middle reorder | okuma | çok düşük | - |
| 6 | Contextual BM25 + abstract ön-ek | 4 | düşük | re-embed sonrası A/B |
| 7 | Decontaminated chunk-düzeyi golden eval | 1 | orta (1x LLM) | - |

Retrieval'ı etkileyen ölçümler **re-embed bitince** (korpus şu an akışkan). Her değişiklik
kazanan değilse GERİ ALINIR (Kural 2).

### Tam kaynak listesi
Contextual: anthropic.com/news/contextual-retrieval · jina.ai late-chunking · arXiv 2409.04701 ·
ollama#5907. Reranker: github FlashRank · clouatre-labs/rag-reranking-benchmarks · answer.ai colbert-small.
Hibrit: BEIR 2104.08663 · Bruch 2210.11934 · BM25S 2407.03618 · Weaviate/OpenSearch fusion.
Embedding/eval: MTEB · RAGAS · Evidently · 2504.14175. Okuma: 2506.21568 · ARAGOG 2404.01037 ·
2307.03172 (lost-in-middle) · CRAG · Self-RAG 2310.11511.
