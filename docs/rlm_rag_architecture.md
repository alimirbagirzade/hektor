# RLM + RAG + Paper Mastery — Mimari (Achilles)

_Son güncelleme: 2026-06-23 · Kaynak talimat: `Desktop/RAG Kaynak/RLM/achilles_rlm_rag_paper_brain_claude_prompt.txt`_

## Amaç

Yerel-öncelikli araştırma beyni: makaleler RAG belleğine alınır, anlaşılır
(Paper Mastery), ve sorulara **çok-adımlı, kaynaklı, doğrulanmış** biçimde cevap
verilir. RLM bir reasoning **kontrol katmanıdır** — yeni bilgi deposu değildir.

```
Local LLM
  + Paper RAG Memory       (mevcut: app/memory, app/ingestion)
  + Paper Mastery Eval      (mevcut: app/learning, app/verification)
  + RLM Controller          (YENİ: app/rlm)
  = Achilles Research Brain
```

## Katmanlar

| Katman | Sorumluluk | Modüller |
|--------|-----------|----------|
| RAG | dış bilgi hafızası (retrieve) | `app/memory/{reranking_retriever,retrieval_service,bm25_corpus,chroma_store}` |
| Doğrulama | atıf/dayanak/çelişki/yeterlilik | `app/verification/{citation,grounding,contradiction,context_sufficiency,confidence,abstention}` |
| **RLM** | **planla → çok-tur retrieve → doğrula → sentezle** | **`app/rlm/*` (YENİ)** |
| Mastery | makale "kullanılabilir mi?" skoru | `app/learning/*`, `app/verification/rag_mastery.py` |

## RLM Controller akışı (`app/rlm/rlm_controller.py`)

```
Kullanıcı sorusu
  → TaskClassifier        (kural-tabanlı, deterministik — task_type)
  → ReasoningPlan         (retrieval turu, must_include bölümler, trading izinleri)
  → Çok-tur retrieval     (RerankingRetriever) ⇄ sorgu yeniden-formülasyonu
  → EvidenceSufficiency   (0–100; eşik altı → "yeterli kaynak yok")
  → Taslak cevap          (LocalLLM, seed=42; LLM yoksa → no_llm yolu)
  → Claim extraction      (GroundingVerifier → iddia listesi)
  → Citation + Grounding + Contradiction doğrulama
  → ConfidenceScorer + AbstentionPolicy
  → Yapısal nihai cevap   (yalnız DESTEKLENEN iddialar) + run logları
```

### Kanıt yeterlilik kapısı (talimat §12)
`app/rlm/evidence_builder.py` — 0–100, LLM-free:
relevance(25) + coverage(20) + section_diversity(15) + citation(15) +
contradiction_risk(10) + method/limit(10) + recency(5).

Eşikler: `≥80` ek tur gereksiz · `60–79` cevap + sınırlama · `40–59` tekrar retrieval ·
`<40` çekimser ("yeterli kaynak yok").

### İki kapılı güvence
1. **Cevap öncesi** — kanıt skoru çok düşükse (`insufficient`) LLM hiç çağrılmaz (uydurma engellenir).
2. **Cevap sonrası** — taslak iddialara bölünür; desteklenmeyenler atılır; güven
   düşükse `AbstentionPolicy` çekimser kalır.

### Trading guard (kural 1, içerik-tabanlı)
Zorunlu uyarı kararı görev sınıflandırıcıya DEĞİL, **çıktı içeriğine** bağlıdır
(`_apply_trading_guard`): soru VEYA nihai cevap trading dili taşıyorsa
(`_TRADING_SIGNAL_RE`) ve uyarı henüz yoksa `_TRADING_DISCLAIMER` eklenir. Böylece
classifier bir trading sorusunu MATH/MULTI/UNCERTAINTY'ye düşürse bile uyarı kaçmaz.
Guard, `settings.rlm_allow_live_trading_signal` bayrağını gerçekten okur (MUTLAK False).

## Veri (talimat §6)

`app/rlm/rlm_store.py` — aynı SQLite dosyasında 4 tablo (mastery_store deseni):
`rlm_runs`, `rlm_steps`, `rlm_evidence`, `rlm_verifications`. Her koşu ayrıca
`reports/rlm_runs/{run_id}.json` olarak kaydedilir.

## Arayüzler

**CLI:**
```bash
uv run achilles rlm-answer "Bu makalelere göre volatilite rejimi momentum'u nasıl etkiler?"
uv run achilles rlm-answer "Bu makalenin metodolojisi?" --paper-ids paper_abc123
uv run achilles rlm-runs
```

**API:**
```
POST /api/rlm/answer            → çok-adımlı kaynaklı cevap
GET  /api/rlm/runs              → son koşular
GET  /api/rlm/runs/{run_id}     → run + steps + evidence + verification
```

## Config (`app/config/settings.py`, env `ACHILLES_RLM_*`)

| Ayar | Vars. | Anlam |
|------|------|-------|
| `rlm_max_retrieval_rounds` | 3 | çok-tur retrieval üst sınırı |
| `rlm_min_evidence_to_retry` | 40 | bu-üstü → tekrar retrieval; altı → yetersiz/abstain |
| `rlm_min_evidence_to_answer` | 60 | bu-üstü → cevap |
| `rlm_min_evidence_to_skip_retry` | 80 | bu-üstü → ek tur gereksiz (değişmez: retry ≤ answer ≤ skip_retry, normalize edilir) |
| `rlm_enable_query_reformulation` | true | yetersiz turda bölüm-odaklı genişletme |
| `rlm_allow_live_trading_signal` | **false** | MUTLAK — asla true (yalnız hipotez) |
| `rlm_seed` | 42 | determinizm (kural 6) |

## Mutlak kural eşlemesi (CLAUDE.md)

| Kural | RLM'de nasıl |
|-------|--------------|
| 1 — yatırım tavsiyesi yok | trading task → yalnız hipotez + zorunlu uyarı bloğu; canlı sinyal asla |
| 2 — test edilmeden "çalışıyor" yok | offline testler (`tests/test_rlm_controller.py`) |
| 4 — look-ahead yok | RLM hesaplama yapmaz; backtest yine `app/trading` disiplinine tabi |
| 5 — eval/exec yok | tüm skorlayıcılar regex/saf-Python |
| 6 — determinizm | tüm LLM çağrıları `seed=42`; sınıflandırma/skorlama saf kural |
| 7 — kaynak uydurma yok | iki kapılı güvence; desteklenmeyen iddia atılır |

## Mevcut altyapı ile eşleme (entegrasyon, sıfırdan değil)

Talimatın çoğu bileşeni Achilles'te **zaten vardı**; RLM bunları orkestre eder:

| Talimat bileşeni | Achilles karşılığı (mevcut) |
|---|---|
| paper ingestion + content-hash id | `app/ingestion/`, `app/memory/paper_indexer.py` |
| hibrit arama (dense+BM25+RRF+rerank) | `app/memory/reranking_retriever.py` |
| understanding card | `app/brain/knowledge_card_builder.py` |
| paper mastery test/skor | `app/learning/*`, `app/verification/rag_mastery.py` |
| verifier'lar | `app/verification/*` (7 modül) |
| L3/L4/L5 + UnderstandingScore | `app/verification/exams/*` |
| GraphRAG (opsiyonel) | `app/research/concept_graph.py`, `app/memory/graph_retriever.py` |

## Sonraki adımlar

- RLM/Mastery loglarından LoRA dataset adayı (talimat §16): yalnız
  `final_confidence≥0.85 ∧ citation≥0.90 ∧ grounding≥0.90 ∧ unsupported=[] ∧ human approved`.
- Web UI'de RLM run dashboard (steps/evidence/verification görselleştirme).
- Query reformulation'ı LLM-destekli (opsiyonel, opt-in) hâle getirme — şimdilik
  deterministik bölüm-odaklı genişletme.
