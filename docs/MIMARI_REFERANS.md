# Hektor Trader AI — Yeniden Yazım Rehberi (Tam Teknik Anatomi)

_Üretim tarihi: 2026-09-02 · Kaynak: depo `alimirbagirzade/achilles`, dal `main` (son merge PR#136) · Bu belge, projeyi sıfırdan yeniden yazacak birinin ihtiyaç duyacağı her şeyi tek yerde toplar: felsefe, mimari, her paket, her ajan, her tablo, her kural, her kapı, her test sözleşmesi, operasyon ve tarihçe._

> **Nasıl okunmalı?** Bölüm 1-2 "neden ve ne"; Bölüm 3-6 "hangi teknolojiyle, nerede saklanır"; Bölüm 7 paket paket iç yapı (en uzun bölüm); Bölüm 8 tüm ajanlar; Bölüm 9 güvenlik; Bölüm 10 eğitim yaşam döngüsü; Bölüm 11 web arayüzü; Bölüm 12 operasyon; Bölüm 13 testler; Bölüm 14 tarihçe ve dersler; Bölüm 15 bilinen kusurlar; Bölüm 16 yeniden yazım sırası; Bölüm 17 bugünkü durum.

---

## İçindekiler

1. Proje kimliği ve felsefe
2. Büyük resim mimarisi
3. Teknoloji yığını ve bağımlılıklar
4. Dizin yapısı (dosya envanteri)
5. Konfigürasyon (Settings ve ortam değişkenleri)
6. Depolama şeması (SQLite, Chroma, dosya sistemi)
7. Alt sistemler paket paket
8. Ajanlar (28 runtime ajanı + 19 Claude ajanı + 20 skill)
9. Güvenlik ve yetki modeli
10. Eğitim yaşam döngüsü (uçtan uca)
11. Web arayüzü (15 sekme)
12. Operasyon (kurulum, güncelleme, zamanlanmış görevler, CI)
13. Test stratejisi
14. Proje tarihi ve dersler
15. Bilinen tutarsızlıklar (yeniden yazımda düzelt)
16. Önerilen yeniden yazım sırası
17. Bugünkü durum ve sıradaki adım
- Ek A: Sabitler tablosu
- Ek B: Akademik referanslar
- Ek C: Komut hızlı referansı

---

# 1. Proje kimliği ve felsefe

## 1.1 Ne?

Hektor Trader AI, **yerel-öncelikli (local-first) bir AI trading araştırma sistemidir**. Akademik finans makalelerini (PDF) okur, bunlardan bilgi kartları ve formüller çıkarır, formülleri birleştirerek yeni strateji hipotezleri üretir, hipotezleri disiplinli backtest'ten geçirir, sonuçtan öğrenir ve isteğe bağlı olarak küçük bir LoRA adaptörü eğitir. macOS Apple Silicon, Windows 10/11 ve Linux'ta çalışır.

Kısa formül: **PDF literatür → RAG / bilgi kartı → (opsiyonel LoRA) → disiplinli backtest.**

## 1.2 Ne değil?

- **Canlı bot değil.** Borsa bağlantısı, emir gönderme, canlı sinyal yok. `rlm_allow_live_trading_signal` ayarı **mutlak olarak `False`** kalır.
- **Yatırım tavsiyesi değil.** Her çıktı "hipotez + test noktası"dır. Tavsiye dili (garanti, kesin, risksiz, "al/sat") tespit edilip reddedilir.
- **Pay-per-token API ürünü değil.** Çalışma zamanı LLM hattı tamamen yereldir (Ollama). Bulut sağlayıcı kodu (OpenAI/Anthropic/Google) vardır ama opsiyoneldir ve **geliştirilmeyecektir**. Geliştirme yardımı aylık abonelikli CLI araçlarıyla (Claude Code, Codex, Gemini CLI) yapılır, API anahtarıyla değil.

## 1.3 Sekiz mutlak kural (CLAUDE.md)

Bu kurallar kodun her katmanına gömülüdür; testler bunları sabitler.

| # | Kural | Kodda nasıl uygulanır |
|---|---|---|
| 1 | **Yatırım tavsiyesi üretme.** Çıktı daima hipotez + test noktası. | `_apply_trading_guard` (RLM), `trading_hypothesis_evaluator` regex'leri, Echo zehir filtresi, `safety_scanner` finansal-direktif taraması, `gate_6_philosophy` tavsiye dili |
| 2 | **Test edilmeden "başarılı/çalışıyor" deme.** backtest + out-of-sample şart. | `evaluator.evaluate` verdict politikası, `adapter_eval` min_n/regresyon vetosu, `_MIN_GRADED_FOR_SCORE=3`, smoke "stub≠runtime" |
| 3 | **Maliyetleri yok sayma** (komisyon + slippage). | `CostSpec` varsayılanı 0.0005+0.0005, `_net_returns`, `risk_manager` maliyet-dahil getiriler, disiplin verisi "maliyet token'ı" şartı |
| 4 | **Look-ahead bias yasak** — pozisyon `shift(1)` ile gecikmeli. | `backtester._net_returns` `eff_pos = position.shift(1)`, `math_verifier` lookahead işareti, `discipline_dataset` look-ahead tuzağı |
| 5 | **`eval`/`exec` yok** — strateji kuralları yalnız güvenli regex ile parse. | `strategy_ir._RULE_RE`, `exams/safe_eval.py` whitelist AST, RLM `production_mode` local-exec yasağı |
| 6 | **Determinizm** — rastgelelik daima `seed` parametresiyle. | Split seed 42, BM25/RRF tie-break, PPR sabit iterasyon, LLM çağrılarına seed (`rlm_seed=42`, concept_graph 42, comprehension 7, cross_paper sha256-türevi), `tools` `requires_seed` |
| 7 | **Kaynak uydurma** — retrieval boşsa açıkça belirt. | RAG "No sources found", RLM `abstained`/`no_llm`, `rag_abstain` kapısı, cross-paper ≥2 makale şartı, formül tespitinde kelime sınırı, `logged_in` daima `null` |
| 8 | **Otomatik ağır eğitim yok** — `train` varsayılan dry-run; gerçek eğitim yalnız açık `--run` **ve** tek-kullanımlık taze insan onayı ile. | `authorize_training_action`, `approvals.require_fresh_approval` (atomik CAS), `require_human` (403), STOP_ALL, orkestrasyon `approval` aşaması, MCP allow-list |

## 1.4 Tasarım felsefesi

- **"Aksi kanıtlanana kadar her strateji güvenilmezdir."** Sistem kasıtlı olarak şüphecidir; `verdict != pass` çıktı **aday**dır, "hazır" değildir.
- **Anlama yüzdeyle ölçülmez, sınavla kanıtlanır.** L3 (uygulama), L4 (karşıolgu), L5 (kompozisyon) sınavları objektif anlama skorudur; kaba "öz-değerlendirme %" yalnız gösterge sayacıdır.
- **RAG bilgiyi, LoRA üslubu sağlar.** İkisi ayrı inşa edilir, zincir olarak birlikte kullanılır. Bilgi için eğitim gerekmez; LoRA yalnız "trader gibi disiplinli düşünme"yi keskinleştirir.
- **Gözlemci üretimi bozmaz.** Tracker/Sentinel/self-heal hiçbir zaman istisna fırlatmaz ve ağır iş başlatmaz.
- **Prompt talimatı güvenlik sınırı değildir.** Doğurulan motorun sınırı araç kısıtıdır (`--safe-mode` + `--strict-mcp-config` + `--disallowedTools`), HTTP auth değil.
- **Motor kendi başarısının tek kaynağı olamaz.** Derin av "PASS"i dosya sistemiyle bağımsız doğrulanır (P8 kanıt JSON + P9 okuma-kanıtı).
- **Kod stili:** Python ≥ 3.12, `from __future__ import annotations`, pydantic v2, SQLAlchemy 2.0 tipli API, ruff (100 kolon, py312), mypy (pydantic plugin), saf pandas/numpy vektörize indikatörler, kullanıcıya dönük metin/log/docstring **Türkçe**.

## 1.5 Kademeli bug-avı kadansı

| Kademe | Ne | Tetikleyici | Otonomi |
|---|---|---|---|
| 0 — Kapı | `make format && lint && typecheck && test` (+ pre-commit/CI) | Her commit | Otomatik |
| 1 — Hafif tarama | Tek `claude -p` rapor-only tarama | Haftalık (`scripts/weekly-bug-scan.ps1`, Pzt 09:00) | Rapor-only |
| 2 — Derin adversarial av | Çok-ajan workflow: alt-sistem başına paralel finder → 2-3 oylu şüpheci doğrulama (varsayılan çürütülmüş) → yalnız onaylananı düzelt → Kademe-0 → commit | Ayda 1 **+ her LoRA eğitiminden ÖNCE (zorunlu)** veya ~25-30 commit | Denetimli |

Kademe-2'nin her eğitimden önce zorunlu olmasının sebebi **v5 regresyonu**dur (bkz. Bölüm 14): eğitim öncesi av atlandığında 46.75 saatlik CPU eğitimi disiplinde gerileyen bir adapter üretti.

---

# 2. Büyük resim mimarisi

## 2.1 Veri akışı

```
📄 PDF (data/papers/raw_pdf/)  ←── kullanıcı yükler / arxiv-fetcher / literature-scout (yalnız gelen-kutusu)
      │
      ▼ [app/ingestion]  parse (PyMuPDF→pypdf) → metadata (sezgisel) → math-aware chunk (1200/200)
      │                  paper_id = "paper_" + sha256(dosya)[:12]  (idempotent)
      ▼ [app/memory]     SQLite (papers/chunks, embedded=0) → Ollama embed (nomic-embed-text) / fake
      │                  → Chroma "paper_chunks" (cosine) → embedded=1 → bm25/graph cache reset
      │                  → FormulaExtractor → ConceptGraph → CrossPaperSynthesizer (best-effort)
      ▼ [app/brain]      RagAnswerer (router|graph|rrf|overfetch+rerank) → LocalLLM (Ollama)
      │                  KnowledgeCardBuilder → knowledge_cards (pending) → İNSAN onayı (06·ONAY)
      ▼ [app/rlm]        RLM Controller: classify → retrieval⇄reformulate → evidence gate → draft
      │                  → claim extraction → citation/grounding → contradiction → confidence → abstention
      ▼ [app/research]   SynthesisEngine (formül birleştir) → StrategyIR → backtest → evaluate → L5 gate
      │                  → ReflectionAgent (tek değişiklik/iterasyon) → research_sessions
      ▼ [app/trading]    StrategyIR (regex kural) → indicators (vektörize) → backtester (shift(1), maliyet)
      │                  → overfit_checks (IS/OOS 70/30) → evaluator (pass/fail/inconclusive)
      ▼ [app/learning]   Paper Mastery: inspect → 20 soru → RAG sınavı → 100 puan → durum
      ▼ [app/verification] Anlama merdiveni: Taban/L1/L2 (RAG) + L3/L4 (LLM sınav) + L5 (kompozisyon)
      ▼ [app/training + app/lora]
                         synth-qa + küratörlü kart + disiplin tuzakları → lora_sft.jsonl (KANONİK)
                         → Gate 0-8 → pretrain-gate GO/NO-GO → lora-split → train/valid
                         → [orkestrasyon 12 aşama] → İNSAN onayı → detached PEFT/MLX eğitim
                         → adapter_eval (base vs adapter) → registry ADAY → terfi (insan)
      ▼ [app/monitoring] Sentinel 10 salt-okuma probe → self-heal (2 runbook) → unattended supervisor
```

## 2.2 Katmanlar

| Katman | Paketler | Sorumluluk |
|---|---|---|
| Girdi | `ingestion` | PDF keşif, parse, metadata, chunk, kalite skoru, arXiv |
| Bellek | `memory` | SQLite ORM (32 tablo), Chroma, embedding, BM25, hibrit/RRF/graf retrieval, rerankers, MasteryStore |
| Beyin | `brain` | LocalLLM (Ollama/OpenAI/Anthropic/Google/MLX), RAG cevaplayıcı, bilgi kartı, sentetik QA, sorgu genişletme |
| Araştırma | `research` | Formül çıkarımı, kavram grafı, sentez motoru, yansıma ajanı, çapraz-makale sentez, öğrenme döngüsü, literatür tarama |
| Öğrenme | `learning` | Paper Mastery ajanı (inspector, soru üretici, sınav koşucu, skor, durum, rapor) |
| Doğrulama | `verification`, `evals`, `reliability` | Atıf/dayanak/yeterlilik/çelişki/güven/çekimserlik; L3-L5 sınavlar; RAGAS-offline; ReleaseGate |
| RLM | `rlm` | Çok-adımlı, kaynaklı, denetimli cevap kontrolcüsü + opsiyonel alexzhang adapter |
| Trading | `trading`, `tools` | StrategyIR, indikatörler, backtester, overfit, risk, Pine/paket export, Monte Carlo/istatistik |
| Eğitim | `lora`, `training`, `registry`, `feedback` | Gate 0-8, dataset üretimi, PEFT/MLX trainer, adapter eval, sürüm/terfi kaydı, Echo geri bildirim |
| Ajan runtime | `agents`, `orchestration`, `monitoring`, `pipeline` | Manifest/registry, tracker, supervisor, approvals, task queue, executor, chain; 12-aşamalı orkestrasyon; AutoDriver (claude -p); engines; verdict audit; Sentinel; self-heal; unattended supervisor |
| Yüzey | `web`, `mcp_server`, `main.py` | FastAPI (~100 route, 15 sekme statik UI), MCP proxy (21 salt-okuma araç), Typer CLI (~118 komut) |

## 2.3 İki depo, tek anahtar

SQLite kaynaktır, Chroma türevdir. İkisi `chunk_id = f"{paper_id}_c{index:04d}"` ile bağlıdır. SQLite dosyası (`storage/sqlite/hektor_trader_ai.db`) dört ayrı store tarafından paylaşılır (SqliteStore, MasteryStore, RlmStore, OrchestrationStore + SentinelStore + FeedbackStore); her biri bağlantıda `PRAGMA journal_mode=WAL` + `busy_timeout=30000` açar. SQLite FK enforcement **bilerek kapalıdır**; ebeveyn doğrulaması uygulama seviyesinde yapılır.

## 2.4 Otomasyon zinciri (topolojik)

`automation_manifest.yaml` içindeki `chain` bölümü ajanların koşma sırasını veri olarak tanımlar; `app/agents/runtime/chain.py` Kahn topolojik sıralamasıyla doğrular (döngü/eksik step → `ChainError`):

```
arxiv-fetcher → rag-learning-loop → {ingestion-quality-scorer, paper-mastery-agent}
paper-mastery-agent → {status-manager, scientific-tool-runtime, hypothesis-evaluator, rlm-controller(yaprak)}
{paper-mastery-agent, status-manager} → lora-control-plane → dataset-quality-gate
dataset-quality-gate → model-data-registry (requires_approval → supervisor BURADA DURUR)
{dataset-quality-gate, model-data-registry} → auto-lora-pipeline → adapter-eval → rules-updater
```

---

# 3. Teknoloji yığını ve bağımlılıklar

## 3.1 pyproject.toml

- `name = "hektor-trader-ai"`, `version = "0.1.0"`, `requires-python = ">=3.12"`, MIT, build backend hatchling, `packages = ["app"]`.
- **Entry points:** `hektor = "app.main:app"` (Typer) · `hektor-web = "app.web.server:run"` (uvicorn).
- `uv.lock` (814 KB) bilerek commit'lenir.

**Zorunlu bağımlılıklar:** pandas≥2.2, numpy≥1.26, pyarrow≥16.0 (kodda import edilmez; chromadb transitif), pydantic≥2.7, pydantic-settings≥2.3, sqlalchemy≥2.0, chromadb≥0.5, pymupdf≥1.24, pypdf≥4.2, typer≥0.12, rich≥13.7, anthropic≥0.40, google-genai≥1.0, requests≥2.32, httpx≥0.27, fastapi≥0.110, uvicorn[standard]≥0.29, python-multipart≥0.0.9, psutil≥6.0.

İndikatörler ve backtest motoru **el yazımıdır**; harici TA/backtest kütüphanesi yoktur.

**Extras:**

| Extra | İçerik | Not |
|---|---|---|
| `train` | `mlx-lm>=0.16` (yalnız Darwin arm64) | Apple Silicon MLX eğitimi |
| `train-cpu` | torch≥2.2, transformers≥4.40, peft≥0.10, accelerate≥0.29 | Windows/Linux PEFT; HF `datasets` kullanılmaz |
| `docs` | markdown-pdf≥1.13 | `scripts/gen_egitim_pdf.py` |
| `rlm` | rlms≥0.1.2 | Opsiyonel alexzhang13/rlm motoru |
| `mcp` | fastmcp≥2.0 | `mcp_server/hektor_mcp.py`; CI'da zorunlu |
| `dev` | pytest≥8.2, pytest-cov≥5.0, ruff≥0.5, mypy≥1.10, pre-commit≥3.7, types-requests | pytest/ruff/mypy **yalnız** bu extra ile gelir → `uv run/sync --extra dev` şart |

## 3.2 Araç ayarları

- **ruff:** `line-length=100`, `target-version="py312"`, `src=["app","tests"]`, `extend-exclude=["*.ipynb"]`; `select=["E","F","I","UP","B","C4","SIM","RUF"]`; `ignore=["B008","RUF001","RUF002","RUF003"]` (typer varsayılanları + Türkçe karakterler kasıtlı); isort `known-first-party=["app"]`.
- **mypy:** `python_version="3.12"`, `warn_unused_ignores`, `warn_redundant_casts`, `ignore_missing_imports`, `plugins=["pydantic.mypy"]`, `exclude=["tests/","scripts/"]`. `mypy app` torch/peft/transformers kurulu olmadan da temiz geçmelidir.
- **pytest:** `testpaths=["tests"]`, `addopts="-q --strict-markers -m 'not ollama and not slow'"`, marker'lar `ollama` ve `slow`. Kodda yalnız 2 `@pytest.mark.ollama` var (`test_knowledge_card_builder.py`, `test_rag_answerer.py`); `slow` tanımlı ama kullanılmıyor.
- **pre-commit:** gitleaks + detect-private-key.

## 3.3 Makefile

`install` (uv sync --extra dev + pre-commit install) · `test` · `lint` · `format` · `typecheck` · `ci` (lint+typecheck+test) · `update` (bash update.sh) · `audit` (pip-audit) · `gen-data` · `web` · `web-start/web-stop/web-log` (macOS launchd) · `clean`.

## 3.4 Çalışma zamanı modelleri

| Rol | Varsayılan | Env |
|---|---|---|
| LLM (Ollama) | `qwen3:4b` (8 GB RAM); 16 GB → `qwen3:8b`; 32 GB → `qwen3:14b`; eğitimde `qwen2.5:1.5b` | `HEKTOR_LLM_MODEL` |
| Embedding | `nomic-embed-text` (Ollama `/api/embed`, 64'lük batch) | `HEKTOR_EMBED_MODEL` |
| PEFT base | `Qwen/Qwen3-4B-Instruct-2507` (1.5B pivotunda `Qwen/Qwen2.5-1.5B-Instruct`) | `HEKTOR_PEFT_BASE_MODEL` |
| MLX base | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` (settings) / `models/mlx/Qwen3-4B-4bit` (.env.example) | `HEKTOR_MLX_BASE_MODEL` |
| Ollama host | `http://127.0.0.1:11434` (localhost DEĞİL, Windows IPv6) | `HEKTOR_OLLAMA_HOST` |
| Uzun vade | Lokal 120B OSS model; tek değişiklik `.env` | — |

---

# 4. Dizin yapısı (dosya envanteri)

Toplam `app/` Python: **48.290 satır, 22 paket**. Test: **180 dosya, 1.657 test fonksiyonu** (parametrize ile son tam koşu 1755 passed, 4 skipped).

```
hektor/
├── CLAUDE.md                 # Claude Code bağlayıcı kurallar (8 kural, kadans, seans protokolü)
├── HANDOFF.md (1431)         # Seans devir defteri — kronolojik tarih + "YENİ SEANS BAŞLANGICI"
├── README.md                 # Kullanıcı kılavuzu (sıfır-varsayım, kopyala-yapıştır)
├── SECURITY.md               # Tehdit modeli + ağa açma checklist'i
├── TRAINING_ROADMAP.md       # Evrensel eğitim yöntemleri şablonu + Hektor durumu
├── automation_manifest.yaml  # 28 runtime ajanının tek bildirimsel kaynağı + chain
├── pyproject.toml / uv.lock / Makefile
├── .env.example              # 38 HEKTOR_* anahtarı
├── setup.sh / setup.ps1 / install.ps1 / update.sh / update.ps1
├── app/                      # 48.290 satır
│   ├── main.py (4838)        # Typer CLI, ~118 komut
│   ├── agents/ (28 dosya, 4409)     # runtime (registry/tracker/supervisor/approvals/task_queue/executor/handlers/chain/preflight/schemas), local_training_* (5A-5E), model_advisor, system_profiler, installer, benchmark, learning (memory, rules_updater)
│   ├── brain/ (10, 1702)     # local_llm, mlx_llm, rag_answerer, knowledge_card_builder, synthetic_qa_builder, query_expander, multi_query_retriever, answer_quality, prompt_loader
│   ├── cli/ (1)              # boş yer tutucu
│   ├── config/ (2, 320)      # settings.py (pydantic-settings), logging
│   ├── evals/ (10, 1267)     # eval_runner, trading_hypothesis_evaluator, metrics, golden_dataset/generator, retrieval_eval, answer_eval, rag_ragas_offline, regression_runner
│   ├── feedback/ (3, 447)    # echo.py, store.py (Echo geri bildirim)
│   ├── ingestion/ (8, 833)   # paper_loader, pdf_parser, metadata_extractor, chunker, clean_text_scorer, quality_scorer, arxiv_fetcher
│   ├── learning/ (8, 962)    # paper_mastery_agent, paper_inspector, question_generator, rag_exam_runner, mastery_scorer, status_manager, report_generator
│   ├── lora/ (14, 2893)      # control_plane, gates, auto_pipeline, dataset_builder, dataset_splitter, curriculum, domain_classifier, math_verifier, quality_filter, safety_scanner, adapter_registry, card_curation, peft_llm_shim
│   ├── memory/ (19, 5060)    # sqlite_store (2384), chroma_store, embedding_service, retrieval_service, bm25_index, bm25_corpus, graph_corpus, graph_retriever, rank_fusion, hybrid_retriever, query_router, reranker, cross_encoder_reranker, flashrank_reranker, reranking_retriever, contextual_chunker, paper_indexer, mastery_store
│   ├── monitoring/ (4, 688)  # sentinel, self_heal, store
│   ├── orchestration/ (14, 4405) # pipeline, orchestrator, store, delegates, driver, engines, engine_procs, verdict_audit, smoke, run_smoke, collision, regression, unattended_supervisor
│   ├── pipeline/ (2, 103)    # auto_researcher
│   ├── prompts/              # knowledge_card.md, rag_answer.md (prompt_loader okur)
│   ├── registry/ (3, 702)    # version_store, promotion_gates
│   ├── reliability/ (2, 81)  # release_gate
│   ├── research/ (12, 3063)  # orchestrator, synthesis_engine, formula_extractor, concept_graph, cross_paper_synthesizer, reflection_agent, chain_data_builder, rag_learning_loop, literature_scout, rag_trend_scanner, synthesis_paper
│   ├── rlm/ (16, 2782)       # rlm_controller (789), answer_pipeline, task_classifier, evidence_builder, claim_extractor, rlm_store, lora_candidate, engine_config, safe_tools, tool_registry, adapters/{base,native,alexzhang_rlm,security}
│   ├── tools/ (5, 584)       # probability_simulator, statistics_checker, result_verifier, tool_registry
│   ├── trading/ (10, 1533)   # strategy_ir, indicators, backtester, evaluator, overfit_checks, risk_manager, market_data_loader, strategy_generator, package_exporter
│   ├── training/ (20, 4073)  # peft_lora_train (828), mlx_lora_train, detached_launch (623), adapter_eval, dataset_builder, dataset_quality, discipline_dataset, sft_assembly, unified_dataset, synthetic (brain'de), reward_signal, dpo_dataset_builder, tool_use_trainer, tool_use_dataset_builder, mastery_sft_builder, evaluate_model, cloud_notebook, backend, adapter_registry, unattended_policy
│   ├── verification/ (19, 2540) # citation_verifier, grounding_verifier, context_sufficiency, contradiction_detector, confidence_scorer, abstention_policy, comprehension_scorer, rag_mastery, exams/{safe_eval, reference_oracle, registry, l3_application, l4_counterfactual, l5_composition, discipline_exam, understanding_score, understanding_record}
│   └── web/ (16, 4984)       # server.py (2625), schemas, security, driver_scope, sse_tickets, version_info, training_manager, lora_chat_service, agent_graph(+routes), engines_routes, orchestration_routes, feedback_routes, sentinel_routes, ai_brain_routes, static/{index.html, ai_brain.html, assets/app.js (5250), app.css (3020), canli.*, fonts/*.woff2}
├── mcp_server/               # hektor_mcp.py (FastMCP proxy), allowlist.py (21 uç)
├── configs/lora/lora_profiles.yaml   # 6 profil
├── evals/                    # discipline_core / overfit_awareness / risk_management .jsonl
├── strategies/{pine,mql5,python}/    # rezerve, kod yazmaz
├── notebooks/                # hektor_lora_stage2.ipynb, Modelfile, KAGGLE_EGITIM_ADIM_ADIM.md
├── scripts/ (35)             # kurulum/doğrulama/eğitim/döngü/PR/tarama scriptleri
├── docs/ (30 md + egitim/ + examples/ + kaynaklar/ + arsiv/)
├── tests/ (180)
├── reports/ (8 .gitkeep dizin + izlenen bug-scan/handoff raporları)
├── data/ · models/ · storage/ · vector_db/   # runtime; yalnız .gitkeep izlenir
├── .claude/ {agents/ (19), skills/ (20), commands/ (speckit.*), settings.json, launch.json}
└── .github/workflows/ {ci.yml, nightly-automation-audit.yml, claude-code-task.yml} + PR/issue şablonları
```

---

# 5. Konfigürasyon

## 5.1 `app/config/settings.py` — tüm alanlar

`Settings(BaseSettings)`, prefix `HEKTOR_`, `.env` okunur, `get_settings()` `lru_cache`'li (testler `cache_clear()` yapar). `ensure_dirs()` şu dizinleri oluşturur: sqlite dizini, chroma_dir, raw_pdf_dir, market_raw_dir, adapters_dir, `reports/{papers,training,backtests,evals,agent_runs}`.

| Alan | Varsayılan | Açıklama |
|---|---|---|
| `llm_backend` | `"auto"` (.env.example: `ollama`) | ollama / auto / openai |
| `ollama_host` | `http://127.0.0.1:11434` | IP, localhost değil |
| `llm_model` | `qwen3:4b` | |
| `ollama_keep_alive` | `30s` | Eğitim sırasında `0` (OOM) |
| `embed_model` | `nomic-embed-text` | |
| `openai_api_key/model/base_url` | `""` / `gpt-4o-mini` / `https://api.openai.com/v1` | Opsiyonel, LiteLLM kancası |
| `anthropic_api_key/model` | `""` / `claude-haiku-4-5-20251001` | Opsiyonel |
| `google_api_key/model` | `""` / `gemini-2.0-flash` | Opsiyonel |
| `mlx_base_model` | `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` | |
| `peft_base_model` | `Qwen/Qwen3-4B-Instruct-2507` | |
| `sqlite_path` | `storage/sqlite/hektor_trader_ai.db` | |
| `chroma_path` | `vector_db/chroma` | |
| `rag_top_k` | 6 | |
| `chunk_size` / `chunk_overlap` | 1200 / 200 | |
| `rag_rerank` / `rag_overfetch` | True / 4 | over-fetch + heuristik rerank |
| `rag_hybrid` | True | BM25 keyword adayları |
| `rag_cross_encoder` / `_model` | False / `BAAI/bge-reranker-base` | opt-in |
| `rag_flashrank` / `_model` | False / `ms-marco-MiniLM-L-12-v2` | opt-in, CE'ye öncelikli |
| `rag_rrf` / `rag_rrf_k` | False / 60 | |
| `rag_graph` / `_damping` / `_iters` | False / 0.85 / 20 | SPRIG-lite |
| `rag_router` / `rag_router_alpha` | False / 0.7 | |
| `rag_contextual_embed` | False | Tüm korpus aynı ayarla embed edilmeli |
| `rlm_max_retrieval_rounds` | 3 | |
| `rlm_min_evidence_to_retry/answer/skip_retry` | 40 / 60 / 80 | Kanıt yeterlilik eşikleri (0-100) |
| `rlm_enable_query_reformulation` | True | |
| `rlm_allow_live_trading_signal` | **False (MUTLAK)** | |
| `rlm_seed` | 42 | |
| `rlm_draft_max_tokens` / `_timeout_s` | 900 / 600 | |
| `rlm_engine_provider` | `native` | native / alexzhang |
| `rlm_production_mode` | True | local-exec/shell/network/fs-write yasak |
| `rlm_alexzhang_*` | enabled=False, backend=`anthropic`, model=`""`, environment=`docker`, allow_local_exec/shell/network/fs_write=False, log_trajectories=True | Opsiyonel motor |
| `rag_reorder_context` | True | Lost-in-the-middle |
| `rag_verify_citations` | True | |
| `rag_abstain` / `_min_similarity` / `_min_margin` | False / 0.55 / 0.02 | CRAG-lite güven kapısı |
| `default_market` / `default_timeframe` | `XAUUSD` / `15m` | |
| `allow_fake_embeddings` | True | False → Ollama yoksa RuntimeError |
| `log_level` | INFO | |
| `synthesis_mirror_dir` | `""` | Boş = aynalama kapalı |
| `scout_inbox_dir` | `""` | Boş = `data/literature_inbox/` |
| `unattended_training_enabled` | **True** | Eğitim yetkisinin tek politika anahtarı |
| `auto_lora_min_cards` | 20 | |
| `auto_lora_check_interval_min` | 60 | |
| `auto_lora_eval_threshold` | 0.5 | |
| `auto_lora_eval_sample_n` | 8 | min_n altı accept bloklanır |
| `web_host` / `web_port` | `127.0.0.1` / 8765 | |
| `api_token` | `""` | Boş = auth KAPALI |
| `cors_origins` | `http://127.0.0.1:8765,http://localhost:8765` | |
| `max_upload_mb` | 100 | |
| `rate_limit_per_min` / `upload_rate_limit_per_min` | 120 / 60 | |
| `trusted_hosts` | `""` | |
| `hsts_enabled` | False | |

## 5.2 Script/runtime ortam değişkenleri (`.env` dışı)

| Değişken | Nerede | Amaç |
|---|---|---|
| `UV_NO_SYNC=1` | start-train.ps1, train-loop.ps1, run-web-service.ps1, verify-install, continuous-learning.sh, systemd unit | `uv run` senkronunun çalışan web `hektor-web.exe`'yi kilitleyip `os error 32` vermesini önler |
| `HEKTOR_TRAIN_DTYPE` | start-train.ps1 | bf16/fp32 |
| `HEKTOR_TRAIN_SUPERVISED=1` | start-train.ps1, `launch()` alt sürecine | Üst katman onay tüketti → iç onay kapısı atlanır; STOP_ALL yine geçerli |
| `HEKTOR_WEB_ISOLATE_CHROMA=1` | run-web-service.ps1 | Windows Chroma native crash izolasyonu; `/api/status` n_chunks=0, RAG döngüsü başlamaz |
| `OPENBLAS/OMP/MKL_NUM_THREADS=1` | run-web-service.ps1 | NumPy stabilizasyonu |
| `HEKTOR_WEB_URL` | MCP proxy | Hedef web (varsayılan 127.0.0.1:8765) |
| `HEKTOR_DRIVER_TOKEN` / `HEKTOR_DRIVER_RUN_ID` | `build_child_env` → motor → MCP proxy | Sürücü kimliği |
| `CLAUDE_CODE_MANAGED_SETTINGS_PATH` / `_REMOTE_SETTINGS_PATH` / `_MOCK_REMOTE_SETTINGS` | `build_child_env` SİLER | Ayar-ezme kanalı |
| `PYTEST_DEBUG_TEMPROOT` | test | Windows Temp izin sorunu; alternatif `--basetemp=.pytest_tmp` |
| `RAG_AB_LIMIT` / `RAG_KW_LIMIT` | rag_ab_*.py | A/B sorgu sınırı |
| `GH_TOKEN` | PR açma | GCM `gho_` token fallback |
| `ANTHROPIC_API_KEY` / `HF_TOKEN` | GitHub secret | Yalnız workflow; repoya yazılmaz |
| `ENABLE_CLAUDE_TASK` / `ENABLE_NIGHTLY_AUDIT` | GitHub repo variable | Workflow aktivasyon kapıları |

---

# 6. Depolama şeması

## 6.1 SQLite — `app/memory/sqlite_store.py` (32 tablo)

Ortak: `_utcnow()` ISO-8601 string; `_sqlite_pragmas` connect listener (WAL + busy_timeout 30000); `create_engine(..., connect_args={"check_same_thread", "timeout": 30.0})`; `_migrate()` idempotent `ALTER TABLE ADD COLUMN` (`knowledge_cards`: trust_level/review_status/lora_eligible/difficulty/stage; `papers`: quality_score/ingest_status).

**Çekirdek RAG**

| Tablo | Anahtar alanlar |
|---|---|
| `papers` | `paper_id` PK, `file_hash` UNIQUE+index, `source_path`, `title`, `authors` (JSON), `year`, `source`, `n_pages`, `n_chars`, `quality_score`, `ingest_status`, `created_at`; chunks cascade delete |
| `chunks` | `chunk_id` PK(80), `paper_id` FK+index, `chunk_index`, `section_name`, `page_number`, `text`, `char_count`, `token_estimate`, **`embedded` 0/1**, `created_at`; `UniqueConstraint(paper_id, chunk_index)` |
| `summaries` | summary_id, paper_id, model, summary_text |
| `knowledge_cards` | `card_id` PK, `paper_id`, `model`, `card_json`, `trust_level` (canonical/verified/draft/unreviewed; default draft), **`review_status`** (pending/approved/rejected), **`lora_eligible`** 0/1, `difficulty` 0-1, `stage` (lora_phase_1..4) |
| `training_examples` | example_id, source_paper_id, example_type, instruction, input_text, output_text |
| `formulas` | formula_id, paper_id, name (index), latex, plain, description, variables_json, category |
| `concept_links` | link_id, from_concept, relation, to_concept, source_paper_id, weight |
| `research_sessions` | session_id, question, iteration, parent_session_id, source_paper_ids_json, synthesis_reasoning, proposed_indicator_json, strategy_ir_json, backtest_result_json, verdict, reflection, improvement_notes, status |

**Trading:** `strategies`, `backtests` (sharpe/sortino/max_drawdown_pct/profit_factor/win_rate_pct/metrics_json/verdict), `risk_reports` (kelly/drawdown ölçekleme).

**Model/eval:** `model_evaluations`, `adapters`, `eval_history` (adapter_name, eval_set, pass_rate, total_items, passed_items).

**RAG izleme:** `arxiv_saved_queries` (query, max_results, auto_ingest, run_count, last_run_at), `rag_queries`, `retrieval_runs`, `retrieval_results`, `chunk_quality_flags` (has_formula/incomplete_formula/incomplete_argument/table/definition/theorem/needs_adjacent_context), `knowledge_entities`, `knowledge_relations`.

**Doğrulama:** `golden_questions`, `failure_logs`, `verification_runs`, `reward_signals` (session_id UNIQUE), `tool_use_examples` (step_type think|call|observe|conclude), `paper_comprehension` (extraction/retrieval/llm/total_score, details_json), `understanding_snapshots` (seed, status, total/passed/failed/skipped/no_data/graded, pass_rate, by_level_json, context_json).

**Agent runtime:** `agent_runs`, `agent_events` (retention 30 gün VEYA 50.000), `automation_tasks`, `approval_requests` (`consumed_at` tek-kullanım).

**Kayıt defteri:** `dataset_versions`, `rag_index_versions`, `embedding_model_versions`, `rlm_reward_versions` (pii_scanned/secret_scanned 0/1/2), `promotion_decisions` (append-only).

**Araçlar:** `tool_runs` (seed), `tool_artifacts` (content_hash), `paper_ingestion_runs`.

**Kritik metotlar:** `upsert_paper`, `get_paper_by_hash`, `find_paper_by_title` (normalize uzunluk <12 → None), `add_chunks` (merge), `mark_chunks_embedded`, `delete_chunks_for_paper`, `has_embedded_chunks`, `list_all_chunks`, `save_knowledge_card`, `approve_card` (içeriksiz kart → False), `reject_card`, `set_card_lora_eligible` (review_status'a dokunmaz), `list_paper_ids` (Gate 0 orphan tespiti), `claim_automation_task_atomic` (CAS), `consume_fresh_approval` (CAS), `prune_agent_events` (500'lük parçalar), `log_tool_run` (contextmanager), `add_ingestion_run` (ebeveyn doğrulama).

## 6.2 Diğer store'lar (aynı dosya, ayrı Base)

| Store | Tablolar |
|---|---|
| `MasteryStore` (`app/memory/mastery_store.py`) | `paper_learning_queue` (priority, attempts, max_attempts=3), `paper_mastery_tests`, `paper_mastery_questions`, `paper_mastery_answers`, `paper_mastery_scores`, `paper_status_history` |
| `RlmStore` (`app/rlm/rlm_store.py`) | `rlm_runs`, `rlm_steps`, `rlm_evidence`, `rlm_verifications` |
| `OrchestrationStore` (`app/orchestration/store.py`) | `orchestration_runs` (`orc_`), `orchestration_stages` (`orst_`; `claim_stage_running` CAS, `heartbeat_at`), `orchestration_events` (`orev_`) |
| `SentinelStore` (`app/monitoring/store.py`) | `sentinel_checks` (keep_last=1000 budama) |
| `FeedbackStore` (`app/feedback/store.py`) | `feedback_corrections` |
| Learning memory (`app/agents/learning/memory.py`) | **Ayrı dosya** `storage/hektor_learning.db`: `model_trials`, `error_patterns`, `rule_suggestions` |

## 6.3 Chroma

Dizin `vector_db/chroma`; koleksiyon `paper_chunks`; `hnsw:space=cosine`; id = `chunk_id`; `document` = orijinal chunk metni (contextual ön-ek dahil değil); metadata `paper_id, chunk_index, page_number (None→-1), section_name (None→""), title (None→"")`. `PersistentClient(anonymized_telemetry=False, allow_reset=True)`; süreç içi paylaşılan client (`_SHARED`), tüm işlemler `_CHROMA_LOCK` (RLock) altında; `add` = `upsert`; `get_all(page=5000)` sayfalı.

## 6.4 Dosya sistemi haritası

| Yol | İçerik | Yazan |
|---|---|---|
| `data/papers/raw_pdf/` | Kaynak PDF'ler (rekürsif) | kullanıcı, arxiv_fetcher |
| `data/papers/extracted_text/{paper_id}.txt`, `metadata/{paper_id}.json` | Tam metin + metadata | PaperIndexer |
| `data/lora_sft/synthetic_qa.jsonl` | Sentetik QA birikimi | synth-qa |
| **`data/lora_sft/lora_sft.jsonl`** | **TEK KANONİK SFT kaynağı** | assemble_sft.py, lora-cloud-prep, build_training_split |
| `data/training/jsonl/{train,valid}.jsonl` | Eğitim bölmesi (seed 42, %5 valid) | ensure_train_split, lora-split |
| `data/training/{unified_sft,mastery_sft,research_chains}.jsonl` | Yardımcı setler | unified_dataset, mastery_sft_builder, chain_data_builder |
| `data/feedback/feedback_sft.jsonl` | Echo aday dosyası (oto-merge YOK) | EchoCollector.export_approved |
| `data/market/raw/*.csv` | OHLCV | gen-data, /api/backtest/csv |
| `data/literature_inbox/{topic}/`, `BULUNANLAR.md` | Keşif ajanı gelen kutusu | literature_scout |
| `models/adapters/<ad>/` | PEFT adapter + checkpoint'ler | trainer |
| `models/adapters/<ad>.meta.json` | Adapter metadata | training/adapter_registry |
| `registry/adapters/registry.jsonl` | LoRA adapter durum makinesi | lora/adapter_registry |
| `reports/papers/{paper_id}_card.json` | Bilgi kartı | KnowledgeCardBuilder |
| `reports/papers/mastery/{paper_id}_mastery_report.{json,md}` | Mastery raporu | ReportGenerator |
| `reports/backtests/{name}_{id}.json` | Backtest | persist_backtest |
| `reports/evals/{adapter_eval_*, pretrain_gate.json, eval_*, understanding/*}` | Eval | eval hattı |
| `reports/training/<ad>_loss.json` | Loss eğrisi | trainer/training_manager |
| `reports/lora/*.md` | Gate raporu | LoRAControlPlane |
| `reports/rlm_runs/{run_id}.json`, `reports/rlm/trajectories/` | RLM | RlmController (CLI), alexzhang |
| `reports/synthesis/sentez_{YYYYMMDD}_{HHMM}.md` (+ ayna dizini) | Sentez makaleleri | synthesis_paper |
| `reports/agent_runs/<run_id>.jsonl` | Ajan olay günlüğü | tracker |
| `reports/local_training_orchestrator/` | 5A-5E raporları | local_training_* |
| `reports/bug-scan/scan-*.md` | Kademe-1/2 raporları (izlenen) | weekly-bug-scan |
| `docs/egitim/{rag,lora,rlm,math-physics}-watchlist.md` | İzleme defterleri | rag_trend_scanner, literature_scout |
| `storage/STOP_ALL` | Küresel kill-switch | supervisor |
| `storage/STOP_TRAINING`, `storage/STOP_LEARNING` | Graceful durdurma sinyalleri | train-loop, continuous-learning |
| `storage/train_status.json` (+pid) | Detached eğitim durumu | launch() / start-train.ps1 |
| `storage/.training_launching` | Atomik başlatma kilidi (TTL 120 s) | detached_launch |
| `storage/auto_lora_state.json` | AutoLoRA aşaması | AutoLoRAPipeline |
| `storage/rag_learning_state.json` | RAG döngüsü config+durum | RagLearningLoop |
| `storage/self_heal_state.json`, `storage/unattended_supervisor_state.json` | Tamir/reconciler durumu | monitoring, orchestration |
| `storage/orchestration/regression_baseline.json` | Regresyon baseline | regression.py |
| `storage/mcp/drive-<run_id>.json` | Sür modu MCP config (sır içermez) | driver.py |
| `logs/{hektor-web, train-full, train-full-err, train-loop, update, synth_qa_seed*, rag-scan-*}.log` | Loglar (ignore) | scriptler |
| `.web.pid` | Web PID | start-server/update |

---

# 7. Alt sistemler paket paket

## 7.1 `app/ingestion/` — İçe-alım boru hattı (8 dosya, 833 satır)

### paper_loader.py
- `DiscoveredPaper(path, file_hash)`; `paper_id` property = `"paper_" + file_hash[:12]`.
- `compute_file_hash(path)` — 64 KiB bloklarla streaming sha256.
- `discover_pdfs(directory=None)` — `raw_pdf_dir` altında rekürsif `rglob("*.pdf")`, sıralı (deterministik).
- **İnvaryant:** aynı bytes → aynı `paper_id`. Farklı PDF export/arXiv sürümü → farklı id; bu boşluğu başlık dedup'ı (`find_paper_by_title`) kapatır.

### pdf_parser.py
- `ParsedPdf(path, pages)`; `text`, `n_pages`, `n_chars`.
- PyMuPDF (`fitz`) öncelikli, yoksa pypdf; ikisi de yoksa `RuntimeError("... uv sync")`. Sayfalar `.strip()`.

### metadata_extractor.py (bağımlılıksız sezgisel)
- `PaperMetadata(title, authors[], year, source="manual")`.
- `guess_title`: ilk 8 boş-olmayan satır; http/@/arxiv/doi içerenler atlanır; `15<=len<=200`, `not isupper()`.
- `guess_year`: `19[7-9]\d|20[0-4]\d`. `guess_authors`: satır 1-12, `AUTHOR_HINT_RE`, en fazla 10.
- Yalnız ilk 4000 karakter incelenir.

### chunker.py (math-aware)
- Sabitler: `_TOKEN_DIVISOR=4`, `_HEADING_RE` (abstract…references), `_MATH_BLOCK_RE` (`$$…$$`, `\[…\]`, `\begin{equation|align|gather|multline}`, inline `$…$` 3-120 karakter), `_MATH_CHARS_RE`, **`_MATH_WHOLE_MAX_CHARS=6000`**.
- `TextChunk(paper_id, chunk_index, text, page_number, section_name)`; `chunk_id=f"{paper_id}_c{index:04d}"`, `token_estimate=max(1, chars//4)`.
- `_is_math_heavy`: blok eşleşmesi VEYA matematik karakter oranı > 0.12.
- `chunk_text`: paragraf-farkındalıklı açgözlü paketleme; math-heavy paragraf `chunk_size`'ı aşabilir ama 6000'i asla aşmaz; aksi halde `". "` → `"\n"` → sert kesim.
- `chunk_parsed_pdf`: sayfa sayfa, sonra global yeniden numaralama.

### clean_text_scorer.py
`score_clean_text(text) -> 0..10`: kontrol karakterleri (−4 max), U+FFFD (−3), tire-satır kırılması (−2), <200 karakter (−1).

### quality_scorer.py (100 puanlık içe-alım rubriği, compute-on-demand)
- Rubrik: parse 15 · metadata 10 · section 15 · formula 15 · table 15 · figure 10 · ocr 10 · cleantext 10.
- Durum: ≥90 `ready_for_rag`, 70-89 `usable`, 50-69 `slow_but_usable`, 40-49 `unstable`, <40 `failed`.
- Formül/tablo/figür yokluğu **nötr** (7.5/7.5/5) — cezalandırılmaz. `n_chunks==0` → çıkarım bileşenleri 0.
- `gather_inputs(store, paper_id)`, `score_paper`, `score_all_papers(record=False, worst_n=10)`; `record=True` → `paper_ingestion_runs` + `papers.quality_score/ingest_status`.

### arxiv_fetcher.py
- `search_arxiv(query, max_results≤50)` (Atom XML, relevance). **Id normalizasyonu kategori korur** (`hep-th/9901001v2` → `hep-th/9901001`).
- `@tracked("arxiv-fetcher") fetch_arxiv_papers(query, max_results=5, dest_dir=None)`: dosya adı `arxiv_{id.replace("/", "_")}.pdf`; **idempotent** (var → skipped); **`%PDF` magic-byte kapısı**; tek makale hatası turu çökertmez.

## 7.2 `app/memory/` — RAG çekirdeği (19 dosya, 5060 satır)

### embedding_service.py
- `EmbeddingService(model, host, allow_fake)`; mod lazy: `ollama` (GET /api/tags 200) veya `fake`.
- `_BATCH_SIZE=64`, `_FAKE_DIM=256`. Toplu `POST /api/embed` (timeout 120); grup patlarsa tekli `/api/embeddings` (timeout 60).
- `_embed_fake`: sha256 tabanlı deterministik 256-dim, L2 normalize. Semantik değildir; yalnız çevrimdışı/test için.

### chroma_store.py
Yukarıda §6.3. `add(upsert)`, `query(top_k, where)`, `delete_by_paper`, `count`, `get_all(page=5000)`, `reset`.

### retrieval_service.py
- `Retriever` Protocol: `retrieve(query, top_k=None) -> list[RetrievedChunk]`.
- `RetrievedChunk(chunk_id, paper_id, text, page_number, section_name, title, distance)`; `citation` = `[{paper_id}:{chunk_id}, s.{page}]`.
- `RetrievalService(chroma, embedder)`: dense sorgu, `k = top_k or rag_top_k`.

### bm25_index.py
Saf Python BM25 (k1=1.5, b=0.75); Türkçe karakter tokenizasyonu; **deterministik sıralama** `(-score, doc_id)`.

### bm25_corpus.py (kritik cache)
- **Kaynak SQLite `Chunk` tablosu, Chroma DEĞİL** (Chroma `get_all()` eşzamanlı erişimde BM25'i sessizce öldürüyordu).
- `_cache = {"sig": None, "pair": (bm25, chunks)}` — tek anahtarda demet (yırtık okuma önlenir). `_build_lock` + çift kontrol (94k chunk ~170 s build; thundering herd).
- Yalnız `embedded=1` chunk'lar (dense yolun görmediği metin lexical yolda alıntılanamaz).
- **İçerik imzası `(count, total_chars)`** (~6.6 ms; tam hash 18 ms reddedildi). `reset_cache()` kilit altında (lost-update).

### graph_corpus.py
Kaynak Chroma; imza **yalnız count** (ucuz `count()`; `get_all()` rebuild'e ertelenir). Aynı-sayıda içerik değişimini yakalamaz → `reset_cache()` otoritatif; ingest her ikisini de resetler.

### graph_retriever.py (GraphRAG, SPRIG-lite, arXiv:2602.23372)
- `extract_terms` (≥3 karakter, ~70 TR+EN stop-word), `build_graph(max_df_ratio=0.5)` hub pruning, `personalized_pagerank(damping=0.85, iterations=20)` iki-adımlı güç iterasyonu (deterministik), `graph_rank`, `seed_weights_from_ids` ({cid: n-i}).

### rank_fusion.py (RRF, Cormack 2009)
`reciprocal_rank_fusion(k=60, weights)` katkı `w/(k+rank+1)`; `fuse_ranked` `(-score, id)` tie-break; k≤0 → ValueError.

### hybrid_retriever.py
`HybridRetriever(semantic, bm25).retrieve(top_k, alpha=0.7)`: min-max normalize (eşitse 0.5), `distance==0.0` doğru işlenir (`is not None`), kesim **sonra** yalnız metni olan chunk'larla doldurulur.

### query_router.py
`classify_query`: ≤2 kelime → lexical; ≤6 kelime ve (kısaltma|rakam|tırnak) → lexical; aksi semantic. `convex_fuse(dense, bm25, alpha)` (Bruch et al. 2210.11934: konveks > RRF).

### reranker.py (heuristik)
Ağırlıklar semantic 0.40 / keyword 0.30 / section 0.20 / formula 0.10; bölüm öncelikleri abstract 1.0 … references 0.1; 9 LaTeX deseni; `tanh(2x)/tanh(2)` sıkıştırma.

### cross_encoder_reranker.py / flashrank_reranker.py
Opt-in; model yüklenemezse heuristiğe düşer; FlashRank ONNX-int8 (~30-100 ms) CPU'da bge-reranker-base'in (>15 s) yerine öncelikli; eksik id'ler sona eklenir (chunk düşmez); `SimpleNamespace` shim ile paketsiz test.

### reranking_retriever.py (orkestratör)
Akış sırası: `enabled=False` → düz dense · router → lexical ise konveks hibrit / semantic ise saf dense · graph → PPR + RRF · rrf → dense+BM25 RRF · varsayılan: `candidate_k = k*overfetch` → dense → (hybrid) BM25 adayları → rerank → `[:k]`. `_default_reranker`: FlashRank → CrossEncoder → heuristik.

### contextual_chunker.py
`ChunkQualityFlags` (formül/eksik formül/tablo/tanım/teorem/eksik argüman/komşu bağlam/prev-next) — sıcak yolda çağrılmaz; `chunk_quality_flags` tablosunu dolduran ayrı yol.

### paper_indexer.py (uçtan uca ingest)
- `build_embed_text(text, title, section, contextual)` — ön-ek yalnız embedding içindir (Anthropic Contextual Retrieval).
- `ingest_one(disc, force=False)`: (1) hash var **ve** `has_embedded_chunks` → skip; hash var ama gömülü chunk yok → **yarım ingest, yeniden işle**; (2) parse+metadata; başlık yoksa dosya adından; (3) başlık dedup; (4) `extracted_text/*.txt` + `metadata/*.json`; (5) `upsert_paper`; (6) eski chunk'ları SQLite+Chroma'dan sil; (7) `add_chunks(embedded=0)`; (8) embed; (9) `chroma.add`; (10) `mark_chunks_embedded`; (11) `bm25_corpus.reset_cache()`; (12) `graph_corpus.reset_cache()`; (13) best-effort FormulaExtractor → ConceptGraph → CrossPaperSynthesizer.
- `ingest_directory`.

### sqlite_store.py / mastery_store.py
§6.1-6.2.

## 7.3 `app/brain/` (10 dosya, 1702 satır)

### local_llm.py
`LocalLLM(model, host)`: `available()`, `active_backend()` (ollama/openai/anthropic/google), `generate(prompt, fmt=None, timeout, max_tokens, seed, temperature, system)`; httpx hataları → `LLMUnavailable`. Backend seçimi `llm_backend` + anahtar varlığına göre; Qwen3 thinking-mode yanıt fix'i.

### mlx_llm.py
MLX-LM subprocess sarmalayıcı (Apple Silicon, LoRA adapter desteği); `/api/ask` `adapter_version` verilirse bu yol.

### rag_answerer.py
`RagAnswerer.answer(question, top_k)`: RerankingRetriever → (opsiyonel) `rag_abstain` güven kapısı (min_similarity 0.55, margin 0.02) → zayıf retrieval'da **LLM çağrılmaz** → `rag_reorder_context` (lost-in-the-middle) → LLM → `rag_verify_citations` dayanaksız atıf uyarısı. `llm_used` bayrağı döner; kaynak yoksa "No sources found".

### knowledge_card_builder.py
- `KnowledgeCard(paper_id, title, year, domain, main_claim, methods[], datasets[], trading_relevance, limitations[], possible_strategy_hypotheses[], risk_warnings[], implementation_notes[])`.
- `_MIN_SOURCE_CHARS=1500` (altı LLM'e gitmez). Retry merdiveni: 6000 char/700 tok → 3000/500 → >12000 ise orantılı kesitler.
- `_classify_card` → `difficulty` (0.1/0.2/0.4 + 0.2) ve `stage` (`lora_phase_1..4`); `trust_level` daima `draft`; kart `pending`, `lora_eligible=0`. `paper_id.strip()` (`\r` savunması).

### synthetic_qa_builder.py
`SyntheticQABuilder`: chunk'tan grounded QA; `_MIN_ANSWER_CHARS=60`, `_is_grounded` (anchor token ∩ ≥1, sayı alt-kümesi), persona rotasyonu, one-shot örnekte **"Pasaja göre" sızıntı öneki YOK** (v5 dersi), JSON coerce, `dedup_jsonl_lines` (hash + Jaccard ≥0.9), seed iletilir, atomik yazım.

### query_expander.py / multi_query_retriever.py / answer_quality.py / prompt_loader.py
Kural tabanlı sorgu genişletme; RRF ile çok-sorgu birleştirme; `answer_quality` deterministik güven/zayıflık/reorder/atıf doğrulama yardımcıları; `app/prompts/*.md` yükleyici.

## 7.4 `app/research/` (12 dosya, 3063 satır)

### orchestrator.py — ResearchOrchestrator
`run(question, iterations=3, paper_ids=None)`: her iterasyon sentez (veya yansıma sonucu IR) → `StrategyIR.model_validate` (hata → `example_ir()`) → `len(df)>10000` ise timeframe 1h → `run_backtest` + `evaluate` → **L5 CompositionGate** (`_signature` ile novelty) → `save_research_session` → `pass` değilse `ReflectionAgent.reflect` → tek değişiklik → tekrar. `research_sessions.verdict` L5 gerçek sinyal kaynağıdır.

### synthesis_engine.py
`_SYNTHESIS_PROMPT` (formüller + kavram ilişkileri + soru + kurallar: "garanti deme", "neden başarısız olabileceğini açıkla", "maliyeti hesaba kat", tek kural, RSI eşiği geniş, yalnız JSON). `SynthesisResult(indicator_name, description, source_papers, formula_components, combination_reasoning, expected_edge, failure_conditions, strategy_ir)`. `_parse_result`: IR varsayılanları (RSI14/ATR14, `rsi_14 > 50`, costs 0.0005+0.0005), `entry_rules[:1]`, RSI eşiği >53 → 50. `_build_failure_hint` son 3 başarısızlık + az-işlem uyarısı.

### formula_extractor.py
LLM (`fmt=json`, text[:3000]) → yoksa kural tabanlı 11 gösterge (**kelime sınırı** `\bRSI\b`; `schema`→EMA yanlış eşleşmesi kapatıldı). `formula_id=fml_<hex12>`, makale içi dedup.

### concept_graph.py
`_VALID_RELATIONS={extends, measures, limits, combines, opposite_of, requires}`; `build_from_papers` önce `delete_concept_links_for_paper` (idempotent); LLM seed=42.

### cross_paper_synthesizer.py
8 fallback şablon (UAM, VNM, RSTF, KMDP, HRRS, UAK, VBTT, RCRSI); `_CATEGORY_ALIASES`; `_example_id=syn_+sha256(ids)[:24]`; `_synthesis_seed=int(sha256(block)[:8],16)`; **≥2 farklı paper_id şartı** (ikili ve üçlü); `merge(TrainingExample(example_type="cross_paper_synthesis"))`.

### reflection_agent.py
Kural tabanlı iyileştirme promptu (az işlem → eşik düşür; aşırı DD → eşik artır/EMA filtresi; >2000 işlem → eşik artır; Sharpe<-1 & >500 işlem → yön ters); **bir iterasyonda tek değişiklik**; `strategy_ir` yoksa None.

### chain_data_builder.py
`research_sessions` → `data/training/jsonl/research_chains.jsonl` ({prompt, completion}; Düşünce/İndikatör/Bileşenler/Giriş kuralları/Beklenen avantaj/Başarısızlık/Backtest/Yansıma/İyileştirme).

### rag_learning_loop.py — sunucu-taraflı otonom döngü (varsayılan KAPALI)
- `is_substantive_card`: title ≥8 ve main_claim ≥40 alfanümerik.
- `RagLoopState` ayarlar: `enabled=False`, `interval_min=30` (5-1440), `fetch_enabled=True`, `fetch_interval_hours=24` (1-720), `max_fetch_per_cycle=5` (0-50), `cards_per_cycle=3`, `scores_per_cycle=5`, `score_use_llm=True`, `rebuild_empty=False`. Durum: stage ∈ idle|fetching|carding|rebuilding|scoring|paused_training|error; `history` son 20 tur; `rebuild_attempts` (≤3).
- Tur: `_fetch_new_papers` (auto_ingest kayıtlı arXiv sorguları) → `_build_missing_cards` → `_approve_if_content` → `_rebuild_empty_cards` (opt-in) → `_score_missing` (kart skordan yeniyse) → mastery.
- **LoRA eğitimi sürerken `paused_training`.** `asyncio.Lock` — eşzamanlı tur yok. `background_loop` 15 s heartbeat.

### literature_scout.py
4 konu paketi (rag/lora/rlm/math-physics); arXiv + çevrimdışı anahtar-kelime skoru; PDF'i gelen kutusuna indirir (`%PDF` doğrulama, idempotent), `BULUNANLAR.md` + `docs/egitim/<konu>-watchlist.md`. **AST testi ile ingest/train çağırmadığı sabit.**

### rag_trend_scanner.py
Güncel RAG yöntemlerini arXiv'de tarar → `docs/egitim/rag-watchlist.md` (deterministik skor, id dedup, `|`→`/`).

### synthesis_paper.py
`research_sessions`'tan sentez makalesi `reports/synthesis/sentez_*.md`; `HEKTOR_SYNTHESIS_MIRROR_DIR` aynası best-effort (OSError yükselmez).

## 7.5 `app/learning/` — Paper Mastery (8 dosya, 962 satır)

| Modül | Öz |
|---|---|
| `paper_inspector.py` | Statik 40 puan: parse 10 (4+3+2+1), metadata 5, chunk kalitesi 15 (≥3 chunk +6, section +3, kısa/uzun oranı <0.10 +6 / <0.25 +3), index 10 (hepsi gömülü +4 / kısmen +2, +3, +3 koşullu). `missing_steps`: paper_not_found, no_chunks, no_extracted_text, missing_title, too_few_chunks, incomplete_embedding |
| `question_generator.py` | LLM'siz şablon: 6 yapısal (main_claim/method/summary/dataset/result/limitation) + karttan (ilk 3 hipotez `trading_hypothesis`, ilk 2 formül) + 2 abstention sorusu ("kaç trilyon dolar kâr", "hangi kripto borsası"); tekilleştirme; `count=20` |
| `rag_exam_runner.py` | `_run_one`: RAG cevabı → `context_precision` (yalnız gözlem, geçme kararına katılmaz) → `no_answer` (boş / "No sources found" / `not llm_used`) → abstention: `no_answer or paper_id ∉ cited` → regular: `passed = not no_answer and context_ok and cit≥0.3 and gnd≥0.4 and not hallucination`; atıf skoru cevaptaki gerçek `[paper:chunk]` atıflarından; boş grounding → 0.0 |
| `mastery_scorer.py` | 100 puan = statik 40 + retrieval 15 + citation 15 + grounding 15 (−halüsinasyon cezası ≤5) + abstention 10 (soru yoksa nötr 5) + formül bonusu 5. Durum: ≥90 learned / ≥75 usable_needs_review / ≥60 partially_learned / ≥40 needs_rechunking / failed |
| `status_manager.py` | 14 geçerli durum; tarihçeden türetilir; `status_from_score` aynı eşikler |
| `report_generator.py` | `reports/papers/mastery/{paper_id}_mastery_report.{json,md}`; idempotent |
| `paper_mastery_agent.py` | 6 adım: create_test → inspect → questions → exam → score → finish + status; rapor ayrı try/except (Windows kilit); `LearningQueue` (`failed` + attempts<3 yeniden seçilir) |

## 7.6 `app/verification/` (19 dosya, 2540 satır)

### Cevap doğrulama
| Modül | Öz |
|---|---|
| `citation_verifier.py` | `[paper:chunk(, s.N)]` regex; her atıf getirilen chunk'larla eşleşiyor mu (`CitationCheck.exists`) |
| `grounding_verifier.py` | Cümle cümle `SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / SPECULATIVE` (token örtüşmesi + spekülatif regex) |
| `context_sufficiency.py` | `SUFFICIENT / PARTIALLY / INSUFFICIENT / CONTRADICTORY / MISSING_FORMULA_CONTINUATION / MISSING_ARGUMENT_CONCLUSION`; `can_answer` (query parametresi kullanılmaz — bilinen kusur) |
| `contradiction_detector.py` | Chunk'lar arası zıt iddia tespiti |
| `confidence_scorer.py` | 4 bileşen: context 0.25 / citation 0.30 / grounding 0.30 / formula integrity 0.15; `_ABSTAIN_THRESHOLD=0.40`, `_WARN_THRESHOLD=0.70` |
| `abstention_policy.py` | Düşük güven / yetersiz bağlam → çekimser mesaj |
| `comprehension_scorer.py` | 3 katman: A kart doluluk (7 alan) 0.30 · B RAG precision@5 0.40 · C LLM anahtar-kelime doğrulama 0.30 (seed 7, temp 0; LLM yoksa 0.5 nötr; `use_llm=False` hızlı mod) |
| `rag_mastery.py` | `mastery = 0.40·coverage + 0.30·comprehension + 0.30·train_readiness(min(1, n_examples/50))`; `empty_cards = n_cards − n_examples` |

### `exams/` — Anlama merdiveni
| Modül | Öz |
|---|---|
| `safe_eval.py` | Whitelist AST aritmetik (`+ − × ÷ ** % //`, unary); `UnsafeExpressionError`; **eval/exec yok** |
| `reference_oracle.py` | Sınavların TEK gerçek kaynağı = `compute_indicator` |
| `registry.py` | Her gösterge için referans + parametreler |
| `l3_application.py` | **L3 UYGULAMA:** formül + tutulan sayılar → model hesaplar → `np.allclose`; LLM yoksa `skipped` |
| `l4_counterfactual.py` | **L4 KARŞIOLGU:** parametre değişiminin yönü koddan türetilir (`_roughness`); model "artar/azalır/aynı" yorumlanır (olumsuzluk dahil); belirgin yön yoksa `no_data` |
| `l5_composition.py` | **L5 KOMPOZİSYON:** 3 kapı — math (`_REGISTRY`={EMA,SMA,RSI,ATR,MACD,ENTROPY,PERMENTROPY}, kural sınırları), novelty (≥2 gösterge tipi, kopya imza yok), maliyet-dahil backtest+OOS → `candidate` |
| `discipline_exam.py` | `evals/{discipline_core,overfit_awareness,risk_management}.jsonl` ile disiplin/dürüstlük basamağı |
| `understanding_score.py` | `pass_rate = passed/(passed+failed)`; skipped/no_data paydaya girmez; `_MIN_GRADED_FOR_SCORE=3` → altı `insufficient_data`; seviyeler Taban (abstention), L1 (citation≥0.3), L2 (grounding≥0.4 & no hallucination), L3, L4, L5; `composition_to_result` (test-edilemez backtest → `skipped`); 2 ardışık LLM hatasında bail; `score_full_ladder(llm=...)` adapter ölçümü; `l5_results_from_sessions` gerçek kompozisyonlara öncelik |
| `understanding_record.py` | `record_understanding` (DB + `reports/evals/understanding/*.json`), `load_understanding_history`, `compare_understanding` (aynı llm_model → `regressed`) |

## 7.7 `app/evals/` (10 dosya, 1267 satır)

| Modül | Öz |
|---|---|
| `metrics.py` | `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k` (saf Python) |
| `golden_dataset.py` / `golden_generator.py` | Sabit sorular; sızıntısız chunk-düzeyi golden-set üretici |
| `retrieval_eval.py` / `answer_eval.py` | Golden'a karşı retrieval ve cevap (citation/grounding/abstention; güven `0.4·cit+0.6·gnd`) |
| `rag_ragas_offline.py` | LLM'siz `faithfulness` (eşik 0.3), `context_precision` (0.06), `context_recall`; proxy, mutlak değil |
| `regression_runner.py` | Golden'a karşı otomatik regresyon |
| `trading_hypothesis_evaluator.py` | Hipotez "sinyal" değil test-edilebilir mi: testable/maliyet/OOS/tavsiye-dili (sure-fire/guarantee/can't-lose; olumsuzlama farkında)/risk → ACCEPT/REJECT |
| `eval_runner.py` | `EvalRunner.run(type)`: `trading-hypothesis`, `rag-retrieval`; `rag-answer/lora/rlm-reward` ERTELENDİ (NotImplementedError); `--strict` → `EvalGateError`; `reports/evals/eval_*.json` |

`app/reliability/release_gate.py`: `ReleaseGate` eşikleri (ör. recall@10 ≥ 0.70); eksik/NaN/inf/None/bool → **fail-closed**.

## 7.8 `app/rlm/` — RLM Controller (16 dosya, 2782 satır)

> RLM yeni bilgi deposu DEĞİLDİR (RAG odur); mevcut retrieval + doğrulama modüllerini reasoning kontrol katmanında birleştirir.

### rlm_controller.py (789)
Akış: `classify → plan → (retrieval ⇄ reformulation)* → evidence gate → draft (LLM) → claim extraction → citation/grounding verify → contradiction → confidence → abstention → yapısal nihai cevap → run logları + JSON rapor`.
- `answer(query, paper_ids, top_k, rounds, write_report)`; durumlar `answered`, `answered_with_limitation`, `abstained`, `no_llm`, `failed`. Bayat `running` reaper.
- `_gather_evidence` her turda farklı sorgu; `_reformulate` bölüm odaklı; `_draft` `max_tokens=900`, `timeout=600`, seed 42.
- **İki-kapılı güvence:** kanıt < `rlm_min_evidence_to_retry` → LLM hiç çağrılmaz (`abstained`); cevap sonrası desteklenmeyen iddia atılır; düşük güven → çekimser.
- `_apply_trading_guard`: soru VEYA cevap trading dili taşıyorsa zorunlu uyarı bloğu (classifier'dan bağımsız).
- Uydurma satır-içi atıf gövdeden temizlenir; `abstained` yüksek güven rozeti taşımaz; kanıt-log hatası başarılı run'ı ezmez.

### Diğer modüller
| Modül | Öz |
|---|---|
| `task_classifier.py` | Kural tabanlı, deterministik `ReasoningPlan` (görev tipi, bölüm hedefleri, atıf zorunluluğu); trading planı hipoteze izin verir, sinyale asla |
| `evidence_builder.py` | `EvidenceSufficiencyScorer` 0-100: relevance, coverage, section_diversity (min(1, n/3)), citation_availability, method_limit_presence, recency_metadata, contradiction_risk (−5/çelişki) |
| `claim_extractor.py` | Taslağı iddialara böler, dayanak durumunu işaretler |
| `rlm_store.py` | `rlm_runs/steps/evidence/verifications` ORM |
| `lora_candidate.py` | §16: `final_confidence≥0.85`, `citation≥0.90`, `grounding≥0.90`, status ∈ {answered, answered_with_limitation}; export JSONL `requires_human_approval=true`; **aday ≠ eğitim verisi** |
| `engine_config.py` | `ALLOWED_TOOL_NAMES`; `public_engine_config()` sır içermez |
| `tool_registry.py` / `safe_tools.py` | Deny-by-default allowlist; araçlar: `rag_search`, `get_paper_metadata`, `get_paper_chunks`, `calculator` (safe_eval), `citation_check`, `grounding_check`, `contradiction_check`, `formula_check`; wrapper istisnası yapısal hata döner |
| `answer_pipeline.py` | Motor seçimi (native/alexzhang); alexzhang yolunda cevap **yalnız desteklenen iddialardan** yeniden kurulur; `support_level` strong/partial/weak |
| `adapters/base.py` / `native.py` | Sözleşme; native = RlmController sarmalayıcı (VARSAYILAN, daima available) |
| `adapters/alexzhang_rlm.py` | Opsiyonel `rlms` paketi; backend `anthropic` veya `local_openai_compatible`; docker preflight; trajektori `reports/rlm/trajectories/{safe_run_id}.json` (>50 MB okunmaz); temperature 0.0 |
| `adapters/security.py` | Üretimde `local`/`ipython`/bilinmeyen ortam bloklanır; shell/network/fs-write bloklanır; `production_mode=False` yükseltmez |

## 7.9 `app/trading/` (10 dosya, 1533 satır)

### strategy_ir.py
```python
_RULE_RE = r"^\s*([a-zA-Z_]\w*)\s*(<|<=|>|>=|==|!=)\s*([a-zA-Z_]\w*|-?\d+(?:\.\d+)?)\s*$"
IndicatorSpec(name, period=14, ...)   RiskSpec(stop_loss="2 * ATR", take_profit=None, position_size="fixed_fractional")
CostSpec(commission=0.0005, slippage=0.0005)
StrategyIR(name, market="XAUUSD", timeframe="15m", indicators[], entry_rules[], exit_rules[], risk, costs)
parse_rule(rule) -> (lhs, op, rhs)     example_ir()     to_pine()  (TradingView v5 taslağı)
```
Kural = `kolon OP kolon|sayı`; başka hiçbir şey parse edilmez (Kural 5).

### indicators.py (vektörize pandas/numpy)
`ema`, `sma`, `rsi`, `macd`, `atr`, `bollinger`, **`entropy`** (yönsel ikili Shannon [0,1]), **`permutation_entropy`** (Bandt-Pompe, order=3), **`forbidden_pattern_rate`** (0711.0729), **`complexity_entropy`** (Jensen-Shannon MPR, 1808.01926). `compute_indicator(name, df, period)` registry: EMA, SMA, RSI, ATR, MACD, ENTROPY, PERMENTROPY, FORBIDDEN(RATE), COMPLEXITY(ENTROPY), BOLLINGER/BB (mid). Yeni indikatör → buraya + test.

### backtester.py
`_compute_columns` (ör. `rsi_14`), `_eval_rules` (AND), `_position_series` (durum makinesi; tek Python döngüsü), `_net_returns(position, bar_ret, cost_per_turn)`: `eff_pos = position.shift(1)` (look-ahead yok), turnover kaydırılmış pozisyondan, maliyet = commission+slippage her pozisyon değişiminde. `BacktestMetrics` (n_trades, total_return_pct, sharpe, sortino, max_drawdown_pct, profit_factor, win_rate_pct). `persist_backtest` → `strategies`/`backtests` + `reports/backtests/*.json`.

### overfit_checks.py / evaluator.py
`in_out_of_sample(split=0.7, min_trades=30)`; `static_checks` (az işlem, aşırı DD, …). `evaluate` verdict: **fail** = OOS n_trades<30 veya DD<−50% veya OOS getiri≤0; **inconclusive** = uyarı var veya OOS Sharpe<0.5; **pass** = aksi. Hiçbir şey OOS olmadan "başarılı" ilan edilmez.

### risk_manager.py
`compute_kelly`, `compute_drawdown_scale`, `compute_fixed_risk`, `analyze_risk` → `RiskReport`; `_extract_trade_returns` maliyet-dahil gecikmeli pozisyondan (Kelly şişme fix'i).

### market_data_loader.py / strategy_generator.py / package_exporter.py
CSV yükleme (open/high/low/close zorunlu), sentetik OHLCV (seed); hipotezden aday IR; `.achpkg` (ACHPKG_VERSION="1") Pine + Python export.

## 7.10 `app/tools/`, `app/pipeline/`, `app/reliability/`

- **tools/** — `tool_registry` (ToolDescriptor, `requires_seed`, `validate_params`, `resolve`; 7 yerleşik araç), `probability_simulator` (seed'li Monte Carlo + risk-of-ruin + VaR/ES; saf numpy; boş/negatif/inf → ValueError), `statistics_checker` (betimsel + permütasyon p-değeri, scipy YOK, p asla 0), `result_verifier` (Sharpe>5/Kelly>1/inf/nan uyarısı). Çalıştırmalar `tool_runs`/`tool_artifacts`'a `log_tool_run` ile.
- **pipeline/auto_researcher.py** — onaylı kartlardan soru çıkar → `ToolUseTrainer` seansları → DPO skorlama; `dry_run`, seed; seans hatası döngüyü kırmaz.
- **reliability/release_gate.py** — §7.7.

## 7.11 `app/lora/` — LoRA Control Plane (14 dosya, 2893 satır)

### gates.py — Gate 0-8
`_card_text(card)` tüm alanları (title, main_claim, methods, datasets, trading_relevance, limitations, hypotheses, risk_warnings, implementation_notes) toplar — eksik alan Gate 7'yi atlatıyordu (BLOCKER-sınıf fix).

| Gate | Fonksiyon | Ne | Sınıf |
|---|---|---|---|
| 0 | `gate_0_source(cards, valid_paper_ids)` | Kaynak bütünlüğü: `paper_id ∉ list_paper_ids()` → **orphan** (varlık denetimi `valid_paper_ids` verilirse) | fail |
| 1 | `gate_1_schema(examples)` | SFT şeması, rol sırası | fail |
| 2 | `gate_2_curriculum(cards)` | `difficulty` aralığı geçerli mi | fail |
| 3 | `gate_3_domain(cards)` | En az bir domain (`domain_classifier`, 8 domain) | fail |
| 4 | `gate_4_quality(cards)` → (result, kept) | `quality_filter`: <50 karakter, soru tekrarı, duplicate (oturum-içi hash) | eleme |
| 5 | `gate_5_math(cards)` | `math_verifier`: lookahead, aşırı-emin dil, >%1000 getiri, >%100 risk, çıplak accuracy/Sharpe iddiası → review; kanıt bağlamı (backtest/dönem) varsa işaretlenmez | uyarı |
| 6 | `gate_6_philosophy(cards)` | Tavsiye dili (`directly applied`, `traders should`, `superior performance`; TR karşılıkları; olumsuzlanmış hâl hariç); **yumuşak blok:** ≥20 kartta >%25 inceleme oranı → FAIL, küçük batch asla bloklamaz | uyarı/soft |
| 7 | `gate_7_safety(cards)` | `safety_scanner`: api_key (entropi+ön-ek), credential, email, phone, national_id (TC checksum+bağlam), finansal direktif; **tek ihlal tüm batch'i reddeder** | **BLOCKER** |
| 8 | `gate_8_split(examples)` → (result, DatasetSplit) | `dataset_splitter`: kaynak-bazlı 0.8/0.1/0.1, seed 42, sızıntı denetimi; boş valid/test → FAIL | fail |

### control_plane.py
`LoRAControlPlane.run_audit()` (Gate 0-7) / `run_full()` (0-8 + split); içeriksiz kartlar (`_card_text()==""`) gate'lerden önce filtrelenir; Gate 7 Gate 4'te elenenleri de tarar; `PipelineReport.passed`; `reports/lora/audit_report.md`. Herhangi gate başarısız → eğitim BAŞLATMAZ.

### auto_pipeline.py — AutoLoRAPipeline
`PipelineStage`: IDLE → CHECKING → GATE_FAILED | READY_TO_TRAIN → TRAINING → TRAIN_FAILED | EVALUATING → EVAL_FAILED | EVAL_SKIPPED (terfi edilemez) | EVAL_PASSED → PROMOTED. Durum `storage/auto_lora_state.json`. `check_and_prepare` (Gate 0-8), `start_training` (yalnız READY_TO_TRAIN + `authorize_training_action` + platform tespiti `detect_lora_backend()`), `_run_eval` (`evaluate_adapter` + anlama-merdiveni base-vs-adapter; `adapter_rate < base_rate − 0.05` → regresyon; bağımlılık yoksa EVAL_SKIPPED), `promote_to_production` (yalnız EVAL_PASSED + onay), `background_loop`. **Efektif `auto_enabled` = `settings.unattended_training_enabled` (True)**; konstrüktör varsayılanı False yalnız doğrudan örnekleme içindir.

### Diğer modüller
| Modül | Öz |
|---|---|
| `dataset_builder.py` | `build_dataset(approved_cards)` → SFT örnekleri (yalnız içerikli kartlar örnek üretir) |
| `dataset_splitter.py` | `split_dataset` (seed 42, 0.8/0.1/0.1, `source_id` gruplaması), `check_leakage` |
| `curriculum.py` | difficulty → LEVEL_0..LEVEL_4 (üst sınır kapsar); `card_json` parametresi kullanılmaz |
| `domain_classifier.py` | 8 domain (mathematics, physics, statistics, philosophy, trading, coding, ai_system_design, risk_management); EN/TR anahtar kelime; noktasız-ı büyük harf; enum sırasında sonuç |
| `math_verifier.py` | Regex tabanlı (§Gate 5) |
| `quality_filter.py` | §Gate 4; skor [0,1]; `_quality_reason` |
| `safety_scanner.py` | §Gate 7; FP korumaları (hacim ≠ TC, çıplak sayı ≠ telefon, uzun hash ≠ api_key, hiperparametre ≠ credential); FN korumaları (AWS base64, `password=`, geçerli TC) |
| `adapter_registry.py` | JSONL durum makinesi `registry/adapters/registry.jsonl`: candidate → smoke_passed → eval_passed → approved → production; `promote(user_approved=True)` şart; aynı anda tek PRODUCTION; reject notu |
| `card_curation.py` | `card_richness`; kanonik = en zengin → en yeni; orphan (`paper_id ∉ valid`) + per-paper version-collapse → `set_card_lora_eligible(0)` (review_status korunur, geri alınabilir, idempotent); `--dry-run` varsayılan |
| `peft_llm_shim.py` | PEFT adapter → `LocalLLM` arayüzü; `load_base_of(adapter_dir)` adapter'ın KENDİ base'i (elmayla-elma; 4B-vs-1.5B sahte-gerileme fix'i) |

## 7.12 `app/training/` (20 dosya, 4073 satır)

### peft_lora_train.py (828) — Windows/Linux trainer
- `PeftTrainConfig`: base_model, train_jsonl, valid_jsonl, adapter_output_path, iterations=300 (**toplam adım**, `max_steps`), batch_size=1, learning_rate=2e-4, lora_r=8, lora_alpha=16, lora_dropout=0.05, max_seq_length=1024, `use_rslora`, `use_dora`, `init_lora_weights="true"` (pissa/olora/corda GGUF-güvensiz uyarısı), `loraplus_lr_ratio`, weight_decay=0.01, warmup_ratio=0.03, lr_scheduler_type=cosine, max_grad_norm=1.0, `neftune_noise_alpha=0.0`, **`assistant_only_loss=False`** (opt-in), **`kl_reg_beta=0.0`**, seed=42, `max_examples=0`.
- `build_lora_kwargs`, `build_training_kwargs(num_epochs, output_dir, on_cuda, max_steps)`, `recipe_summary` (vanilya vs teknik listesi), `load_lora_profile(name)` (`configs/lora/lora_profiles.yaml`), `dry_run(cfg)`.
- Satır formatları: `{messages}`, `{prompt, completion}` (MLX), `text`.
- **`build_masked_labels(prompt_ids, full_ids)`** → prompt token'ları `-100` (tekrar döngüsü kökten çözüldü); `_chat_input_ids` (list ve BatchEncoding); `_MaskedDataCollator(pad_token_id)` label pad `-100`; `sample_rows(rows, max_examples, seed)`.
- `_KLRegTrainer` (`disable_adapter` ile ek model yüklemez; ikinci forward ~1.5-2×); `_make_trainer_cls(beta)`.
- `train(cfg)`: AutoModelForCausalLM + PEFT → Trainer → `_write_loss_curve` → `reports/training/<ad>_loss.json`.
- `generate_colab_notebook`, `build_command`. `__main__` `--run` gerektirir (`--profile` kabul etmez — bilinen kusur).
- `target_modules` yedi projeksiyonla sınırlı (`lm_head`/`embed_tokens` GGUF'ta düşer).

### mlx_lora_train.py / backend.py
Apple Silicon `python -m mlx_lm.lora` sarmalayıcı (dry-run varsayılan); `detect_lora_backend()` → `mlx` (Darwin arm64) / `peft`.

### detached_launch.py (623)
- `_SPLIT_SEED=42`, `_VALID_RATIO=0.05`, `_LAUNCH_LOCK_TTL=120`.
- `ensure_train_split(settings)`: `lora_sft.jsonl` → `train/valid` (kaynak boşsa mevcut korunur = clobber guard); `build_training_split()` (kaynak yoksa `assemble_sft_lines` ile bir kez üret) + `_auto_register_dataset` (DatasetVersion pending, best-effort); `readiness()`; `training_status()` (web/CLI/detached fark etmez); `is_running()`.
- `_build_train_cmd(...)` test edilebilir; **`launch(adapter_name, iterations, base_model, profile="discipline_safe_local", max_examples)`** — profil varsayılanı v5 fix'i (`profile=""` bilinçli vanilya kaçışı); atomik kilit; `HEKTOR_TRAIN_SUPERVISED` alt sürece; `storage/train_status.json` (pid dahil); detached spawn.
- `read_detached_training_status`, `_pid_alive`, `is_detached_training_running(root)` (log-tazeliği yedeği root'a bağlı), `_terminate_tree`, `request_stop_detached_training` (pid kill + `STOP_TRAINING`).

### adapter_eval.py — dürüst gate
`_is_degenerate` (n-gram tekrar), `_flags_for` (boş cevap → `empty_answer` flag, kategorik veto; meşru çekimserlik non-empty), `_resolve_base_model(adapter_dir)` (adapter_config.json), `_load_model`, `_generate` (greedy, 220 token), `_MIN_EVAL_N=5`, `_decide_verdict`: degenerate → reject; adapter<base → reject (her n); n<min_n → inconclusive; adapter>base → accept; eşit → inconclusive. `evaluate_adapter(adapter_dir, eval_set, n, min_n)` → `reports/evals/adapter_eval_<ad>_<set>.json`. **Terfi etmez.**

### Diğer modüller
| Modül | Öz |
|---|---|
| `dataset_builder.py` | `training_examples` SQLite → JSONL `{prompt,completion}`; curriculum pacing %60 current / %30 prev / %10 next (`phase` ve >20 örnek); boş build clobber etmez; valid ⟂ train, valid min 4 |
| `dataset_quality.py` | **pretrain-gate #3**: `audit_dataset` GO/NO-GO — garanti vaadi zehiri → NO-GO; **>%40 tek açılış-bigramı** → NO-GO (v5 mekanizması); sızıntı öneki/maliyet-körü/küçük set → WARN; `recommend_epochs(n)` (1-3) |
| `discipline_dataset.py` | 9 tuzak (garanti/backtest'siz/maliyetsiz/kaynak-yok/bağlam-uyumsuz/look-ahead/overfit/kaldıraç/grounded-belirsizlik + R-Tuning abstain tuzakları `gelecek_tahmin`, `canli_veri_yok`) × 16 strateji × 3 varyant = **432 deterministik adversarial örnek**; açılışlar çeşitli; 1/3 system-prompt'suz; cevaplar naif `check_flags`'i geçer; `mix_discipline(ratio)` |
| `sft_assembly.py` | `assemble_sft_lines(settings, discipline=True, ratio=0.25, seed=0)`: synth-qa + küratörlü kart → dedup → **disiplin dedup'tan SONRA** karışır → `lora_sft.jsonl`; `to_jsonl_line` U+2028/2029/0085 kaçışı |
| `unified_dataset.py` | Tüm SFT kaynaklarını `data/training/unified_sft.jsonl`'de birleştirir (UTF-8 fix) |
| `evaluate_model.py` | `ModelEvaluator` (Ollama base'i ölçer; flag'ler `success_without_test`, garanti; temperature 0 + seed) |
| `reward_signal.py` / `dpo_dataset_builder.py` | 6 kriter (execution, trade_count, sharpe, drawdown, return, win_rate) → composite → chosen/rejected çiftleri → DPO JSONL; `reward_signals` tablosu |
| `tool_use_trainer.py` / `tool_use_dataset_builder.py` | THINK→CALL→OBSERVE→CONCLUDE döngüsü; `tool_use_examples`; MODEL eğitmez |
| `mastery_sft_builder.py` | Mastery sınavından SFT (min_score 75, citation 0.5) |
| `cloud_notebook.py` | Stage 2 unsloth notebook + Ollama Modelfile üretici (nbformat 4, base 2507'ye pinli) |
| `adapter_registry.py` | SQLite `adapters` metadata (+ `.meta.json`); **`app/lora/adapter_registry.py` ile aynı isimde ayrı sistem** |
| `unattended_policy.py` | `authorize_training_action(action, summary, gates_passed, agent_id)` → `TrainingAuthorization(authorized, mode, reason, approval_id)`: STOP_ALL → blocked; `unattended_training_enabled ∧ gates_passed` → unattended_policy; aksi `require_fresh_approval` (tek kullanımlık) — **tüm eğitim yüzeylerinin tek yetki kaynağı** |

### configs/lora/lora_profiles.yaml — 6 profil

| Alan | small_smoke_test | standard_reasoning | high_capacity_reasoning | discipline_safe | **discipline_safe_local** | discipline_safe_kl |
|---|---|---|---|---|---|---|
| r / alpha | 8/16 | 16/32 | 32/64 | 16/32 | 16/32 | 16/32 |
| dropout | 0.05 | 0.05 | 0.05 | 0.1 | 0.1 | 0.1 |
| epochs | 1 | 2 | 3 | 1 | 1 | 1 |
| max_seq_length | 2048 | 2048 | 2048 | 2048 | 1024 | 1024 |
| learning_rate | 2e-4 | 2e-4 | 2e-4 | 1e-4 | 1e-4 | 1e-4 |
| use_rslora | — | — | true | false | false | false |
| neftune_noise_alpha | — | — | — | 5 | 5 | 5 |
| warmup_ratio | — | 0.03 | 0.03 | 0.05 | 0.05 | 0.05 |
| assistant_only_loss | — | — | — | false (bulut `train_on_responses_only` maskeler; çift-maskeleme olmasın) | **true** | true |
| kl_reg_beta | — | — | — | — | — | 0.01 (arXiv:2512.22337, deneysel) |
| max_examples | 200 | — | — | — | 300 | 300 |
| target_modules | 4 (q,k,v,o) | 7 | 7 | 7 | 7 | 7 |

`discipline_safe_local` = `launch()`, `start-train.ps1` ve `training-watchdog.ps1` güvenli varsayılanı (v5 dersi: düşük lr + az epoch + yüksek dropout + NEFTune + grad-clip + gerçek maskeleme).

## 7.13 `app/registry/` (3 dosya, 702 satır)

- `version_store.py` — `RegistryStore`: `register_dataset(_from_file)` (SHA-256, idempotent), `find_dataset_by_hash`, `set_dataset_status`, **`cas_dataset_status`** (atomik), `register_rag_index` / `snapshot_rag_index` (SQLite sayımından), `register_embedding` / `snapshot_embedding`, `register_reward` / `set_reward_scan_flags`, `log_decision` (append-only), listeler.
- `promotion_gates.py` — `scan_secret_pii` (kendi mini regex seti; safety_scanner import edilmez), `approve_dataset` / `reject_dataset` (pending → approved|rejected terminal; CAS tek kazanan; çifte onay tek karar loglar), `check_rag_index_eval` (ReleaseGate), `gate_reward_dataset` (sır/PII → blocked).
- Kural 8: dataset onaylanmadan eğitime giremez; `promote` İNSAN onayı (CLI `registry-promote-dataset --approver`).

## 7.14 `app/feedback/` — Echo (3 dosya, 447 satır)

- `correction_safety_reason(*texts)` — Kural-1 zehir filtresi (garanti/kesinlik/advice/risk-free/canlı-sinyal dili) **tüm alanlarda** (soru SFT'de user-turn'e sızar).
- `EchoCollector.record(correction, question, bad_answer, source, correction_type)` → zehir → `rejected`; `approve` (güvenlik yeniden kontrol), `reject`, `to_sft_line`, `export_approved(out_path)` (proje kökü içinde `_ensure_within`; **ayrı aday dosya** `data/feedback/feedback_sft.jsonl`; kanonik sete oto-merge YOK; yazım penceresinde yeniden tarar), `summary`.
- `store.py` — `feedback_corrections` (created_at eşitliğinde deterministik sıra).

## 7.15 `app/monitoring/` — Sentinel (4 dosya, 688 satır)

- `sentinel.py` — `ProbeResult(name, status ok|warn|fail|skip, detail)`, `SentinelReport(overall, probes)`. **10 salt-okuma probe:** `probe_llm` (Ollama tags), `probe_web` (127.0.0.1:8765), `probe_training` (train_status + log), `probe_orchestration` (stale `running` peek, mutasyonsuz), `probe_stop_all`, `probe_disk`, `probe_sqlite` (`PRAGMA quick_check`, journal_mode'a dokunmaz), `probe_feedback` (bekleyen kuyruk), `probe_contention` (**DANIŞMAN** Resource Negotiator: CPU çekişmesi raporlar, eğitimi DURAKLATMAZ), `probe_rag_loop`. Agregasyon fail>warn>ok; hepsi skip → skip; probe istisnası skip (nöbetçi düşmez); geçersiz status normalize. `Sentinel.run(persist=True)`, `history(limit)`.
- `store.py` — `sentinel_checks`, `keep_last=1000` budama (aynı-timestamp bağında yeni kayıt silinmez).
- `self_heal.py` — `SelfHealingController`: 2 kayıtlı idempotent runbook (`_repair_orchestration` → `recover_stale(30)` + Sentinel doğrulama; `_repair_rag_loop` → `run_one_cycle`); **3 ardışık sağlıksız tur** şart; sağlıklı tur streak'i sıfırlar; başarısız onarım → üstel cooldown `min(21600, 300·2^(attempt−1))`; durum `storage/self_heal_state.json` (atomik); `background_loop(60 s)`; eğitim/onay/terfi ASLA.

## 7.16 `app/agents/` (28 dosya, 4409 satır)

### runtime/ — Phase 1 gözlem + Phase 2 kontrol düzlemi
| Modül | Öz |
|---|---|
| `schemas.py` | `AgentAutonomy` (manual/semi_auto/autonomous/requires_approval/dangerous_without_approval), `AgentRunStatus`, `AgentEventKind`, `AgentSpec`, `AgentEvent`, `AgentRun`, `TaskStatus` (pending/claimed/running/completed/failed/cancelled/blocked_approval/blocked_stop_all), `ApprovalStatus`, `RiskLevel`, `AutomationTask`, `ApprovalRequest`, `ApprovalDecision`, `SupervisorDecision` |
| `registry.py` | `load_agent_registry(manifest)` → `dict[agent_id, AgentSpec]`; bozuk → `ManifestError` (sessiz boş liste yok); `dangerous_agents`, `agents_requiring_approval` |
| `tracker.py` | `RunTracker` (SQLite `agent_runs/agent_events` + `reports/agent_runs/<run_id>.jsonl`); `run_id=arun_YYYYMMDD_HHMMSS_<8hex>`; `track_agent_run` ctx, `@tracked(agent_id, trigger_type)` async-farkında dekoratör (`CancelledError` yakalar), `log_step`, `log_system_event`; retention 30 gün / 50.000; `cancel_stale_running_agent_runs(6h)`; **asla fırlatmaz** |
| `supervisor.py` | TEK kapı: registry'de mi → iptal mi → STOP_ALL (yalnız `dangerous`) → zaten çalışıyor mu → taze onay; `is_stop_all_active`, `create_stop_all(reason)`, `clear_stop_all`, `can_run_agent`, `run_with_supervision`. STOP_ALL = `storage/STOP_ALL` |
| `approvals.py` | `request_approval` (`apr_`), `list/get`, `approve`, `reject`, **`require_fresh_approval(agent_id, action, risk, summary)`** → `consume_fresh_approval` atomik CAS (`consumed_at`); `has_fresh_approval`; standing yetki YOK |
| `task_queue.py` | `create_task` (`atask_`), `claim_task` / `try_claim_task` (CAS), `complete/fail/cancel/mark_blocked/requeue_task` (yalnız blocked_*) |
| `executor.py` | Phase 2.5 hibrit yürütücü: `_HANDLERS` allow-list (boş başlar; bilinmeyen agent_id → `fail_task("handler yok")`), `run_task`, `run_pending(limit, retry_blocked)` → `run_with_supervision` |
| `handlers.py` | `register_default_handlers` (idempotent) — yalnız `model-advisor`; tehlikeli ajanlar (auto-lora-pipeline, arxiv-fetcher, rag-learning-loop) KAYDEDİLMEZ |
| `chain.py` | `load_chain`, `_topo_sort` (Kahn), `resolve_chain` → `ResolvedStep` (order, needs_approval); `ChainError` |
| `preflight.py` | `runtime_preflight`: manifest + 4 Phase-2 tablosu + STOP_ALL doğrula |

### local_training_*.py — Phase 5A-5E (salt-rapor)
| Faz | Modül | Sözleşme |
|---|---|---|
| 5A | `local_training_orchestrator.py` | Eğitim-hazırlık denetimi (skor ≥70 READY; <500 satır risk); DAİMA salt-rapor |
| 5B | `local_training_request.py` | Onay-kapılı İSTEK; `--create-approval` yalnız PENDING oluşturur, TÜKETMEZ |
| 5C | `local_training_dryrun.py` | Onayı READ-ONLY okur, adapter-eval MOCK, execution planı |
| 5D | `local_training_handoff.py` | Gerçek komutu YAZDIRIR (`uv run hektor train --run`), çalıştırmaz; `ready_for_human_execution` |
| 5E | `local_training_postcheck.py` | Eğitim sonrası READ-ONLY; `promotion_recommendation` DAİMA `human_review_required` |
Raporlar `reports/local_training_orchestrator/<ts>_{audit,request,dryrun,handoff,postcheck}.{md,json}`.

### Diğer ajanlar
| Modül | Öz |
|---|---|
| `system_profiler/profiler.py` | RAM/GPU/CPU profili (Windows GPU `wmic` — modern Windows'ta yok, "unknown"a düşer) |
| `model_advisor/advisor.py` | `recommend(profile, task, top_k)` → RAM/VRAM sert reddleri; `/api/recommend` |
| `installer/ollama_installer.py` | Yalnız güvenli whitelist komutlar; `pull_model` timeout 600 s |
| `benchmark/runner.py` | tps>15∧q>0.7 / tps>5∧q>0.5 / tps>2 eşikleri |
| `learning/memory.py` | Ayrı `storage/hektor_learning.db` (model_trials, error_patterns, rule_suggestions) |
| `learning/rules_updater.py` | Başarısız trial'lardan kural önerisi (`_MIN_FAILURES_FOR_BLACKLIST=3`, `_THROTTLE=2`, `_MIN_ERROR_OCCURRENCES=2`, `_MIN_TPS=5.0`); `pending_review`; `--approve` STOP_ALL + taze onay ister; HİÇBİR şeyi otomatik uygulamaz |

## 7.17 `app/orchestration/` (14 dosya, 4405 satır)

### pipeline.py — 12 kanonik aşama (frozen dataclass)
`preflight → collision → smoke → deep-hunt → data-gate → curriculum → dry-run → regression → approval → train → evaluate → registry`. `StageKind`, `StageStatus` (pending/running/completed/blocked/failed/skipped), `RunStatus`, `StageDef(autonomous: bool)`; `autonomous=False` aşamada orkestratör DURUR (Kural 8). `stage_order(name)`.

### store.py — OrchestrationStore
`orchestration_runs/stages/events` (`orc_`/`orst_`/`orev_`); WAL + busy_timeout; `create_run` tüm aşamaları per-run snapshot'lar; **`claim_stage_running` atomik CAS**; `touch_heartbeat` (bugün çağıran yok); `get_events(limit=200)`, `list_runs(limit=50)`.

### orchestrator.py — TrainingOrchestrator (saf durum makinesi)
`start(params)`, `step(run_id)` (TAM OLARAK BİR aşama; claim → delege → sonuç; koşu finalize olduysa yazmaz; `output_json` `default=str`), `run_until_blocked(max_steps=50)`, `recover_stale(timeout_min=30)` (heartbeat'i durmuş `running` → failed; terminal koşuları clobber etmez), `cancel` (asılı running → skipped), `status`, `timeline`, `list_runs`. Resume: tamamlanan atlanır, blocked/failed yeniden denenir.

### delegates.py — gerçek bağlama (savunmacı import)
`preflight` (STOP_ALL, veri, bağımlılık) · `collision` · `smoke` · `deep_hunt` (**hunt_ack yoksa blocked**) · `data_gate` (`audit_dataset` GO?) · `curriculum` · `dry_run` (komut önizleme) · `regression` · `approval` (`authorize_training_action("train_run", gates_passed=True)`; STOP_ALL → blocked; unattended ∧ gates → completed; aksi `has_fresh_approval`? — **tüketmez**) · `train_handoff` (unattended? completed : blocked) · `eval_handoff` / `registry_handoff` (skipped; auto_pipeline devralır). `default_delegates()`.

### driver.py — AutoDriver (claude -p / codex / gemini)
- Sabitler: `HUNT_TIMEOUT_S=1800`, `DRIVE_TIMEOUT_S=3600`, `DRIVE_TOKEN_TTL_S=3900`, `STOPPED_RC=-99`, `STOP_POLL_S=1.0`, `_HUMAN_SECRET_ENV=("HEKTOR_API_TOKEN",)`, `_SETTINGS_OVERRIDE_ENV` (3 `CLAUDE_CODE_*`).
- `drive(run_id, execute=False, engine=None, mode="hunt"|"drive")`:
  - **hunt:** hardened motor şart; koşu deep-hunt@blocked mi; `driver_scope.mint(run_id, ttl=2100)`; `build_child_env` (insan token'ı BOŞ, override env silinir, sürücü token+run_id eklenir); `_default_runner`: `Popen(argv, shell=False, cwd=_REPO_ROOT)` + `engine_procs.register` + 1 s STOP_ALL yoklama → `terminate_run` → rc=-99 → `stopped=True`; `parse_hunt_verdict` (son satır `HEKTOR_HUNT_VERDICT: PASS|FAIL`, yoksa FAIL); `verdict_audit.audit_hunt_evidence`; **PASS ∧ audit.ok → `hunt_ack=true` → `run_until_blocked`** (onay kapısında durur); `finally revoke_run`.
  - **drive:** `drive_hardened` şart; `write_mcp_config(storage/mcp/drive-<safe_run_id>.json)` (sır yazmaz); `mint(ttl=3900)`; `--tools Read,Grep,Glob --mcp-config <path>`; `parse_drive_verdict` (`HEKTOR_DRIVE_VERDICT`) — **hunt_ack YAZMAZ**; config unlink.
  - Prompt (hunt): SALT-RAPOR, kod/git/eğitim yasak, weekly-bug-scan deseni; (drive): eğitimi yasaklar, MCP kullan, dosya düzenleme yok, 403'ü aşmaya çalışma, CLAUDE.md oku.
- `_resolve_executable` (PATH'ten cwd çıkarılır, mutlak yol) — sahte binary savunması.

### engines.py — motor kayıt tablosu
- `PROMPT`/`MCP_CONFIG` sentinel'leri (`\x00…\x00`) tam-eşleşme tek argv öğesi; `PROBE_TTL_S=60`; `Engine(name, label, probe_cmd, argv_template, hardened, drive_argv_template, drive_hardened, quota_warning)`; `DEFAULT_ENGINE="claude"`.
- Motorlar: **claude** (hardened + drive_hardened), **codex** (hardened + drive_hardened; 5 saatlik yuvarlanan kota), **gemini** (doğrulanmamış), **local** (Ollama; süreç doğurmaz).
- `DISALLOWED_TOOLS = (Bash, Edit, Write, NotebookEdit, WebFetch, WebSearch, Task)`; av argv: `claude -p PROMPT --safe-mode --strict-mcp-config --disallowedTools <liste>` (variadic EN SONDA); sür argv: `--setting-sources "" --disable-slash-commands --strict-mcp-config --tools Read,Grep,Glob --mcp-config MCP_CONFIG` (**`--safe-mode` MCP'yi kapatır, `--bare` API-key'e indirger → ikisi de kullanılmaz**).
- `available()` = PATH'te kurulu mu (giriş durumu bilinemez); `run_blocked_reason`, `drive_supported`, `describe()` (kimlik alanı yok, `logged_in=None`), `LOGIN_UNKNOWN_NOTE`.

### engine_procs.py
Canlı motor süreç kaydı (`run_id` başına): `register/unregister`, `is_run_live`, `live_count`, `terminate_run`, `terminate_all(grace=5.0)`.

### verdict_audit.py — bağımsız verdict oracle (P8 + P9)
- `EVIDENCE_MARKER="HEKTOR_HUNT_EVIDENCE"` JSON bloğu: `scanned_files: [{path, line, quote}]`, `subsystems`, `findings: [{severity,...}]`.
- `extract_evidence` **her istisnada None** (RecursionError dahil; fail-closed).
- `audit_hunt_evidence(evidence, verdict)`: ≥`MIN_SCANNED_FILES=5` var olan dosya (yol-geçişi reddi, tekil), ≥`MIN_SUBSYSTEMS=2`, PASS + `_BLOCKING_SEVERITIES={HIGH,BLOCKER,CRITICAL,SEVERE}` bulgu → iç-tutarsız red; **okuma-kanıtı:** ≥`MIN_READ_PROVEN=5` dosya için `lines[line-1].strip()==quote.strip()` (`MIN_QUOTE_LEN=12`, `line:true` bool sayılmaz, aynı alıntı+dosya bir kez, `MAX_PROOF_FILE_BYTES=5_000_000` satır-satır okuma).
- Sınır: "hiç açmadan PASS" kapatıldı; "açtı ama düşünmedi" ikinci LLM doğrulayıcı ister (ertelendi).

### smoke.py / run_smoke.py
`SmokeRunner(llm, retriever)`: backend canlı → gerçek küçük üretim (seed 42, timeout 60, 32 token, temp 0; degenere değil) → gerçek retrieval (≥1 chunk; boş → warn). Verdict: çevrimdışı → **skip**; canlı+sağlıklı → pass; canlı ama boş/degenere/hata → fail. `run_smoke.py`: ⚡ RUN sözleşmeleri (10 yoklama: engine-registry, drive-mode-wiring, allow-list, MCP config sır, `drive(execute=False)` varsayılanı, …) + `LiveDriveSmoke --allow-live-spawn` (240 s; CI'da ASLA).

### collision.py / regression.py
- Collision (git salt-okuma, enjekte `git_runner`, 30 s): `.git/index.lock` / aynı-branch çoklu worktree / HEAD-drift → **fail (blok)**; kirli izlenen ağaç → **warn**; git yok → skip; block warn'a baskın.
- Regression: aday set metrikleri (`top_opening_share`, garanti vaadi, sızıntı, maliyet-körü, disiplin kapsamı, GO/NO-GO) son geçen baseline ile; toleranslar `0.0 / 0.05 / 0.02 / 0.0 / 0.02 / 0.0`; baseline yok → skip; **yalnız `--commit`** günceller (`storage/orchestration/regression_baseline.json`); provider istisnası → skip.

### unattended_supervisor.py
Desired-state reconciler (web lifespan 60 s): STOP_ALL? → motor canlı? → backoff? → `_ensure_run()` → `AutoDriver.drive(mode="drive")` (yalnız hardened); tek canlı motor; üstel backoff 300 s → 6 sa; `storage/unattended_supervisor_state.json` (göreli yol — bilinen kusur); varsayılan `engine="codex"` (DEFAULT_ENGINE ile uyuşmaz — bilinen kusur).

## 7.18 `app/web/` — FastAPI (16 dosya, 4984 satır + statik)

### Kurulum
`app = FastAPI(title, version="0.1.0", docs_url="/api/docs" if token boş else None, openapi_url=…, lifespan)`. `_expose_schema` import anında hesaplanır (token varsa `/api/docs` 404).

**Lifespan:** logging → `ensure_dirs` → NumPy tek-thread ön yükleme (Windows `0xC0000005`) → `warn_if_auth_disabled` → `cancel_stale_running_agent_runs` → arka plan: `auto_pipeline.background_loop`, `rag_loop.background_loop` (ISOLATE_CHROMA değilse), `self_healer.background_loop`, `unattended_supervisor.background_loop` → BM25 ısıtma thread'i (router/hybrid/rrf/graph açıksa).

**Middleware:** `TrustedHostMiddleware` (trusted_hosts doluysa) → `CORSMiddleware` (cors_origins doluysa) → `_security_middleware` (`/api/` rate limit; `_UPLOAD_PATHS` ek limit; `SECURITY_HEADERS`; HSTS).

**Router'lar:** ai_brain, ai_brain_ui, orchestration, feedback, sentinel, agent_graph, engines.

### security.py
- `require_auth` (`api_auth`): token boş → geç; Bearer/X-Api-Token `compare_digest`; geçerli **sürücü token'ı da kimlik olarak kabul** (motor ortamında insan token'ı boş); aksi 401.
- `resolve_scope`: `X-Hektor-Driver-Token` yoksa `human`; varsa `driver_scope.verify(token, run_id)` → geçersiz → **401** (human'a düşmez).
- `require_human` (`human_only`): scope driver → **403**.
- `SECURITY_HEADERS`: nosniff, `X-Frame-Options: DENY`, no-referrer, Permissions-Policy, COOP same-origin, CSP `default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'` (inline script yasak; fontlar self-host).
- `RateLimiter` (IP başına 60 s kayan pencere, 60 s'de sweep; proxy başlıklarına güvenmez).
- `validate_pdf_upload` (`.pdf`, boş, boyut, `%PDF-`), `validate_csv_upload` (`.csv`, ilk 8 KB decode, OHLC başlık), `sanitize_filename` (NFKD, 128 kırpma), `safe_destination` (traversal).

### driver_scope.py / sse_tickets.py
- Driver token: `sha256(token) → (run_id, expires_at)`; `mint(run_id, ttl_s=2100)` (önceki iptal), `verify` (tüketmez; run_id `compare_digest`), `revoke_run`, `reset`; restart → tümü geçersiz. Header'lar `x-hektor-driver-token`, `x-hektor-run-id`.
- SSE bileti: `DEFAULT_TTL_S=60`, `mint()` (`token_urlsafe(32)`), `consume()` (tek kullanım). `/api/training/stream?ticket=` — insan api_token query'de **kabul edilmez**.

### Route envanteri (A = api_auth, H = human_only)
- **Sistem:** `GET /api/status` A · `GET /api/version` A (30 dk throttle'lı `git fetch`) · `GET /api/healthz` A · `GET /api/profile` **auth yok** · `GET /api/recommend` **auth yok** · `GET /api/supervisor/status` A · `POST /api/supervisor/stop-all` A (bayrak + `terminate_all()`, `engines_terminated` döner) · `POST /api/supervisor/clear-stop-all` A+**H** (gizli) · `GET /api/events` A.
- **Makale:** `GET /api/papers` · `POST /api/papers/upload` (sha256 dedup; BackgroundTasks ingest) · `POST /api/ingest` · `GET/POST /api/papers/{id}/comprehension` · `GET /api/papers/comprehension/all` · `POST /api/papers/comprehension/batch` · `GET/POST /api/card/{paper_id}` · `POST /api/cards/batch` · `POST /api/card/{paper_id}/backtest` · `GET /api/arxiv/search` · `POST /api/arxiv/fetch` · `/api/arxiv/queries*`.
- **Onay:** `GET /api/cards/pending` · `GET /api/cards/approved` · `POST /api/card/{card_id}/approve` A+**H** (approved/not_found/**empty**) · `POST /api/card/{card_id}/reject` A+**H**.
- **RAG/RLM:** `POST /api/ask` (`AskRequest{question 3-2000, top_k 1-20, adapter_version}`) · `POST /api/rlm/answer` (write_report=False) · `GET /api/rlm/runs` (1-100) · `GET /api/rlm/runs/{id}` · `.../trajectory` · `GET /api/rlm/config` · `POST /api/rlm/test-adapter` · `GET /api/rag-mastery` · `GET /api/understanding-score` (`record=true` YAZAR) · `GET /api/understanding-score/history` (1-200) · `GET /api/learning/{summary,eval-history,training-runs,card-growth}`.
- **RAG döngüsü:** `GET /api/rag-loop/status` · `POST /api/rag-loop/enable` · `POST /api/rag-loop/run-once` · `POST /api/rag-loop/config` (8 kelepçeli parametre).
- **Backtest:** `POST /api/backtest` (yalnız sentetik; CSV için ayrı uç) · `POST /api/backtest/csv` (≥50 bar) · `GET /api/backtests` · `GET /api/backtest/{id}/risk` (**YAZAR** `rr_<id>`) · `GET /api/risk-reports` · `GET /api/backtest/{id}/pine` · `GET /api/backtest/{id}/download-pkg` · `GET /api/strategy/{name}/export` · `POST /api/package/export`.
- **Araştırma:** `GET /api/research/formulas` · `GET /api/research/graph` · `POST /api/research/extract` · `POST /api/research/run` · `GET /api/research/sessions` · `POST /api/research/chain-dataset` · `POST /api/synthesis/cross-paper` · `GET /api/synthesis/reports[/{name}]` · `POST /api/synthesis/reports/generate` · `GET /api/lora-adapters` · `POST /api/lora-chat`.
- **Eğitim:** `GET /api/training/status` · `POST /api/training/dataset` (kanonik `build_training_split`) · `GET/DELETE /api/training/examples[/{id}]` · `POST /api/training/dry-run` · **`POST /api/training/run` A+H (gizli)** · `POST /api/training/stop` · `GET /api/training/colab-notebook` · `GET /api/training/progress` · `GET /api/training/live` · `GET /api/training/logs` · `POST /api/training/stream-ticket` · `GET /api/training/stream` (bilet).
- **Auto-LoRA:** `GET /api/auto-lora/status` · `POST .../enable` · `POST .../check` · `POST .../train` A+**H** · `POST .../promote` A+**H** · `POST .../reset`.
- **Runtime:** `GET /api/agents` · `GET /api/agents/runs[/{id}]` · `GET /api/agents/graph` · `GET/POST /api/automation/tasks` (sürücü → `requires_approval` zorla True) · `POST .../{id}/cancel` · `GET /api/approvals` · `POST /api/approvals/{id}/approve` A+**H** · `.../reject` A+**H**.
- **Motor:** `GET /api/engines` · `POST /api/engines/rescan` (kimlik bilgisi dönmez; `logged_in` null).
- **Orkestrasyon:** `POST /api/orchestration/start` (`hunt_ack=true` ise H) · `GET .../status/{id}` (+`driver_running`) · `GET .../timeline/{id}` · `POST .../resume/{id}` · `GET .../runs` · `POST .../recover` · **`POST .../autodrive/{id}` A+H** (`{execute=False, engine, mode="drive"}`; bilinmeyen motor 400; `run_blocked_reason` → 503; drive için `drive_hardened` → 503; BackgroundTasks).
- **Feedback:** `POST /api/feedback/submit` · `GET .../list` · `GET .../summary` · `POST .../approve/{id}` H · `.../reject/{id}` H · `POST .../export` H.
- **Sentinel:** `POST /api/sentinel/run` · `GET .../overview` (persist=True YAZAR) · `GET .../history` (1-200) · `GET .../self-heal` · `GET .../unattended`.
- **AI-brain:** `GET /api/registry/{datasets|rag-indices|embeddings|rewards|decisions}` · `GET /api/tools[/runs]` · `GET /api/ingestion-quality/{paper_id}` · `POST /api/eval/trading-hypothesis`; `GET /ai-brain` (auth yok, statik sayfa).
- **Statik:** `/assets` mount; `GET /` → `index.html`, `?v=` **içerik sha256[:12]** ile değiştirilir + `Cache-Control: no-cache`.

### `/api/training/run` kapısı
`authorize_training_action("train_run", …, agent_id="lora-trainer")` → `stop_all` → blocked; yetkisiz → `needs_approval` + `approve_command`; yetkili → onay TÜKETİLDİ → `detached_launch.launch(...)`. CLI `train --run` ile **aynı anahtar**.

### Diğer web modülleri
`training_manager.py` (in-process eğitim + SSE; MLX/HF regex'leri; `reports/training/<ad>_loss.json`), `lora_chat_service.py` (PEFT adapter lazy-load + thread-safe cache), `version_info.py` (offline git drift rozeti), `agent_graph.py` (manifest → nodes/edges(chain/data/control)/groups(8)/`main_agent=orchestration-autodrive`; canlı durum best-effort; `live_count()>0` → `running`), `schemas.py` (pydantic istek/yanıt).

## 7.19 `mcp_server/` — MCP proxy

- `hektor_mcp.py`: spec **in-process** `app.web.server.app.openapi()`'den; çağrılar `httpx.AsyncClient(base_url=HEKTOR_WEB_URL, timeout=120, headers={**auth_headers(), **driver_headers()})` ile çalışan web'e proxy. `auth_headers()` → Bearer (token boşsa yok); `driver_headers(env)` → sürücü token/run_id başlıkları (kimlik aklama önlemi). `_warn_if_token_leaves_loopback` (stderr). **Tembel `__getattr__`** ile `mcp` (import fastmcp gerektirmez). Çalıştırma: `uv sync --extra mcp` + `uv run python mcp_server/hektor_mcp.py`; `claude mcp add hektor -- uv run --project <repo> python mcp_server/hektor_mcp.py`; `scripts/sync-mcp.sh`.
- `allowlist.py`: **varsayılan kapalı** spec budama (route_maps yerine). `ALLOWED` (21): `POST /api/ask`, `GET /api/rag-loop/status`, `POST /api/rag-loop/run-once`, `GET /api/cards/pending`, `GET /api/cards/approved`, `GET /api/card/{paper_id}`, `GET /api/backtests`, `GET /api/backtest/{id}/pine`, `GET /api/status`, `/healthz`, `/version`, `/profile`, `GET /api/sentinel/history`, `GET /api/agents`, `/agents/graph`, `/agents/runs`, `/agents/runs/{run_id}`, `GET /api/papers`, `GET /api/learning/summary`, `GET /api/rag-mastery`, `GET /api/understanding-score/history`. Dışarıda: `/backtest/{id}/risk`, `/sentinel/overview`, `/understanding-score` (yazan GET'ler). `FORBIDDEN_SUBSTRINGS` (12): approvals, stop-all, clear-stop-all, training/run, orchestration/{autodrive,start,resume}, auto-lora/{promote,train,enable}, rag-loop/{enable,config}. `verify_allowlist` (drift), `filter_spec`, `AllowlistError` (fail-closed).

## 7.20 `app/main.py` — Typer CLI (~118 komut)

`app = typer.Typer(no_args_is_help=True)`; `--verbose`; ağır importlar fonksiyon içinde. Çıkış kodu: `2` STOP_ALL/fail-verdict/sapma, `3` taze onay gerekli, `1` bulunamadı/geçersiz.

| Grup | Komutlar |
|---|---|
| Kurulum/teşhis | `init`, `status`, `doctor` (salt-okuma git drift; Windows görev yolu; sapma → 2), `runtime-init`, `chain-status [--live]` |
| Makale/RAG | `ingest [--directory --force]`, `papers`, `ask <soru> --top-k`, `card <id>`, `cards pending|approve|reject`, `arxiv <q> --max-results --search-only --auto-ingest`, `arxiv-sync`, `rag-scan`, `lit-scan -t <topic> --download`, `reindex-contextual`, `rag-mastery`, `synth-paper`, `ingestion-quality[-scan]` |
| Araştırma | `extract-formulas`, `formulas`, `research <soru> --iterations`, `research-sessions`, `chain-dataset`, `auto-research --dry-run` |
| RLM | `rlm-answer <q> --engine native|alexzhang --rounds`, `rlm-runs`, `rlm-trajectory <id>`, `rlm-lora-candidates --export`, `rlm-engine`, `rlm-test-adapter`, `rlm-tools --call` |
| Trading | `gen-data`, `backtest <csv> --strategy-json`, `pine [name]`, `export-package`, `risk <backtest_id>` |
| Anlama | `exam-l3 --indicator`, `exam-l4`, `exam-l5`, `understanding-score --full --with-rag --record`, `understanding-history --compare` |
| Mastery | `mastery-run <id>`, `mastery-queue --enqueue-all --run-next --run-all`, `mastery-score`, `mastery-report`, `mastery-to-sft` |
| Veri üretimi | `dataset`, `synth-qa`, `synth-qa-bulk --target`, `discipline-dataset --write`, `unified-dataset`, `lora-dataset`, `lora-curate --run`, `lora-split`, `lora-readiness --threshold 1000`, `pretrain-gate --json`, `lora-cloud-prep`, `lora-audit --run`, `tool-use-train`, `tool-use-dataset`, `reward-analyze --build-dpo` |
| Eğitim | **`train [--run] --backend auto|mlx|peft --profile --max-examples`** (dry-run varsayılan; `--run`: STOP_ALL→2, onay→3, `ensure_train_split`), `evaluate <set>`, `lora-eval <adapter> --n`, `lora-chat`, `lora-registry`, `lora-status` |
| Yerel eğitim 5A-5E | `local-training-audit`, `-request --create-approval`, `-dry-run`, `-handoff`, `-postcheck` (hepsi salt-rapor) |
| Orkestrasyon | `orchestrate-start --hunt-ack`, `-status <id> --timeline`, `-resume`, `-list`, `-recover`, `-autodrive <id> [--execute]`, `-smoke [--skip-runtime --skip-run-pipeline]`, `-drive-live --allow-live-spawn`, `-collision`, `-regression [--commit]` |
| Runtime/onay | `agents-list`, `agents-runs`, `agents-log <id>`, `task-create`, `tasks-list`, `task-cancel`, `tasks-run --retry-blocked`, `approvals-list`, `approval-approve <id>`, `approval-reject`, `stop-all`, `clear-stop-all` |
| İzleme | `sentinel [--history --no-persist]` (fail→2) |
| Feedback | `feedback-add/list/approve/reject/export/status` |
| Registry/araçlar | `registry-list --kind`, `registry-snapshot`, `registry-register-dataset`, `registry-promote-dataset --approver [--reject --reason]`, `tools-list`, `montecarlo --seed`, `stats-check --seed`, `eval-runner --type --strict` |
| OSS model | `profile`, `recommend`, `install --auto-safe`, `benchmark`, `rules-update --approve` |

---

# 8. Ajanlar

Hektor'te "ajan" üç ayrı katmanda yaşar: (a) **runtime ajanları** — `automation_manifest.yaml`'de kayıtlı, kodda çalışan 28 bileşen; (b) **Claude ajan tanımları** — `.claude/agents/*.md`, geliştirme oturumunda alt-ajan olarak çalışır (19); (c) **skill'ler** — `.claude/skills/*/SKILL.md`, prosedür paketleri (20). Ayrıca AutoDriver'ın doğurduğu **motorlar** (claude/codex/gemini/local) vardır.

## 8.1 Runtime ajanları (automation_manifest.yaml — 28)

Her kayıt: `agent_id, name, file, entrypoint, trigger, autonomy, dangerous, default_enabled, writes[], reads[], safety_gates[], approval_required, stop_method, status_location, known_failure_modes[]`. Registry bunu `AgentSpec`'e maplar. `phase: 1 # observer only` etiketi bayattır (Phase 2/2.5 devrede).

| # | agent_id | Dosya / giriş | Otonomi | Tehlikeli | Tetik | Ne yapar / kapılar |
|---|---|---|---|---|---|---|
| 1 | `auto-lora-pipeline` | `lora/auto_pipeline.py` | autonomous | **evet** | web `/api/auto-lora/*` + background_loop | Gate 0-8 → READY_TO_TRAIN → (tek politika) eğitim → eval → terfi; STOP_ALL üstün; regresyon → EVAL_FAILED |
| 2 | `training-orchestrator` | `orchestration/orchestrator.py` | autonomous | hayır | CLI `orchestrate-*`, web | 12 aşama; deep-hunt hunt_ack'siz blocked; approval tek politika; train HANDOFF; checkpoint/resume |
| 3 | `orchestration-autodrive` | `orchestration/driver.py` | autonomous | hayır | CLI `orchestrate-autodrive`, web `/autodrive/{id}` | `claude -p` doğurur (execute=False varsayılan); av PASS → bağımsız denetim → hunt_ack; sür modu MCP'li |
| 4 | `echo-feedback` | `feedback/echo.py` | semi_auto | hayır | CLI `feedback-*`, web | Düzeltme → SFT adayı; zehir filtresi 3 kez; ayrı dosya, oto-merge yok |
| 5 | `sentinel-monitor` | `monitoring/sentinel.py` | semi_auto | hayır | CLI `sentinel`, web, 14·NÖBETÇİ | 10 salt-okuma probe; hiçbir şeyi durdurmaz/başlatmaz |
| 6 | `self-healing-controller` | `monitoring/self_heal.py` | autonomous | hayır | web lifespan 60 s | 3 ardışık hata → 2 idempotent runbook; doğrulama; backoff |
| 7 | `training-guardian` | `scripts/training-watchdog.ps1` | autonomous | hayır | Task Scheduler 5 dk | Ölen detached eğitimi son checkpoint'ten yeniden başlatır (mutex + log-tazeliği); profil açıkça geçirilir; STOP_ALL üstün |
| 8 | `unattended-supervisor` | `orchestration/unattended_supervisor.py` | autonomous | hayır | web lifespan 60 s | Tek canlı hardened motor; backoff; restart sonrası reconcile |
| 9 | `rag-learning-loop` | `research/rag_learning_loop.py` | autonomous | hayır | background 15 s + web + MCP run-once | fetch→card→(rebuild)→score; eğitimde duraklar; substantive vetosu; **varsayılan KAPALI** |
| 10 | `research-orchestrator` | `research/orchestrator.py` | semi_auto | hayır | CLI `research`, web | sentez → IR → backtest → L5 → yansıma; verdict≠pass → aday |
| 11 | `literature-scout` | `research/literature_scout.py` | semi_auto | hayır | CLI `lit-scan`, Task Scheduler günlük 08:30 | arXiv keşif → gelen kutusu + BULUNANLAR.md; ingest/eğitim YOK |
| 12 | `rag-trend-scanner` | `research/rag_trend_scanner.py` | semi_auto | hayır | CLI `rag-scan`, Task Scheduler 24 s | RAG yenilikleri → watchlist |
| 13 | `reflection-agent` | `research/reflection_agent.py` | manual | hayır | orchestrator | Başarısız iterasyonda tek değişiklik önerir |
| 14 | `paper-mastery-agent` | `learning/paper_mastery_agent.py` | semi_auto | hayır | CLI `mastery-*`, web | 100 puanlık ustalık sınavı |
| 15 | `status-manager` | `learning/status_manager.py` | manual | hayır | mastery agent | 14 durum, skor→durum |
| 16 | `lora-control-plane` | `lora/control_plane.py` | semi_auto | hayır | CLI `lora-audit`, auto-lora | Gate 0-8; Gate 7 BLOCKER |
| 17 | `adapter-eval` | `training/adapter_eval.py` | semi_auto | hayır | auto-lora `_run_eval`, CLI `lora-eval` | Gerçek base-vs-adapter; regresyon/degenerasyon reject; terfi etmez |
| 18 | `dataset-quality-gate` | `training/dataset_quality.py` | semi_auto | hayır | `pretrain-gate` | GO/NO-GO (garanti zehiri, açılış ezberi) |
| 19 | `tool-use-trainer` | `training/tool_use_trainer.py` | semi_auto | hayır | CLI `tool-use-train` | THINK→CALL→OBSERVE→CONCLUDE SFT verisi |
| 20 | `auto-researcher` | `pipeline/auto_researcher.py` | semi_auto | hayır | CLI `auto-research` | Kartlardan soru → tool-use seansları → DPO |
| 21 | `arxiv-fetcher` | `ingestion/arxiv_fetcher.py` | autonomous | hayır | web `/api/arxiv/*`, saved query, CLI | İdempotent indirme, %PDF doğrulama |
| 22 | `rules-updater` | `agents/learning/rules_updater.py` | **requires_approval** | hayır | CLI `rules-update` | Başarısız trial → öneri (pending_review); uygulamaz |
| 23 | `model-advisor` | `agents/model_advisor/advisor.py` | autonomous | hayır | CLI `recommend`, web `/api/recommend` | RAM/VRAM'a göre model önerisi (tek kayıtlı executor handler'ı) |
| 24 | `rlm-controller` | `rlm/rlm_controller.py` | manual | hayır | CLI `rlm-answer`, web | Kaynaklı denetimli cevap; iki-kapılı güvence; trading uyarısı |
| 25 | `ingestion-quality-scorer` | `ingestion/quality_scorer.py` | semi_auto | hayır | CLI `ingestion-quality` | 100 puan rubrik; `--record` ile kalıcı |
| 26 | `scientific-tool-runtime` | `tools/` | manual | hayır | CLI `tools-list/montecarlo/stats-check` | Seed'li deterministik doğrulama; `tool_runs` |
| 27 | `model-data-registry` | `registry/` | **requires_approval** | hayır | CLI `registry-*` | Sürümleme + terfi kapısı (CAS, terminal) |
| 28 | `hypothesis-evaluator` | `evals/eval_runner.py` | semi_auto | hayır | CLI `eval-runner` | Hipotez test-edilebilirlik + ReleaseGate |

**Chain (topolojik sıra):** §2.4. **Kontrol kenarları (agent_graph):** orchestration-autodrive→rag-learning-loop; unattended-supervisor→{autodrive, self-heal, auto-lora, training-guardian}; training-guardian→auto-lora; sentinel→self-heal; self-heal→{training-orchestrator, rag-learning-loop}.

**Manifest kaymaları (2026-09-02'de düzeltildi, bkz. SADELESTIRME_RAPORU.md):** `literature-scout` kaydına eksik 3 alan eklendi; `auto-lora-pipeline.stop_method` notu güncellendi; `agent_graph._GROUP` içindeki ölü `makale-arastirma` anahtarı silindi (`auto-researcher` zaten kayıtlıydı).

## 8.2 Motorlar (AutoDriver'ın doğurduğu abonelikli CLI'lar)

| Motor | Kurulu-mu yoklaması | Av (hunt) | Sür (drive) | Kota notu |
|---|---|---|---|---|
| `claude` (varsayılan) | PATH | `claude -p PROMPT --safe-mode --strict-mcp-config --disallowedTools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task` | `claude -p PROMPT --setting-sources "" --disable-slash-commands --strict-mcp-config --tools Read,Grep,Glob --mcp-config <path>` | İnteraktif Claude Code kotasını yer |
| `codex` | PATH | hardened | drive_hardened (config yoksa fail-closed) | 5 saatlik yuvarlanan kota |
| `gemini` | PATH | doğrulanmadı (hardened=False → RUN'a kapalı) | — | Günlük istek kotası |
| `local` | — (Ollama) | süreç doğurmaz | — | Kota yok |

Kimlik toplanmaz/saklanmaz; `logged_in` daima `null`.

## 8.3 Claude ajan tanımları (`.claude/agents/`, 19)

| Ajan | Rol |
|---|---|
| `lora-control-orchestrator` | LoRA yaşam döngüsünü koordine eder, raporlar, onay isteyen kararları yükseltir; ağır eğitim BAŞLATMAZ |
| `lora-dataset-auditor` | Gate 0-4: kaynak bütünlüğü, şema, duplicate, kalite |
| `lora-curriculum-classifier` | Kartları 0-4 müfredat seviyesine sınıflar |
| `lora-domain-verifier` | 8 domain sınıflaması |
| `lora-math-physics-statistics-verifier` | Gate 5: hesap hatası, lookahead, survivorship, yanıltıcı istatistik |
| `lora-logic-philosophy-reviewer` | Gate 6: mantık tutarlılığı, nedensellik, belirsizlik ifadesi |
| `lora-safety-secret-scanner` | Gate 7 BLOCKER: sır, PII, tehlikeli finansal tavsiye |
| `lora-trainer-configurator` | MLX/LLaMA-Factory/Axolotl config üretir; eğitim başlatmaz |
| `lora-evaluation-reviewer` | base/rag_only/lora_only/rag+lora kıyası; accept/reject önerisi |
| `lora-adapter-registry-manager` | candidate→smoke_passed→eval_passed→approved→production; production insan onayı |
| `lora-arastirma` | Güncel LoRA/SFT literatürü (günlük hafif + haftalık derin), adversarial doğrulama, `docs/egitim/LORA_ARASTIRMA_LOG.md` |
| `makale-arastirma` | Haftalık arXiv keşfi → `Desktop\RAG Kaynak\Gerekli kaynaklar` + `00_NEDEN_ONEMLI_oku_once.md` |
| `rlm-answer` | RLM Controller ile çok-adımlı doğrulanmış cevap |
| `rlm-integration-agent` | RLM motor-adapter mimarisi (alexzhang) additive uygulama/inceleme |
| `rlm-security-reviewer` | exec/shell/network/fs/secret riskleri; PASS/FAIL raporu (P7/P8/P9 denetimlerini yaptı) |
| `scientific-tool-runtime` | Hesap/olasılık/risk iddialarını deterministik araçla doğrular |
| `hypothesis-evaluator` | "sinyal" değil test-edilebilir hipotez denetimi + eval-runner |
| `ingestion-quality-scorer` | 100 puanlık içe-alım kalitesi |
| `model-data-registry` | Sürüm kaydı + terfi kapıları |

## 8.4 Skill'ler (`.claude/skills/`, 20)

| Skill | Ne zaman | Öz |
|---|---|---|
| `/trading-research` | Araştırma döngüsü | formül çıkar → sentez → backtest → rapor (BTCUSD 1H örneği) |
| `/backtest-auditor` | Backtest değerlendirme | `shift(1)` look-ahead, OOS, maliyet, overfit denetimi |
| `/codegen-review` | Yeni indikatör/strateji kodu | registry'ye ekle + test; ruff+mypy+pytest |
| `/rlm-answer` | Kaynaklı+doğrulanmış cevap | çok-tur retrieval → iddia doğrula → çekimser; `rlm-runs` |
| `/rlm-integration` | RLM/RAG/Mastery/adapter kodu değişikliği | native varsayılan; OpenAI default değil; local exec yasak |
| `/hektor-web` | Web API'yi MCP tool olarak kullan | web değişince MCP'yi senkronla (`sync-mcp.sh`) |
| `/tv-bridge` | TradingView MCP köprüsü | Pine export ↔ TV karşılaştırma (>%10 fark uyarısı) |
| `/lora-training-control-plane` | Her LoRA hattı işi | audit, curriculum, gates, smoke, config, eval, registry, safe promote |
| `/veri-uretim-protokolu` | Stage 1 | sentetik QA 15→1000+, gece döngüsü, eşik izleme; CPU eğitimi YAPMAZ |
| `/bulut-egitim-protokolu` | Stage 2 | Kaggle/Colab T4 unsloth → GGUF → Ollama; eval gate + promote |
| `/paper-mastery-agent` | Ustalık ölçümü | mastery-run/queue/score/report |
| `/model-data-registry` | Sürüm/terfi | registry-list/snapshot/promote |
| `/scientific-tool-runtime` | Hesap doğrulama | tools-list, montecarlo (seed ZORUNLU), stats-check |
| `/hypothesis-evaluator` | Hipotez denetimi | eval-runner --type trading-hypothesis |
| `/ingestion-quality-scorer` | İçe-alım kalitesi | ingestion-quality[-scan] --record |
| `/advanced-rag-optimizer` | RAG ayarı | chunk/rerank/hybrid optimizasyonu |
| `/rag-reliability-engineer` | RAG güvenilirliği | citation/grounding/sufficiency bileşenlerini koştur |
| `/scientific-rag-reasoning` | Bilimsel akıl yürütme | formül/argüman bütünlüğü ile RAG |
| `/formula-and-argument-integrity` | Formül/argüman doğrulama | (örnek kod var olmayan modüllere atıf yapar — bilinen drift) |
| speckit.* (`.claude/commands/`) | Spec-kit iş akışı | specify/plan/tasks/implement/analyze/checklist/clarify/constitution/taskstoissues |

Ayrıca CLAUDE.md'de önerilen genel skill'ler: `/health`, `/investigate`, `/deep-research`, `/claude-mem:*`.

---

# 9. Güvenlik ve yetki modeli

## 9.1 Tehdit modeli (SECURITY.md)

| Varlık | Tehdit | Savunma |
|---|---|---|
| Yerel makine | Ağa açılma | Varsayılan bind `127.0.0.1` |
| API | Yetkisiz erişim | Opsiyonel Bearer token, sabit-zamanlı karşılaştırma |
| Upload PDF/CSV | Sahte dosya, enjeksiyon | Uzantı + `%PDF-` + boyut; CSV başlık sniff; kural çalıştırma yok |
| Dosya sistemi | Path traversal | `sanitize_filename` + `safe_destination` |
| Tarayıcı | XSS/clickjacking | CSP `script-src 'self'`, `X-Frame-Options: DENY`, nosniff |
| Servis | DoS | IP başına kayan pencere + ayrı upload limiti |
| DB | SQLi | SQLAlchemy ORM |
| Strateji | Kod enjeksiyonu | regex-only parse |
| Sırlar | Sızıntı | `.env` ignore; gitleaks + detect-private-key |
| Host başlığı | DNS rebinding | TrustedHostMiddleware |
| Aktarım | Düz HTTP token | TLS + HSTS |

Ağa açarken: P0 token (`openssl rand -hex 32`) + VPN/SSH tüneli (en iyi) veya reverse proxy + TLS; P1 trusted hosts, rate limit, CORS; P2 pip-audit, pre-commit, yedek. Kapsam dışı: canlı borsa, çoklu kullanıcı, parola saklama.

## 9.2 Kimlik ve scope

- İki kimlik seviyesi: **human** (UI/CLI, api_token) ve **driver** (doğurulan motor; `X-Hektor-Driver-Token` + `X-Hektor-Run-Id`, sha256-hash'li, run_id-bağlı, TTL'li, koşu bitince iptal).
- `require_human` ile korunan uçlar (driver → 403): onay ver/reddet, kart onayla/reddet, feedback approve/reject/export, `training/run`, `auto-lora/train|promote`, `clear-stop-all`, `orchestration/autodrive`, `hunt_ack=true` içeren start/resume. `stop-all` kasıtlı olarak human_only **değil** (motor kendini durdurabilir, frenini çözemez).
- Sürücünün açtığı `automation/tasks` zorla `requires_approval=True`.
- **Dürüstlük sınırı:** `api_token` boşken bu katman kriptografik sınır değildir; öneri `HEKTOR_API_TOKEN` atamak. `include_in_schema=False` koruma değildir.

## 9.3 Doğurulan motora karşı asıl sınır = araç kısıtı

HTTP auth motora karşı "tiyatro"dur (4 bypass kanalı: CLI auth'suz, `.env` fallback, MCP `mcp__*`, hook'lar). Sınıf-düzeyi çözüm:
- **Av:** `--safe-mode` (hook/plugin/MCP/özel ajan kanallarını kapatır) + `--strict-mcp-config` (sıfır MCP) + `--disallowedTools` (yerleşik araçlar; `Task` dahil).
- **Sür:** `--safe-mode` kullanılamaz (MCP'yi kapatır) → `--setting-sources ""` + `--disable-slash-commands` + `--strict-mcp-config` + `--tools Read,Grep,Glob` + `--mcp-config`. Motorun MCP yüzeyi = 21 salt-okuma uç; human_only uçlar MCP'de hiç yok, çağrılsa 403.
- `--bare` yasak (API-key'e indirger).
- `build_child_env`: insan token'ı boşaltılır, `CLAUDE_CODE_*` ayar-ezme env'leri silinir.
- `argv[0]` mutlak yol + `cwd=_REPO_ROOT` (sahte `claude.exe` savunması); `shell=False`.
- Üç spawn noktası farklı rejim: `AutoDriver` tam kısıt + sürücü token; `weekly-bug-scan.ps1` tam kısıt; `rag-research-loop.ps1` **kuşatılamaz** (yazma/push işlevsel şart; sahte kısıt eklenmedi, test bunu sabitler).
- `Engine.hardened` varsayılan False → fail-closed; yalnız claude ve codex doğrulanmış.
- Açık kalan: av motoru `Read` ile `.env` okuyabilir (azaltım: `.env`'i repo dışına al / izole çalışma alanı); aynı run_id'de eşzamanlı autodrive için koşu-başına kilit yok (UI kilidi güvenlik sınırı değil).

## 9.4 Onay ve kill-switch

- **Standing yetki yok.** Her tehlikeli aksiyon (`train_run`, `promote`, `rules_apply`) `consumed_at` damgalı tek-kullanımlık taze onay tüketir; tüketim atomik CAS. Kanonik anahtar `lora-trainer / train_run` (CLI, web, orkestrasyon, 5B aynı).
- Orkestrasyon `approval` aşaması onayı **gözler, tüketmez**.
- **STOP_ALL** (`storage/STOP_ALL`): yalnız `dangerous` ajanları bloklar; salt-okuma ajanlar çalışır; bayrak yazımı + `engine_procs.terminate_all()`; `clear-stop-all` insan-yalnız. `STOP_TRAINING`/`STOP_LEARNING` graceful döngü sinyalleri.
- **verdict_audit** (P8/P9): av PASS'i ≥5 var olan dosya + ≥2 alt-sistem + HIGH bulgu çelişkisi yok + ≥5 okuma-kanıtı.
- MCP allow-list drift testi FastAPI'nin gerçek dependency grafiğini tarar; yan-etkili GET'ler handler **kaynak kodu** taranarak yakalanır ("GET = salt-okuma" bu depoda yanlış varsayım).

---

# 10. Eğitim yaşam döngüsü (uçtan uca)

## 10.1 Aşamalı eğitim protokolü
**Stage 1 (lokal, eğitim YOK):** makale chunk'larından grounded sentetik QA (`synth-qa`, `synth-qa-bulk --target 1000`), disiplin tuzakları (`discipline-dataset`), kart kürasyonu (`lora-curate`), `assemble_sft.py` → `lora_sft.jsonl` (≥1000 örnek hedefi, `lora-readiness`).
**Stage 2 (gerçek LoRA):** bulut-GPU (Kaggle T4×2, `lora-cloud-prep` → notebook → GGUF → `ollama create`) **veya** yerel küçük model (Qwen2.5-1.5B, `discipline_safe_local`, CPU ~35 s/adım). 4B CPU eğitimi YAPILMAZ (haftalar + overfit; v5 46.75 saat).

## 10.2 Veri hattı
```
knowledge_cards (approved, lora_eligible=1, kanonik)  ─┐
synthetic_qa.jsonl (grounded, dedup)                   ─┼─ assemble_sft_lines → dedup → +%25 disiplin → lora_sft.jsonl (KANONİK)
discipline_dataset (432 adversarial)                   ─┘
lora_sft.jsonl → Gate 0-8 (lora-audit) → pretrain-gate (GO/NO-GO) → lora-split (seed 42, %5 valid) → train.jsonl / valid.jsonl
                                                                    → _auto_register_dataset (DatasetVersion pending) → registry-promote-dataset (insan)
```
Clobber guard üç katmanlı; `train.jsonl` her başlatmada `lora_sft.jsonl`'den yeniden türetilir (iki-hat drifti kapatıldı).

## 10.3 Orkestrasyon + onay + eğitim
```
İNSAN ⚡ RUN (15·AJAN HARİTASI) veya CLI orchestrate-start
  → preflight (STOP_ALL?) → collision (git) → smoke (Ollama canlı?) → deep-hunt [BLOCKED: hunt_ack yok]
  → [insan: 12·ORKESTRASYON "Otonom AV" (mode=hunt)] AutoDriver → claude -p (safe-mode) → HEKTOR_HUNT_VERDICT + EVIDENCE
  → verdict_audit bağımsız doğrular → hunt_ack=true
  → data-gate (audit_dataset GO?) → curriculum → dry-run → regression (baseline kıyası) → approval [BLOCKED: taze onay yok]
  → [insan: approval-approve <id> veya UI "✅ ONAYLA VE EĞİTİMİ BAŞLAT" (iki-tık)]
  → train (handoff) → İNSAN: train --run / POST /api/training/run → authorize_training_action → onay TÜKETİLİR
  → detached_launch.launch(profile=discipline_safe_local) → Popen detached (web/terminal kapansa da sürer)
  → training-guardian (5 dk) çökmede checkpoint'ten devam
  → adapter_eval (base vs adapter, min_n≥5, degenerasyon/boş-cevap vetosu) + anlama merdiveni regresyon kıyası
  → accept → registry ADAY → promote (insan, user_approved=True) → PRODUCTION (tek)
```
Sür modu (⚡ RUN varsayılan) veri hattını (carding→RLM→curate→assemble) MCP araçlarıyla ilerletir; eğitim adımında **onay kapısında durur**; `hunt_ack` yazmaz.

## 10.4 v5 dersi (neden bu kadar kapı var)
`hektor_lora_v5` (Qwen3-4B, CPU PEFT, 1203 adım, 46.75 saat) base'den DAHA KÖTÜ çıktı: tekrar döngüsü, "pasaja göre" ezberi, maliyetsiz rakam uydurma. Kökler: (1) sentetik QA one-shot'ında her cevaba "Pasaja göre" öneki (%68 açılış ezberi), (2) adversarial disiplin örneği yok, (3) prompt maskeleme yok (assistant_only_loss=False), (4) eval adapter'ı yüklemiyordu (base Ollama'yı ölçüyordu), (5) eval n=1 ile sahte accept. Her biri ayrı kapıyla kapatıldı: sızıntı öneki kaldırıldı, 432 disiplin örneği, `build_masked_labels`, `evaluate_adapter` gerçek PEFT yükleme, `_MIN_EVAL_N=5`, `pretrain-gate` açılış-bigram bloğu, `regression.py` `top_opening_share`, profile-drop fix (`discipline_safe_local` varsayılan), `empty_answer` vetosu, `load_base_of` (elmayla-elma). İlk 1.5B eğitimi (`hektor_lora_15b_v1`, 400 adım, loss 2.63→1.54) eval'de base 0.125 → adapter 1.0 ile **ACCEPT** aldı.

---

# 11. Web arayüzü (15 sekme)

`uv run hektor-web` → `http://127.0.0.1:8765`. Statik `index.html` + `app.js` (5250) + `app.css` (3020); CSP nedeniyle inline script/onclick yok; fontlar self-host (Spectral, Plus Jakarta Sans, JetBrains Mono); Okabe-Ito renk-körü-güvenli palet (pass `#0a7d55`, fail `#cf4014`) + ✓/✕/≈ şekil ipuçları; WCAG AA.

**Üst şerit:** canlılık rozeti (4 s), bağlantı/embed/makale sayısı (30 s), RAG ustalık %, "obj. anlama" (tıkla → tam merdiven + kayıt), eğitim göstergesi (🔴 çalışıyor / ▶ EĞİTİME HAZIR — BAŞLAT / yok; 15 s), sürüm/sapma rozeti, disclaimer.

**Grup navigasyonu:** Keşfet & sor (01, 11) · Kütüphane (02) · Trader & backtest (03, 04) · Eğitim hattı (06, 05, 07, 12, 13) · İzleme & sağlık (09, 10, 14, 08, 15; "gelişmiş" toggle ile gizlenir — yalnız nav katmanı). Aktif sekme `localStorage` + `#sekme=` hash.

| # | Sekme | Ne yapar |
|---|---|---|
| 01 | ARAŞTIRMA | RAG soru-cevap, top_k, model/adapter seçici, kaynak listesi, ilk-açılış 3 adım |
| 02 | MAKALELER | PDF sürükle-bırak (çoklu, 429'da retry), toplu kart/skor, çapraz sentez, arXiv arama/çekme, kayıtlı sorgular, filtre/sıralama |
| 03 | TRADER BEYİN | Formül çıkarımı, agentic araştırma (1-5 iterasyon), LoRA sohbet (PEFT), zincir veri seti, sentez makaleleri, araştırma geçmişi |
| 04 | BACKTEST | Sentetik/CSV, özel StrategyIR JSON, geçmiş, risk raporu, Pine, .achpkg |
| 05 | EĞİTİM | Veri seti, ayarlar, Başlat/DURDUR/Colab notebook, canlı ilerleme (SSE bilet), adapter'lar, örnekler, Auto-LoRA paneli |
| 06 | ONAY | Bekleyen bilgi kartlarını onayla/reddet (insan-yalnız) |
| 07 | DEĞERLENDİRME | Eval seti + adapter → kural ihlali taraması |
| 08 | SİSTEM | Canlı durum, donanım profili + model önerisi, katman diyagramı, kurallar, API token |
| 09 | ÖĞRENME | RAG öğrenme döngüsü paneli (aç/kapat/tur/8 ayar), SVG grafikler (loss, kart büyümesi, eval, anlama geçmişi) |
| 10 | AGENTS | Supervisor + STOP_ALL, onay istekleri, ajanlar, koşular, görevler, olay akışı |
| 11 | RLM | Motor paneli, koşu tablosu, adım/kanıt/doğrulama detayı (**silinmemeli — PR#111 dersi, test kilidi**) |
| 12 | ORKESTRASYON | Yeni koşu (hunt_ack kutusu), Sürdür, **🤖 Otonom Sürüş (mode=hunt)**, Panic Recovery, aşama grafiği, timeline |
| 13 | GERİ BİLDİRİM | Echo: düzeltme gir, onayla/reddet, export |
| 14 | NÖBETÇİ | 🩺 ŞİMDİ YOKLA, probe tablosu, geçmiş |
| 15 | AJAN HARİTASI | Işıklı-yol SVG grafiği (7 s poll), motor seçici, **⚡ EĞİTİMİ DEVREYE SOK**, canlı şerit + ⛔ DURDUR, kuru-çalıştırma kutusu, iki-tık eğitim kapısı |

**⚡ RUN akışı:** tekrar-giriş kilidi → motor seçilebilir mi → koşu yoksa `POST /orchestration/start` → **atlanamaz onay modalı** (odak Vazgeç'te, onay kutusu her açılışta sıfır, `isTrusted` şart, Esc/arka plan = iptal; kota uyarısı; "Eğitim BAŞLAMAZ" güvencesi) → `POST /orchestration/autodrive/{id} {execute: go, engine, mode: "drive"}` → canlı şerit `driver_running`'e bağlı. Testler her `execute:true` çağrısının 1200 karakter geriye `amConfirmOpen(`/`window.confirm(` içerdiğini sabitler.

**Cache-bust:** `?v=` içerik hash'i; HTML `no-cache`; kod çekince sunucu RESTART şart (route'lar başlangıçta yüklenir), tarayıcıda `Ctrl+Shift+R`.

`/ai-brain` — bağımsız dashboard (registry/tools/ingestion/eval, 4 sekme, salt-okuma).

---

# 12. Operasyon

## 12.1 Kurulum zinciri

**Windows:** `install.ps1` (tek satır `irm … | iex`; Git yoksa winget; hedef `%USERPROFILE%\hektor`; mevcut checkout'u `main`'e ff-only günceller, yabancı kopya tespiti) → `setup.ps1` (Python/uv/Ollama winget, 18 model menüsü, `ollama pull` model + `nomic-embed-text`, `.env`, `hektor init`) → `scripts\start-server.ps1 -Install` (önce `verify-install.ps1` kapısı; HKCU Run `HektorWeb` + Task Scheduler `HektorUpdate` 03:00 + `HektorTrainingWatchdog` 5 dk).
**macOS/Linux:** `setup.sh` (uv → uv sync → Ollama/Homebrew → `.env` → init → erişim modu [yerel | uzaktan: `0.0.0.0` + otomatik token] → `verify-install.sh` → opsiyonel `install-autostart.sh` systemd/launchd/cron).

**verify-install:** init → status → gen-data → backtest → pytest (offline); exit 0/1/2; autostart yalnız geçerse.

## 12.2 Güncelleme (`update.ps1` / `update.sh`)
Web'i durdur (pid/port 8765) → `git fetch origin main` → dal main değilse `git switch main` (kirliyse ve `-Force` yoksa HATA; main başka worktree'de ise HATA) → `git pull --ff-only` (ıraksaksa AUTO-MERGE YOK; `-Force` → `reset --hard origin/main`, yalnız izlenen kod; veriler silinmez) → drift raporu → `uv sync --extra dev` → web'i başlat → sağlık → "Ctrl+Shift+R". **Eğitime dokunmaz.** Teşhis: `uv run hektor doctor` (sapma → exit 2); Windows görev yolu onarımı `start-server.ps1 -Repair`.

## 12.3 Zamanlanmış görevler (Windows)

| Görev | Kuran | Sıklık | Çalıştırır |
|---|---|---|---|
| `HektorWeb` (HKCU Run) | `start-server.ps1 -Install` | Logon | `hektor-autostart.vbs` → web |
| `HektorLoop` (HKCU Run) | `start-loop.ps1 -Install` | Logon | sürekli öğrenme döngüsü |
| `HektorUpdate` | `start-server.ps1 -Install/-Repair` | Günlük 03:00 | `update.ps1` |
| `HektorTrainingWatchdog` | `start-server.ps1` | 5 dk | `training-watchdog.ps1` |
| `Hektor-WeeklyBugScan` | `install-bug-scan-task.ps1` | Pzt 09:00 | `weekly-bug-scan.ps1` (Kademe-1) |
| `Hektor-LiteratureScout` | `install-literature-scout-task.ps1` | Günlük 08:30 | `literature-scout.ps1` |
| `Hektor-RAG-Scan` | `rag-research-loop.ps1 -Mode Scan -Register` | 24 s | `rag-scan` |
| `Hektor-RAG-Integrate` | `rag-research-loop.ps1 -Mode Integrate -Register` | 168 s | entegrasyon turu |

macOS: `launchctl` `com.hektor.web.plist`; Linux: systemd kullanıcı servisi `hektor-web.service` (`Restart=on-failure`, linger).

## 12.4 Scriptler (35)

| Script | Amaç |
|---|---|
| `start-server.ps1` | `-Install/-Restart/-Repair/-Uninstall/-Stop/-Status`; pid `.web.pid`; loglar `logs/hektor-web*.log` |
| `run-web-service.ps1` | `.venv\Scripts\python.exe -c "from app.web.server import run; run()"` + `UV_NO_SYNC`, thread=1, `ISOLATE_CHROMA=1` |
| `start-train.ps1` | Detached eğitim: `-Adapter -Iterations -Dtype -Profile discipline_safe_local -Stop -Status`; `lora-split` önce; `HEKTOR_TRAIN_SUPERVISED=1`; `train_status.json` (pid'siz) |
| `train-loop.ps1` | assemble_sft → lora-split → train --run döngüsü (STOP_TRAINING ile durur; `--profile` GEÇMEZ — kusur) |
| `training-watchdog.ps1` | Mutex + log-tazeliği (10 dk) → ölmüşse `start-train.ps1 … -Profile discipline_safe_local` |
| `assemble_sft.py` | `assemble_sft_lines` → `lora_sft.jsonl`; eğitim başlatmaz |
| `start-loop.ps1` / `continuous-learning.sh` / `auto-chain.sh` / `mac-loop.sh` | Sürekli öğrenme döngüsü (kart → skor → sentez → synth-qa; eğitim değil; eğitim sürerken ertelenir; `STOP_LEARNING`) |
| `synth_qa_chain.ps1` | Zincirleme synth-qa (`-Target -StartSeed`) |
| `weekly-bug-scan.ps1` | Kademe-1 `claude -p` rapor-only (tam kısıt) → `reports/bug-scan/scan-<tarih>.md` + HANDOFF satırı |
| `install-bug-scan-task.ps1` / `install-literature-scout-task.ps1` / `literature-scout.ps1` | Task Scheduler kurulumları |
| `rag-research-loop.ps1` (+ `rag-research-scan.md`, `rag-research-cycle.md`) | Scan/Integrate modları (kuşatılamaz: yazma/push işlevsel) |
| `rag_ab_*.py`, `rag_keyword_eval.py`, `rag_retrieval_ab.py` | Retrieval A/B ölçümleri (`RAG_AB_LIMIT`) |
| `gen_egitim_pdf.py` | `docs/egitim/*.md` → PDF (`docs` extra) |
| `open-pr.sh/.ps1` | push + PR + CI yeşilse oto squash-merge (`--no-merge`) |
| `setup-pr-automation.sh` | Bir kerelik: `allow_auto_merge` + main koruması (required ctx `lint · types · tests (offline)`, `enforce_admins=false`) |
| `sync-mcp.sh` | MCP tool üretimini doğrula + `claude mcp remove/add` |
| `check_protected_paths.py` | `data/ storage/ vector_db/ models/ .env*` değişikliği guard'ı (CI'da PRE/POST) |
| `bootstrap.sh`, `verify-install.sh/.ps1`, `install-autostart.sh` | Kurulum/doğrulama |
| `enrichment-topics.txt` | Kart zenginleştirme konuları |

## 12.5 CI ve PR otomasyonu

- **`ci.yml`** — push main + PR; job `lint · types · tests (offline)`: `uv sync --extra dev --extra mcp` → `ruff check app tests` → `ruff format --check` → `mypy app` → `pytest -m "not ollama"`. `mcp` extra zorunlu (allow-list testleri skip olmasın). Auto-merge (squash + delete-branch) açık; owner doğrudan push edebilir.
- **`nightly-automation-audit.yml`** — rapor-only şablon; cron Pzt 04:17 UTC yalnız `vars.ENABLE_NIGHTLY_AUDIT=='true'`; gate + `understanding-score --record` + `understanding-history --compare` + `pretrain-gate` → artifact (14 gün). Kod/branch/eğitim/terfi/main'e dokunmaz.
- **`claude-code-task.yml`** — Phase 4C, `vars.ENABLE_CLAUDE_TASK` olmadan INERT; `anthropics/claude-code-action` SHA-pin (`51705da…`); `CLAUDE_HARD_RULES` (train --run yok, terfi yok, data/storage/vector_db/models/.env değişmez, main'e push yok, auto-merge yok, backtest+OOS'suz başarı iddiası yok); `--allowedTools` allow-list (Read/Edit/Write/Glob/Grep + sınırlı Bash) + `--disallowedTools`; PRE/POST `check_protected_paths.py`; bağımlılık/workflow değişikliği → `needs-approval` etiketi; **PR yalnız insan onayıyla**.
- PR şablonu: Kademe-0 kapısı checklist'i; issue şablonu `claude_task.yml` (allowed/forbidden files, risk, gates).
- Gotcha'lar: açık PR'a push CI'ı tetiklemeyebilir (close/reopen); global RateLimiter cross-test birikimi → conftest fixture; Node 20 deprecation uyarısı.

## 12.6 Dokümanlar (`docs/`)

| Dosya | İçerik |
|---|---|
| `ROADMAP_MOTOR_BAGLAMA.md` (767) | P1-P9 motor bağlama paketleri (scope izolasyonu, MCP token, allow-list, engines, RUN kapısı, E2E, sür modu, bağımsız verdict, okuma-kanıtı) + kopyala-yapıştır prompt'lar |
| `SCOPE_ISOLATION.md` (232) | human/driver ayrımı, 4 bypass kanalı, araç kısıtı |
| `UNATTENDED_ARCHITECTURE.md` | Unattended Supervisor desired-state reconciler |
| `AGENT_RUNTIME_OBSERVER.md` | Tracker/registry tasarımı, Phase 1/2 |
| `PROTOKOL_ASAMALI_EGITIM.md` (master) → `PROTOKOL_VERI_URETIM.md` (Stage 1), `PROTOKOL_BULUT_EGITIM.md` (Stage 2) | Aşamalı eğitim |
| `PROTOKOL_RAG_LORA_ZINCIR.md` + `RAG_LORA_ENTEGRASYON.md` | RAG bilgi / LoRA üslup zinciri; PEFT→GGUF tek çalışan yol (Qwen3 `ADAPTER` desteklenmez) |
| `RAG_EGITIM_YENIDEN_TASARIM.md` | Büyük pivot gerekçesi (CPU eğitimi durdur, RAG-first) |
| `PROTOKOL_BACKTEST.md`, `PROTOKOL_MCP.md`, `PROTOKOL_RAG_GUNCEL_ARASTIRMA.md`, `PROTOKOL_LORA_ARASTIRMA.md`, `PROTOKOL_MAKALE_ARASTIRMA.md` | Süreç protokolleri |
| `EGITIM_PROTOKOLU.md` | Windows yerel eğitim: donanım (i7-1165G7, GPU yok), ölçülmüş süreler (Qwen3-4B Q4 ~74 s/adım) |
| `GUNCELLEME_KILAVUZU.md` | Çok-makine güncelleme, doctor, -Repair |
| `LOCAL_TRAINING_{REQUEST_FLOW,DRYRUN_PIPELINE,HANDOFF,POSTCHECK}.md` | Phase 5B-5E |
| `PHASE4_GITHUB_AUTOMATION.md`, `PHASE4B_DRYRUN.md`, `PHASE4C_ACTIVATION.md` | GitHub otomasyonu (testlerle zorunlu yönetişim belgeleri) |
| `rlm_rag_architecture.md`, `rlm_runtime_modes.md`, `rlm_security_model.md`, `rlm_alexzhang_integration.md` | RLM |
| `HANDOFF_2026-08-03.md` | `hektor_lora_v5_durable` bf16 discipline_safe_local hedef 2406, 25/2406 checkpoint |
| `egitim/` | `LORA_EGITIM_DETAYLI_ANLATIM.md/.pdf`, `RAG_EGITIM_DETAYLI_ANLATIM.md/.pdf` (sürümlü), `LORA_ARASTIRMA_LOG.md`, `rag-watchlist.md` |
| `examples/raft_discipline_seed.jsonl`, `kaynaklar/00_NEDEN_ONEMLI_oku_once.md`, `arsiv/` (6 eski belge) | |

---

# 13. Test stratejisi

- **180 dosya, 1657 test fonksiyonu; %100 çevrimdışı** (fake embedding, sentetik OHLCV, enjekte stub'lar, `tmp_path`). Ollama gerektiren 2 test `@pytest.mark.ollama`.
- `conftest.py`: `_ollama_running()` → yoksa ollama testleri skip; `_isolate_storage` (session, autouse: `HEKTOR_SQLITE_PATH/CHROMA_PATH` tmp, `ALLOW_FAKE_EMBEDDINGS=true`, `get_settings.cache_clear()`); `_reset_web_rate_limiter` (function, autouse: `_rate_limiter._hits.clear()`); `store` fixture.
- Windows: `--basetemp=.pytest_tmp` (WinError 5) veya `PYTEST_DEBUG_TEMPROOT`; `uv run --extra dev` şart.
- Kategoriler: orkestrasyon/motor (~180), güvenlik/scope (~90), LoRA/gate (~180), veri hattı (~110), eğitim/eval (~70), 5A-5E (70), runtime (~110), izleme (~35), RAG retrieval (~120), RAG kalite (~100), anlama merdiveni (~70), RLM (~90), trading (~120), web/UI (~130), depolama (~50), araştırma (~70), yönetişim (34).
- **Yönetişim testleri:** `test_phase4b_dryrun_docs.py`, `test_phase4c_activation_docs.py`, `test_github_workflows_static.py` dokümanları/YAML'ları zorunlu kılar (doküman silmeden önce `tests/` içinde ara).
- **Kilit sözleşmeler (özet):** dry-run kapısız, `--run` onay ister; taze onay tek kullanımlık + CAS; STOP_ALL her şeyi ezer; driver 403 listesi; allow-list kapalı küme + dependency-graph taraması + yan-etkili GET kaynak taraması; her `execute:true` insan kapılı; motor kimlik alanı dönmez; verdict fail-closed + okuma-kanıtı; paper_id içerik hash + yarım ingest onarımı; BM25 embedded=1; cache reset kilit altında; determinizm tie-break'leri; `_MIN_GRADED=3`; L5 test-edilemez → skipped; adapter min_n=5 + degenerasyon/boş vetosu; profile-drop; clobber guard; U+2028 kaçışı; Gate 7 tüm alanlar; RLM trading uyarısı her zaman; alexzhang paketsiz çalışır; RLM paneli ve çekirdek gruplar korunur; asset `?v=` hash (numara pinleme yok).

---

# 14. Proje tarihi ve dersler

## 14.1 Zaman çizelgesi

| Tarih | Olay |
|---|---|
| 2026-06-07 | Tool-use trainer, reward signal + DPO builder, Paper Mastery, arxiv-sync, unified-dataset |
| 06-09/10 | Windows kurulum (`install.ps1`, Task Scheduler), macOS LaunchAgent, PEFT fix, Qwen3 thinking fix, PDF upload BackgroundTasks |
| 06-11 | İçeriksiz kart filtresi (Gate), Windows PEFT backend tespiti, eğitim UI, 7 runtime hata; math-aware chunker + formül pipeline; 405 test |
| 06-13/14 | **BÜYÜK PİVOT:** RAG-first + aşamalı eğitim; sürekli CPU eğitimi durdu; reranker/hybrid/cross-encoder/contextual/dedup/synth-qa-bulk (12 commit) |
| 06-16 | **v5 REJECT** (46.75 h); eval harness fix (`lora-eval` gerçek PEFT); **Anlama Doğrulama sistemi** (safe_eval, L3/L4/L5, ENTROPY); v5 kök sebep teşhisi; discipline_dataset (432); pretrain-gate ilk NO-GO (%68 "pasaja göre") |
| 06-17/18 | Anlama merdiveni kalıcı (`understanding_snapshots`); geniş denetim 24 bulgu → 11 fix (L5 yanlış-negatif, sessions L5, compare); `\r` bug; 6 arXiv |
| 06-19 | Kademe-2 Sprint 1-5, 18 fix (BM25 tie-break, embedded erken yazım, komisyon eksik, IS+OOS) |
| 06-20 | Cache-bust (içerik hash), exam timeout 240 s, otonom başlangıç zinciri (executor, chain, preflight, verify-install) |
| 06-21 | Git toparlama (15→2 worktree), taşınabilirlik, otomatik PR (setup-pr-automation), CI RED→YEŞİL; upload 429 retry (PR#10) |
| 06-22 | İki-hat train.jsonl drifti kapatıldı (`build_training_split`) |
| 06-23 | Kademe-2 (8 finder) → 5 fix (root izolasyonu, Gate 7 alanlar, concept seed, Kelly şişme, min-graded) |
| 06-24 | alexzhang RLM opsiyonel adapter (4 PR); AI Brain 4 modül (registry, tools, eval_runner, quality_scorer; ~16 PR, 3 tur bug-fix); assistant_only_loss maskeleme + 1.5B pivotu + web LoRA chat |
| 06-25 | Çok-makine yakınsama (update main'e geçer, doctor, sürüm rozeti; PR#54-58); Web UI redesign 5 PR (nav, Türkçe, Okabe-Ito) |
| 06-26 | Kademe-2 av (PR#67) → **ilk 1.5B LoRA ACCEPT** (`hektor_lora_15b_v1`, base 0.125 → 1.0); 71 arXiv ingest |
| 06-29 | Orkestrasyon Faz-1 (PR#72, 9 aşama), Echo (PR#74), AutoDriver (PR#79); Kademe-2 → #75/#76/#77/#82 |
| 06-30 | Faz-4 SmokeRunner; Faz-5 Collision + Regression (PR#86); 12 aşama; graph_corpus cache bulgusu |
| 07-03 | Faz-6 Sentinel (PR#90); eğitim-öncesi Kademe-2: profile-drop, boş-cevap sahte-kabul, 4B-vs-1.5B (PR#95); veri hattı KAPANDI (2000 örnek, GO); UTF-8 fix; 6 arXiv |
| 07-04 | 15·AJAN HARİTASI (PR#101), worktree hazard (PR#102), UI declutter (PR#103), kart yerleşimi (PR#105) |
| 07-21 | Genel temizlik (PR#106: 17 ölü modül; PR#107: make install, fastmcp, CORS sahte-guard); Motor bağlama roadmap; **Scope izolasyonu** (PR#116/117/118); MCP allow-list 116→21 (PR#119); literature-scout (PR#120); sür-modu safe-mode çelişkisi (PR#121); RUN onay kapısı (PR#122); P6 E2E (DURDUR fix, yazan GET'ler, mutlak yol) |
| 07-22 | P7 sür modu fişe (PR#125), SSE bilet; P8 bağımsız verdict (PR#126); P9 okuma-kanıtı (PR#129); 1755 passed |
| 08-03 | Unattended training handoff (`hektor_lora_v5_durable`), training guardian, self-heal, unattended supervisor (PR#136 "codex drive subscription") |

## 14.2 Kanonik dersler

1. Prompt talimatı güvenlik sınırı değildir; kısıt teknik olmalı.
2. "GET = salt-okuma" bu depoda yanlıştır; handler kaynak kodunu tara.
3. Doküman silmeden önce `tests/` içinde ara.
4. UI asset testi `?v=N` pinlemesin.
5. Dal sayımından önce `git fetch --prune`; `--no-merged` içerik ölçmez.
6. `git add -A`/`git clean` ASLA (eşzamanlı oturum WIP'i); `.claude/worktrees` orphan olabilir → `git rev-parse --show-toplevel` kontrolü, gerçek `git worktree add`.
7. Sabitin büyüklüğünü ölçen test sahte güvencedir; çağrının yapıldığını assert et.
8. Motor kendi başarısının tek kaynağı olamaz.
9. `uv run` çalışan web'i kilitler (`os error 32`) → `UV_NO_SYNC=1`.
10. `PYTEST_DEBUG_TEMPROOT` alt-kabuğa geçmez → `--basetemp`.
11. Stub geçse de runtime bozuk olabilir (smoke "stub≠runtime").
12. Eval adapter'ı gerçekten yüklemeli; n küçükken accept verilmez; boş cevap = başarısızlık.
13. Kanonik veri dosyası tek olmalı; her yol aynı üreticiyi çağırmalı (iki-hat drifti).
14. Harness `run_in_background` oturum kapanınca ölür → uzun eğitim detached.
15. Chroma `get_all()` eşzamanlı erişimde SQLite kilidine takılır → BM25 kaynağı SQLite.
16. Veri kapıları (GO) kod-avından ortogonaldir; eğitim ayrıca Kademe-2 + Kural-8 bekler.
17. API asla; bulut yolu opt-in + native fallback.

---

# 15. Bilinen tutarsızlıklar (yeniden yazımda düzelt)

> **Güncelleme 2026-09-02:** Bu listedeki 1, 4 (kısmen), 7 (train-loop), 8, 11 (docstring'ler), 14, 16 (şema), 17-21 maddeleri [SADELESTIRME_RAPORU.md](SADELESTIRME_RAPORU.md) §A ile uygulandı; ölü modüller §B'de silme komutuyla, politika gerektirenler §C'de karar listesiyle bekliyor.

**Kod**
1. `evaluator.evaluate` çıplak `assert oos is not None` — n≤1 veride AssertionError (`-O` altında AttributeError); `inconclusive` dönmeli.
2. `_position_series` Python döngüsü (vektörize iddiasının tek istisnası).
3. L5 `_REGISTRY` (7) ile `compute_indicator` (12+) uyuşmaz (FORBIDDEN/COMPLEXITY/BB math kapısında reddedilir).
4. Pine ↔ Python sapması: VWAP/STOCH/SUPERTREND farklı; slippage Pine'a aktarılmaz; `IndicatorSpec`'te `smooth_k/multiplier` yok.
5. `context_sufficiency.classify` `query`'yi kullanmaz; `answer_eval` ve `ConfidenceScorer` iki farklı güven formülü.
6. `auto_researcher` sayım/kaynak uyuşmazlığı (`training_examples` vs `list_cards`).
7. `scripts/train-loop.ps1` `--profile` geçmez (vanilya reçete); `start-train.ps1` `pid` yazmaz; `peft_lora_train.__main__` `--profile` almaz.
8. `curriculum.classify_curriculum(card_json)` ve `LoRAControlPlane.run_full(dry_run)` parametreleri yok sayılır.
9. İki `AdapterRegistry` (lora JSONL vs training SQLite) aynı isimde.
10. `promotion_gates` kendi mini sır/PII regex'i taşır (safety_scanner ile birleştirilebilir).
11. `unattended_supervisor` göreli `storage/` yolu ve `engine="codex"` varsayılanı (`DEFAULT_ENGINE=claude`).
12. `agents/learning/memory.py` ayrı DB + `__file__` göreli yol; `profiler.py` `wmic`.
13. `store.touch_heartbeat` çağıran yok → `recover_stale` heartbeat'siz aşamalarda tetiklenmez.
14. `static/assets/canli.html` Google Fonts linki CSP ile çelişir.
15. `/api/profile` ve `/api/recommend` auth-muaf (kurulum popup) — test kilidi yok; api_auth EKLEME.
16. `TrainingProgressResponse` şeması kullanılmıyor; `/api/automation/tasks` query-param tabanlı.
17. `eval_runner.DEFERRED_TYPES` `rlm-reward` gerekçesi bayat (app/rlm tamam).

**Doküman**
18. `TRAINING_ROADMAP.md` §6 çelişkili durum başlıkları; Paper Mastery yanlış bölümde; 2026-06-07'de kalmış.
19. `LORA_EGITIM_DETAYLI_ANLATIM.md` "müfredat pacing kodda yok" iddiası yanlış (`training/dataset_builder.py` %60/30/10); `auto_enabled` efektif True; bayat satır referansları.
20. `formula-and-argument-integrity` skill'i var olmayan modüllere atıf yapar; `rlm-answer` skill'i timeout 300 der (kod 600).
21. `automation_manifest.yaml` `phase: 1`, `literature-scout` eksik alanlar; `agents/runtime/__init__.py` "Phase 2 HENÜZ YOK"; `driver.py` "19 uç" (21).
22. `app/pipeline/__init__.py` boş; `app/reliability/__init__.py` docstring'i fazla modül sayar; `strategies/` kullanılmıyor.
23. `reports/bug-scan/` gitignore kararı bekliyor; "test var ama üretim yolu yok" dosyalar: query_expander, multi_query_retriever, hybrid_retriever, regression_runner, answer_eval, golden_generator (kanıt ve silme komutu SADELESTIRME_RAPORU.md §B). Not: `ollama_installer` ölü DEĞİL (`hektor install` kullanır); `self_refining_rag` diye bir dosya yok.

---

# 16. Önerilen yeniden yazım sırası

1. **Çekirdek sözleşmeler ve config** — `Settings`, `_sqlite_pragmas`, `paper_id`/`chunk_id` türetimi, 8 kural sabitleri, seed politikası. Testleri önce yaz (determinizm, tie-break, clobber guard).
2. **Trading çekirdeği** — `StrategyIR` + regex parser, indikatör registry (vektörize), backtester (`shift(1)`, maliyet), overfit IS/OOS, evaluator, risk manager. Bunlar dış bağımlılıksız ve saf; ilk yeşil.
3. **Depolama** — SqliteStore (32 tablo + migrate + CAS metotları), ChromaStore (lock, upsert, paging), EmbeddingService (fake yolu).
4. **Ingestion → RAG** — loader/parser/metadata/chunker(math-aware) → PaperIndexer (idempotency + yarım-ingest onarımı + cache reset) → BM25 corpus (SQLite, imza, lock) → retrieval orkestratörü (router/graph/rrf/rerank) → RagAnswerer (abstain, reorder, citation).
5. **Doğrulama katmanı** — citation/grounding/sufficiency/contradiction/confidence/abstention; RAGAS-offline; safe_eval; L3/L4/L5; understanding_score (+record/compare); comprehension + rag_mastery.
6. **Brain** — LocalLLM (Ollama öncelikli, LLMUnavailable), KnowledgeCardBuilder (min-source, retry merdiveni), SyntheticQABuilder (grounding, dedup, sızıntısız one-shot).
7. **RLM** — controller akışı, evidence gate, claim/verify, trading guard, store, safe tools allowlist, native adapter; alexzhang opsiyonel + security gate.
8. **Research + Learning** — formula extractor (kelime sınırı), concept graph (seed), synthesis engine + reflection + orchestrator (L5 gate), cross-paper (≥2 makale), chain data; Paper Mastery 7 modül; RAG öğrenme döngüsü (varsayılan kapalı, eğitimde duraklar).
9. **Eğitim veri hattı** — discipline_dataset (432), sft_assembly (kanonik `lora_sft.jsonl`, U+2028), dataset_quality (pretrain-gate), Gate 0-8 + safety scanner, card_curation, splitter (seed 42), detached_launch (`ensure_train_split`, launch profil varsayılanı, kilit, stop), profiles yaml.
10. **Trainer + eval** — peft_lora_train (masked labels, collator, KL opsiyonel, max_steps), mlx wrapper, adapter_eval (gerçek yükleme, min_n, vetolar, load_base_of), registry (CAS terminal), unattended_policy (tek yetki), auto_pipeline.
11. **Agent runtime** — schemas, registry (manifest), tracker (asla fırlatmaz), supervisor + STOP_ALL, approvals (CAS tek-kullanım), task_queue, executor (allow-list), chain (Kahn), preflight; manifest YAML.
12. **Orkestrasyon** — pipeline 12 aşama, store (CAS claim), orchestrator, delegates, smoke, collision, regression, engines (+sentinel argv), engine_procs, driver (hunt/drive, child env, mutlak yol, STOP_ALL yoklama), verdict_audit (P8/P9), unattended_supervisor.
13. **Monitoring** — Sentinel 10 probe + store budama, self_heal (3 tur, cooldown).
14. **Feedback** — Echo (3 kez zehir kontrolü, ayrı dosya).
15. **Web** — security (scope, headers, rate limit, upload), driver_scope, sse_tickets, server (lifespan, middleware, route'lar, cache-bust), router'lar, agent_graph, statik UI (CSP-güvenli, 15 sekme, RUN modalı, iki-tık kapı).
16. **MCP** — allowlist (kapalı küme + verify), proxy (bearer + driver headers, lazy import).
17. **CLI** — Typer komutları, çıkış kodları, dry-run varsayılanları, 5A-5E.
18. **Operasyon** — setup/install/update scriptleri, autostart, scheduled tasks, CI (dev+mcp extra), PR otomasyonu, protected paths guard, weekly bug-scan.
19. **Dokümanlar** — CLAUDE.md, HANDOFF.md, README (sıfır-varsayım), SECURITY.md, protokoller, skill/agent tanımları, manifest.

Her adımda Kademe-0 kapısı (`make format && make lint && make typecheck && make test`) yeşil olmalı; bir alt sistem tamamlanınca Kademe-2 adversarial av.

---

# 17. Bugünkü durum ve sıradaki adım (HANDOFF, 2026-07-22 → 08-03)

- **Motor bağlama P1-P9 KAPANDI.** ⚡ RUN sür modunda motor doğuruyor; MCP 21 uç; av verdict'i bağımsız + okuma-kanıtlı; DURDUR gerçekten kesiyor. Kalan: DOK (doküman senkronu).
- **Veri hattı KAPALI:** carding ✅ (185 makale; 483 onaylı / 3 bekleyen / 26 red), RLM ✅ (33 answered, 10 abstained, 5 §16 aday), curate ✅ (40 çok-versiyon düştü, 183 kanonik), assemble ✅ (**2000 örnek = 1324 synth-qa + 176 kart + 500 disiplin**), split ✅ (1900/100), pretrain-gate **GO** (2 epoch), Sentinel 9/9.
- **Audit uyarıları (bloklamaz):** Gate 0: 98 orphan → elendi; Gate 5: 57 performans iddiası (review); Gate 6: 94 felsefe (review); Gate 7: 1 red → elendi.
- **Sıradaki (insan kararları, Kural 8):** (1) 3 bekleyen kartı 06·ONAY'dan onayla/reddet; (2) `uv run hektor approval-approve <id>`; (3) `scripts/start-train.ps1 -Profile discipline_safe_local` (DETACHED); eğitim sonrası `evaluate_adapter` → registry ADAY → terfi insan onayı.
- 2026-08-03 handoff: `hektor_lora_v5_durable` (bf16, discipline_safe_local, hedef 2406 adım, checkpoint-25) unattended training guardian ile sürüyordu.
- Açık kararlar: `reports/bug-scan/` gitignore; graph_corpus count-only cache (chip); Contradiction Broker / Artemis RAG-drift ajan adayları (öncelik değil).

---

# Ek A — Sabitler tablosu

| Sabit | Değer | Dosya |
|---|---|---|
| chunk_size / overlap | 1200 / 200 | settings |
| `_MATH_WHOLE_MAX_CHARS` | 6000 | ingestion/chunker |
| `_MIN_SOURCE_CHARS` (kart) | 1500 | brain/knowledge_card_builder |
| `_BATCH_SIZE` / `_FAKE_DIM` | 64 / 256 | memory/embedding_service |
| Chroma `get_all(page)` | 5000 | memory/chroma_store |
| BM25 k1/b | 1.5 / 0.75 | memory/bm25_index |
| RRF k | 60 | memory/rank_fusion |
| PPR damping/iters | 0.85 / 20 | memory/graph_retriever |
| Reranker ağırlıkları | 0.40/0.30/0.20/0.10 | memory/reranker |
| `_SHORT_MAX_WORDS` | 6 | memory/query_router |
| `rag_abstain_min_similarity/margin` | 0.55 / 0.02 | settings |
| Mastery eşikleri | 90/75/60/40 | learning/mastery_scorer |
| Exam geçme | cit≥0.3, gnd≥0.4 | learning/rag_exam_runner |
| `_MIN_GRADED_FOR_SCORE` | 3 | verification/exams/understanding_score |
| Confidence abstain/warn | 0.40 / 0.70 | verification/confidence_scorer |
| RLM evidence retry/answer/skip | 40 / 60 / 80 | settings |
| §16 aday | conf≥0.85, cit≥0.90, gnd≥0.90 | rlm/lora_candidate |
| CostSpec | 0.0005 + 0.0005 | trading/strategy_ir |
| OOS split / min_trades / max DD | 0.7 / 30 / −50% | trading/overfit_checks, evaluator |
| `_MIN_EVAL_N` | 5 | training/adapter_eval |
| `_SPLIT_SEED` / `_VALID_RATIO` / `_LAUNCH_LOCK_TTL` | 42 / 0.05 / 120 s | training/detached_launch |
| pretrain-gate açılış-bigram | >%40 → NO-GO | training/dataset_quality |
| Disiplin örnek | 9×16×3 = 432; ratio 0.25 | training/discipline_dataset |
| Gate 6 yumuşak blok | ≥20 kart, >%25 | lora/gates |
| Splitter | 0.8/0.1/0.1, seed 42 | lora/dataset_splitter |
| Event retention | 30 gün / 50.000 | agents/runtime/tracker |
| Stale run sweep | 6 saat | agents/runtime/tracker |
| Pipeline aşama | 12; `max_steps` 50; `recover_stale` 30 dk | orchestration |
| HUNT/DRIVE timeout | 1800 / 3600 s | orchestration/driver |
| DRIVE_TOKEN_TTL / driver DEFAULT_TTL | 3900 / 2100 s | driver, web/driver_scope |
| STOPPED_RC / STOP_POLL | −99 / 1.0 s | driver |
| verdict_audit | 5 dosya / 2 alt-sistem / quote≥12 / 5 MB | orchestration/verdict_audit |
| Engine PROBE_TTL / grace | 60 s / 5 s | engines, engine_procs |
| Regression toleransları | 0.0/0.05/0.02/0.0/0.02/0.0 | orchestration/regression |
| Backoff | 300 s → 21600 s | unattended_supervisor, self_heal |
| Self-heal | 3 ardışık tur | monitoring/self_heal |
| Sentinel keep_last | 1000 | monitoring/store |
| SSE ticket TTL | 60 s | web/sse_tickets |
| Rate limit / upload | 120 / 60 dk | settings |
| Max upload | 100 MB | settings |
| MCP allow-list / forbidden | 21 / 12 | mcp_server/allowlist |
| Mastery queue max_attempts | 3 | memory/mastery_store |
| RAG loop rebuild attempts | 3 | research/rag_learning_loop |

# Ek B — Akademik referanslar (kodda atıf yapılan)
Cormack/Clarke/Büttcher 2009 (RRF) · Raudaschl 2024 (RAG-Fusion) · Bruch et al. arXiv:2210.11934 (konveks füzyon) · SPRIG arXiv:2602.23372 (CPU GraphRAG) · Lost in the Middle arXiv:2307.03172 · RAGAS arXiv:2309.15217 · CRAG arXiv:2401.15884 · RAFT arXiv:2403.10131 · R-Tuning arXiv:2311.09677 · Anthropic Contextual Retrieval · BEIR · Bandt-Pompe permütasyon entropisi · Forbidden patterns arXiv:0711.0729 · MPR complexity arXiv:1808.01926 · KL-regularized LoRA arXiv:2512.22337 · Deflated Sharpe · OPLoRA 2510.13003 · GRACE 2601.04525 · Pre-Gen Hallucination 2606.21917 · Conformal TS 2509.02844 / 2606.15953 · Global PE 2508.19955.

# Ek C — Komut hızlı referansı
```bash
# kurulum / doğrulama
bash setup.sh | irm .../install.ps1 | iex ; uv run hektor init ; uv run hektor status ; uv run hektor doctor
make format && make lint && make typecheck && make test      # Kademe-0 kapısı
uv run pytest -q --basetemp=.pytest_tmp                        # Windows

# web / mcp
uv run hektor-web                                            # http://127.0.0.1:8765
uv sync --extra mcp && uv run python mcp_server/hektor_mcp.py

# içerik
uv run hektor ingest ; uv run hektor arxiv "momentum" --max-results 5 ; uv run hektor ask "..."
uv run hektor rlm-answer "..." ; uv run hektor card <paper_id> ; uv run hektor research "..."

# veri hattı
uv run hektor synth-qa-bulk --target 1000 ; uv run hektor lora-curate --run
uv run python scripts/assemble_sft.py ; uv run hektor lora-audit ; uv run hektor pretrain-gate
uv run hektor lora-split ; uv run hektor lora-readiness

# orkestrasyon / eğitim (Kural 8)
uv run hektor orchestrate-start ; uv run hektor orchestrate-smoke
uv run hektor orchestrate-autodrive <run_id> --execute       # av (hunt) veya web ⚡ RUN (drive)
uv run hektor approval-approve <id> ; uv run hektor train --run --profile discipline_safe_local
.\scripts\start-train.ps1 -Profile discipline_safe_local      # detached
uv run hektor lora-eval <adapter> --n 8 ; uv run hektor registry-promote-dataset --version <id> --approver <kim>

# izleme
uv run hektor sentinel ; uv run hektor stop-all ; uv run hektor clear-stop-all
```
