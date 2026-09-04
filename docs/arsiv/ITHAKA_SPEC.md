# Ithaka — Yerel-Öncelikli Trading Araştırma Sistemi · Proje Spesifikasyonu

_Sürüm 1.0 · 2026-09-02 · Bu belge, sıfırdan yazılacak yeni projenin tam tasarım sözleşmesidir. Achilles Trader AI'ın kanıtlanmış mimarisinden türetilmiş, ölü/çelişik parçalar atılmış ve bilinen kusurlar tasarıma düzeltilmiş hâliyle gömülmüştür._

> **Ad:** "Ithaka" çalışma adıdır. Paket `ithaka`, CLI `ithaka`, web `ithaka-web`, ortam değişkeni öneki `ITHAKA_`. Başka ad seçilirse bu dört yer toplu değiştirilir.

---

## İçindekiler

1. Amaç, sınırlar ve sekiz mutlak kural
2. Mimari
3. Teknoloji yığını
4. Dizin yapısı
5. Konfigürasyon
6. Veri şeması
7. Modül sözleşmeleri (paket paket)
8. Ajanlar ve motorlar
9. Güvenlik ve yetki modeli
10. Eğitim yaşam döngüsü
11. Web arayüzü
12. Operasyon (kurulum, güncelleme, zamanlanmış işler, CI)
13. Test stratejisi
14. İnşa sırası
- Ek A: Sabitler
- Ek B: Achilles'ten bilinçli çıkarılanlar
- Ek C: Akademik referanslar

---

# 1. Amaç, sınırlar ve sekiz mutlak kural

## 1.1 Amaç
Ithaka, akademik finans makalelerini (PDF) okuyan, bunlardan bilgi kartı ve formül çıkaran, formülleri birleştirerek strateji hipotezleri üreten, hipotezleri disiplinli backtest'ten geçiren ve isteğe bağlı olarak küçük bir LoRA adaptörü eğiten **yerel-öncelikli araştırma sistemidir**. Windows, macOS (Apple Silicon) ve Linux'ta çalışır.

Kısa formül: **PDF → RAG / bilgi kartı → hipotez → backtest → (opsiyonel) LoRA.**

## 1.2 Sınırlar (ne DEĞİL)
- **Canlı bot değil.** Borsa bağlantısı, emir, canlı sinyal yok. `allow_live_trading_signal` ayarı yoktur; kod yolu hiç yazılmaz.
- **Yatırım tavsiyesi değil.** Her çıktı "hipotez + test noktası"dır; tavsiye dili tespit edilip reddedilir.
- **Bulut API ürünü değil.** Çalışma zamanı LLM'i yalnız yerel Ollama'dır. OpenAI/Anthropic/Google istemcisi **yazılmaz** (Achilles'teki opsiyonel bulut kodu ve alexzhang adapter'ı bilinçli olarak dışarıda bırakıldı; bkz. Ek B). Geliştirme yardımı abonelikli CLI motorlarıyla (Claude Code, Codex CLI) yapılır.

## 1.3 Sekiz mutlak kural ve kodda karşılıkları

| # | Kural | Zorlayan mekanizma |
|---|---|---|
| 1 | Yatırım tavsiyesi üretme; çıktı hipotez + test noktası | `guards.trading_language` tek regex modülü (RLM uyarısı, hipotez değerlendirici, Echo zehir filtresi, Gate 6, safety scanner hepsi bunu kullanır) |
| 2 | Test edilmeden "başarılı" deme; backtest + out-of-sample şart | `evaluator` verdict politikası (`pass/fail/inconclusive`; veri yetersizse `inconclusive`), adapter eval `min_n` + regresyon vetosu, `MIN_GRADED_FOR_SCORE=3`, smoke "stub≠runtime" |
| 3 | Maliyetleri yok sayma (komisyon + slippage) | `CostSpec` varsayılanı 5+5 bps, `_net_returns`, risk yöneticisi maliyet-dahil getiriler, disiplin verisi "maliyet token'ı" şartı |
| 4 | Look-ahead yasak; pozisyon `shift(1)` | `backtester._net_returns` `eff_pos = position.shift(1)`, matematik doğrulayıcı lookahead işareti, disiplin tuzağı |
| 5 | `eval`/`exec` yok; kurallar güvenli regex ile | `strategy_ir.RULE_RE`, `safe_eval` whitelist AST, RLM araçlarında kod yürütme yok |
| 6 | Determinizm; rastgelelik daima `seed` ile | Split seed 42, BM25/RRF tie-break, PPR sabit iterasyon, tüm LLM çağrılarına seed, araç registry'sinde `requires_seed` |
| 7 | Kaynak uydurma; retrieval boşsa açıkça söyle | "Kaynak bulunamadı" yanıtı, RLM `abstained`, çapraz sentez ≥2 makale şartı, formül tespitinde kelime sınırı, motor `logged_in` daima `null` |
| 8 | Otomatik ağır eğitim yok; `train` varsayılan dry-run; gerçek eğitim `--run` + tek-kullanımlık insan onayı | `authorize_training_action` (tek yetki kaynağı), `approvals.require_fresh_approval` (atomik CAS), `require_human` (403), `STOP_ALL`, orkestrasyon `approval` aşaması, MCP allow-list |

## 1.4 Tasarım ilkeleri
- **Şüphecilik varsayılan:** `verdict != pass` çıktı **aday**dır, "hazır" değil.
- **Anlama sınavla kanıtlanır:** L3 uygulama, L4 karşıolgu, L5 kompozisyon sınavları objektif skordur; kart doluluk yüzdesi yalnız göstergedir.
- **RAG bilgi, LoRA üslup:** ayrı inşa edilir, zincir olarak birlikte kullanılır.
- **Gözlemci üretimi bozmaz:** tracker/sentinel/self-heal istisna fırlatmaz, ağır iş başlatmaz.
- **Prompt talimatı sınır değildir:** doğurulan motorun sınırı araç kısıtıdır, HTTP auth değil.
- **Motor kendi başarısının tek kaynağı olamaz:** av "PASS"i dosya sistemiyle bağımsız doğrulanır.
- **Tek kanonik kaynak:** SFT verisi için tek dosya, eğitim yetkisi için tek fonksiyon, tavsiye-dili için tek regex modülü, adapter kaydı için tek registry.
- **Testler ağaca yazmaz:** tüm veri/durum yolları `settings.root`'tan türer ve test oturumunda tmp'ye yönlendirilir.
- Kod stili: Python ≥ 3.12, `from __future__ import annotations`, pydantic v2, SQLAlchemy 2.0, ruff (100 kolon), mypy (pydantic plugin), vektörize pandas/numpy, kullanıcıya dönük metin **Türkçe**.

## 1.5 Bug-avı kadansı
| Kademe | Ne | Tetik |
|---|---|---|
| 0 | `make ci` (format + lint + typecheck + test) | Her commit |
| 1 | Tek `claude -p` rapor-only tarama (araç-kısıtlı) | Haftalık zamanlanmış görev |
| 2 | Çok-ajan adversarial av (paralel finder → ≥2 oylu şüpheci doğrulama → yalnız onaylananı düzelt) | Ayda 1 **+ her LoRA eğitiminden önce (zorunlu)** |

---

# 2. Mimari

## 2.1 Veri akışı
```
PDF (data/papers/raw_pdf/)  ← kullanıcı / arxiv-fetcher / literature-scout (yalnız gelen kutusu)
  │ ingestion   parse (PyMuPDF→pypdf) · metadata · math-aware chunk (1200/200) · paper_id = "paper_"+sha256[:12]
  │ memory      SQLite (papers/chunks embedded=0) → Ollama embed → Chroma "paper_chunks" → embedded=1 → cache reset
  │             → FormulaExtractor → ConceptGraph → CrossPaperSynthesizer (best-effort)
  │ brain       RagAnswerer (router | graph | rrf | overfetch+rerank) → LocalLLM (Ollama)
  │             KnowledgeCardBuilder → knowledge_cards (pending) → İNSAN onayı
  │ rlm         classify → retrieval⇄reformulate → evidence gate → draft → claims → verify → confidence → abstain
  │ research    SynthesisEngine → StrategyIR → backtest → evaluate → L5 gate → Reflection (tek değişiklik/iterasyon)
  │ trading     StrategyIR (regex kural) → indicators (vektörize) → backtester (shift(1), maliyet) → IS/OOS → verdict
  │ learning    Paper Mastery: inspect → 20 soru → RAG sınavı → 100 puan → durum
  │ verification Anlama merdiveni: Taban/L1/L2 (RAG) + L3/L4 (LLM sınav) + L5 (kompozisyon)
  │ training    synth-qa + küratörlü kart + disiplin tuzakları → sft.jsonl (KANONİK) → Gate 0-8 → pretrain-gate
  │             → orkestrasyon (12 aşama) → İNSAN onayı → detached PEFT/MLX → adapter eval → registry aday → terfi (insan)
  └ monitoring  Sentinel (salt-okuma probe'lar) → self-heal (2 runbook) → unattended supervisor
```

## 2.2 Katmanlar
| Katman | Paket | Sorumluluk |
|---|---|---|
| Girdi | `ingestion` | PDF keşif, parse, metadata, chunk, kalite skoru, arXiv |
| Bellek | `memory` | SQLite ORM, Chroma, embedding, BM25, hibrit/RRF/graf retrieval, rerank, mastery store |
| Beyin | `brain` | LocalLLM (Ollama, MLX), RAG cevaplayıcı, bilgi kartı, sentetik QA |
| Araştırma | `research` | Formül çıkarımı, kavram grafı, sentez, yansıma, çapraz sentez, öğrenme döngüsü, literatür tarama, otomatik araştırma |
| Öğrenme | `learning` | Paper Mastery |
| Doğrulama | `verification`, `evals` | Atıf/dayanak/yeterlilik/çelişki/güven/çekimserlik; L3-L5; RAGAS-offline; ReleaseGate; hipotez değerlendirici |
| RLM | `rlm` | Çok-adımlı kaynaklı cevap kontrolcüsü + güvenli araç allowlist'i |
| Trading | `trading`, `tools` | IR, indikatörler, backtester, overfit, risk, Pine/paket export, Monte Carlo/istatistik |
| Eğitim | `lora`, `training`, `registry`, `feedback` | Gate 0-8, dataset üretimi, PEFT/MLX trainer, adapter eval, tek sürüm/terfi kaydı, Echo |
| Runtime | `runtime`, `orchestration`, `monitoring` | Manifest/registry, tracker, supervisor, approvals, task queue, executor, chain; 12-aşamalı orkestrasyon; AutoDriver; motorlar; verdict audit; Sentinel; self-heal; unattended supervisor |
| Ortak | `guards`, `config` | Tavsiye-dili/sır/PII regex'leri (tek yer), ayarlar |
| Yüzey | `web`, `mcp_server`, `cli` | FastAPI + 15 sekme statik UI, MCP proxy (allow-list), Typer CLI |

## 2.3 İki depo, tek anahtar
SQLite kaynak, Chroma türev; bağ `chunk_id = f"{paper_id}_c{index:04d}"`. Tek SQLite dosyası (`storage/sqlite/ithaka.db`) tüm store'larca paylaşılır; her bağlantı `PRAGMA journal_mode=WAL` + `busy_timeout=30000` açar. FK enforcement uygulama seviyesinde (SQLite'ta kapalı).

## 2.4 Otomasyon zinciri (manifest `chain`, Kahn topolojik)
```
arxiv-fetcher → rag-learning-loop → {ingestion-quality-scorer, paper-mastery-agent}
paper-mastery-agent → {scientific-tool-runtime, hypothesis-evaluator, rlm-controller(yaprak)}
paper-mastery-agent → lora-control-plane → dataset-quality-gate → model-data-registry (İNSAN ONAYI — supervisor durur)
{dataset-quality-gate, model-data-registry} → auto-lora-pipeline → adapter-eval
```

---

# 3. Teknoloji yığını

## 3.1 pyproject
- `name = "ithaka"`, `requires-python = ">=3.12"`, hatchling, `packages = ["ithaka"]`.
- Entry points: `ithaka = "ithaka.cli.main:app"`, `ithaka-web = "ithaka.web.server:run"`.
- `uv.lock` commit'lenir.

**Zorunlu:** pandas≥2.2, numpy≥1.26, pydantic≥2.7, pydantic-settings≥2.3, sqlalchemy≥2.0, chromadb≥0.5, pymupdf≥1.24, pypdf≥4.2, typer≥0.12, rich≥13.7, httpx≥0.27, fastapi≥0.110, uvicorn[standard]≥0.29, python-multipart≥0.0.9, psutil≥6.0, pyyaml≥6.0.
(`anthropic`, `google-genai`, `requests`, `pyarrow` **yok** — pyarrow chromadb transitif isterse kendisi çeker.)

| Extra | İçerik |
|---|---|
| `train-cpu` | torch≥2.2, transformers≥4.40, peft≥0.10, accelerate≥0.29 |
| `train-mlx` | mlx-lm≥0.16 (Darwin arm64) |
| `mcp` | fastmcp≥2.0 (CI'da zorunlu) |
| `rerank` | flashrank (opsiyonel ONNX reranker) |
| `dev` | pytest≥8.2, pytest-cov, ruff≥0.5, mypy≥1.10, pre-commit, types-PyYAML |

## 3.2 Araç ayarları
- ruff: `line-length=100`, `target-version="py312"`, `select=["E","F","I","UP","B","C4","SIM","RUF"]`, `ignore=["B008","RUF001","RUF002","RUF003"]`, `extend-exclude=["*.ipynb"]`.
- mypy: `python_version="3.12"`, `warn_unused_ignores`, `ignore_missing_imports`, `plugins=["pydantic.mypy"]`, `exclude=["tests/","scripts/"]`; torch/peft kurulu olmadan temiz geçmeli.
- pytest: `testpaths=["tests"]`, `addopts="--strict-markers -m 'not ollama and not slow'"` (**`-q` addopts'ta DEĞİL** — özet satırı gizlenmesin), marker `ollama`, `slow`.
- pre-commit: gitleaks + detect-private-key.
- Makefile: `install`, `test`, `lint`, `format`, `typecheck`, `ci`, `audit` (pip-audit), `web`, `clean`.

## 3.3 Çalışma zamanı modelleri
| Rol | Varsayılan | Ayar |
|---|---|---|
| LLM (Ollama) | `qwen3:4b` (8 GB); 16 GB `qwen3:8b`; 32 GB `qwen3:14b` | `llm_model` |
| Embedding | `nomic-embed-text` | `embed_model` |
| Eğitim base (PEFT) | `Qwen/Qwen2.5-1.5B-Instruct` (CPU'da eğitilebilir) | `peft_base_model` |
| Eğitim base (MLX) | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `mlx_base_model` |
| Ollama host | `http://127.0.0.1:11434` (IP; Windows IPv6 sorunu) | `ollama_host` |

---

# 4. Dizin yapısı

```
ithaka/
├── CLAUDE.md · README.md · SECURITY.md · HANDOFF.md (kısa; tarihçe docs/arsiv/)
├── automation_manifest.yaml       # runtime ajanları + chain (tek bildirimsel kaynak)
├── pyproject.toml · uv.lock · Makefile · .env.example · .pre-commit-config.yaml
├── setup.sh · setup.ps1 · install.ps1 · update.sh · update.ps1
├── ithaka/
│   ├── config/settings.py         # pydantic-settings, ITHAKA_ öneki, root-göreli yollar
│   ├── guards/                    # trading_language.py, secrets_pii.py (TEK regex kaynağı)
│   ├── ingestion/                 # paper_loader, pdf_parser, metadata, chunker, clean_text, quality_scorer, arxiv_fetcher
│   ├── memory/                    # sqlite_store, chroma_store, embedding, retrieval, bm25_index, bm25_corpus, graph_corpus, graph_retriever, rank_fusion, query_router, reranker, flashrank_reranker, reranking_retriever, contextual_flags, paper_indexer, mastery_store
│   ├── brain/                     # local_llm, mlx_llm, rag_answerer, knowledge_card_builder, synthetic_qa_builder, answer_quality, prompt_loader, prompts/*.md
│   ├── research/                  # orchestrator, synthesis_engine, formula_extractor, concept_graph, cross_paper_synthesizer, reflection_agent, chain_data_builder, rag_learning_loop, literature_scout, rag_trend_scanner, synthesis_paper, auto_researcher
│   ├── learning/                  # paper_mastery_agent, paper_inspector, question_generator, rag_exam_runner, mastery_scorer, status_manager, report_generator
│   ├── verification/              # citation, grounding, context_sufficiency, contradiction, confidence, abstention, comprehension_scorer, rag_mastery, exams/{safe_eval, reference_oracle, registry, l3, l4, l5, discipline_exam, understanding_score, understanding_record}
│   ├── evals/                     # metrics, golden_dataset, retrieval_eval, rag_ragas_offline, trading_hypothesis_evaluator, eval_runner, release_gate
│   ├── rlm/                       # controller, task_classifier, evidence_builder, claim_extractor, store, lora_candidate, safe_tools, tool_registry
│   ├── trading/                   # strategy_ir, indicators, backtester, evaluator, overfit_checks, risk_manager, market_data, strategy_generator, package_exporter
│   ├── tools/                     # tool_registry, probability_simulator, statistics_checker, result_verifier
│   ├── lora/                      # control_plane, gates, auto_pipeline, dataset_builder, dataset_splitter, curriculum, domain_classifier, math_verifier, quality_filter, card_curation, peft_llm_shim
│   ├── training/                  # peft_lora_train, mlx_lora_train, backend, detached_launch, adapter_eval, dataset_quality, discipline_dataset, sft_assembly, reward_signal, dpo_dataset_builder, tool_use_trainer, mastery_sft_builder, unattended_policy
│   ├── registry/                  # adapter_registry (TEK), version_store, promotion_gates
│   ├── feedback/                  # echo, store
│   ├── runtime/                   # schemas, registry, tracker, supervisor, approvals, task_queue, executor, handlers, chain, preflight, system_profiler, model_advisor
│   ├── orchestration/             # pipeline, store, orchestrator, delegates, driver, engines, engine_procs, verdict_audit, smoke, collision, regression, unattended_supervisor
│   ├── monitoring/                # sentinel, self_heal, store
│   ├── web/                       # server, schemas, security, driver_scope, sse_tickets, version_info, training_manager, lora_chat_service, agent_graph, routes/{engines, orchestration, feedback, sentinel, registry_tools}, static/
│   ├── mcp_server/                # proxy.py, allowlist.py
│   └── cli/                       # main.py (Typer) + alt modüller (ingest, research, training, orchestrate, runtime)
├── configs/lora_profiles.yaml     # 4 profil
├── evals/                         # discipline_core / overfit_awareness / risk_management .jsonl
├── scripts/                       # kurulum/doğrulama/eğitim/döngü/PR/tarama
├── docs/                          # protokoller, mimari, güncelleme kılavuzu, arsiv/
├── tests/                         # %100 çevrimdışı
├── data/ · models/ · storage/ · vector_db/ · reports/ · logs/   # runtime; yalnız .gitkeep izlenir
├── .claude/ {agents/, skills/, settings.json, launch.json}
└── .github/workflows/ {ci.yml, weekly-audit.yml}
```

Achilles'e göre farklar: `app/cli` boş paketi ve `strategies/` gitti; `agents/` → `runtime/` (local_training_5A-5E, benchmark, installer, rules_updater çıkarıldı — orkestrasyon aşamaları aynı işi yapıyor); `pipeline/` → `research/auto_researcher`; `reliability/` → `evals/release_gate`; iki `AdapterRegistry` → `registry/adapter_registry` (tek); `promotion_gates` sır/PII regex'i → `guards/secrets_pii`; ölü `query_expander`/`multi_query_retriever`/`hybrid_retriever`/`regression_runner`/`answer_eval`/`golden_generator` yok; cross-encoder reranker yok (FlashRank yeterli, CPU'da >15 s olan bge kaldırıldı).

---

# 5. Konfigürasyon (`ithaka/config/settings.py`)

`Settings(BaseSettings)`, prefix `ITHAKA_`, `.env` okunur, `get_settings()` lru_cache. **Tüm yollar `root`'tan türer**; `root` env ile (`ITHAKA_ROOT`) değiştirilebilir → testler tmp'ye yönlendirir.

| Alan | Varsayılan | Not |
|---|---|---|
| `root` | proje kökü | Tüm göreli yolların tabanı |
| `ollama_host` / `llm_model` / `embed_model` / `ollama_keep_alive` | `http://127.0.0.1:11434` / `qwen3:4b` / `nomic-embed-text` / `30s` | Eğitim sırasında keep_alive `0` |
| `peft_base_model` / `mlx_base_model` | §3.3 | |
| `sqlite_path` / `chroma_path` | `storage/sqlite/ithaka.db` / `vector_db/chroma` | |
| `rag_top_k` / `chunk_size` / `chunk_overlap` | 6 / 1200 / 200 | |
| `rag_rerank` / `rag_overfetch` / `rag_hybrid` | True / 4 / True | |
| `rag_flashrank` / `rag_flashrank_model` | False / `ms-marco-MiniLM-L-12-v2` | |
| `rag_rrf` / `rag_rrf_k` | False / 60 | |
| `rag_graph` / `rag_graph_damping` / `rag_graph_iters` | False / 0.85 / 20 | |
| `rag_router` / `rag_router_alpha` | False / 0.7 | |
| `rag_contextual_embed` | False | Tüm korpus aynı ayarla embed edilmeli |
| `rag_reorder_context` / `rag_verify_citations` | True / True | |
| `rag_abstain` / `rag_abstain_min_similarity` / `rag_abstain_min_margin` | False / 0.55 / 0.02 | |
| `rlm_max_retrieval_rounds` | 3 | |
| `rlm_min_evidence_to_retry/answer/skip_retry` | 40 / 60 / 80 | |
| `rlm_enable_query_reformulation` | True | |
| `rlm_seed` / `rlm_draft_max_tokens` / `rlm_draft_timeout_s` | 42 / 900 / 600 | |
| `default_market` / `default_timeframe` | `XAUUSD` / `15m` | |
| `allow_fake_embeddings` | True | False → Ollama yoksa `RuntimeError` |
| `synthesis_mirror_dir` / `scout_inbox_dir` | `""` / `""` | Boş = kapalı / `data/literature_inbox` |
| `unattended_training_enabled` | **False** | Eğitim yetkisinin tek politika anahtarı (Achilles'te True'ydu; yeni projede kapalı başlar) |
| `unattended_engine` | `"claude"` | Unattended supervisor motoru (kod içinde sabit değil) |
| `auto_lora_min_cards` / `auto_lora_check_interval_min` / `auto_lora_eval_threshold` / `auto_lora_eval_sample_n` | 20 / 60 / 0.5 / 8 | |
| `background_loops_enabled` | True | Test oturumu **False** yapar → web lifespan hiçbir döngü başlatmaz |
| `web_host` / `web_port` / `api_token` | `127.0.0.1` / 8765 / `""` | Token boş = auth kapalı (yalnız localhost) |
| `cors_origins` / `trusted_hosts` / `hsts_enabled` | `http://127.0.0.1:8765,http://localhost:8765` / `""` / False | |
| `max_upload_mb` / `rate_limit_per_min` / `upload_rate_limit_per_min` | 100 / 120 / 60 | |
| `log_level` | INFO | |

`ensure_dirs()` şunları oluşturur: `storage/sqlite`, `storage/mcp`, `vector_db/chroma`, `data/papers/{raw_pdf,extracted_text,metadata}`, `data/market/raw`, `data/sft`, `data/training/jsonl`, `data/feedback`, `models/adapters`, `reports/{papers,training,backtests,evals,agent_runs,rlm_runs,synthesis}`, `logs`.

**Script/runtime env'leri:** `UV_NO_SYNC=1` (çalışan web `.exe` kilidi), `ITHAKA_TRAIN_SUPERVISED=1` (üst katman onay tüketti; STOP_ALL yine geçerli), `ITHAKA_WEB_ISOLATE_CHROMA=1` (Windows native crash izolasyonu), `ITHAKA_WEB_URL` (MCP hedefi), `ITHAKA_DRIVER_TOKEN` / `ITHAKA_DRIVER_RUN_ID` (sürücü kimliği), `PYTEST_BASETEMP` yerine `--basetemp=.pytest_tmp`.

---

# 6. Veri şeması

## 6.1 SQLite tabloları (tek dosya; ORM `memory/sqlite_store.py` + alt store'lar)

**RAG çekirdeği:** `papers` (paper_id PK, file_hash UNIQUE, title, authors JSON, year, source, n_pages, n_chars, quality_score, ingest_status) · `chunks` (chunk_id PK, paper_id FK, chunk_index, section_name, page_number, text, char_count, token_estimate, **embedded** 0/1; UNIQUE(paper_id, chunk_index)) · `knowledge_cards` (card_id, paper_id, model, card_json, trust_level draft/verified/canonical, **review_status** pending/approved/rejected, **lora_eligible** 0/1, difficulty, stage) · `formulas` · `concept_links` · `research_sessions` (question, iteration, synthesis, proposed_indicator_json, strategy_ir_json, backtest_result_json, verdict, reflection) · `arxiv_saved_queries` · `chunk_quality_flags`.

**Trading:** `strategies`, `backtests` (metrikler + metrics_json + verdict), `risk_reports`.

**Doğrulama/ölçüm:** `paper_comprehension`, `understanding_snapshots`, `eval_history`, `model_evaluations`, `reward_signals`, `tool_use_examples`, `golden_questions`, `tool_runs`, `tool_artifacts`, `paper_ingestion_runs`.

**Mastery:** `paper_learning_queue` (attempts, max_attempts=3), `paper_mastery_tests/questions/answers/scores`, `paper_status_history`.

**RLM:** `rlm_runs`, `rlm_steps`, `rlm_evidence`, `rlm_verifications`.

**Runtime:** `agent_runs`, `agent_events` (retention 30 gün VEYA 50.000), `automation_tasks`, `approval_requests` (`consumed_at` tek-kullanım).

**Orkestrasyon:** `orchestration_runs` (`orc_`), `orchestration_stages` (`orst_`; `claim_stage_running` CAS; `heartbeat_at` — orkestratör her delege çağrısı **öncesi ve sonrası** heartbeat yazar), `orchestration_events` (`orev_`).

**Kayıt defteri (TEK):** `adapters` (adapter_name, version, base_model, profile, status candidate→smoke_passed→eval_passed→approved→production, eval_json, meta_json, promoted_by, promoted_at), `dataset_versions`, `rag_index_versions`, `embedding_model_versions`, `promotion_decisions` (append-only).

**Diğer:** `feedback_corrections`, `sentinel_checks` (keep_last=1000).

Kritik metotlar: `upsert_paper`, `get_paper_by_hash`, `find_paper_by_title` (normalize uzunluk <12 → None), `add_chunks` (merge), `mark_chunks_embedded`, `delete_chunks_for_paper`, `has_embedded_chunks`, `list_all_chunks` (BM25 kaynağı), `approve_card` (içeriksiz kart → False), `set_card_lora_eligible` (review_status'a dokunmaz), `list_paper_ids`, `consume_fresh_approval` (CAS), `claim_automation_task_atomic` (CAS), `prune_agent_events` (500'lük parçalar), `log_tool_run` (contextmanager).

## 6.2 Chroma
`vector_db/chroma`, koleksiyon `paper_chunks`, cosine; id = chunk_id; document = orijinal chunk metni; metadata `paper_id, chunk_index, page_number(None→-1), section_name(None→""), title(None→"")`. Süreç içi paylaşılan client, tüm işlemler RLock altında, `add`=upsert, `get_all(page=5000)`.

## 6.3 Dosya sistemi (hepsi `settings.root` altında; runtime içerik gitignore, `storage/*.json` dahil)
| Yol | İçerik |
|---|---|
| `data/papers/raw_pdf/`, `extracted_text/{paper_id}.txt`, `metadata/{paper_id}.json` | Kaynak PDF ve türevleri |
| `data/sft/synthetic_qa.jsonl` | Sentetik QA birikimi |
| **`data/sft/sft.jsonl`** | **TEK kanonik SFT kaynağı** (`assemble_sft` üretir; başka hiçbir şey yazmaz) |
| `data/training/jsonl/{train,valid}.jsonl` | Her başlatmada `sft.jsonl`'den türetilir (seed 42, %5 valid) |
| `data/feedback/feedback_sft.jsonl` | Echo aday dosyası (oto-merge yok) |
| `data/market/raw/*.csv` | OHLCV |
| `data/literature_inbox/{topic}/`, `BULUNANLAR.md` | Keşif ajanı gelen kutusu |
| `models/adapters/<ad>/` | PEFT adapter + checkpoint'ler (metadata SQLite `adapters`'ta; sidecar JSON yok) |
| `reports/papers/{paper_id}_card.json`, `reports/papers/mastery/…`, `reports/backtests/`, `reports/evals/`, `reports/training/<ad>_loss.json`, `reports/lora/`, `reports/rlm_runs/`, `reports/synthesis/`, `reports/agent_runs/` | Raporlar |
| `storage/STOP_ALL` | Küresel kill-switch |
| `storage/STOP_TRAINING`, `storage/STOP_LEARNING` | Graceful döngü sinyalleri |
| `storage/train_status.json` (pid dahil, **tüm başlatıcılar yazar**) | Detached eğitim durumu |
| `storage/.training_launching` | Atomik başlatma kilidi (TTL 120 s) |
| `storage/{auto_lora,rag_learning,self_heal,unattended_supervisor}_state.json` | Ajan durumları |
| `storage/orchestration/regression_baseline.json` | Regresyon baseline (yalnız `--commit`) |
| `storage/mcp/drive-<run_id>.json` | Sür modu MCP config (sır içermez) |
| `logs/*.log`, `.web.pid` | Loglar |

---

# 7. Modül sözleşmeleri

Her alt bölüm: amaç → ana tipler/fonksiyonlar → invaryantlar. Sabitler Ek A'da toplu listelenir.

## 7.1 `guards/` — tek regex kaynağı
- `trading_language.py`: `advice_present(text) -> bool`, `advice_reasons(text) -> list[str]` — garanti/kesinlik/risksiz/"al-sat"/canlı-sinyal dili; EN+TR; **olumsuzlanmış hâl işaretlenmez** ("garanti değildir"); kanıt bağlamı ("backtest'te", "dönemde") çıplak performans iddiasını affeder.
- `secrets_pii.py`: `scan(texts) -> ScanResult(has_secret, has_pii, hits[])` — api_key (entropi + bilinen ön-ek), private key, `password=`, e-posta, telefon (bağlam şartı), ulusal kimlik (checksum + bağlam), finansal direktif. FP korumaları: hacim sayısı ≠ kimlik, uzun hash ≠ api_key, hiperparametre ≠ credential.
- Tüketiciler: RLM trading guard, hipotez değerlendirici, Echo, Gate 6, Gate 7, registry terfi kapısı, disiplin sınavı. **Başka yerde tavsiye/sır regex'i yazılmaz.**

## 7.2 `ingestion/`
- `paper_loader`: `DiscoveredPaper(path, file_hash)`; `paper_id = "paper_" + sha256[:12]`; `discover_pdfs` rekürsif, sıralı.
- `pdf_parser`: PyMuPDF → pypdf fallback → `RuntimeError`; `ParsedPdf(pages)`.
- `metadata_extractor`: sezgisel başlık (ilk 8 satır, 15-200 karakter, büyük harf değil), yıl regex, ≤10 yazar; yalnız ilk 4000 karakter.
- `chunker`: paragraf-farkındalıklı, math-aware; `_MATH_BLOCK_RE` (`$$`, `\[`, `\begin{equation|align|gather|multline}`, inline `$`), matematik oranı >0.12 → bütün emit; **`MATH_WHOLE_MAX_CHARS=6000` asla aşılmaz**; `chunk_id = f"{paper_id}_c{index:04d}"`; sayfa numarası korunur.
- `clean_text_scorer`: 0-10 (kontrol karakteri −4, U+FFFD −3, tire kırılması −2, <200 karakter −1).
- `quality_scorer`: 100 puan (parse 15 · metadata 10 · section 15 · formula 15 · table 15 · figure 10 · ocr 10 · cleantext 10); yokluk nötr (7.5/7.5/5); durum ≥90 ready_for_rag / ≥70 usable / ≥50 slow / ≥40 unstable / failed; `record=True` ile `paper_ingestion_runs`.
- `arxiv_fetcher`: Atom API, id normalizasyonu **kategori korur**; dosya `arxiv_{id}.pdf`; idempotent; `%PDF` sihirli bayt; tek hata turu çökertmez.

## 7.3 `memory/`
- `embedding_service`: Ollama `/api/embed` 64'lük batch (grup patlarsa tekli), fake mod sha256-tabanlı 256-dim L2-normalize (yalnız çevrimdışı/test).
- `chroma_store`: §6.2.
- `retrieval`: `Retriever` Protocol; `RetrievedChunk(chunk_id, paper_id, text, page_number, section_name, title, distance)`; `citation` = `[{paper_id}:{chunk_id}, s.{page}]`.
- `bm25_index`: saf BM25 (k1 1.5, b 0.75), Türkçe token; sıralama `(-score, doc_id)`.
- `bm25_corpus`: kaynak **SQLite** (`list_all_chunks`, yalnız `embedded=1`); cache `{"sig": (count, total_chars), "pair": (bm25, chunks)}`; `_build_lock` + çift kontrol; `reset_cache()` kilit altında.
- `graph_corpus` + `graph_retriever` (SPRIG-lite): terim-chunk bipartite graf, hub pruning `max_df_ratio=0.5`, PPR damping 0.85 / 20 iterasyon; imza yalnız count → `reset_cache()` otoritatif; ingest her iki cache'i resetler.
- `rank_fusion`: RRF `w/(k+rank+1)`, k=60, tie-break `(-score, id)`; k≤0 → ValueError.
- `query_router`: ≤2 kelime veya (≤6 kelime ∧ kısaltma|rakam|tırnak) → lexical, aksi semantic; `convex_fuse(dense, bm25, alpha)`.
- `reranker` (heuristik): semantic 0.40 / keyword 0.30 / section 0.20 / formula 0.10; bölüm öncelikleri abstract 1.0 … references 0.1; `tanh(2x)/tanh(2)`.
- `flashrank_reranker` (opt-in): ONNX-int8; paket yoksa heuristiğe düşer; eksik id'ler sona eklenir (chunk düşmez).
- `reranking_retriever`: sıra — `enabled=False` düz dense · router → lexical konveks hibrit / semantic saf dense · graph → PPR+RRF · rrf → dense+BM25 RRF · varsayılan: `candidate_k = k*overfetch` → dense → (hybrid) BM25 adayları (yalnız metni olan) → rerank → `[:k]`.
- `contextual_flags`: `ChunkQualityFlags` (formül / eksik formül / tablo / tanım / teorem / eksik argüman / komşu bağlam / prev-next); `annotate(chunks, paper_title)`.
- `paper_indexer.ingest_one(disc, force=False)`: (1) hash var ∧ `has_embedded_chunks` → skip; hash var ∧ gömülü yok → **yarım ingest, yeniden işle**; (2) parse + metadata; başlık yoksa dosya adı; (3) başlık dedup; (4) txt + json yaz; (5) `upsert_paper`; (6) eski chunk'ları SQLite+Chroma'dan sil; (7) `add_chunks(embedded=0)`; (8) embed (`build_embed_text` — ön-ek yalnız embedding'e); (9) `chroma.add`; (10) `mark_chunks_embedded`; (11) `bm25_corpus.reset_cache()`; (12) `graph_corpus.reset_cache()`; (13) best-effort formül → kavram grafı → çapraz sentez.
- `mastery_store`: §6.1; `get_next_queued()` `pending` **ve** `failed` (attempts<3) kayıtları seçer.

## 7.4 `brain/`
- `local_llm.LocalLLM(model, host)`: yalnız Ollama; `available()`, `generate(prompt, fmt, timeout, max_tokens, seed, temperature, system)`; httpx hataları → `LLMUnavailable`; Qwen3 thinking-mode yanıtı ayıklanır. `mlx_llm`: Apple Silicon adapter'lı üretim (subprocess).
- `rag_answerer.answer(question, top_k)`: retrieval → (`rag_abstain`) güven kapısı → zayıf retrieval'da **LLM çağrılmaz** → `rag_reorder_context` → LLM → dayanaksız atıf uyarısı; `llm_used` bayrağı; kaynak yoksa "Kaynak bulunamadı".
- `knowledge_card_builder`: `KnowledgeCard(paper_id, title, year, domain, main_claim, methods[], datasets[], trading_relevance, limitations[], possible_strategy_hypotheses[], risk_warnings[], implementation_notes[])`; `MIN_SOURCE_CHARS=1500` altı LLM'e gitmez; retry merdiveni 6000/700 → 3000/500 → orantılı kesitler; `difficulty`/`stage` sınıflaması; daima `pending`, `lora_eligible=0`, `trust_level=draft`; `paper_id.strip()`.
- `synthetic_qa_builder`: chunk'tan grounded QA; `MIN_ANSWER_CHARS=60`; `_is_grounded` (anchor ∩ ≥1, sayı alt-kümesi); persona rotasyonu; one-shot örnekte **sabit açılış öneki yok**; JSON coerce; hash + Jaccard≥0.9 dedup; seed iletilir; atomik yazım.
- `answer_quality`: deterministik güven/zayıflık/reorder/atıf doğrulama yardımcıları. `prompt_loader`: `brain/prompts/*.md`.

## 7.5 `research/`
- `orchestrator.run(question, iterations=3, paper_ids)`: sentez (veya yansıma IR'ı) → `StrategyIR.model_validate` (hata → `example_ir()`) → `len(df)>10000` ise timeframe 1h → backtest + evaluate → **L5 CompositionGate** (novelty imzası) → `save_research_session` → `pass` değilse `ReflectionAgent` (tek değişiklik) → tekrar.
- `synthesis_engine`: prompt kuralları ("garanti deme", "neden başarısız olabileceğini yaz", "maliyeti hesaba kat", tek kural, RSI eşiği geniş, yalnız JSON); `_parse_result` IR varsayılanları (RSI14/ATR14, `rsi_14 > 50`, costs 5+5 bps), `entry_rules[:1]`, RSI eşiği >53 → 50; son 3 başarısızlık ipucu.
- `formula_extractor`: LLM (text[:3000], json) → yoksa kural tabanlı 11 gösterge, **kelime sınırı** `\bRSI\b`; `fml_<hex12>`; makale içi dedup.
- `concept_graph`: ilişkiler {extends, measures, limits, combines, opposite_of, requires}; build öncesi makale linklerini sil (idempotent); seed 42.
- `cross_paper_synthesizer`: 8 fallback şablonu, kategori alias'ları; `syn_+sha256(ids)[:24]`; seed `int(sha256(block)[:8],16)`; **≥2 farklı paper_id şartı** (ikili ve üçlü); `merge(TrainingExample(cross_paper_synthesis))`.
- `reflection_agent`: kural tabanlı (az işlem → eşik düşür; aşırı DD → eşik artır/EMA filtresi; >2000 işlem → eşik artır; Sharpe<−1 ∧ >500 işlem → yön ters); **iterasyonda tek değişiklik**.
- `chain_data_builder`: `research_sessions` → `research_chains.jsonl` ({prompt, completion}).
- `rag_learning_loop` (varsayılan **kapalı**): ayarlar `interval_min=30 (5-1440)`, `fetch_enabled`, `fetch_interval_hours=24`, `max_fetch_per_cycle=5`, `cards_per_cycle=3`, `scores_per_cycle=5`, `score_use_llm`, `rebuild_empty=False`; tur fetch → card → `approve_if_substantive` (title ≥8 ∧ main_claim ≥40 alfanümerik) → (rebuild ≤3 deneme) → score; **LoRA eğitimi sürerken duraklar**; asyncio lock; durum `storage/rag_learning_state.json`; `history` son 20 tur.
- `literature_scout`: 4 konu paketi (rag/lora/rlm/math-physics); arXiv + çevrimdışı skor; gelen kutusuna indirir (`%PDF`, idempotent), `BULUNANLAR.md` + watchlist; **ingest/eğitim çağırmaz (AST testi)**.
- `rag_trend_scanner`: arXiv → `docs/watchlist/rag.md` (deterministik skor, id dedup).
- `synthesis_paper`: `reports/synthesis/sentez_*.md` + opsiyonel ayna (OSError yükselmez).
- `auto_researcher.run_pipeline`: onaylı kartlardan soru → tool-use seansları → DPO skorlama; `dry_run`, seed; seans hatası döngüyü kırmaz; sayım ve soru kaynağı **aynı** liste (`list_cards`).

## 7.6 `learning/` — Paper Mastery
- `paper_inspector`: statik 40 puan (parse 10, metadata 5, chunk kalitesi 15, index 10 — koşullu); `missing_steps`.
- `question_generator` (LLM'siz): 6 yapısal + karttan (3 hipotez, 2 formül) + 2 abstention sorusu; tekilleştirme; `count=20`.
- `rag_exam_runner._run_one`: `context_precision` yalnız gözlem; `no_answer` = boş / "Kaynak bulunamadı" / `not llm_used`; abstention: `no_answer or paper_id ∉ cited`; regular: `passed = not no_answer ∧ context_ok ∧ cit≥0.3 ∧ gnd≥0.4 ∧ not hallucination`; atıf skoru gerçek `[paper:chunk]` atıflarından; boş grounding → 0.0.
- `mastery_scorer`: statik 40 + retrieval 15 + citation 15 + grounding 15 (−halüsinasyon ≤5) + abstention 10 (soru yoksa nötr 5) + formül bonusu 5; durum ≥90 learned / ≥75 usable_needs_review / ≥60 partially_learned / ≥40 needs_rechunking / failed.
- `status_manager`: 14 durum, tarihçeden türetilir. `report_generator`: `{paper_id}_mastery_report.{json,md}`, idempotent. `paper_mastery_agent.run`: 6 adım; rapor ayrı try/except; `LearningQueue.run_next/run_all`.

## 7.7 `verification/`
- `citation_verifier` (`[paper:chunk(, s.N)]`), `grounding_verifier` (SUPPORTED / PARTIALLY / UNSUPPORTED / SPECULATIVE), `context_sufficiency.classify(query, chunks)` — **sorgu-bağlam token örtüşmesi hesaba katılır** (Achilles'te parametre kullanılmıyordu), `contradiction_detector`, `confidence_scorer` (context 0.25 / citation 0.30 / grounding 0.30 / formula 0.15; abstain <0.40, warn <0.70 — **tek güven formülü**, `answer_eval` yok), `abstention_policy`.
- `comprehension_scorer`: A kart doluluk (7 alan) 0.30 · B RAG precision@5 0.40 · C LLM anahtar-kelime 0.30 (seed 7, temp 0; LLM yoksa 0.5); `use_llm=False` hızlı mod.
- `rag_mastery`: `0.40·coverage + 0.30·comprehension + 0.30·min(1, n_examples/50)`.
- `exams/`: `safe_eval` (whitelist AST); `reference_oracle` = `compute_indicator`; `registry` (her gösterge için parametreler — **`compute_indicator` ile aynı küme**; L5 `_REGISTRY` ayrı tutulmaz, registry'den okunur); `l3_application` (`np.allclose`, LLM yoksa skipped); `l4_counterfactual` (yön koddan; olumsuzluk farkında; belirgin yön yoksa `no_data`); `l5_composition` (math + novelty ≥2 tip/kopya yok + maliyet-dahil backtest+OOS → `candidate`; test-edilemez backtest → `skipped`); `discipline_exam` (`evals/*.jsonl`); `understanding_score` (`pass_rate = passed/(passed+failed)`, `MIN_GRADED_FOR_SCORE=3`, 2 ardışık LLM hatasında bail, `llm=` ile adapter ölçümü, `l5_results_from_sessions` öncelikli); `understanding_record` (DB + JSON, `compare_understanding` aynı model → `regressed`).

## 7.8 `evals/`
`metrics` (recall@k, precision@k, MRR, nDCG) · `golden_dataset` · `retrieval_eval` · `rag_ragas_offline` (faithfulness eşik 0.3, context_precision 0.06, context_recall; proxy) · `trading_hypothesis_evaluator` (`guards.trading_language` kullanır; testable/maliyet/OOS/risk → ACCEPT/REJECT) · `eval_runner` (`trading-hypothesis`, `rag-retrieval`; `--strict` → `EvalGateError`; ertelenen tipler açık NotImplementedError) · `release_gate` (eşikler; eksik/NaN/inf/None/bool → **fail-closed**).

## 7.9 `rlm/`
- `controller.answer(query, paper_ids, top_k, rounds, write_report)`: classify → plan → (retrieval ⇄ reformulate)* → evidence gate → draft (max_tokens 900, timeout 600, seed) → claims → citation/grounding → contradiction → confidence → abstention → yapısal cevap → loglar. Durumlar `answered`, `answered_with_limitation`, `abstained`, `no_llm`, `failed`; bayat `running` reaper. **İki kapı:** kanıt < retry eşiği → LLM hiç çağrılmaz; sonrasında desteklenmeyen iddia atılır. `apply_trading_guard` soru **veya** cevap trading dili taşıyorsa uyarı bloğu (guards'tan). Uydurma atıf temizlenir; `abstained` yüksek güven rozeti taşımaz.
- `task_classifier` (deterministik `ReasoningPlan`), `evidence_builder` (0-100: relevance, coverage, section_diversity, citation_availability, method_limit_presence, recency, contradiction_risk −5/çelişki), `claim_extractor`, `store`, `lora_candidate` (§16: conf≥0.85, cit≥0.90, gnd≥0.90; export `requires_human_approval=true`), `tool_registry` (deny-by-default `ALLOWED_TOOL_NAMES`), `safe_tools` (`rag_search`, `get_paper_metadata`, `get_paper_chunks`, `calculator`=safe_eval, `citation_check`, `grounding_check`, `contradiction_check`, `formula_check`; istisna yapısal hata döner).
- Dış motor adapter'ı **yok** (Ek B).

## 7.10 `trading/`
- `strategy_ir`: `RULE_RE = ^\s*(\w+)\s*(<|<=|>|>=|==|!=)\s*(\w+|-?\d+(?:\.\d+)?)\s*$`; `IndicatorSpec(name, period=14, params: dict = {})` — **ek parametreler (`smooth_k`, `multiplier`) açık alanda taşınır, getattr yok**; `RiskSpec`, `CostSpec(commission=0.0005, slippage=0.0005)`; `StrategyIR(name, market, timeframe, indicators, entry_rules, exit_rules, risk, costs)`; `parse_rule`; `example_ir`; `to_pine()` yalnız Python tarafıyla **birebir aynı tanıma sahip** göstergeleri çevirir (EMA/SMA/RSI/ATR/MACD/BB), diğerleri için açık `// desteklenmiyor` yorumu; **slippage Pine `strategy()` çağrısına aktarılır**.
- `indicators`: `ema, sma, rsi, macd, atr, bollinger, entropy, permutation_entropy, forbidden_pattern_rate, complexity_entropy`; `compute_indicator(name, df, period)` registry — sınav registry'si ile **aynı** kümeyi yayınlar (`INDICATOR_NAMES`).
- `backtester`: `_compute_columns`, `_eval_rules` (AND), `_position_series` **vektörize** (giriş/çıkış sinyallerinden `ffill` ile durum türetimi; Python döngüsü yok), `_net_returns` (`eff_pos = position.shift(1)`, turnover kaydırılmış pozisyondan, maliyet her pozisyon değişiminde); `BacktestMetrics(n_trades, total_return_pct, sharpe, sortino, max_drawdown_pct, profit_factor, win_rate_pct)`; `persist_backtest`.
- `overfit_checks`: `in_out_of_sample(split=0.7, min_trades=30)`; veri bölünemiyorsa `out_sample=None`. `evaluator.evaluate`: **`oos is None → inconclusive`**; fail = OOS n_trades<30 ∨ DD<−50% ∨ OOS getiri≤0; inconclusive = uyarı ∨ OOS Sharpe<0.5; pass aksi.
- `risk_manager`: Kelly, drawdown ölçekleme, sabit risk; `_extract_trade_returns` maliyet-dahil gecikmeli pozisyondan.
- `market_data` (CSV OHLC zorunlu, sentetik seed'li), `strategy_generator`, `package_exporter` (`.ithpkg` v1: IR + Pine + Python).

## 7.11 `tools/`
`tool_registry` (`ToolDescriptor`, `requires_seed`, `validate_params`, `resolve`), `probability_simulator` (Monte Carlo + risk-of-ruin + VaR/ES; saf numpy; boş/negatif/inf → ValueError), `statistics_checker` (betimsel + permütasyon p-değeri; p asla 0), `result_verifier` (Sharpe>5, Kelly>1, inf/nan uyarısı). Çalışmalar `tool_runs`/`tool_artifacts`'a.

## 7.12 `lora/` — Gate 0-8
`_card_text` **tüm** kart alanlarını toplar.

| Gate | Ne | Sınıf |
|---|---|---|
| 0 kaynak | `paper_id ∉ list_paper_ids()` → orphan | fail |
| 1 şema | SFT rol sırası | fail |
| 2 curriculum | `difficulty` aralığı | fail |
| 3 domain | ≥1 domain (8 domain, EN/TR) | fail |
| 4 kalite | <50 karakter, soru tekrarı, duplicate (hash) | eleme |
| 5 matematik | lookahead, aşırı-emin dil, >%1000 getiri, >%100 risk, çıplak iddia → review; kanıt bağlamı affeder | uyarı |
| 6 felsefe | `guards.trading_language`; yumuşak blok ≥20 kartta >%25 | uyarı/soft |
| 7 güvenlik | `guards.secrets_pii`; **tek ihlal batch'i reddeder** | **BLOCKER** |
| 8 split | kaynak-bazlı 0.8/0.1/0.1, seed 42, sızıntı; boş valid/test → fail | fail |

- `control_plane.run_audit()` (0-7) / `run_full()` (0-8 + split); içeriksiz kartlar önce filtrelenir; Gate 7 Gate 4'te elenenleri de tarar; rapor `reports/lora/audit_report.md`.
- `auto_pipeline`: `PipelineStage` IDLE→CHECKING→GATE_FAILED|READY_TO_TRAIN→TRAINING→TRAIN_FAILED|EVALUATING→EVAL_FAILED|EVAL_SKIPPED|EVAL_PASSED→PROMOTED; durum `storage/auto_lora_state.json`; `start_training` yalnız READY_TO_TRAIN + `authorize_training_action`; `_run_eval` = `evaluate_adapter` + anlama merdiveni (adapter < base − 0.05 → regresyon); bağımlılık yoksa EVAL_SKIPPED (terfi edilemez); `auto_enabled = settings.unattended_training_enabled` (**varsayılan False**).
- `dataset_builder` (onaylı kartlar → SFT; curriculum pacing %60/30/10 **burada**, `phase` ve >20 örnekte), `dataset_splitter`, `curriculum.classify(difficulty)` (LEVEL_0..4), `domain_classifier`, `math_verifier`, `quality_filter`, `card_curation` (orphan + per-paper version-collapse → `lora_eligible=0`, idempotent, dry-run varsayılan), `peft_llm_shim` (`load_base_of(adapter_dir)` — adapter'ın kendi base'i).

## 7.13 `training/`
- `peft_lora_train`: `PeftTrainConfig` (iterations = **toplam adım** → `max_steps`; lr 2e-4; r 8/alpha 16; dropout 0.05; max_seq 1024; rslora/dora/init opsiyonları; neftune 0; `assistant_only_loss=False` opt-in; `kl_reg_beta=0`; seed 42; `max_examples`); `load_lora_profile(name)`; `build_masked_labels` (prompt token'ları −100), `_MaskedDataCollator`, `sample_rows(seed)`; `_KLRegTrainer` (adapter kapalı forward, ek model yok); `train(cfg)` → `reports/training/<ad>_loss.json`; `dry_run(cfg)`; `__main__` **`--profile` kabul eder**; `target_modules` 7 projeksiyon.
- `mlx_lora_train` (Apple Silicon sarmalayıcı), `backend.detect()` → `mlx`/`peft`.
- `detached_launch`: `ensure_train_split(settings)` (`sft.jsonl` → train/valid; kaynak boşsa dokunma), `build_training_split()` (kaynak yoksa `assemble_sft_lines` ile bir kez üret; `DatasetVersion pending` best-effort), `readiness()`, `training_status()`, **`launch(adapter_name, iterations, base_model, profile="discipline_safe_local", max_examples)`** (atomik kilit TTL 120; `ITHAKA_TRAIN_SUPERVISED`; `train_status.json` **pid dahil**; detached spawn), `request_stop` (pid kill + `STOP_TRAINING`), `is_running(root)`.
- `adapter_eval`: `_is_degenerate` (n-gram tekrar), boş cevap → `empty_answer` vetosu, `_resolve_base_model` (adapter_config.json), greedy üretim, `MIN_EVAL_N=5`; verdict: degenerate → reject; adapter<base → reject (her n); n<min_n → inconclusive; adapter>base → accept; eşit → inconclusive. **Terfi etmez.**
- `dataset_quality` (pretrain-gate): garanti vaadi → NO-GO; **>%40 tek açılış-bigramı** → NO-GO; sızıntı/maliyet-körü/küçük set → WARN; `recommend_epochs`.
- `discipline_dataset`: 9 tuzak × 16 strateji × 3 varyant = 432 deterministik örnek (R-Tuning abstain tuzakları dahil); açılışlar çeşitli; 1/3 system-prompt'suz; `mix_discipline(ratio)`.
- `sft_assembly.assemble_sft_lines(settings, discipline=True, ratio=0.25, seed=0)`: synth-qa + küratörlü kart → dedup (hash + Jaccard 0.9) → **disiplin dedup'tan sonra** → `data/sft/sft.jsonl`; `to_jsonl_line` U+2028/2029/0085 kaçışı. **Kanonik dosyaya yazan tek fonksiyon.**
- `reward_signal` / `dpo_dataset_builder` (6 kriter → chosen/rejected), `tool_use_trainer` (THINK→CALL→OBSERVE→CONCLUDE), `mastery_sft_builder`, `unattended_policy.authorize_training_action(action, summary, gates_passed, agent_id)` → `TrainingAuthorization(authorized, mode, reason, approval_id)`: STOP_ALL → blocked; `unattended_training_enabled ∧ gates_passed` → unattended; aksi `require_fresh_approval`. **Tüm eğitim yüzeylerinin tek yetki kaynağı.**

**`configs/lora_profiles.yaml` — 4 profil**

| Alan | smoke | standard | discipline_safe_local (varsayılan) | discipline_safe_kl (deneysel) |
|---|---|---|---|---|
| r / alpha | 8/16 | 16/32 | 16/32 | 16/32 |
| dropout | 0.05 | 0.05 | 0.1 | 0.1 |
| epochs | 1 | 2 | 1 | 1 |
| max_seq_length | 2048 | 2048 | 1024 | 1024 |
| learning_rate | 2e-4 | 2e-4 | 1e-4 | 1e-4 |
| neftune_noise_alpha | — | — | 5 | 5 |
| warmup_ratio | — | 0.03 | 0.05 | 0.05 |
| assistant_only_loss | — | — | **true** | true |
| kl_reg_beta | — | — | — | 0.01 |
| max_examples | 200 | — | 300 | 300 |

`discipline_safe_local` = `launch()`, `start-train.ps1`, `train-loop.ps1` ve `training-watchdog.ps1` için **ortak** varsayılan (hepsi aynı `_build_train_cmd` yardımcısını kullanır).

## 7.14 `registry/` (tek kayıt defteri)
- `adapter_registry` (SQLite `adapters`): `register(candidate)`, `set_status`, `promote(user_approved: bool)` (**False → hata**; aynı anda tek PRODUCTION), `reject(note)`, `get_production()`. Sidecar JSON ve JSONL yok.
- `version_store`: dataset/RAG-indeks/embedding sürümleri (SHA-256, idempotent), `cas_dataset_status`, `log_decision` (append-only).
- `promotion_gates`: `approve_dataset`/`reject_dataset` (pending → approved|rejected terminal, CAS tek kazanan), `check_rag_index_eval` (ReleaseGate), `gate_dataset` (`guards.secrets_pii`).

## 7.15 `feedback/` — Echo
`EchoCollector.record(correction, question, bad_answer, source, type)` → `guards.trading_language` **tüm alanlarda** → zehir `rejected`; `approve` (yeniden kontrol), `reject`, `export_approved(out_path)` (proje kökü içinde; **ayrı aday dosya** `data/feedback/feedback_sft.jsonl`; kanonik sete oto-merge yok; yazım penceresinde yeniden tarama); `feedback_corrections` deterministik sıra.

## 7.16 `monitoring/`
- `sentinel`: `ProbeResult(name, status ok|warn|fail|skip, detail)`; **10 salt-okuma probe**: llm, web, training, orchestration (stale peek), stop_all, disk, sqlite (`quick_check`), feedback, contention (DANIŞMAN; eğitimi duraklatmaz), rag_loop; agregasyon fail>warn>ok; probe istisnası skip; `run(persist=True)`, `history`.
- `store`: `sentinel_checks`, keep_last=1000 (aynı zaman damgasında yeni kayıt silinmez).
- `self_heal`: 2 idempotent runbook (`recover_stale` + doğrulama; `rag_loop.run_one_cycle`); **3 ardışık sağlıksız tur** şart; başarısız onarım → `min(21600, 300·2^(attempt−1))` cooldown; `storage/self_heal_state.json` atomik; eğitim/onay/terfi asla.

## 7.17 `runtime/`
- `schemas`: `AgentAutonomy` (manual/semi_auto/autonomous/requires_approval), `AgentSpec`, `AgentRun`, `AgentEvent`, `TaskStatus` (pending/claimed/running/completed/failed/cancelled/blocked_approval/blocked_stop_all), `ApprovalRequest`, `SupervisorDecision`.
- `registry.load_agent_registry(manifest)` → `ManifestError` (sessiz boş liste yok).
- `tracker.RunTracker` (SQLite + `reports/agent_runs/<run_id>.jsonl`; `arun_YYYYMMDD_HHMMSS_<8hex>`; `@tracked`; retention; `cancel_stale(6h)`; **asla fırlatmaz**).
- `supervisor`: registry'de mi → iptal → STOP_ALL (yalnız `dangerous`) → zaten çalışıyor mu → taze onay; `create_stop_all`, `clear_stop_all`, `run_with_supervision`.
- `approvals.require_fresh_approval(agent_id, action, risk, summary)` → `consume_fresh_approval` CAS (`consumed_at`); standing yetki yok.
- `task_queue` (CAS claim, requeue yalnız blocked_*), `executor` (allow-list handler; bilinmeyen `agent_id` çalışmaz), `handlers` (yalnız `model-advisor`), `chain` (Kahn; `ChainError`), `preflight`.
- `system_profiler` (RAM/GPU/CPU; Windows GPU için `Get-CimInstance Win32_VideoController`, `wmic` değil), `model_advisor.recommend(profile)`.

## 7.18 `orchestration/`
- `pipeline`: 12 aşama `preflight → collision → smoke → deep-hunt → data-gate → curriculum → dry-run → regression → approval → train → evaluate → registry`; `StageDef(autonomous)`; `autonomous=False` → orkestratör durur.
- `store`: §6.1; `claim_stage_running` CAS; **heartbeat orkestratör tarafından yazılır** (delege öncesi/sonrası ve uzun delegelerde 30 s'de bir) → `recover_stale` gerçekten çalışır.
- `orchestrator`: `start`, `step` (tam bir aşama; finalize olduysa yazmaz; `output_json default=str`), `run_until_blocked(max_steps=50)`, `recover_stale(30 dk)` (terminal clobber yok), `cancel`, `status`, `timeline`.
- `delegates`: preflight (STOP_ALL/veri/bağımlılık) · collision · smoke · deep_hunt (**hunt_ack yoksa blocked**) · data_gate (`audit_dataset` GO?) · curriculum · dry_run · regression · approval (`authorize_training_action(gates_passed=True)`; onayı **tüketmez**) · train_handoff (unattended? completed : blocked) · eval/registry handoff.
- `engines`: `Engine(name, label, probe, argv_template, hardened, drive_argv_template, drive_hardened, quota_warning)`; `PROMPT`/`MCP_CONFIG` sentinel'leri tek argv öğesi; `DEFAULT_ENGINE="claude"`; motorlar claude (hardened+drive), codex (hardened+drive), local (Ollama, spawn yok). Av argv: `claude -p PROMPT --safe-mode --strict-mcp-config --disallowedTools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Task`. Sür argv: `--setting-sources "" --disable-slash-commands --strict-mcp-config --tools Read,Grep,Glob --mcp-config MCP_CONFIG`. `--bare` yasak. `available()` = PATH'te mi; `describe()` kimlik alanı yok, `logged_in=None`; `PROBE_TTL_S=60`.
- `engine_procs`: run_id başına canlı süreç kaydı; `terminate_run`, `terminate_all(grace=5)`.
- `driver.AutoDriver.drive(run_id, execute=False, engine=None, mode="hunt"|"drive")`: hardened kontrolü → `driver_scope.mint` (hunt 2100 s, drive 3900 s) → `build_child_env` (insan token'ı boş, `CLAUDE_CODE_*` ayar-ezme env'leri silinir, sürücü token/run_id) → `Popen(argv, shell=False, cwd=root, argv[0]=mutlak yol)` + `engine_procs.register` + 1 s STOP_ALL yoklama → `STOPPED_RC=-99` → hunt: `parse_hunt_verdict` (son satır, yoksa FAIL) + `verdict_audit` → PASS ∧ audit.ok → `hunt_ack=true` → `run_until_blocked`; drive: `write_mcp_config` (sır yazmaz) → `parse_drive_verdict` (**hunt_ack yazmaz**) → config unlink; `finally revoke_run`. Koşu başına **kilit** (aynı run_id'de eşzamanlı sürüş reddedilir).
- `verdict_audit`: `EVIDENCE_MARKER="ITHAKA_HUNT_EVIDENCE"` JSON (`scanned_files:[{path,line,quote}]`, `subsystems`, `findings`); `extract_evidence` her istisnada None; kapılar: ≥5 var olan dosya (yol-geçişi reddi, tekil), ≥2 alt-sistem, PASS + {HIGH,BLOCKER,CRITICAL,SEVERE} bulgu → red, ≥5 okuma-kanıtı (`lines[line-1].strip()==quote.strip()`, `MIN_QUOTE_LEN=12`, bool line sayılmaz, aynı alıntı+dosya bir kez, `MAX_PROOF_FILE_BYTES=5_000_000`).
- `smoke.SmokeRunner(llm, retriever)`: backend canlı → küçük üretim (seed 42, 32 token, temp 0, degenere değil) → retrieval (boş → warn); çevrimdışı → **skip**; canlı+bozuk → fail. `run_smoke`: ⚡ RUN sözleşmeleri (10 yoklama) + `--allow-live-spawn` (CI'da asla).
- `collision` (git: index.lock / aynı-branch çoklu worktree / HEAD-drift → fail; kirli ağaç → warn; git yok → skip) · `regression` (v5 sinyalleri: `top_opening_share`, garanti, sızıntı, maliyet-körü, disiplin kapsamı, GO/NO-GO; toleranslar 0/0.05/0.02/0/0.02/0; baseline yalnız `--commit`).
- `unattended_supervisor`: 60 s reconciler; STOP_ALL → motor canlı? → backoff? → `drive(mode="drive", engine=settings.unattended_engine)`; tek canlı motor; backoff 300 s→6 sa; durum `settings.root/storage/unattended_supervisor_state.json`; `enabled` varsayılan **False** (kullanıcı web'den açar); `background_loops_enabled=False` ise hiç başlamaz.

## 7.19 `web/`
- **Kurulum:** `FastAPI(docs_url=/api/docs if token boş else None)`; lifespan: logging → `ensure_dirs` → `warn_if_auth_disabled` → `cancel_stale_running_agent_runs` → **`background_loops_enabled` ise** auto-lora / rag-loop (isolate değilse) / self-heal / unattended döngüleri → BM25 ısıtma thread'i. Middleware: TrustedHost (doluysa) → CORS (doluysa) → `_security_middleware` (`/api/` rate limit, upload limiti, `SECURITY_HEADERS`, HSTS).
- **security:** `require_auth` (token boş → geç; Bearer/X-Api-Token compare_digest; geçerli sürücü token'ı kimlik olarak geçer), `resolve_scope` (`X-Ithaka-Driver-Token` yoksa human; geçersizse 401), `require_human` (driver → 403), CSP `default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'` (**tüm statik sayfalar self-host font**), `RateLimiter` (IP başına 60 s pencere, sweep), `validate_pdf_upload` (`%PDF-`), `validate_csv_upload` (OHLC başlık), `sanitize_filename` (NFKD, 128), `safe_destination`.
- **driver_scope:** sha256(token) → (run_id, expires_at); `mint` (öncekini iptal), `verify` (tüketmez), `revoke_run`. **sse_tickets:** 60 s tek-kullanımlık bilet; insan api_token'ı query'de kabul edilmez.
- **Route grupları (A = auth, H = human_only):** sistem (`status`, `version`, `healthz`, `profile`/`recommend` auth-muaf, `supervisor/stop-all` A + `terminate_all`, `clear-stop-all` A+H) · makale (`papers`, `upload` sha256 dedup + BackgroundTasks, `ingest`, `comprehension`, `card`, `cards/batch`, `arxiv/*`) · onay (`cards/pending|approved`, `card/{id}/approve|reject` A+H; `empty` yanıtı) · RAG/RLM (`ask`, `rlm/*`, `rag-mastery`, `understanding-score` (`record` yalnız POST `/understanding-score/record`), `/history`, `learning/*`) · RAG döngüsü (`status`, `enable`, `run-once`, `config` kelepçeli) · backtest (`backtest` sentetik, `backtest/csv`, `backtests`, **`POST /backtest/{id}/risk`** (yazan uç GET değil), `risk-reports`, `pine`, `download-pkg`, `package/export`) · araştırma (`research/*`, `synthesis/*`, `lora-adapters`, `lora-chat`) · eğitim (`training/status|dataset|examples|dry-run|stop|progress|live|logs|stream-ticket|stream`, **`POST /training/run` A+H**) · auto-lora (`status`, `enable`, `check`, `train` A+H, `promote` A+H, `reset`) · runtime (`agents`, `agents/runs`, `agents/graph`, `automation/tasks` (sürücü → `requires_approval` zorla), `approvals/{id}/approve|reject` A+H) · motor (`engines`, `engines/rescan`) · orkestrasyon (`start` (`hunt_ack=true` → H), `status`, `timeline`, `resume`, `runs`, `recover`, **`autodrive/{id}` A+H** (bilinmeyen motor 400; blocked 503; drive için `drive_hardened`)) · feedback (`submit`, `list`, `summary`, `approve|reject|export` H) · sentinel (`run` POST, **`overview` salt-okuma — persist etmez**, `history`, `self-heal`, `unattended`) · registry/araçlar (`registry/{kind}`, `tools`, `ingestion-quality/{id}`, `eval/trading-hypothesis`).
- **`/api/training/run` kapısı:** `authorize_training_action("train_run", agent_id="lora-trainer")` → stop_all → blocked; yetkisiz → `needs_approval` + komut; yetkili → onay tüketildi → `launch(...)`. CLI ile aynı anahtar.
- **Statik:** `/assets` mount; `GET /` → index.html, `?v=` içerik sha256[:12], `Cache-Control: no-cache`.
- Diğer: `training_manager` (in-process eğitim + SSE), `lora_chat_service` (PEFT lazy-load), `version_info` (git drift, 30 dk throttle fetch), `agent_graph` (manifest → nodes/edges chain/data/control, gruplar; canlı motor → running).

## 7.20 `mcp_server/`
- `proxy.py`: spec in-process `app.openapi()`; çağrılar `httpx` ile çalışan web'e; `auth_headers()` (Bearer) + `driver_headers(env)` (sürücü kimliği aklanmaz); loopback dışı hedefte stderr uyarısı; tembel `__getattr__` (fastmcp import gerektirmez).
- `allowlist.py`: **varsayılan kapalı** spec budama; `ALLOWED` (21): `POST ask`, `GET rag-loop/status`, `POST rag-loop/run-once`, `GET cards/pending|approved`, `GET card/{id}`, `GET backtests`, `GET backtest/{id}/pine`, `GET status|healthz|version|profile`, `GET sentinel/history`, `GET agents|agents/graph|agents/runs|agents/runs/{id}`, `GET papers`, `GET learning/summary`, `GET rag-mastery`, `GET understanding-score/history`; `FORBIDDEN_SUBSTRINGS` (approvals, stop-all, clear-stop-all, training/run, orchestration/{autodrive,start,resume}, auto-lora/{promote,train,enable}, rag-loop/{enable,config}); `verify_allowlist` (drift), `AllowlistError` fail-closed. Yazan uçlar GET olmadığı için "yan-etkili GET" sınıfı tasarımda yok; test yine kaynak kodu tarar.

## 7.21 `cli/` — Typer (alt modüllere bölünmüş)
Çıkış kodları: 0 ok · 1 bulunamadı/geçersiz · 2 STOP_ALL / fail-verdict / sapma · 3 taze onay gerekli.

| Grup | Komutlar |
|---|---|
| Sistem | `init`, `status`, `doctor`, `runtime-init`, `chain-status [--live]` |
| Makale/RAG | `ingest`, `papers`, `ask`, `card`, `cards pending|approve|reject`, `arxiv`, `arxiv-sync`, `rag-scan`, `lit-scan`, `reindex-contextual`, `rag-mastery`, `synth-paper`, `ingestion-quality[-scan]` |
| Araştırma | `extract-formulas`, `formulas`, `research`, `research-sessions`, `chain-dataset`, `auto-research --dry-run` |
| RLM | `rlm-answer`, `rlm-runs`, `rlm-trajectory`, `rlm-lora-candidates --export`, `rlm-tools` |
| Trading | `gen-data`, `backtest`, `pine`, `export-package`, `risk` |
| Anlama | `exam-l3`, `exam-l4`, `exam-l5`, `understanding-score --full --with-rag --record`, `understanding-history --compare` |
| Mastery | `mastery-run`, `mastery-queue`, `mastery-score`, `mastery-report`, `mastery-to-sft` |
| Veri | `synth-qa`, `synth-qa-bulk --target`, `discipline-dataset --write`, `assemble-sft`, `lora-curate --run`, `lora-split`, `lora-readiness`, `pretrain-gate --json`, `lora-audit --run`, `tool-use-train`, `tool-use-dataset`, `reward-analyze --build-dpo` |
| Eğitim | **`train [--run] --backend auto|mlx|peft --profile --max-examples`** (dry-run varsayılan; `--run`: STOP_ALL→2, onay→3, `ensure_train_split`), `evaluate`, `lora-eval <adapter> --n`, `lora-chat`, `lora-status` |
| Orkestrasyon | `orchestrate-start --hunt-ack`, `-status`, `-resume`, `-list`, `-recover`, `-autodrive <id> [--execute] [--mode]`, `-smoke`, `-drive-live --allow-live-spawn`, `-collision`, `-regression [--commit]` |
| Runtime/onay | `agents-list`, `agents-runs`, `agents-log`, `task-create`, `tasks-list`, `task-cancel`, `tasks-run`, `approvals-list`, `approval-approve`, `approval-reject`, `stop-all`, `clear-stop-all` |
| İzleme | `sentinel [--history --no-persist]` |
| Feedback | `feedback-add/list/approve/reject/export/status` |
| Registry/araçlar | `registry-list`, `registry-snapshot`, `registry-register-dataset`, `registry-promote-dataset --approver`, `adapter-list`, `adapter-promote --approve`, `tools-list`, `montecarlo --seed`, `stats-check --seed`, `eval-runner --type --strict` |
| Model | `profile`, `recommend`, `pull-model` |

---

# 8. Ajanlar ve motorlar

## 8.1 Runtime ajanları (`automation_manifest.yaml`, 24)
Her kayıt: `agent_id, name, file, entrypoint, trigger, autonomy, dangerous, default_enabled, writes[], reads[], safety_gates[], approval_required, stop_method, status_location, known_failure_modes[]` (**tüm alanlar zorunlu**; registry eksik alanda `ManifestError` verir). `phase: 2`.

| # | agent_id | Modül | Otonomi | Tehlikeli | Öz / kapılar |
|---|---|---|---|---|---|
| 1 | `auto-lora-pipeline` | lora/auto_pipeline | autonomous | **evet** | Gate 0-8 → READY → (tek politika) eğitim → eval → terfi; STOP_ALL üstün; varsayılan kapalı |
| 2 | `training-orchestrator` | orchestration/orchestrator | autonomous | hayır | 12 aşama; deep-hunt hunt_ack'siz blocked; approval tek politika; checkpoint/resume; heartbeat |
| 3 | `orchestration-autodrive` | orchestration/driver | autonomous | hayır | Motor doğurur (execute=False varsayılan); av PASS → bağımsız denetim → hunt_ack; sür modu MCP'li |
| 4 | `unattended-supervisor` | orchestration/unattended_supervisor | autonomous | hayır | Tek canlı hardened motor; backoff; varsayılan kapalı |
| 5 | `training-guardian` | scripts/training-watchdog | autonomous | hayır | Ölen detached eğitimi checkpoint'ten sürdürür; profil açık; STOP_ALL üstün |
| 6 | `echo-feedback` | feedback/echo | semi_auto | hayır | Düzeltme → SFT adayı; zehir 3 kez; ayrı dosya |
| 7 | `sentinel-monitor` | monitoring/sentinel | semi_auto | hayır | 10 salt-okuma probe |
| 8 | `self-healing-controller` | monitoring/self_heal | autonomous | hayır | 3 ardışık hata → 2 runbook; doğrulama; backoff |
| 9 | `rag-learning-loop` | research/rag_learning_loop | autonomous | hayır | fetch→card→score; eğitimde duraklar; varsayılan kapalı |
| 10 | `research-orchestrator` | research/orchestrator | semi_auto | hayır | sentez → IR → backtest → L5 → yansıma |
| 11 | `literature-scout` | research/literature_scout | semi_auto | hayır | Keşif → gelen kutusu; ingest/eğitim yok |
| 12 | `rag-trend-scanner` | research/rag_trend_scanner | semi_auto | hayır | RAG yenilikleri → watchlist |
| 13 | `reflection-agent` | research/reflection_agent | manual | hayır | Tek değişiklik önerir |
| 14 | `auto-researcher` | research/auto_researcher | semi_auto | hayır | Kartlardan soru → tool-use → DPO |
| 15 | `paper-mastery-agent` | learning/paper_mastery_agent | semi_auto | hayır | 100 puanlık ustalık |
| 16 | `status-manager` | learning/status_manager | manual | hayır | Skor → durum |
| 17 | `lora-control-plane` | lora/control_plane | semi_auto | hayır | Gate 0-8; Gate 7 BLOCKER |
| 18 | `adapter-eval` | training/adapter_eval | semi_auto | hayır | base vs adapter; terfi etmez |
| 19 | `dataset-quality-gate` | training/dataset_quality | semi_auto | hayır | GO/NO-GO |
| 20 | `tool-use-trainer` | training/tool_use_trainer | semi_auto | hayır | SFT verisi; model eğitmez |
| 21 | `arxiv-fetcher` | ingestion/arxiv_fetcher | autonomous | hayır | İdempotent indirme |
| 22 | `model-advisor` | runtime/model_advisor | autonomous | hayır | RAM/VRAM'a göre model önerisi (tek executor handler'ı) |
| 23 | `rlm-controller` | rlm/controller | manual | hayır | Kaynaklı denetimli cevap |
| 24 | `model-data-registry` | registry/ | **requires_approval** | hayır | Sürümleme + terfi kapısı |

Ayrıca zincirde düğüm olarak: `ingestion-quality-scorer` (ingestion/quality_scorer), `scientific-tool-runtime` (tools/), `hypothesis-evaluator` (evals/eval_runner) — manuel, salt-hesap.

**Kontrol kenarları (ajan haritası):** autodrive → rag-learning-loop; unattended-supervisor → {autodrive, self-heal, auto-lora, training-guardian}; training-guardian → auto-lora; sentinel → self-heal; self-heal → {training-orchestrator, rag-learning-loop}.

## 8.2 Motorlar
| Motor | Kurulu-mu | Av | Sür | Not |
|---|---|---|---|---|
| `claude` (varsayılan) | PATH | `--safe-mode --strict-mcp-config --disallowedTools …` | `--setting-sources "" --disable-slash-commands --strict-mcp-config --tools Read,Grep,Glob --mcp-config` | İnteraktif kotayı yer |
| `codex` | PATH | hardened | drive_hardened | 5 saatlik yuvarlanan kota |
| `local` | Ollama | süreç doğurmaz | — | Kota yok |

Kimlik toplanmaz/saklanmaz; giriş durumu bilinemez (`logged_in=null`). Doğrulanmamış motor `hardened=False` → RUN'a kapalı (fail-closed).

## 8.3 Geliştirme ajanları ve skill'ler (`.claude/`)
Kod tabanının parçası değil, geliştirme oturumunda kullanılan prosedürler. Yeni projede **yalnız kodda karşılığı olanlar** taşınır:

| Tanım | Rol |
|---|---|
| Ajan `dataset-auditor`, `safety-scanner`, `math-verifier`, `philosophy-reviewer`, `domain-verifier`, `curriculum-classifier` | Gate 0-7'nin insan-okur açıklamaları; kod `lora/gates.py` |
| Ajan `evaluation-reviewer`, `adapter-registry-manager` | base/adapter kıyası ve terfi; kod `training/adapter_eval`, `registry/adapter_registry` |
| Ajan `rlm-answer`, `security-reviewer`, `scientific-tool-runtime`, `hypothesis-evaluator`, `ingestion-quality-scorer`, `model-data-registry` | İlgili modüllerin kullanım rehberi |
| Ajan `literature-scout`, `lora-research` | Periyodik literatür/yöntem taraması (yalnız kaynak/yöntem besler) |
| Skill `/trading-research`, `/backtest-auditor`, `/codegen-review`, `/rlm-answer`, `/lora-control-plane`, `/data-generation`, `/paper-mastery`, `/model-data-registry`, `/scientific-tool-runtime`, `/hypothesis-evaluator`, `/ingestion-quality-scorer`, `/web-mcp`, `/tv-bridge` | Prosedür paketleri; her biri gerçek CLI komutlarına atıf yapar (var olmayan modül örneği yok) |

---

# 9. Güvenlik ve yetki modeli

## 9.1 Tehdit modeli
| Varlık | Tehdit | Savunma |
|---|---|---|
| Yerel makine | Ağa açılma | Varsayılan bind `127.0.0.1` |
| API | Yetkisiz erişim | Opsiyonel Bearer token, compare_digest |
| Upload | Sahte dosya / enjeksiyon | `%PDF-`, CSV başlık sniff, boyut, ayrı limit; kural çalıştırma yok |
| Dosya sistemi | Traversal | `sanitize_filename` + `safe_destination` |
| Tarayıcı | XSS / clickjacking | CSP `'self'`, DENY, nosniff, self-host font |
| Servis | DoS | IP kayan pencere + upload limiti |
| DB | SQLi | ORM |
| Strateji | Kod enjeksiyonu | regex-only |
| Sırlar | Sızıntı | `.env` ignore; gitleaks; `guards.secrets_pii` eğitim verisinde |
| Host | DNS rebinding | TrustedHost |
| Aktarım | Düz HTTP token | TLS + HSTS; SSE tek-kullanımlık bilet |

Ağa açarken: token (`openssl rand -hex 32`) + VPN/SSH tüneli (en iyi) veya reverse proxy + TLS; trusted hosts, rate limit, CORS; pip-audit, pre-commit, yedek. Kapsam dışı: canlı borsa, çoklu kullanıcı, parola saklama.

## 9.2 Kimlik ve scope
- **human** (UI/CLI, api_token) ve **driver** (doğurulan motor; `X-Ithaka-Driver-Token` + `X-Ithaka-Run-Id`; sha256-hash'li, run_id-bağlı, TTL'li, koşu bitince iptal).
- `require_human` (driver → 403): onay ver/reddet, kart onayla/reddet, feedback approve/reject/export, `training/run`, `auto-lora/train|promote`, `clear-stop-all`, `orchestration/autodrive`, `hunt_ack=true` içeren start/resume, `understanding-score/record`, `backtest/{id}/risk`. `stop-all` kasıtlı olarak human_only **değil**.
- Sürücünün açtığı görevler zorla `requires_approval=True`.
- Dürüstlük sınırı: `api_token` boşken bu katman kriptografik sınır değildir; localhost dışına açmadan token atanır.

## 9.3 Doğurulan motora karşı asıl sınır = araç kısıtı
HTTP auth motora karşı yetersizdir (CLI auth'suz, `.env` fallback, MCP, hook'lar). Sınıf-düzeyi çözüm: av modunda `--safe-mode` + `--strict-mcp-config` + `--disallowedTools` üçü birlikte; sür modunda `--safe-mode` yerine ayar/komut kanallarını tek tek kapatma + `--tools Read,Grep,Glob` + allow-list'li MCP. `--bare` yasak. `build_child_env` insan token'ını boşaltır, ayar-ezme env'lerini siler. `argv[0]` mutlak yol, `cwd=root`, `shell=False`. Üç spawn noktası: AutoDriver (tam kısıt + sürücü token), haftalık tarama scripti (tam kısıt), araştırma döngüsü scripti (yazma/push işlevsel → **kuşatılamaz**, yalnız güvenilir içerikle koşar; sahte kısıt eklenmez). Kalan bilinen sınır: av motoru `Read` ile `.env` okuyabilir → `.env` repo dışında tutulur (`ITHAKA_ENV_FILE` yolu) veya motor izole çalışma kopyasında koşar.

## 9.4 Onay ve kill-switch
- Standing yetki yok; her tehlikeli aksiyon (`train_run`, `promote`) `consumed_at` damgalı tek-kullanımlık onay tüketir (CAS). Kanonik anahtar `lora-trainer / train_run` (CLI, web, orkestrasyon aynı).
- Orkestrasyon `approval` aşaması onayı gözler, tüketmez.
- `STOP_ALL`: yalnız `dangerous` ajanları bloklar; bayrak yazımı + `terminate_all()`; `clear-stop-all` insan-yalnız.
- `verdict_audit`: §7.18.
- MCP allow-list drift testi FastAPI'nin gerçek dependency grafiğini tarar; yan-etki testi handler kaynak kodunu tarar.

---

# 10. Eğitim yaşam döngüsü

## 10.1 Yol
Tek yol: **yerel PEFT** (Qwen2.5-1.5B, `discipline_safe_local`, CPU ~35 s/adım) veya Apple Silicon MLX. 4B+ modelin CPU'da eğitimi desteklenmez (haftalar + overfit); bulut-GPU notebook üretimi çekirdekte yoktur (Ek B).

## 10.2 Veri hattı
```
knowledge_cards (approved, lora_eligible=1, kanonik)  ─┐
synthetic_qa.jsonl (grounded, dedup)                   ─┼─ assemble_sft → dedup → +%25 disiplin → data/sft/sft.jsonl (KANONİK)
discipline_dataset (432 adversarial)                   ─┘
sft.jsonl → Gate 0-8 → pretrain-gate → ensure_train_split (seed 42, %5) → train/valid
         → DatasetVersion pending → registry-promote-dataset (insan)
```
Clobber guard üç katmanlı: dataset_builder boş build'de yazmaz; `build_training_split` boş assembly'de dokunmaz; `ensure_train_split` kaynak boşsa korur.

## 10.3 Orkestrasyon + onay + eğitim
```
İNSAN ⚡ RUN (web) veya `ithaka orchestrate-start`
  preflight → collision → smoke → deep-hunt [BLOCKED]
  → İNSAN: Otonom AV (mode=hunt) → motor → ITHAKA_HUNT_VERDICT + EVIDENCE → verdict_audit → hunt_ack
  → data-gate → curriculum → dry-run → regression → approval [BLOCKED]
  → İNSAN: approval-approve <id> / UI "Onayla ve başlat" (iki tık)
  → train handoff → İNSAN: `train --run` / POST /api/training/run → authorize → onay TÜKETİLİR
  → launch(profile=discipline_safe_local) detached (pid kaydı) → training-guardian
  → adapter_eval (min_n≥5, degenerasyon/boş-cevap vetosu) + anlama merdiveni kıyası
  → accept → adapter_registry candidate → promote(user_approved=True) → PRODUCTION (tek)
```
Sür modu (⚡ RUN varsayılan) veri hattını MCP araçlarıyla ilerletir; eğitim adımında durur; `hunt_ack` yazmaz.

## 10.4 Kapıların gerekçesi (tek paragraf)
Bir önceki projede maskesiz, sabit-açılışlı sentetik veriyle 46 saatlik CPU eğitimi base'den daha kötü bir adapter üretmiş ve eval bunu yakalayamamıştı (adapter yüklenmiyor, n=1 ile accept). Bu yüzden: sabit açılış öneki yasak; 432 adversarial disiplin örneği; `assistant_only_loss` yerel varsayılan; eval adapter'ı gerçekten yükler; `MIN_EVAL_N=5`; boş/degenere cevap kategorik veto; pretrain-gate açılış-bigram bloğu; regresyon blocker; profil tüm başlatıcılarda ortak varsayılan; base kıyası adapter'ın kendi base'iyle.

---

# 11. Web arayüzü

`ithaka-web` → `http://127.0.0.1:8765`. Statik `index.html` + `app.js` + `app.css`; CSP nedeniyle inline script yok; fontlar self-host; renk-körü-güvenli palet (pass `#0a7d55`, fail `#cf4014`) + şekil ipuçları; WCAG AA.

**Üst şerit:** canlılık, bağlantı/embed/makale sayısı (30 s), RAG ustalık %, "obj. anlama" (tıkla → tam merdiven; kayıt ayrı POST), eğitim göstergesi (🔴 çalışıyor / ▶ hazır — başlat / yok), sürüm/sapma rozeti, disclaimer.

**Gruplar:** Keşfet & sor (01, 11) · Kütüphane (02) · Trader & backtest (03, 04) · Eğitim hattı (06, 05, 07, 12, 13) · İzleme & sağlık (09, 10, 14, 08, 15; "gelişmiş" toggle yalnız nav'ı gizler).

| # | Sekme | Öz |
|---|---|---|
| 01 | ARAŞTIRMA | RAG soru-cevap, top_k, adapter seçici, kaynaklar |
| 02 | MAKALELER | PDF sürükle-bırak (429'da retry), toplu kart/skor, çapraz sentez, arXiv, kayıtlı sorgular |
| 03 | TRADER BEYİN | Formül çıkarımı, agentic araştırma, LoRA sohbet, zincir veri seti, sentez makaleleri, geçmiş |
| 04 | BACKTEST | Sentetik/CSV, özel IR JSON, geçmiş, risk raporu (POST), Pine, paket |
| 05 | EĞİTİM | Veri seti, ayarlar, Başlat/DURDUR, canlı ilerleme (SSE bilet), adapter'lar, Auto-LoRA paneli |
| 06 | ONAY | Bilgi kartlarını onayla/reddet (insan-yalnız) |
| 07 | DEĞERLENDİRME | Eval seti + adapter → ihlal taraması |
| 08 | SİSTEM | Durum, donanım profili + model önerisi, kurallar, API token |
| 09 | ÖĞRENME | RAG öğrenme döngüsü paneli, grafikler |
| 10 | AGENTS | Supervisor + STOP_ALL, onaylar, ajanlar, koşular, görevler, olaylar |
| 11 | RLM | Koşu tablosu, adım/kanıt/doğrulama detayı |
| 12 | ORKESTRASYON | Yeni koşu, sürdür, **Otonom AV (hunt)**, recovery, aşama grafiği, timeline |
| 13 | GERİ BİLDİRİM | Echo |
| 14 | NÖBETÇİ | Yokla, probe tablosu, geçmiş |
| 15 | AJAN HARİTASI | Işıklı-yol grafiği, motor seçici, **⚡ RUN (drive)**, canlı şerit + ⛔ DURDUR, iki-tık eğitim kapısı |

**⚡ RUN akışı:** tekrar-giriş kilidi → motor seçilebilir mi → koşu yoksa `start` → atlanamaz onay modalı (odak Vazgeç'te, kutu her açılışta sıfır, `isTrusted`, Esc = iptal, kota uyarısı, "eğitim başlamaz" güvencesi) → `autodrive {execute, engine, mode:"drive"}` → canlı şerit `driver_running`'e bağlı. Testler her `execute:true` çağrısının insan onayına bağlı olduğunu sabitler.

Kod çekince sunucu yeniden başlatılır (route'lar başlangıçta yüklenir); tarayıcıda sert yenileme.

---

# 12. Operasyon

## 12.1 Kurulum
- **Windows:** `install.ps1` (tek satır; Git yoksa winget; hedef `%USERPROFILE%\ithaka`; mevcut checkout ff-only) → `setup.ps1` → `scripts\start-server.ps1 -Install` (önce `verify-install.ps1`).
- **macOS/Linux:** `setup.sh` → `verify-install.sh` → opsiyonel `install-autostart.sh` (systemd/launchd/cron).
- **Sihirbaz menüsü yalnız yerel modeller** (qwen3 4b/8b/14b/30b, llama3.1 8b/70b, mistral 7b, deepseek-r1 8b/14b); varsayılan `qwen3:4b`; RAM/disk kontrolü; `ollama pull <model>` + `nomic-embed-text`; `.env`'e `ITHAKA_LLM_BACKEND=ollama`, `ITHAKA_LLM_MODEL=<model>`; erişim modu (yerel / uzaktan: `0.0.0.0` + otomatik token); `ithaka init`.
- **verify-install:** init → status → gen-data → backtest → pytest; exit 0/1/2; autostart yalnız geçerse.

## 12.2 Güncelleme
`update.ps1` / `update.sh`: web'i durdur → `git fetch origin main` → dal main değilse `switch` (kirliyse ve `--force` yoksa HATA; main başka worktree'de ise HATA) → `pull --ff-only` (ıraksaksa merge yok; `--force` → `reset --hard origin/main`, yalnız izlenen kod) → drift raporu → `uv sync --extra dev` → web başlat → sağlık. Eğitime dokunmaz. Teşhis `ithaka doctor` (sapma → exit 2).

## 12.3 Zamanlanmış görevler
| Görev | Sıklık | Çalıştırır |
|---|---|---|
| Web autostart | Logon | web servisi (`UV_NO_SYNC=1`, thread=1, `ISOLATE_CHROMA=1`) |
| Update | Günlük 03:00 | `update.ps1` |
| TrainingWatchdog | 5 dk | `training-watchdog.ps1` (mutex, log-tazeliği 10 dk, profil açık) |
| WeeklyBugScan | Pzt 09:00 | `weekly-bug-scan.ps1` (Kademe-1, araç-kısıtlı) |
| LiteratureScout | Günlük 08:30 | `literature-scout.ps1` |
| RAG-Scan / RAG-Integrate | 24 s / 168 s | `rag-research-loop.ps1` |

## 12.4 Scriptler
`start-server.ps1` (-Install/-Restart/-Repair/-Uninstall/-Stop/-Status), `run-web-service.ps1`, `start-train.ps1` (detached; `-Profile discipline_safe_local`; `train_status.json` **pid dahil**), `train-loop.ps1` (assemble → split → `train --run --profile …`), `training-watchdog.ps1`, `assemble_sft.py`, `start-loop.ps1` / `continuous-learning.sh` (kart → skor → sentez → synth-qa; eğitim değil; eğitim sürerken ertelenir), `weekly-bug-scan.ps1`, `install-*-task.ps1`, `rag-research-loop.ps1`, `rag_ab_*.py` (A/B), `open-pr.sh/.ps1`, `setup-pr-automation.sh`, `sync-mcp.sh`, `check_protected_paths.py`, `verify-install.*`, `install-autostart.sh`.

## 12.5 CI ve PR
- `ci.yml`: push main + PR; `uv sync --extra dev --extra mcp` → `ruff check` → `ruff format --check` → `mypy ithaka` → `pytest -m "not ollama"` (özet satırı görünür). Auto-merge (squash + delete-branch); required context `lint · types · tests (offline)`.
- `weekly-audit.yml`: rapor-only; repo variable ile açılır; `understanding-score --record` + `pretrain-gate` → artifact; kod/eğitim/terfi/main'e dokunmaz.
- PR şablonu: Kademe-0 checklist (format/lint/typecheck/test; yeni indikatör → registry + test; backtest → shift(1) + maliyet + OOS).
- `check_protected_paths.py`: `data/ storage/ vector_db/ models/ .env*` değişikliğini PR'da engeller.

---

# 13. Test stratejisi

- **%100 çevrimdışı**; Ollama gerektiren testler `@pytest.mark.ollama` (varsayılan seçimde dışarıda).
- **`conftest.py` izolasyonu (session, autouse):** `ITHAKA_ROOT=tmp`, `ITHAKA_SQLITE_PATH`, `ITHAKA_CHROMA_PATH`, `ITHAKA_ALLOW_FAKE_EMBEDDINGS=true`, **`ITHAKA_BACKGROUND_LOOPS_ENABLED=false`**, `ITHAKA_UNATTENDED_TRAINING_ENABLED=false`; `get_settings.cache_clear()`. Sonuç: **hiçbir test gerçek `data/`, `storage/`, `models/` dizinine yazmaz; web lifespan hiçbir döngü/motor başlatmaz.** Bir test bunu ihlal ederse session-sonu fixture'ı gerçek ağaçta yeni dosya tespit edip **FAIL** verir.
- `_reset_web_rate_limiter` (function, autouse): `_hits.clear()`.
- Windows: `--basetemp=.pytest_tmp`.
- Kategoriler ve kilit sözleşmeler: dry-run kapısız, `--run` onay ister; taze onay tek kullanımlık + CAS; STOP_ALL her şeyi ezer; driver 403 listesi; allow-list kapalı küme + dependency-graph taraması + kaynak-kodu yan-etki taraması; her `execute:true` insan kapılı; motor kimlik alanı yok; verdict fail-closed + okuma-kanıtı; `paper_id` içerik hash + yarım ingest onarımı; BM25 `embedded=1`; cache reset kilit altında; determinizm tie-break'leri; `MIN_GRADED=3`; L5 test-edilemez → skipped; adapter `min_n=5` + vetolar; profil tüm başlatıcılarda; clobber guard; U+2028 kaçışı; Gate 7 tüm alanlar; RLM trading uyarısı her zaman; RLM paneli ve çekirdek nav grupları korunur; asset `?v=` hash (numara pinleme yok); heartbeat yazılıyor (`recover_stale` gerçek senaryoda tetiklenir); `evaluate` yetersiz veride `inconclusive`; `to_pine` çıktısı Python metrikleriyle ±%1 içinde (sentetik veri, desteklenen göstergeler).
- Yönetişim testleri: workflow YAML'ları ve güvenlik dokümanları statik testlerle zorunlu kılınır.

---

# 14. İnşa sırası

1. `config` + `guards` + test izolasyon fixture'ları (ilk gün testler ağaca yazamaz).
2. `trading` (IR, indikatörler, vektörize backtester, IS/OOS, evaluator, risk) — dış bağımlılıksız, ilk yeşil.
3. `memory` depolama (SqliteStore + migrate + CAS, ChromaStore, EmbeddingService fake).
4. `ingestion` → `paper_indexer` → BM25 corpus → retrieval orkestratörü → `rag_answerer`.
5. `verification` + `evals` (atıf/dayanak/güven, RAGAS-offline, safe_eval, L3/L4/L5, understanding_score).
6. `brain` (LocalLLM, KnowledgeCardBuilder, SyntheticQABuilder).
7. `rlm` (controller, evidence, claims, safe tools).
8. `research` + `learning`.
9. `training` veri hattı (discipline, sft_assembly, dataset_quality, Gate 0-8, curation, splitter, detached_launch, profiles).
10. Trainer + eval + registry + unattended_policy + auto_pipeline.
11. `runtime` (schemas, registry, tracker, supervisor, approvals, task_queue, executor, chain, preflight) + manifest.
12. `orchestration` (pipeline, store + heartbeat, orchestrator, delegates, smoke, collision, regression, engines, engine_procs, driver, verdict_audit, unattended_supervisor).
13. `monitoring`, `feedback`.
14. `web` (security, scope, tickets, server, route'lar, agent_graph, statik UI).
15. `mcp_server` (allowlist + proxy).
16. `cli`.
17. Operasyon scriptleri, CI, PR otomasyonu.
18. Dokümanlar (CLAUDE.md, README, SECURITY, protokoller, skill/agent tanımları).

Her adımda `make ci` yeşil; bir alt sistem tamamlanınca Kademe-2 adversarial av.

---

# Ek A — Sabitler

| Sabit | Değer | Yer |
|---|---|---|
| chunk_size / overlap | 1200 / 200 | settings |
| `MATH_WHOLE_MAX_CHARS` | 6000 | ingestion/chunker |
| `MIN_SOURCE_CHARS` (kart) | 1500 | brain/knowledge_card_builder |
| Embedding batch / fake dim | 64 / 256 | memory/embedding |
| Chroma `get_all` page | 5000 | memory/chroma_store |
| BM25 k1 / b | 1.5 / 0.75 | memory/bm25_index |
| RRF k | 60 | memory/rank_fusion |
| PPR damping / iters | 0.85 / 20 | memory/graph_retriever |
| Reranker ağırlıkları | 0.40 / 0.30 / 0.20 / 0.10 | memory/reranker |
| Router kısa sorgu | ≤6 kelime | memory/query_router |
| Abstain similarity / margin | 0.55 / 0.02 | settings |
| Mastery eşikleri | 90 / 75 / 60 / 40 | learning/mastery_scorer |
| Exam geçme | cit≥0.3, gnd≥0.4 | learning/rag_exam_runner |
| `MIN_GRADED_FOR_SCORE` | 3 | verification/exams |
| Confidence abstain / warn | 0.40 / 0.70 | verification/confidence_scorer |
| RLM evidence retry / answer / skip | 40 / 60 / 80 | settings |
| §16 aday | conf≥0.85, cit≥0.90, gnd≥0.90 | rlm/lora_candidate |
| CostSpec | 0.0005 + 0.0005 | trading/strategy_ir |
| OOS split / min_trades / max DD | 0.7 / 30 / −50% | trading |
| `MIN_EVAL_N` | 5 | training/adapter_eval |
| Split seed / valid ratio / launch lock TTL | 42 / 0.05 / 120 s | training/detached_launch |
| pretrain-gate açılış-bigram | >%40 → NO-GO | training/dataset_quality |
| Disiplin örnek / oran | 9×16×3=432 / 0.25 | training/discipline_dataset |
| Gate 6 yumuşak blok | ≥20 kart, >%25 | lora/gates |
| Splitter | 0.8/0.1/0.1, seed 42 | lora/dataset_splitter |
| Event retention / stale run | 30 gün · 50.000 / 6 sa | runtime/tracker |
| Pipeline aşama / max_steps / recover_stale | 12 / 50 / 30 dk | orchestration |
| Heartbeat aralığı | 30 s | orchestration/orchestrator |
| HUNT / DRIVE timeout | 1800 / 3600 s | orchestration/driver |
| Driver token TTL (hunt / drive) | 2100 / 3900 s | web/driver_scope, driver |
| `STOPPED_RC` / `STOP_POLL_S` | −99 / 1.0 | driver |
| verdict_audit | 5 dosya / 2 alt-sistem / quote ≥12 / 5 MB | orchestration/verdict_audit |
| Engine probe TTL / grace | 60 s / 5 s | engines, engine_procs |
| Regression toleransları | 0 / 0.05 / 0.02 / 0 / 0.02 / 0 | orchestration/regression |
| Backoff | 300 s → 21600 s | unattended_supervisor, self_heal |
| Self-heal eşiği | 3 ardışık tur | monitoring/self_heal |
| Sentinel keep_last | 1000 | monitoring/store |
| SSE ticket TTL | 60 s | web/sse_tickets |
| Rate limit / upload / max upload | 120 / 60 dk / 100 MB | settings |
| MCP allow-list / forbidden | 21 / 12 | mcp_server/allowlist |
| Mastery queue attempts / RAG loop rebuild | 3 / 3 | memory/mastery_store, research/rag_learning_loop |

# Ek B — Achilles'ten bilinçli çıkarılanlar
| Parça | Neden |
|---|---|
| OpenAI / Anthropic / Google LLM istemcileri, ilgili ayarlar ve bağımlılıklar, kurulum menüsündeki 9 bulut seçeneği | "Bulut API asla" kuralıyla çelişiyordu; hiç kullanılmıyordu |
| alexzhang13/rlm opsiyonel motor adapter'ı, güvenlik kapısı, docker preflight, trajektori logları, `rlm` extra | Varsayılan kapalıydı; gerçek kullanım API anahtarı istiyordu |
| Bulut-GPU (Kaggle/Colab) notebook üretici, Modelfile, `lora-cloud-prep`, ilgili protokol/skill | Yerel küçük-model eğitimine pivot edildi; tek yol PEFT/MLX |
| `query_expander`, `multi_query_retriever`, `hybrid_retriever`, `regression_runner`, `answer_eval`, `golden_generator` | Hiçbir üretim yolu çağırmıyordu |
| Cross-encoder reranker (bge) | CPU'da kullanılamaz yavaş; FlashRank yeterli |
| Yerel eğitim 5A-5E "salt-rapor" faz modülleri | Orkestrasyon aşamaları aynı işi yapıyor |
| `benchmark`, `installer` (whitelist ollama komutları dışında), `rules_updater` + ayrı öğrenme DB'si | OSS-agent MVP kalıntıları; değer/karmaşıklık oranı düşük |
| İkinci `AdapterRegistry` (sidecar JSON), `promotion_gates` içindeki kopya sır/PII regex'i | Tek registry, tek guard modülü |
| `strategies/` boş dizinleri, boş `app/cli` paketi, kullanılmayan `TrainingProgressResponse` şeması | Ölü |
| HANDOFF tarihçesi, PR numaraları, oturum notları, makineye özgü yollar | Tarihçe `docs/arsiv/`'e; spesifikasyon güncel tasarımı anlatır |

# Ek C — Akademik referanslar
Cormack/Clarke/Büttcher 2009 (RRF) · Bruch et al. arXiv:2210.11934 (konveks füzyon) · SPRIG arXiv:2602.23372 (CPU GraphRAG) · Lost in the Middle arXiv:2307.03172 · RAGAS arXiv:2309.15217 · CRAG arXiv:2401.15884 · RAFT arXiv:2403.10131 · R-Tuning arXiv:2311.09677 · Anthropic Contextual Retrieval · Bandt-Pompe permütasyon entropisi · Forbidden patterns arXiv:0711.0729 · MPR complexity arXiv:1808.01926 · KL-regularized LoRA arXiv:2512.22337 · Deflated Sharpe (Bailey & López de Prado).
