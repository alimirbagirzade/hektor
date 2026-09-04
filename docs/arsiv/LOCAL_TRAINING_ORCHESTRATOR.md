# Local Training-Agent Orchestrator (Phase 5A)

> **This command is report-only. It never starts training.**

## Purpose
Achilles'in eğitim hazırlık durumunu **lokal, salt-okuma** olarak denetler ve bir rapor
üretir. Amaç: gerçek eğitime geçmeden önce "hazır mıyız?" sorusunu objektif sinyallerle
(veri, kalite kapısı, STOP_ALL, onaylar, AutoLoRA durumu) yanıtlamak — **hiçbir tehlikeli
aksiyon başlatmadan**.

## What it does
- STOP_ALL kill-switch durumunu okur (`supervisor.is_stop_all_active`).
- Detached eğitim çalışıyor mu bakar (`detached_launch.is_detached_training_running`).
- Bekleyen onayları **listeler** + taze `train_run` onayı var mı diye bakar
  (`approvals.list_approvals` / `has_fresh_approval`) — **tüketmeden**.
- AutoLoRA durumunu okur (`AutoLoRAPipeline.get_status`).
- Veri hazırlığını sayar (lora_sft.jsonl satır, train/valid, onaylı kart).
- Ön-eğitim kalite kapısını **salt-okuma** çağırır (`dataset_quality.audit_dataset` → GO/NO-GO).
- Adapter eval için hazır koşulları **listeler** (adapter mevcut mu) — eval başlatmaz.
- Eğitim-hazırlık skoru (0–100) + karar (`READY`/`NOT_READY`/`BLOCKED`) + riskler üretir.
- Raporu `reports/local_training_orchestrator/` altına **markdown + json** yazar.

## What it never does
- `detached_launch.launch()` **çağırmaz**.
- `achilles train --run` / training subprocess **başlatmaz**.
- `AutoLoRAPipeline.start_training()` **çağırmaz**.
- `promote_to_production()` **çağırmaz** (adapter terfisi yok).
- `require_fresh_approval()` **çağırmaz** → onay **tüketmez**, yeni pending onay açmaz.
- Cloud / Kaggle / Colab tetiklemez.
- Korumalı yollara (`data/`, `storage/`, `vector_db/`, `models/`, `.env`) **yazmaz**.
- Ağ çağrısı yapmaz; LLM gerektirmez (çevrimdışı çalışır).

## CLI usage
```bash
uv run achilles local-training-audit                 # rapor üret + reports/ altına yaz
uv run achilles local-training-audit --json          # makine-okunabilir JSON
uv run achilles local-training-audit --out reports/local_training_orchestrator
uv run achilles local-training-audit --no-write      # yazmadan yalnız ekrana
```
Komut her durumda şunu basar:
```text
No training was started. This is a local report-only audit.
```
`--run` benzeri bir mod **bu fazda desteklenmez**; `--no-dry-run` verilse bile salt-rapor kalır.

## Safety model
Orkestratör yalnızca **read-only accessor**'ları çağırır (hepsi pure SELECT / dosya-okuma /
filesystem-existence). Her sonda (probe) `try/except` ile savunmacıdır: bir bağımlılık
yoksa/başarısızsa "unavailable" döner, audit'i çökertmez. Tehlikeli fonksiyonlara modül
kaynağında **referans bile yoktur** (statik test ile doğrulanır).

## Approval model
Onay **tek kullanımlıktır** (CLAUDE.md Kural 8). Bu denetim onayı yalnız **gözlemler**:
- `has_fresh_approval("lora-trainer","train_run")` → taze onay VAR MI (tüketmeden).
- `list_approvals(status="pending")` → bekleyen onaylar (raporlanır, tüketilmez).
Gerçek eğitim için ayrı **taze manuel onay** + `achilles train --run` (ya da onaylı web
`/api/training/run`, Phase 4D-1) gerekir. Orkestratör bu kapıyı **açmaz**.

## STOP_ALL behavior
`storage/STOP_ALL` aktifse rapor bunu **kırmızı risk** olarak işaretler ve kararı
`BLOCKED` yapar. (STOP_ALL ayrıca tüm gerçek tehlikeli aksiyonları zaten bloklar.)

## Output reports
- `reports/local_training_orchestrator/<YYYYMMDD_HHMMSS>_report.md`
- `reports/local_training_orchestrator/<YYYYMMDD_HHMMSS>_report.json`

`reports/local_training_orchestrator/` `.gitignore`'dadır (yalnız `.gitkeep` izlenir) →
üretilen raporlar repoya commit edilmez. JSON şeması: `generated_at`, `banner`, `probes`,
`risks`, `readiness_score`, `readiness_verdict`, `notes`.

## How this replaces cloud nightly automation
GitHub cloud nightly audit (`nightly-automation-audit.yml`) **inert** (local-first karar).
Onun yerine bu komut **lokal, on-demand** rapor-only denetim sağlar: kullanıcı dilediğinde
`uv run achilles local-training-audit` çalıştırır, çıktıyı inceler. Otomatik tetik yok,
secret yok, cloud yok. İstenirse Windows Task Scheduler ile periyodik **rapor-only** koşu
ayarlanabilir (yine eğitim başlatmaz).

## Phase 5B — request flow (UYGULANDI)
- **Phase 5B request flow:** [LOCAL_TRAINING_REQUEST_FLOW.md](LOCAL_TRAINING_REQUEST_FLOW.md)
  — `local-training-request` bu audit sonucundan **onay-kapılı bir eğitim İSTEĞİ** üretir
  (`--create-approval` ile PENDING onay oluşturur; onayı **TÜKETMEZ**, eğitim **BAŞLATMAZ**).
- Yaşam döngüsü: **audit → request → human approval → (sonra) gerçek eğitim** (insan elinde).
- **Phase 5C (UYGULANDI):** [LOCAL_TRAINING_DRYRUN_PIPELINE.md](LOCAL_TRAINING_DRYRUN_PIPELINE.md)
  — `local-training-dry-run` onaylı isteği READ-ONLY okur + adapter-eval **mocked** simüle
  eder; eğitim/onay-tüketimi YOK.
- **Phase 5D (UYGULANDI):** [LOCAL_TRAINING_HANDOFF.md](LOCAL_TRAINING_HANDOFF.md) —
  `local-training-handoff` son checklist + gerçek eğitim komutunu YALNIZ METİN verir;
  ÇALIŞTIRMAZ. Gerçek eğitim yalnız İNSAN tarafından + açık taze onayla.
- **Phase 5E (UYGULANDI):** [LOCAL_TRAINING_POSTCHECK.md](LOCAL_TRAINING_POSTCHECK.md) —
  `local-training-postcheck` eğitim-sonrası artefakt/eval/score'u READ-ONLY denetler;
  terfi ÖNERMEZ (`human_review_required`). Otomatik terfi/eğitim/model-yazma YOK.
- Web UI'de "eğitim hazırlık" kartı (bu raporu Agents/Otomasyon sekmesinde göster).
- Readiness skoru eşiği geçtiğinde kullanıcıya **bildirim** (otomatik başlatma YOK).
