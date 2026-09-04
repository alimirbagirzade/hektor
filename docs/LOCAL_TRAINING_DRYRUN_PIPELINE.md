# Local Training Dry-Run Pipeline (Phase 5C)

> **This command can inspect an approval and simulate a training pipeline, but it never consumes approval and never starts training.**

## Purpose
5B'de oluşturulan onay-kapılı eğitim isteğini bir adım ileri taşır: onaylı (veya pending)
bir isteği **READ-ONLY** okur, approval durumunu **tüketmeden** kontrol eder ve gerçek
eğitim başlatmadan bir **dry-run pipeline simülasyonu** + **execution plan** üretir.

## Difference between audit, request, and dry-run
| | audit (5A) | request (5B) | dry-run (5C) |
|---|---|---|---|
| Ne yapar | durumu okur, readiness | uygunsa onay isteği OLUŞTURUR | onayı READ-ONLY okur + pipeline simüle eder |
| Onay | listeler | **pending oluşturur** | **yalnız okur** (tüketmez/oluşturmaz) |
| Adapter eval | varlık listeler | — | **mocked_ready** (gerçek model yok) |
| Çıktı | readiness raporu | istek raporu | **execution plan** + dry-run raporu |
| Eğitim | yok | yok | yok |

## CLI usage
```bash
uv run achilles local-training-dry-run                          # audit + son istek + plan
uv run achilles local-training-dry-run --approval-id apr_xxx    # onayı READ-ONLY kontrol et
uv run achilles local-training-dry-run --request-json reports/local_training_orchestrator/..._request.json
uv run achilles local-training-dry-run --mock-adapter-eval      # adapter-eval mock (varsayılan)
uv run achilles local-training-dry-run --json
uv run achilles local-training-dry-run --out reports/local_training_orchestrator
```

## Approval ID behavior
`--approval-id` verilirse `approvals.get_approval` ile durumu **okunur** (pending /
approved_not_consumed / approved_consumed / rejected / not_found). `approved_not_consumed`
ise pipeline `dry_run_passed` olur; aksi halde `needs_approval`.

## Approval is never consumed
Bu faz yalnız `get_approval` / `list_approvals` / `has_fresh_approval` (read-only) kullanır.
`require_fresh_approval` (tüketir) ve `request_approval` (oluşturur) **çağrılmaz**. Onayı
insan, ayrı bir adımda verir/tüketir.

## STOP_ALL behavior
STOP_ALL aktifse (veya audit kararı `BLOCKED`) pipeline `blocked` döner, plan üretilmez.

## Pretrain gate behavior
Audit'in `pretrain-gate` (LLM-free, read-only) verdict'i rapora girer. `NO-GO` ise
pipeline `not_ready` olur; `GO` + readiness `READY` + onaylı → `dry_run_passed`.

## Mock adapter eval behavior
Adapter-eval **gerçek model çalıştırmadan** `mocked_ready` döner. Bu fazda gerçek eval
**desteklenmez** (`--no-mock-adapter-eval` → `real_eval_unsupported`).

## Execution plan (yalnız PLAN — uygulanmaz)
```text
1. validate dataset
2. run pretrain gate
3. prepare LoRA config
4. wait for explicit real-training command
```

## What it never does
`launch()` · `achilles train --run` / subprocess · `AutoLoRAPipeline.start_training()` ·
`promote_to_production()` · `require_fresh_approval()` (tüketim) · `request_approval()`
(oluşturma) · gerçek adapter-eval / model write / adapter write · cloud/Kaggle/Colab ·
canlı trading. Korumalı yollara yazmaz.

## Output reports
- `reports/local_training_orchestrator/<YYYYMMDD_HHMMSS>_dryrun.{md,json}`
- Aynı gitignore'lu dizin (yalnız `.gitkeep` izlenir) → raporlar commit edilmez.
- JSON: `status`, `approval_id`, `approval_status`, `readiness_score/verdict`,
  `pretrain_gate`, `adapter_eval`, `execution_plan`, `risks`, `note`.

## Phase 5D (UYGULANDI) — human-gated handoff
- **5D handoff:** [LOCAL_TRAINING_HANDOFF.md](LOCAL_TRAINING_HANDOFF.md) —
  `local-training-handoff` bu dry-run sonucunu okur ve uygunsa gerçek eğitim komutunu
  YALNIZ METİN olarak + son insan checklist'i verir. Komutu ÇALIŞTIRMAZ, onayı TÜKETMEZ.
- Gerçek eğitim yalnız İNSAN tarafından elle: `train --run` (taze onayı CLI o adımda
  tüketir) ya da onaylı web `/api/training/run`.
- Yaşam döngüsü tracker'ı: audit → request → approval → **dry-run** → (insan) gerçek eğitim.
- **Phase 5E (UYGULANDI):** [LOCAL_TRAINING_POSTCHECK.md](LOCAL_TRAINING_POSTCHECK.md) —
  eğitimden SONRA `local-training-postcheck` artefakt/eval/score'u READ-ONLY denetler; terfi YOK.
- Web UI'de dry-run sonucu kartı (read-only).
