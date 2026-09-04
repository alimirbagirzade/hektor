# Local Training Handoff (Phase 5D)

> **This command may print a real training command, but it never executes it.**

## Purpose
5C dry-run sonucu `dry_run_passed` ise, gerçek eğitime geçmeden önce **son insan kontrol
checklist'i** ve **gerçek eğitim komutunu (yalnız metin)** üretir. Komutu **çalıştırmaz**,
onayı **tüketmez**. Bu faz yalnız handoff / checklist / komut-yazdırma fazıdır.

## Difference between audit, request, dry-run, and handoff
| | audit (5A) | request (5B) | dry-run (5C) | handoff (5D) |
|---|---|---|---|---|
| Ne yapar | readiness okur | onay isteği OLUŞTURUR | pipeline simüle eder | **son checklist + komut metni** |
| Onay | listeler | pending oluşturur | read-only okur | read-only okur |
| Çıktı | readiness | istek | execution plan | **recommended_command + checklist** |
| Eğitim | yok | yok | yok | **yok (yalnız metin)** |

## CLI usage
```bash
uv run achilles local-training-handoff                       # son dry-run + checklist
uv run achilles local-training-handoff --approval-id apr_xxx # onayı READ-ONLY kontrol et
uv run achilles local-training-handoff --dryrun-json reports/local_training_orchestrator/..._dryrun.json
uv run achilles local-training-handoff --json
uv run achilles local-training-handoff --out reports/local_training_orchestrator
```
`ready_for_human_execution` durumunda ekrana basılır:
```text
READY FOR HUMAN EXECUTION
Recommended command:
uv run achilles train --run
This command was NOT executed.
```

## Approval ID behavior
`--approval-id` (veya dry-run raporundaki approval_id) `approvals.get_approval` ile
**READ-ONLY** okunur. `approved_not_consumed` → handoff hazır; `approved_consumed` →
blocked; pending/none/rejected → `needs_approval`.

## Approval is never consumed
Yalnız `get_approval` (read) kullanılır. `require_fresh_approval` (tüketir) ve
`request_approval` (oluşturur) **çağrılmaz**. Onayı CLI/web'de gerçek eğitim adımında
**insan** tüketir.

## STOP_ALL behavior
STOP_ALL aktifse (canlı read-only kontrol) handoff `blocked` döner, komut önerilmez —
dry-run daha önce geçmiş olsa bile.

## Recommended command behavior
`recommended_command = "uv run achilles train --run"` **yalnızca string** olarak döner;
bu araç onu **çalıştırmaz**. Web alternatifi yalnız dokümante edilir (onaylı
`/api/training/run`, Phase 4D-1) — otomatik değil.

## Human checklist
```text
[ ] STOP_ALL is not active
[ ] Dry-run passed
[ ] Approval is approved and not consumed
[ ] You understand this will start real training
[ ] You have enough disk/RAM/GPU
[ ] You have reviewed dataset/pretrain-gate result
```

## What it never does
`launch` · `achilles train --run` (subprocess) · `subprocess.Popen` · `os.system` ·
`AutoLoRAPipeline.start_training` · `promote_to_production` · `require_fresh_approval` ·
`request_approval` · approval consumption · cloud/Kaggle/Colab · model/adapter write ·
live trading. Korumalı yollara yazmaz.

## Output reports
- `reports/local_training_orchestrator/<YYYYMMDD_HHMMSS>_handoff.{md,json}`
- Aynı gitignore'lu dizin (yalnız `.gitkeep` izlenir) → raporlar commit edilmez.
- JSON: `status`, `approval_id`, `approval_status`, `dryrun_status`, `recommended_command`,
  `checklist`, `note`.

## How real training is manually started later
1. Handoff `ready_for_human_execution` verir + komutu **yazar**.
2. **İnsan** checklist'i gözden geçirir.
3. **İnsan** elle çalıştırır: `uv run achilles train --run` — CLI gate taze onayı **o
   adımda tüketir** ve gerçek eğitimi başlatır. (Ya da onaylı web `/api/training/run`.)
   Bu 5D handoff'u 3. adımı **yapmaz**; yalnız komutu önerir.

## Phase 5E (UYGULANDI) — post-training postcheck
- **5E postcheck:** [LOCAL_TRAINING_POSTCHECK.md](LOCAL_TRAINING_POSTCHECK.md) —
  `local-training-postcheck` eğitim İNSAN tarafından çalıştırıldıktan SONRA adapter-eval /
  understanding-score / training artefaktını **read-only** raporlar; terfi ÖNERMEZ
  (`human_review_required`). Otomatik terfi YOK; promote insan + ayrı taze onay.
- Tüm zincirin (audit→request→approval→dry-run→handoff→insan eğitim→eval) tracker'da izi.
- Web UI'de "ready for human training" kartı (read-only).
