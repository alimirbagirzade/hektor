# Local Training Postcheck (Phase 5E)

> **This command can inspect training artifacts, but it never promotes an adapter and never starts training.**

## Purpose
İnsan gerçek eğitimi (`train --run`) elle çalıştırdıktan **sonra**, sonucu **SALT-OKUMA**
denetler ve bir postcheck raporu üretir. Terfi (promotion) **önermez** — yalnız
`human_review_required` der. Eğitim/terfi/model-yazma **yapmaz**.

## Difference between audit, request, dry-run, handoff, and postcheck
| | audit (5A) | request (5B) | dry-run (5C) | handoff (5D) | postcheck (5E) |
|---|---|---|---|---|---|
| Ne zaman | önce | önce | önce | eğitimden ÖNCE | eğitimden **SONRA** |
| Ne yapar | readiness | onay isteği | pipeline simüle | komut metni + checklist | **sonucu read-only denetler** |
| Çıktı | readiness | istek | plan | recommended_command | **artifact/eval/score özeti** |
| Eğitim/terfi | yok | yok | yok | yok | **yok** |

## CLI usage
```bash
uv run achilles local-training-postcheck                                    # son artefaktları ara
uv run achilles local-training-postcheck --handoff-json reports/local_training_orchestrator/..._handoff.json
uv run achilles local-training-postcheck --dryrun-json  reports/local_training_orchestrator/..._dryrun.json
uv run achilles local-training-postcheck --training-report reports/training/...json
uv run achilles local-training-postcheck --adapter-path models/adapters/adapter_...   # yalnız metadata/stat
uv run achilles local-training-postcheck --json
uv run achilles local-training-postcheck --out reports/local_training_orchestrator
```
Sonuç yoksa `no_training_run_found`; varsa `postcheck_ready_for_human_review`.

## Training artifact behavior
`--training-report` verilirse o JSON **read-only** okunur; verilmezse
`storage/train_status.json` (varsa) okunur. Eğitim **başlatılmaz**.

## Adapter eval behavior
`reports/evals/adapter_eval_*.json` raporları **read-only** taranır — gerçek adapter-eval
(model yükleme/çalıştırma) **YAPILMAZ**. Yalnız varsa önceki rapor okunur.

## Understanding score behavior
`reports/evals/understanding/*.json` kayıtları **read-only** okunur (varsa). LLM çağrısı
**yok**; offline çalışır.

## Promotion safety
`promotion_recommendation` **her zaman `human_review_required`**'dır. Otomatik terfi
**YOK**; `promote_to_production()` **çağrılmaz**. Terfi yalnız insan + ayrı taze onayla.

## Human review checklist
```text
[ ] Adapter-eval base'i geçti mi (regresyon yok)?
[ ] Understanding-score düştü mü (disiplin/dürüstlük)?
[ ] Pretrain-gate GO idi (zehir/ezber yok)?
[ ] Maliyet-dahil backtest/OOS kabul edilebilir mi?
[ ] Terfi kararı: yalnız insan + ayrı taze onay.
```

## What it never does
`launch` · `train --run`/subprocess · `subprocess.Popen` · `os.system` ·
`AutoLoRAPipeline.start_training` · `promote_to_production` · `require_fresh_approval` ·
`request_approval` · approval consumption · model load / real eval · model/adapter write ·
cloud/Kaggle/Colab · live trading. Korumalı yollara **yazmaz** (yalnız okur/stat).

## Output reports
- `reports/local_training_orchestrator/<YYYYMMDD_HHMMSS>_postcheck.{md,json}`
- Aynı gitignore'lu dizin (yalnız `.gitkeep` izlenir) → raporlar commit edilmez.
- JSON: `status`, `handoff_status`, `training_artifacts_found`, `adapter_eval_found`,
  `understanding_score_found`, `promotion_recommendation`, `review_checklist`, `note`.

## Phase 5F path
- 5F (öneri): tüm zincirin (audit→request→approval→dry-run→handoff→insan eğitim→postcheck)
  **agent-runtime tracker** ile uçtan uca izlenmesi + Web UI'de read-only yaşam-döngüsü kartı.
- İsteğe bağlı: postcheck sonucundan **insan-onaylı terfi isteği** (yine `request_approval`
  ile pending; terfiyi insan ayrı adımda + taze onayla yapar — otomatik terfi YOK).
