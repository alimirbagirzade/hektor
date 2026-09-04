# Local Training Request Flow (Phase 5B)

> **This command can create an approval request, but it never consumes approval and never starts training.**

## Purpose
5A audit'in (eğitim-hazırlık denetimi) sonucunu kullanarak **onay-kapılı bir eğitim
İSTEĞİ** akışı sağlar. Uygunsa (readiness `READY`) tek-kullanımlık bir **PENDING onay
isteği oluşturur** ve kullanıcıya onay komutunu gösterir. Onay verilse bile bu faz onayı
**tüketmez** ve gerçek eğitimi **başlatmaz** — akış dry-run/mocked kalır.

## Difference between audit and request
| | `local-training-audit` (5A) | `local-training-request` (5B) |
|---|---|---|
| Ne yapar | Durumu OKUR, readiness raporu üretir | Audit'i çalıştırır + uygunsa **onay isteği OLUŞTURUR** |
| Yazar mı | reports/ (rapor) | reports/ (istek raporu) **+** opsiyonel PENDING approval (DB) |
| Onay | sadece listeler | **pending oluşturabilir** (tüketmez) |
| Eğitim | başlatmaz | başlatmaz |

## CLI usage
```bash
uv run achilles local-training-request                 # default: audit + ÖN İZLEME (onay yok)
uv run achilles local-training-request --create-approval  # READY ise PENDING onay isteği oluştur
uv run achilles local-training-request --preview          # her zaman yalnız ön izleme
uv run achilles local-training-request --json             # makine-okunabilir JSON
uv run achilles local-training-request --out reports/local_training_orchestrator
```

## Preview mode (güvenli varsayılan)
`--create-approval` verilmedikçe (veya `--preview` ile) komut yalnız **ön izleme** döner:
audit çalışır, readiness raporlanır, **onay oluşturulmaz**, eğitim başlamaz.
```json
{ "status": "preview", "readiness_verdict": "...", "note": "No approval request was created. No training was started." }
```

## Create-approval mode
`--create-approval` + readiness `READY` → tek-kullanımlık **PENDING** onay isteği oluşturulur:
```json
{
  "status": "approval_required",
  "approval_id": "apr_…",
  "readiness_score": 82,
  "readiness_verdict": "READY",
  "risks": [],
  "approve_command": "uv run achilles approval-approve apr_…",
  "note": "No training was started."
}
```
STOP_ALL aktif / risk / readiness `READY` değilse onay oluşturulmaz:
```json
{ "status": "blocked", "reason": "...", "note": "No training was started." }
```

## Approval model
Aynı anahtar kullanılır: `agent_id="lora-trainer"`, `action="train_run"`, `risk="critical"`
— böylece oluşturulan onay, CLI/web'deki gerçek eğitim kapısıyla **değiştirilebilir**.
İstek akışı yalnız `request_approval` (PENDING oluştur) çağırır; `require_fresh_approval`
(**tüketir**) **asla** çağrılmaz. Onayı insan, ayrı bir adımda verir/tüketir.

## STOP_ALL behavior
STOP_ALL aktifse audit kararı `BLOCKED` olur → istek akışı `blocked` döner ve **onay
oluşturmaz**. (STOP_ALL ayrıca tüm gerçek tehlikeli aksiyonları zaten bloklar.)

## What it never does
- `detached_launch.launch()` çağırmaz · `achilles train --run`/subprocess başlatmaz.
- `AutoLoRAPipeline.start_training()` / `promote_to_production()` çağırmaz.
- `require_fresh_approval()` çağırmaz → **onay tüketmez**.
- Cloud/Kaggle/Colab tetiklemez · model/adapter yazmaz · korumalı yollara yazmaz.

## Output reports
- `reports/local_training_orchestrator/<YYYYMMDD_HHMMSS>_request.{md,json}`
- Dizin `.gitignore`'dadır (yalnız `.gitkeep` izlenir) → istek raporları commit edilmez.

## How to move from request to real training later
1. `local-training-request --create-approval` → PENDING `approval_id`.
2. **İnsan** inceler → `uv run achilles approval-approve <approval_id>` (onayı verir).
3. **İnsan** gerçek eğitimi açıkça başlatır: `uv run achilles train --run` (taze onayı o
   adımda CLI **tüketir**) veya onaylı web `/api/training/run` (Phase 4D-1).
   Bu 5B akışı 2. ve 3. adımları **yapmaz**; yalnız 1. adımı (istek/öneri) üretir.

## Phase 5C (UYGULANDI) — dry-run pipeline
- **5C dry-run pipeline:** [LOCAL_TRAINING_DRYRUN_PIPELINE.md](LOCAL_TRAINING_DRYRUN_PIPELINE.md)
  — `local-training-dry-run` onaylı/pending isteği **READ-ONLY** okur, onayı **tüketmeden**
  kontrol eder, pretrain-gate read-only + adapter-eval **mocked** çalıştırır ve bir
  execution PLANI üretir (uygulamaz). Eğitim/onay-tüketimi YOK.
- Web UI'de "eğitim isteği" kartı (approval_required durumunu Agents sekmesinde göster).
- İstek→onay→(insan)→gerçek eğitim yaşam döngüsünün uçtan uca izlenmesi (tracker).
- Zincir devam etti: 5C dry-run + 5D handoff ([LOCAL_TRAINING_HANDOFF.md](LOCAL_TRAINING_HANDOFF.md))
  + 5E postcheck ([LOCAL_TRAINING_POSTCHECK.md](LOCAL_TRAINING_POSTCHECK.md)) — handoff komutu
  YALNIZ METİN verir; postcheck eğitim-sonrası sonucu READ-ONLY denetler. Çalıştırma/terfi/onay-tüketimi yok.
