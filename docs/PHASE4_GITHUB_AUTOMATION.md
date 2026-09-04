# Phase 4 — GitHub + Claude Code PR Automation (Tasarım)

> **Phase 4A durumu:** Yalnız **tasarım + ŞABLON dosyalar + guard + test**. Hiçbir
> workflow GERÇEK çalıştırılmadı/tetiklenmedi. Aktivasyon (secret ekleme + workflow
> enable) **kullanıcı onayındadır**. `data/storage/vector_db/models` elle değişmedi;
> push yok; main'e dokunulmadı.

## 1. Purpose
Achilles'in kendi kendini **güvenli** biçimde GitHub üzerinden geliştirmesi:
issue'dan görev al → güvenli iş-branch'i aç → Claude Code ile **sınırlı**
kod/doküman/test değişikliği yap → offline CI çalıştır → **PR aç** → insan incele →
**yalnız elle merge**. main'e otomatik merge ASLA.

## 2. Strict Safety Model
**Claude Code YAPABİLİR:** `app/`, `tests/`, `docs/` içinde kod/doküman/test
değişikliği; offline testleri çalıştırmak; bug-scan raporu hazırlamak; PR açmak;
issue'ya sonuç yazmak.

**Claude Code YAPAMAZ:** main'e push · auto-merge · `train --run` · LoRA training ·
adapter promotion · cloud training · Kaggle/Colab run · secret yazmak · `.env`
değiştirmek · `data/`, `storage/`, `vector_db/`, `models/`, `models/adapters/`
değiştirmek · canlı trading entegrasyonu açmak · `ACHILLES_API_TOKEN` /
`ANTHROPIC_API_KEY` / `HF_TOKEN` loglamak.

Bu kurallar workflow içinde `env.CLAUDE_HARD_RULES` olarak prompt'a eklenir +
**iki kez protected-path guard** (Claude'dan önce ve sonra) ile mekanik zorlanır.

## 3. Branch / PR Flow
```text
GitHub issue + label claude-task
        ↓
workflow starts (dispatch veya 'claude-task' label)
        ↓
checkout repo (fetch-depth: 0)
        ↓
protected path guard (PRE)
        ↓
Claude Code works on iş-branch (app/tests/docs)
        ↓
ruff / format / mypy / pytest (offline)
        ↓
protected path guard (POST)
        ↓
PR opened (iş-branch → main)   ← MERGE YOK
        ↓
human review
        ↓
manual merge only
```

## 4. Issue Labels
| Label | Ne zaman | Claude çalışır mı | İnsan onayı |
|-------|----------|-------------------|-------------|
| `claude-task` | Sınırlı, güvenli kod/doküman/test görevi | ✅ (workflow tetikler) | PR merge'de |
| `automation` | Otomasyon/altyapı görevi (genel) | ✅ etiket akışına göre | PR merge'de |
| `bug-scan` | Bug-scan bulgusu izleme | rapor-only | — |
| `needs-approval` | pyproject/dep veya hassas değişiklik | ⛔ insan kararı şart | **evet** (çalışmadan) |
| `safe-refactor` | Düşük riskli refactor | ✅ | PR merge'de |
| `dangerous-change` | Eğitim/veri/model/güvenlik dokunuşu | ⛔ **yalnız insan** | **evet** |
| `no-claude` | Claude bu issue'ya dokunmasın | ⛔ | — |
| `human-only` | Yalnız insan yapacak | ⛔ | — |

Workflow YALNIZ `claude-task` etiketiyle (veya elle dispatch) tetiklenir; diğerleri
otomatik koşmaz. `dangerous-change` / `no-claude` / `human-only` mutlak engeldir.

## 5. Protected Paths (guard FAIL eder)
```text
data/**
storage/**
vector_db/**
models/**          (models/adapters/** dahil)
.env, .env.*
*.key, *.pem, *.p12
*.sqlite, *.db
```
`scripts/check_protected_paths.py` bu yolları değiştiren bir diff görürse **exit 2**
→ workflow fail, PR açılmaz, issue'ya güvenlik ihlali yazılır.

## 6. Allowed Paths (varsayılan)
```text
app/**, tests/**, docs/**, .github/workflows/**, scripts/**, mcp_server/**
automation_manifest.yaml, pyproject.toml, README.md, CLAUDE.md, HANDOFF.md
```
**`pyproject.toml`** dependency değişikliği yaparsa ayrıca **`needs-approval`** istenir
(lock/deps insan onayı olmadan değişmez).

## 7. CI Gate
```bash
uv sync --extra dev
uv run ruff check app tests
uv run ruff format --check app tests
uv run mypy app
uv run pytest -m "not ollama" --basetemp=.pytest_tmp
```
- **Phase 3.5 sonrası `mypy app` torch/peft KURULU OLMADAN temiz** (CI `train-cpu` kurmaz).
- Ollama / network / cloud gerektiren testler bu workflow'da KOŞMAZ (`-m "not ollama"`).

## 8. Nightly Audit (rapor-only)
`nightly-automation-audit.yml` YALNIZ rapor üretir:
- **Yapabilir:** CI çalıştır; varsa `understanding-score --record` + `understanding-history
  --compare` + `pretrain-gate`; bug-scan/rapor; raporu **artifact** olarak yükle;
  (opsiyonel) bir `automation` issue'sunu güncelle.
- **Yapamaz:** kod değiştir; branch açıp fix yap; training başlat; adapter promote et;
  raporu repoya **commit etme** (artifact olarak kalır).
- Cron WIRED ama **varsayılan KAPALI**: zamanlanmış koşu yalnız repo variable
  `ENABLE_NIGHTLY_AUDIT == 'true'` ise çalışır; elle dispatch her zaman.

## 9. Required Secrets
Yalnız GitHub Actions **secret** olarak: `ANTHROPIC_API_KEY` (zorunlu, aktivasyonda),
`HF_TOKEN` (opsiyonel). Secret'lar: repoya YAZILMAZ · loglanmaz · workflow output'a
basılmaz · dokümanda örnek değer verilmez. Workflow'da yalnız `${{ secrets.* }}`
referansı kullanılır (literal değer yok).

## 10. Rollout Plan
- **Phase 4A (BU FAZ):** doküman + workflow ŞABLONLARI + guard + test; lokal YAML
  doğrulama; **trigger YOK**.
- **Phase 4B:** elle bir test issue'su; `claude-task` etiketi; dry-run tarzı PR;
  **auto-merge YOK**; insan incelemesi.
- **Phase 5:** nightly self-improvement issue döngüsü (rapor → insan onaylı görev).

### Aktivasyon öncesi TODO (kullanıcı)
- [ ] `anthropics/claude-code-action` TAM input adlarını/sürümünü doğrula + pinle
  (workflow'da `TODO` işaretli; uydurma syntax ile etkinleştirme).
- [ ] `ANTHROPIC_API_KEY` secret'ını repo ayarlarından ekle.
- [ ] main için branch protection (PR review zorunlu, force-push kapalı).
- [ ] Nightly için `ENABLE_NIGHTLY_AUDIT` repo variable'ını bilinçli set et.
