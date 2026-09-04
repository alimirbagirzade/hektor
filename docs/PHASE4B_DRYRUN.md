# Phase 4B — Claude Code Manual Dry-Run Plan

> **Durum:** Hazırlık + doğrulama. Bu fazda GERÇEK workflow tetiklenmedi, test issue
> açılmadı, label ile canlı koşu başlatılmadı. Aktivasyon ve ilk dry-run **kullanıcı
> kontrolünde** (Phase 4C). `claude-code-action@v1` input'ları resmi README ile
> doğrulandı; Claude adımı `vars.ENABLE_CLAUDE_TASK == 'true'` olmadan INERT (skip).

## Goal
İlk canlı deneme **yalnız zararsız bir docs değişikliği** olacak: app/test/veri
mantığına dokunmadan, guard + CI + PR akışının uçtan uca çalıştığını kanıtlamak.

## Before enabling — checklist (kullanıcı)
- [ ] GitHub secret **`ANTHROPIC_API_KEY`** eklendi mi? (repoya YAZILMAZ, loglanmaz)
- [ ] main için **branch protection** açık mı? (force-push kapalı)
- [ ] **Required review** (en az 1 insan onayı) açık mı?
- [ ] **Auto-merge kapalı** mı? (workflow zaten merge etmez)
- [ ] Workflow dosyaları main'e **elle merge** edildi mi? (Actions yalnız main'deki
      workflow'u çalıştırır)
- [ ] Protected-path guard testleri geçiyor mu? (`pytest tests/test_protected_paths_guard.py`)
- [ ] `data/ storage/ vector_db/ models/` koruması aktif mi? (guard + hard rules)
- [ ] `anthropics/claude-code-action` **SHA-pin** + `claude_args` tool-kısıtlaması
      docs/configuration.md ile doğrulandı mı? (workflow'daki TODO'lar)
- [ ] Aktivasyon: repo variable **`ENABLE_CLAUDE_TASK = true`** set edildi mi?

## Test issue örneği
**Issue title:**
```text
Claude dry-run: add one documentation sentence
```
**Issue body:**
```text
Task:
Add one harmless sentence to docs/PHASE4_GITHUB_AUTOMATION.md under a "Dry-run note" section.

Allowed files:
- docs/PHASE4_GITHUB_AUTOMATION.md

Forbidden:
- app/
- tests/
- data/
- storage/
- vector_db/
- models/
- .env

Acceptance:
- Only one docs file changed.
- CI passes.
- Protected-path guard passes.
- PR created.
- No auto-merge.
```
**Labels:**
```text
claude-task
safe-refactor
```

## Expected result
- iş-branch oluşturulur (main'e push YOK)
- yalnız docs değişikliği (`docs/PHASE4_GITHUB_AUTOMATION.md`)
- CI koşar (ruff + format + mypy + `pytest -m "not ollama"`)
- protected-path guard (pre + post) GEÇER
- PR açılır (iş-branch → main)
- **merge YOK** — insan incelemesi zorunlu
- PR gövdesi: değişen dosyalar + test sonucu + guard sonucu + kalan risk +
  "manual review required"

## Rollback
- PR'ı kapat (merge etme)
- iş-branch'i sil
- issue label'larını kaldır (`claude-task`, `safe-refactor`)
- gerekirse `vars.ENABLE_CLAUDE_TASK` değişkenini sil → workflow yeniden INERT
- secret'ı kaldır (tam geri alma istenirse)

## Negatif testler (öneri — yine MANUEL, Phase 4C)
- `dangerous-change` / `human-only` / `no-claude` / `needs-approval` etiketli issue →
  Claude ATLANMALI (skip-notice yorumu).
- Protected path'e dokunan bir görev → guard FAIL, PR kırmızı, merge edilemez.
- `pyproject.toml` değişen bir görev → "needs-approval" uyarısı.
