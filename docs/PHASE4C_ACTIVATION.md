# Phase 4C — Claude Code Automation: Activation Runbook (KULLANICI)

> Bu runbook **ilk gerçek GitHub dry-run** içindir ve adımların tamamı **kullanıcı
> tarafından** yapılır. Asistan (Claude/ChatGPT) bu adımların HİÇBİRİNİ yapmaz —
> yalnız kodu/dokümanı hazırlar ve sonucu analiz eder.

## 1. Purpose
Hazırlanan `claude-code-task.yml` workflow'unu **kontrollü** biçimde etkinleştirip,
**zararsız docs-only** bir görevle uçtan uca akışı (guard → CI → PR, merge YOK) ilk kez
canlı denemek.

## 2. What the assistant will NOT do
- Asistan **secret EKLEMEZ** (`ANTHROPIC_API_KEY` dahil).
- Asistan **issue AÇMAZ**.
- Asistan **label KOYMAZ**.
- Asistan **workflow TETİKLEMEZ** / repo variable SET ETMEZ.
- Asistan **PR MERGE ETMEZ**.
- Asistan **main'e PUSH yapmaz**.
- Asistan eğitim/terfi/cloud çalıştırmaz.
Tüm canlı GitHub eylemleri **senin** kontrolündedir.

## 3. Pre-activation checklist (kullanıcı)
- [ ] `feat/agent-runtime-phase4c` kodları gözden geçirildi.
- [ ] Workflow dosyaları main'e **PR ile elle merge** edildi (Actions yalnız main'deki
      workflow'u çalıştırır).
- [ ] main için **branch protection** açık.
- [ ] **Required review** (≥1 insan onayı) açık.
- [ ] **Force-push kapalı**.
- [ ] **Auto-merge kapalı**.
- [ ] **`ANTHROPIC_API_KEY`** GitHub Secret olarak eklendi (repoya YAZILMAZ).
- [ ] `ENABLE_CLAUDE_TASK` repo variable hâlâ **false / yok** (aktivasyon en sonda).
- [ ] Protected-path guard testleri geçti: `uv run pytest tests/test_protected_paths_guard.py`.
- [ ] GitHub Actions permissions gözden geçirildi (workflow minimum yetki ister).
- [x] **SHA-pin UYGULANDI** (branch'te, supply-chain): `claude-code-task.yml` artık
      `anthropics/claude-code-action@51705da45eecce209d4700538bf8377d5b5fc695`
      (v1 @ 2026-06-19, Claude Code 2.1.183) kullanır. `v1` **hareketli** tag (o gün
      taşındı) → pin sürüm sabitler. Action'ı **YÜKSELTMEK** istediğinde yeniden pinle:
      ```bash
      gh api repos/anthropics/claude-code-action/git/ref/tags/v1 --jq '.object.sha'
      # Çıktı bir annotated tag SHA'sı ise commit'e deref et:
      gh api repos/anthropics/claude-code-action/git/tags/<TAG_SHA> --jq '.object.sha'
      # Sonra: uses: anthropics/claude-code-action@<COMMIT_SHA>   # v1 (tarih)
      ```
      (`gh` yoksa: GitHub API'yi tarayıcıdan/`curl` ile aynı yollardan çağır.)
- [ ] `needs-approval` ve `claude-task` / `safe-refactor` / `human-only` / `no-claude` /
      `dangerous-change` label'ları repo'da **oluşturuldu** (workflow etiket ekleyebilsin).

## 4. Enable workflow
Tüm checklist tamamlandıktan SONRA, repo variable'ı set et:
```bash
gh variable set ENABLE_CLAUDE_TASK --body true
```
(GitHub UI: Settings → Secrets and variables → Actions → Variables → New variable →
`ENABLE_CLAUDE_TASK` = `true`.)

## 5. First dry-run issue
**Issue title:**
```text
Claude dry-run: docs-only automation test
```
**Issue body:**
```text
Task:
Add one harmless sentence to docs/PHASE4_GITHUB_AUTOMATION.md under a new "Dry-run note" heading.

Allowed files:
- docs/PHASE4_GITHUB_AUTOMATION.md

Forbidden files:
- app/**
- tests/**
- data/**
- storage/**
- vector_db/**
- models/**
- .env
- .github/workflows/**

Acceptance criteria:
- Only docs/PHASE4_GITHUB_AUTOMATION.md changed.
- Protected-path guard passes.
- CI passes.
- PR opens.
- No auto-merge.
- Manual review required.
```
**Labels:**
```text
claude-task
safe-refactor
```

## 6. Expected result
- workflow başlar · Claude adımı koşar · iş-branch oluşturulur
- yalnız docs değişikliği · protected guard (pre+post) GEÇER · CI GEÇER
- PR açılır · **merge YOK** · manuel insan incelemesi zorunlu

## 7. What to inspect in PR
- [ ] tam olarak tek docs dosyası değişti
- [ ] `data/ storage/ vector_db/ models/` YOK
- [ ] `.env` YOK
- [ ] log'da eğitim komutu YOK (`train --run`, `achilles lora` vb.)
- [ ] auto-merge YOK
- [ ] CI yeşil
- [ ] protected-path guard GEÇTİ
- [ ] PR gövdesinde test özeti var
- [ ] issue PR'a bağlı

## 8. Rollback
Bir şey yanlışsa:
```bash
gh variable delete ENABLE_CLAUDE_TASK
gh pr close <PR_NUMBER> --comment "Closing dry-run PR."
git push origin --delete <BRANCH_NAME>
```
(UI: variable'ı sil, PR'ı kapat (merge etme), branch'i sil, label'ları kaldır.)

## 9. What to send back to the assistant
Dry-run sonrası şunları paylaş (asistan analiz etsin):
- Actions log özeti (özellikle guard pre/post + CI adımları)
- açılan PR linki
- değişen dosya listesi
- CI sonucu (ruff / mypy / pytest)
- protected-path guard sonucu
- herhangi bir hata mesajı
