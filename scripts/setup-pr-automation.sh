#!/usr/bin/env bash
# Achilles -- PR otomasyonu BİR KERELİK repo kurulumu.
#
# Ne yapar:
#   1) Repo ayarları: "Allow auto-merge" + squash + merge sonrası dalı sil.
#   2) main branch koruması: CI kontrolü ("lint · types · tests (offline)")
#      GEÇMEDEN PR merge OLAMAZ.  enforce_admins=FALSE → owner (sen) yine
#      doğrudan `git push origin main` yapabilir; otonom loop'lar etkilenmez.
#      PR'lar ise CI yeşil olunca otomatik merge olur (gh pr merge --auto).
#
# Ön koşul:  gh auth login  (bir kerelik, kendi terminalinde)
# Çalıştır:  bash scripts/setup-pr-automation.sh
# Geri al:   bash scripts/setup-pr-automation.sh --undo   (branch korumasını kaldırır)
set -euo pipefail

CI_CONTEXT="lint · types · tests (offline)"   # .github/workflows/ci.yml job adı

if ! gh auth status >/dev/null 2>&1; then
  echo "[HATA] gh giriş yapılmamış. Önce:  gh auth login" >&2
  exit 2
fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo ">> Repo: $REPO"

if [ "${1:-}" = "--undo" ]; then
  echo ">> main branch koruması kaldırılıyor..."
  gh api "repos/$REPO/branches/main/protection" --method DELETE >/dev/null 2>&1 \
    && echo "[OK] Branch koruması kaldırıldı." \
    || echo "[i] Koruma zaten yoktu."
  exit 0
fi

# --- 1) Repo ayarları: auto-merge + squash + merge sonrası dal sil ---
echo ">> Repo ayarları (allow_auto_merge, squash, delete_branch_on_merge)..."
gh api "repos/$REPO" --method PATCH \
  -F allow_auto_merge=true \
  -F allow_squash_merge=true \
  -F delete_branch_on_merge=true >/dev/null
echo "[OK] Auto-merge repo düzeyinde açıldı."

# --- 2) main branch koruması: CI zorunlu, admin (owner) muaf ---
echo ">> main branch koruması (CI zorunlu, owner muaf)..."
if gh api "repos/$REPO/branches/main/protection" --method PUT --input - >/dev/null 2>prot.err <<JSON
{
  "required_status_checks": { "strict": true, "contexts": ["$CI_CONTEXT"] },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
then
  rm -f prot.err
  echo "[OK] main koruması kuruldu: PR'lar '$CI_CONTEXT' geçmeden merge OLAMAZ."
  echo "     (owner muaf → doğrudan 'git push origin main' hâlâ çalışır.)"
else
  echo "[!] Branch koruması kurulamadı (özel repo + ücretsiz plan olabilir):" >&2
  sed 's/^/      /' prot.err >&2 2>/dev/null || true
  rm -f prot.err
  echo "[i] Yine de auto-merge AÇIK; ama CI 'zorunlu' olmadan --auto, PR mergeable"
  echo "    olur olmaz birleştirir (CI'yi beklemeyebilir). Repo'yu public yaparsan"
  echo "    veya GitHub Pro ile koruma açılır, o zaman CI gerçekten beklenir."
fi

echo ""
echo "TAMAM. Artık tek komutla tam-otomatik PR:"
echo "    bash scripts/open-pr.sh        (push + PR + CI geçince oto-merge)"
