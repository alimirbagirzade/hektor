#!/usr/bin/env bash
# Achilles -- TEK KOMUTLA PR (elle GitHub web'e girmeye son).
#
# Mevcut daldaki işi push eder ve otomatik PR açar (başlık/gövde commit'lerden
# doldurulur). İstersen CI yeşil olunca otomatik squash-merge ayarlar.
#
# Kullanım (VARSAYILAN = tam otomatik: push + PR + CI geçince oto squash-merge):
#   bash scripts/open-pr.sh                      # push + PR + oto-merge (önerilen)
#   bash scripts/open-pr.sh "Özel PR başlığı"    # başlığı sen ver
#   bash scripts/open-pr.sh --no-merge           # sadece PR aç, merge'i sen yap
#   bash scripts/open-pr.sh --base develop       # farklı hedef dal (vars. main)
#
# Ön koşul (bir kerelik):  gh auth login  +  bash scripts/setup-pr-automation.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.."

BASE="main"; MERGE=true; TITLE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base)     BASE="${2:?--base bir dal ister}"; shift 2 ;;
    --merge)    MERGE=true; shift ;;
    --no-merge) MERGE=false; shift ;;
    -h|--help)  sed -n '2,16p' "$0"; exit 0 ;;
    *) TITLE="$1"; shift ;;
  esac
done

# --- gh giriş yapılmış mı? ---
if ! gh auth status >/dev/null 2>&1; then
  echo "[HATA] gh giriş yapılmamış. Önce bir kerelik:  gh auth login" >&2
  exit 2
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "$BASE" ]; then
  echo "[!] '$BASE' dalındasın — PR için önce bir özellik dalına geç:" >&2
  echo "      git switch -c feat/yeni-ozellik" >&2
  exit 1
fi

# Kaydedilmemiş değişiklik varsa uyar (push edilmez).
if [ -n "$(git status --porcelain)" ]; then
  echo "[!] Kaydedilmemiş değişiklik var — önce commit et (yoksa PR'a girmez)." >&2
fi

echo ">> '$BRANCH' dalı push ediliyor..."
git push -u origin "$BRANCH"

if gh pr view "$BRANCH" >/dev/null 2>&1; then
  echo "[i] Bu dal için PR zaten var (güncellendi)."
else
  echo ">> PR oluşturuluyor (hedef: $BASE)..."
  if [ -n "$TITLE" ]; then
    gh pr create --base "$BASE" --head "$BRANCH" --title "$TITLE" \
      --body "Otomatik PR (scripts/open-pr.sh) — değişiklikler \`$BRANCH\` dalında."
  else
    gh pr create --base "$BASE" --head "$BRANCH" --fill
  fi
fi

URL="$(gh pr view "$BRANCH" --json url -q .url)"
echo "[OK] PR: $URL"

if [ "$MERGE" = true ]; then
  echo ">> Oto-merge ayarlanıyor (CI yeşil olunca squash + dalı sil)..."
  gh pr merge "$BRANCH" --squash --delete-branch --auto \
    && echo "[OK] Oto-merge aktif (gerekli kontroller geçince birleşir)." \
    || echo "[!] Oto-merge ayarlanamadı (branch koruması/checks gerekebilir) — PR açık kaldı."
fi
