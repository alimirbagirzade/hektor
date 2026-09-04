#!/usr/bin/env bash
# Achilles Trader AI -- Linux/macOS cevrimdisi kurulum dogrulama kapisi.
#
# Amac: autostart KURULMADAN / "hazir" denmeden ONCE "sistem gercekten ayaga
# kalkiyor mu" kanitla. verify-install.ps1'in (Windows) bire-bir bash karsiligi:
# Ollama GEREKTIRMEZ -- fake embedding + sentetik veri ile uctan uca duman testi
# (init -> status -> gen-data -> backtest -> offline pytest).
#
# Tasinabilir: script-konumu tabanli, hardcoded yol YOK. Idempotent.
# Cikis kodu sozlesmesi (cagiran buna gore kapi uygular):
#   0  = GECTI  (kullanima/autostart icin hazir)
#   1  = KALDI  (bir adim basarisiz -- hazir DEGIL)
#   2  = ORTAM  (uv bulunamadi -- on kosul eksik)
#
# Kullanim:
#   bash scripts/verify-install.sh              # tam dogrulama (gerekirse uv sync)
#   bash scripts/verify-install.sh --skip-sync  # bagimlilik senkronunu atla (cevrimdisi/CI)
set -u

SKIP_SYNC=false
[ "${1:-}" = "--skip-sync" ] && SKIP_SYNC=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR" || { echo "[HATA] proje dizinine girilemedi: $PROJECT_DIR"; exit 2; }

# ---------------------------------------------------------------- uv bul
# PATH'te yoksa bilinen konumlari dene (taze makinede setup.sh sonrasi PATH gec gelebilir).
find_uv() {
    if command -v uv >/dev/null 2>&1; then command -v uv; return 0; fi
    for p in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "/usr/local/bin/uv" "/opt/homebrew/bin/uv"; do
        [ -x "$p" ] && { echo "$p"; return 0; }
    done
    return 1
}
UV="$(find_uv)" || {
    echo "[HATA] uv bulunamadi. Kur: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 2
}

# ---------------------------------------------------------------- cevrimdisi mod
# Fake embedding: Ollama/ag olmadan RAG/embedding kod yollari calisabilsin.
export ACHILLES_ALLOW_FAKE_EMBEDDINGS=true

echo "Achilles cevrimdisi kurulum dogrulamasi"
echo "  Proje : $PROJECT_DIR"
echo "  uv    : $UV"

# ---------------------------------------------------------------- 0) bagimliliklar
if [ "$SKIP_SYNC" = false ]; then
    echo ""
    echo "==> [0] Bagimliliklar (uv sync --extra dev)"
    "$UV" sync --extra dev --project "$PROJECT_DIR" || \
        echo "  [!] uv sync basarisiz (cevrimdisi olabilir) -- mevcut ortamla devam."
fi
# Sonraki cagrilar yeniden senkron DENEMESIN (calisan sunucu kilidini bozmasin).
export UV_NO_SYNC=1

# ---------------------------------------------------------------- adim koscusu
step_no=0
failed=""
run_step() {
    local name="$1"; shift
    [ -n "$failed" ] && return 0   # fail-fast: onceki adim kaldiysa gerisini atla
    step_no=$((step_no + 1))
    echo ""
    echo "==> [$step_no] $name"
    if "$UV" "$@"; then
        echo "  [OK] $name"
    else
        echo "  [KALDI] $name (cikis kodu $?)"
        failed="$name"
    fi
}

# ---------------------------------------------------------------- duman testi zinciri
run_step "Sistem baslat (init)"      run --no-sync achilles init
run_step "Durum (status)"            run --no-sync achilles status
run_step "Sentetik veri (gen-data)"  run --no-sync achilles gen-data
run_step "Backtest (ornek strateji)" run --no-sync achilles backtest data/market/raw/synthetic.csv
run_step "Testler (offline)"         run --no-sync pytest -q -m "not ollama and not slow" --basetemp .pytest_tmp

# ---------------------------------------------------------------- sonuc
echo ""
if [ -n "$failed" ]; then
    echo "SONUC: KALDI -- '$failed' adiminda basarisiz."
    echo "       Sistem 'hazir' DEGIL. Hatayi duzeltip tekrar calistirin."
    exit 1
fi
echo "SONUC: GECTI -- $step_no adim cevrimdisi dogrulandi. Kullanima hazir."
exit 0
