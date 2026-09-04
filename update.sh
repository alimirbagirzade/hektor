#!/usr/bin/env bash
# Achilles Trader AI — TEK KOMUT güncelleme (macOS / Linux; KURULU makinede çalıştır)
#
#   ./update.sh            -- normal: origin/main'e GÜVENLİ yakınsama (ff-only)
#   ./update.sh --force    -- yereli AT, origin/main ile birebir eşitle (salt-kopya kurulum)
#
# Yapar: web sunucusunu durdur (port 8765) -> 'main' dalına yakınsa (origin/main) ->
#        uv sync --extra dev -> web'i yeniden başlat -> sağlık kontrolü.  EĞİTİME DOKUNMAZ.
#
# NOT (kök-neden düzeltmesi): Bu betik artık MEVCUT dal ne olursa olsun makineyi
# 'main' dalına + origin/main'e yakınsatır. Eskiden bir feature dalına parklanmış
# makinede 'git pull origin main' origin/main'i o dala MERGE ediyor, makine asla
# main'e geçmiyordu -> "güncelleme oturmuyor". Tanı için:  uv run achilles doctor
#
# Tarayıcıda son halini görmek için sonunda: Cmd+Shift+R (sert yenileme).

set -u

FORCE=0
case "${1:-}" in
  -f|--force) FORCE=1 ;;
  "") ;;
  *) echo "Bilinmeyen argüman: $1 (kullanım: ./update.sh [--force])"; exit 1 ;;
esac

# Proje dizini = bu script'in bulunduğu yer
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$PROJECT_DIR" || { echo "[HATA] Proje dizinine girilemedi"; exit 1; }

# --- araçları bul ---
UV="$(command -v uv || true)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
[ -z "$UV" ] && [ -x "$HOME/.cargo/bin/uv" ] && UV="$HOME/.cargo/bin/uv"
GIT="$(command -v git || true)"
[ -z "$UV" ]  && { echo "[HATA] uv bulunamadı (önce setup.sh)."; exit 1; }
[ -z "$GIT" ] && { echo "[HATA] git bulunamadı."; exit 1; }

"$GIT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "[HATA] Bu klasör bir git deposu değil: $PROJECT_DIR"; exit 1;
}

mkdir -p logs
LOG="logs/update.log"
echo "[$(date '+%Y-%m-%d %H:%M')] Güncelleme başlıyor (force=$FORCE)..." >> "$LOG"

# --------------------------------------------------------------------------
# Yardımcı fonksiyonlar (çağrılmadan ÖNCE tanımlı)
# --------------------------------------------------------------------------
# 'main' başka bir worktree'de checkout mu?
main_elsewhere() {
  "$GIT" worktree list --porcelain 2>/dev/null | grep -q 'branch refs/heads/main$'
}

# Dal + HEAD + origin/main'e göre ahead/behind raporla (drift görünür olsun)
show_drift() {
  local b h counts behind ahead
  b="$("$GIT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  h="$("$GIT" rev-parse --short HEAD 2>/dev/null)"
  counts="$("$GIT" rev-list --left-right --count origin/main...HEAD 2>/dev/null || true)"
  behind="${counts%%[[:space:]]*}"; ahead="${counts##*[[:space:]]}"
  case "$behind" in ''|*[!0-9]*) behind=0 ;; esac
  case "$ahead" in ''|*[!0-9]*) ahead=0 ;; esac
  echo "[DURUM] dal=$b HEAD=$h | origin/main'e gore: +${ahead} / -${behind}"
  echo "[$(date '+%Y-%m-%d %H:%M')] DURUM dal=$b HEAD=$h ahead=${ahead} behind=${behind}" >> "$LOG"
}

# Mevcut dal ne olursa olsun 'main' + origin/main'e DETERMİNİSTİK yakınsa.
# Kullanıcı verisini ASLA atmaz (--force hariç); ıraksak dalı AUTO-MERGE ETMEZ.
sync_to_main() {
  "$GIT" fetch origin main >/dev/null 2>&1 || echo "[UYARI] fetch basarisiz (cevrimdisi?) -- yerel ref'lerle devam."

  local cur dirty mainhash remote head
  cur="$("$GIT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  dirty=0; [ -n "$("$GIT" status --porcelain 2>/dev/null)" ] && dirty=1

  if [ "$cur" != "main" ]; then
    echo "[UYARI] Bu makine 'main' DEGIL, '$cur' dalinda PARKLANMIS — main'e geciliyor..."
    echo "[$(date '+%Y-%m-%d %H:%M')] Parkli dal '$cur' -> main gecisi." >> "$LOG"

    if [ "$dirty" -eq 1 ] && [ "$FORCE" -eq 0 ]; then
      echo "[HATA] Yerel degisiklik var; 'main'e guvenle gecemiyorum."
      echo "       Cozum: commit/stash et, ya da yereli ATMAK icin: ./update.sh --force"
      return
    fi
    if main_elsewhere; then
      echo "[HATA] 'main' baska bir worktree'de checkout — bu kopya gecemez. origin/main'e DOKUNULMADI."
      return
    fi

    if "$GIT" show-ref --verify --quiet refs/heads/main; then
      if [ "$FORCE" -eq 1 ]; then "$GIT" switch -f main >/dev/null 2>&1; else "$GIT" switch main >/dev/null 2>&1; fi
    else
      if [ "$FORCE" -eq 1 ]; then "$GIT" switch -C main --track origin/main >/dev/null 2>&1; else "$GIT" switch -c main --track origin/main >/dev/null 2>&1; fi
    fi

    if [ "$("$GIT" rev-parse --abbrev-ref HEAD 2>/dev/null)" != "main" ]; then
      echo "[HATA] 'main' dalina gecilemedi — origin/main'e MERGE EDILMEDI (veri korundu)."
      return
    fi
    echo "[OK] 'main' dalina gecildi."
  fi

  remote="$("$GIT" rev-parse origin/main 2>/dev/null || true)"
  head="$("$GIT" rev-parse HEAD 2>/dev/null || true)"
  if [ "$FORCE" -eq 1 ]; then
    echo "[!] --force: yerel degisiklikler ATILIYOR, origin/main'e (HEAD=main) esitleniyor."
    "$GIT" reset --hard origin/main >/dev/null 2>&1
  elif [ -z "$remote" ]; then
    echo "[UYARI] origin/main yerel ref'i yok (ilk fetch basarisiz olabilir) -- atlandi."
  elif [ "$head" != "$remote" ]; then
    if ! "$GIT" pull --ff-only origin main >/dev/null 2>&1; then
      echo "[HATA] ff-only ilerleyemedi: yerel 'main' ile origin/main IRAKSAK. Cozum (yereli ATAR): ./update.sh --force"
      echo "[$(date '+%Y-%m-%d %H:%M')] ff-only IRAKSAK HATASI." >> "$LOG"
    fi
  else
    echo "[OK] Kod zaten guncel (${head:0:7})."
  fi
}

# --- 1. Web sunucusunu durdur (port 8765 + achilles-web). EĞİTİME dokunma. ---
stop_web() {
  if [ -f .web.pid ]; then
    pid="$(cat .web.pid 2>/dev/null)"
    case "$pid" in
      ""|*[!0-9]*) ;;
      *) kill "$pid" 2>/dev/null || true ;;
    esac
    rm -f .web.pid
  fi
  # port 8765'i dinleyeni durdur (macOS/Linux: lsof)
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti tcp:8765 2>/dev/null || true)"
    [ -n "$pids" ] && kill $pids 2>/dev/null || true
  fi
  # achilles-web süreçleri (EĞİTİM 'achilles train' DEĞİL — ona dokunma)
  pkill -f 'achilles-web' 2>/dev/null || true
}
stop_web
sleep 1

# --- 2. origin/main'e DETERMİNİSTİK yakınsama (parklanmış dalı zorla main'e al) ---
LOCAL="$("$GIT" rev-parse HEAD 2>/dev/null)"
sync_to_main
show_drift

NEW="$("$GIT" rev-parse HEAD 2>/dev/null)"
UPDATED=0
[ "$NEW" != "$LOCAL" ] && UPDATED=1
[ "$UPDATED" -eq 1 ] && echo "[OK] Kod güncellendi: ${LOCAL:0:7} -> ${NEW:0:7}"

# --- 3. Bağımlılıklar (WEB extra DAHIL — düz 'uv sync' web paketlerini budar) ---
if [ "$UPDATED" -eq 1 ] || [ "$FORCE" -eq 1 ]; then
  echo "[..] Bağımlılıklar eşitleniyor (uv sync --extra dev)..."
  "$UV" sync --extra dev >/dev/null 2>&1 || true
fi

# --- 4. Web'i yeniden başlat (arka plan) ---
nohup "$UV" run --project "$PROJECT_DIR" achilles-web \
  > logs/achilles-web.log 2> logs/achilles-web-err.log &
echo $! > .web.pid
echo "[$(date '+%Y-%m-%d %H:%M')] Sunucu başlatıldı (PID $(cat .web.pid))." >> "$LOG"

# --- 5. Sağlık kontrolü (port dinliyor mu / status yanıt veriyor mu) ---
OK=0
i=0
while [ "$i" -lt 15 ]; do
  if command -v lsof >/dev/null 2>&1 && lsof -ti tcp:8765 >/dev/null 2>&1; then OK=1; break; fi
  if curl -sf "http://127.0.0.1:8765/api/status" >/dev/null 2>&1; then OK=1; break; fi
  i=$((i + 1))
  sleep 2
done
echo ""
if [ "$OK" -eq 1 ]; then
  echo "[OK] Web çalışıyor: http://127.0.0.1:8765"
  echo "     >> Son halini görmek için tarayıcıda: Cmd+Shift+R (sert yenileme!)"
else
  echo "[UYARI] Web 30 sn'de açılmadı — log: logs/achilles-web-err.log"
fi
