#!/usr/bin/env bash
# Achilles otomatik zincir (Windows Git Bash / Linux):
#   kartsız makalelere kart üret (LLM) → içerikli pending kartları onayla
#   → anlama skorları hesapla (web API, LLM) → Markov araştırması (hipotez+backtest)
#   → sentez makalesi üret → dataset tazele → 24 saatlik eğitim döngüsünü başlat.
#
# RAM disiplini: LLM adımları eğitimden ÖNCE sıralı çalışır (4B eğitim ~16GB,
# qwen3:4b ~4GB — aynı anda OOM riski). Eğitim en sonda otomatik başlar.
# Durdurma: storage/STOP_TRAINING dosyası (eğitim döngüsü için).
set -u
cd "$(dirname "$0")/.." || exit 1
LOG=logs/auto-chain.log
mkdir -p logs
# uv her `uv run`'da paketi yeniden senkronlayip calisan web sunucusunun kilitledigi
# achilles-web.exe'yi silmeye ugrasir -> "os error 32" -> adimlar sessizce coker.
# Senkronu kapat; bagimliliklar zaten kurulu. (bkz. continuous-learning.sh)
export UV_NO_SYNC=1
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== OTOMATİK ZİNCİR BAŞLADI ==="

# --- 1) Kartsız makalelere kart üret (LLM, makale başına dakikalar) ---------
PIDS=$(uv run python -c "
from app.memory.sqlite_store import SqliteStore
s = SqliteStore()
for p in s.list_papers():
    if not s.has_knowledge_card(p.paper_id):
        print(p.paper_id)
" 2>/dev/null)
N=$(printf '%s' "$PIDS" | grep -c . || true)
log "1) Kartsız makale: $N"
for pid in $PIDS; do
  log "   kart üretiliyor: $pid"
  timeout 900 uv run achilles card "$pid" >> "$LOG" 2>&1 || log "   kart HATA: $pid"
done

# --- 2) İçerikli pending kartları onayla (boş kabuk kartlar onaylanmaz) -----
uv run python -c "
from app.memory.sqlite_store import SqliteStore
s = SqliteStore()
approved = skipped = 0
for c in s.list_pending_cards():
    cj = c.get('card_json') or {}
    title = str(cj.get('title') or '').strip()
    claim = str(cj.get('main_claim') or '').strip()
    if title and claim:
        s.approve_card(c['card_id']); approved += 1
    else:
        skipped += 1
print(f'2) Kart onayı: {approved} onaylandı, {skipped} içeriksiz atlandı')
" >> "$LOG" 2>&1

# --- 3) Anlama skorları (yalnız İÇERİKLİ kartı olan makaleler; LLM) ---------
PIDS2=$(uv run python -c "
from app.memory.sqlite_store import SqliteStore
s = SqliteStore()
for p in s.list_papers():
    c = s.get_latest_knowledge_card(p.paper_id)
    if c and str((c.get('card_json') or {}).get('title') or '').strip():
        if s.get_comprehension_score(p.paper_id) is None:
            print(p.paper_id)
" 2>/dev/null)
N2=$(printf '%s' "$PIDS2" | grep -c . || true)
log "3) Anlama skoru hesaplanacak makale: $N2"
for pid in $PIDS2; do
  curl -s -m 600 -X POST "http://127.0.0.1:8765/api/papers/$pid/comprehension" >> "$LOG" 2>&1
  log "   anlama skorlandı: $pid"
done

# --- 4) Markov odaklı agentic araştırma (hipotez + backtest + yansıma) ------
log "4) Markov araştırması başlıyor (2 iterasyon)"
timeout 5400 uv run achilles research \
  "Markov zinciri rejim degisimi (regime-switching) sinyalleri momentum ve volatilite gostergeleriyle nasil birlestirilir? Rejim gecis olasiligina dayali yeni bir indikator oner ve test et." \
  --iterations 2 >> "$LOG" 2>&1 || log "   research HATA/timeout"

# --- 5) Sentez makalesi üret (web'den indirilebilir) ------------------------
log "5) Sentez makalesi üretiliyor"
uv run achilles synth-paper >> "$LOG" 2>&1

# --- 6) Dataset tazele -------------------------------------------------------
log "6) LoRA dataset tazeleniyor"
uv run achilles lora-dataset >> "$LOG" 2>&1
uv run achilles rag-mastery >> "$LOG" 2>&1

# --- 7) 24 saatlik eğitim döngüsü --------------------------------------------
log "7) === EĞİTİM DÖNGÜSÜ BAŞLIYOR (24h, iters=40, cooldown=120sn) ==="
rm -f storage/STOP_TRAINING
END=$(( $(date +%s) + 86400 ))
cycle=0
while [ ! -f storage/STOP_TRAINING ] && [ "$(date +%s)" -lt "$END" ]; do
  cycle=$((cycle+1))
  log "   Eğitim döngü $cycle"
  uv run achilles lora-dataset >> "$LOG" 2>&1
  uv run achilles train --run --backend peft --adapter-name achilles_auto --iterations 40 >> "$LOG" 2>&1
  for i in $(seq 1 24); do [ -f storage/STOP_TRAINING ] && break; sleep 5; done
done
log "Eğitim döngüsü bitti ($cycle döngü). ZİNCİR TAMAM."
rm -f storage/STOP_TRAINING
