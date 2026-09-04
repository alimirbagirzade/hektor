#!/usr/bin/env bash
# Achilles macOS MLX EĞİTİM DÖNGÜSÜ — Apple Silicon için.
#
# Her tur:
#   1) Kartsız makalelere kart üret (LLM)
#   2) İçerikli pending kartları onayla
#   3) Sentetik QA üret (synth-qa)
#   4) Dataset tazele
#   5) MLX ile kısa eğitim (300 iter) → adapter kaydet
#   → 5 dakika dinlen → tekrar
#
# Kullanım: bash scripts/mac-loop.sh [MAX_SAAT]   (varsayılan: 24)
# Durdurma: touch storage/STOP_LEARNING
#
# Log: logs/mac-loop.log
# Adapter: models/adapters/mac_loop_v<TUR>

set -u
cd "$(dirname "$0")/.." || exit 1

MAX_HOURS="${1:-24}"
LOG=logs/mac-loop.log
STOP=storage/STOP_LEARNING
mkdir -p logs storage

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== MAC LOOP BAŞLADI (max ${MAX_HOURS}sa) ==="
rm -f "$STOP"

END=$(( $(date +%s) + MAX_HOURS*3600 ))
round=0

while [ ! -f "$STOP" ] && [ "$(date +%s)" -lt "$END" ]; do
  round=$((round+1))
  log "──── TUR $round / $(( (END - $(date +%s)) / 3600 ))sa kaldı ────"

  # --- 1) Kartsız makalelere kart üret ---------------------------------------
  PIDS=$(uv run python -c "
from app.memory.sqlite_store import SqliteStore
s = SqliteStore()
for p in s.list_papers():
    if not s.has_knowledge_card(p.paper_id):
        print(p.paper_id)
" 2>/dev/null)
  N=$(printf '%s' "$PIDS" | grep -c . || echo 0)
  log "1) Kartsız makale: $N"
  for pid in $PIDS; do
    log "   kart: $pid"
    timeout 900 uv run achilles card "$pid" >> "$LOG" 2>&1 || log "   HATA: $pid"
  done

  # --- 2) İçerikli pending kartları onayla -----------------------------------
  uv run python -c "
from app.memory.sqlite_store import SqliteStore
s = SqliteStore()
a = k = 0
for c in s.list_pending_cards():
    cj = c.get('card_json') or {}
    if str(cj.get('title') or '').strip() and str(cj.get('main_claim') or '').strip():
        s.approve_card(c['card_id']); a += 1
    else:
        k += 1
print(f'2) Kart: {a} onaylandı, {k} içeriksiz atlandı')
" >> "$LOG" 2>&1

  # --- 3) Sentetik QA üret (2 makale, 6 chunk, 3 soru/chunk) ----------------
  log "3) synth-qa üretiliyor (2 makale × 6 chunk × 3 soru)..."
  timeout 3600 uv run achilles synth-qa \
    --per-chunk 3 --max-chunks 6 --max-papers 2 >> "$LOG" 2>&1 \
    || log "   synth-qa HATA/timeout"

  # --- 4) Dataset tazele (kart + synth-qa) -----------------------------------
  log "4) Dataset tazeleniyor..."
  uv run achilles lora-dataset >> "$LOG" 2>&1

  NEXAMPLES=$(wc -l < data/lora_sft/lora_sft.jsonl 2>/dev/null || echo 0)
  log "   Dataset: $NEXAMPLES örnek"

  # --- 5) MLX eğitimi (300 iter, Apple Silicon) ------------------------------
  ADAPTER="mac_loop_v${round}"
  log "5) MLX eğitimi: adapter=$ADAPTER iters=300"
  # train_status.json → web UI rozeti için adapter adını yaz
  printf '{"adapter":"%s","running":true}' "$ADAPTER" > storage/train_status.json
  # stderr → train-full-err.log (web UI canlı ilerleme buradan okur) + mac-loop.log
  timeout 7200 uv run achilles train --run \
    --adapter-name "$ADAPTER" \
    --iterations 300 \
    --batch-size 2 \
    --num-layers 8 \
    --backend mlx \
    >> "$LOG" 2> >(tee -a logs/train-full-err.log >> "$LOG") \
    && log "   Eğitim OK: $ADAPTER" \
    || log "   Eğitim HATA/timeout"
  # eğitim bitti → rozeti temizle
  printf '{"adapter":"%s","running":false}' "$ADAPTER" > storage/train_status.json

  log "──── TUR $round bitti — 5dk dinlenme ────"
  for i in $(seq 1 60); do
    [ -f "$STOP" ] && break
    sleep 5
  done
done

log "=== MAC LOOP DURDU ($round tur tamamlandı) ==="
rm -f "$STOP"
