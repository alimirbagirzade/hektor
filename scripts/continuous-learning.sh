#!/usr/bin/env bash
# Achilles SÜREKLİ ÖĞRENME DÖNGÜSÜ — "trader uzmanı" protokolü.
#
#   ZENGİNLEŞTİR (arXiv: psikoloji/belirsizlik/felsefe/trading, dönüşümlü konu)
#     → KAVRA (kart üret + içerikli onayla + anlama skorla)
#     → SENTEZLE (her 3 turda: research hipotezi + sentez makalesi)
#     → EĞİT (LoRA, 2 koşu × 20 iter)
#     → tekrar.
#
# RAM disiplini: LLM fazları ile eğitim fazı ASLA çakışmaz (sıralı).
# Devralma: başka bir eğitim döngüsü çalışıyorsa STOP_TRAINING ile nazikçe
# durdurur, mevcut koşunun bitmesini bekler, sonra kendisi başlar.
# Durdurma: storage/STOP_LEARNING dosyası (tur sonunda temiz çıkar).
#
# Kullanım:  bash scripts/continuous-learning.sh [MAX_SAAT (vars. 72)]
set -u
cd "$(dirname "$0")/.." || exit 1
MAX_HOURS="${1:-72}"
LOG=logs/continuous-learning.log
STATE=storage/learning_topic_index
STOP=storage/STOP_LEARNING
mkdir -p logs storage
# uv, her `uv run` cagrisinda paketi yeniden senkronlamaya calisir; bu da calisan
# web sunucusunun kilitledigi achilles-web.exe'yi silmeye ugrasip "os error 32" ile
# patlar -> dongunun TUM adimlari sessizce coker (research/synth-qa HATA). Cozum:
# senkronu kapat; bagimliliklar zaten kurulu, dongu yalniz mevcut ortami kullanir.
export UV_NO_SYNC=1
log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== SÜREKLİ ÖĞRENME BAŞLADI (max ${MAX_HOURS}sa) ==="

# --- Devralma: eski eğitim döngüsünü nazikçe durdur -------------------------
# TAŞINABİLİR süreç kontrolü (eski sürüm gömülü `powershell Win32_Process`
# çağırıyordu → Linux/bulutta `powershell` yok, komut hep başarısız olup döngü
# her zaman ~90dk boş bekliyor, çakışma koruması bozuluyordu). Sıra:
#   1) pgrep  (Linux/macOS + Git Bash procps-ng)  2) powershell/WMI (yalnız Windows)
#   3) ikisi de yoksa "çalışmıyor" varsay (güvenli ilerle).
is_training_running() {
  if command -v pgrep >/dev/null 2>&1; then
    pgrep -f 'peft_lora_train' >/dev/null 2>&1 && return 0
    pgrep -f 'achilles.*train.*--run' >/dev/null 2>&1 && return 0
    return 1
  elif command -v powershell >/dev/null 2>&1; then
    powershell -NoProfile -Command \
      "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*peft_lora_train*' -or (\$_.CommandLine -like '*achilles*' -and \$_.CommandLine -like '*train*--run*') }) { exit 0 } else { exit 1 }" \
      >/dev/null 2>&1 && return 0
    return 1
  fi
  return 1
}
touch storage/STOP_TRAINING
log "Devralma: STOP_TRAINING bırakıldı; mevcut eğitim koşusunun bitmesi bekleniyor"
for i in $(seq 1 360); do  # max ~90 dk (uzun eğitim koşusunu da kapsar)
  if is_training_running; then sleep 15; else break; fi
done
# Eski döngünün cooldown'da STOP'u görüp temiz çıkması için ek bekleme
sleep 150
rm -f storage/STOP_TRAINING
log "Devralma tamam — döngü başlıyor"

END=$(( $(date +%s) + MAX_HOURS*3600 ))
round=0
while [ ! -f "$STOP" ] && [ "$(date +%s)" -lt "$END" ]; do
  round=$((round+1))
  log "──── TUR $round ────"

  # --- 1) ZENGİNLEŞTİR: KAPATILDI (2026-06-14, kullanıcı isteği) -----------
  # Sistem kendi kendine arxiv'den makale ÇEKMEZ. Makaleler YALNIZCA kullanıcının
  # web arayüzünden elle yüklediğiyle gelir. Döngü dış kaynak indirmez; yalnız
  # MEVCUT (kullanıcı-onaylı) makaleler üzerinde kart/kavrama/synth-qa yapar.
  log "1) Otomatik arxiv çekme DEVRE DIŞI (yalnız elle yükleme)"

  # --- 2) KAVRA: kartsızlara kart + içerikli onay + anlama skoru -----------
  PIDS=$(uv run python -c "
from app.memory.sqlite_store import SqliteStore
s = SqliteStore()
for p in s.list_papers():
    if not s.has_knowledge_card(p.paper_id):
        print(p.paper_id)
" 2>/dev/null)
  log "2) Kartsız makale: $(printf '%s' "$PIDS" | grep -c . || true)"
  for pid in $PIDS; do
    pid="${pid//$'\r'/}"   # Windows: python stdout CRLF -> pid'e takilan \r dosya adina sizip Errno 22 yapar
    [ -z "$pid" ] && continue
    timeout 900 uv run achilles card "$pid" >> "$LOG" 2>&1 || log "   kart HATA: $pid"
  done
  uv run python -c "
from app.memory.sqlite_store import SqliteStore
from app.research.rag_learning_loop import is_substantive_card
s = SqliteStore()
a = k = 0
for c in s.list_pending_cards():
    # GÜÇLÜ gate: dejenere '...'/'unknown' kartları onaylama (eski non-empty ölçütü
    # coverage'ı çöple şişiriyordu — v5-tipi risk). Anlamlı title≥8 + main_claim≥40.
    if is_substantive_card(c.get('card_json') or {}):
        s.approve_card(c['card_id']); a += 1
    else:
        k += 1
print(f'   onay: {a} onaylandı, {k} içeriksiz atlandı')
" >> "$LOG" 2>&1

  # İçerikli (örnek üreten) makalelerden skoru olmayanları skorla — LLM.
  PIDS2=$(uv run python -c "
from app.memory.sqlite_store import SqliteStore
from app.lora.dataset_builder import build_dataset
s = SqliteStore()
pids = {str(e.metadata.get('paper_id','')) for e in build_dataset(s.list_approved_cards())}
for pid in sorted(p for p in pids if p):
    if s.get_comprehension_score(pid) is None:
        print(pid)
" 2>/dev/null)
  N2=$(printf '%s' "$PIDS2" | grep -c . || true)
  log "   anlama skorlanacak: $N2"
  for pid in $PIDS2; do
    pid="${pid//$'\r'/}"   # CRLF -> \r temizligi (yukaridaki ile ayni Windows sebebi)
    [ -z "$pid" ] && continue
    RESP=$(curl -s -m 600 -X POST "http://127.0.0.1:8765/api/papers/$pid/comprehension" 2>&1)
    log "   anlama[$pid]: $(printf '%s' "$RESP" | head -c 160)"
  done

  # --- 3) SENTEZLE: her 3 turda hipotez + makale ----------------------------
  if [ $(( round % 3 )) -eq 1 ]; then
    log "3) Research + sentez makalesi"
    timeout 3600 uv run achilles research \
      "Markov zinciri / gizli Markov modeli (HMM) ile piyasa REJIM DEGISIMINI (trend / yatay / yuksek-volatilite gecisleri) modelleyen YENI bir trading indikatoru veya filtresi oner; davranissal yanlilik ve belirsizlik kavramlariyla birlestir; backtest ile test et (maliyet dahil, look-ahead yok)." \
      --iterations 1 >> "$LOG" 2>&1 || log "   research HATA/timeout"
    uv run achilles synth-paper >> "$LOG" 2>&1
  fi

  # --- 4) VERİ-ÜRET: kart dataset'i tazele + sentetik QA üret ---------------
  # DEĞİŞİKLİK (2026-06-13): Sürekli CPU-LoRA DURDURULDU. Kanıt: 4B CPU'da
  # ~76sn/adım = haftalar; 15-50 örnek overfit eder (anlamlı LoRA ~1000 örnek
  # ister). Döngü artık EĞİTMEZ; VERİ ÜRETİR (15→1000+ büyüme motoru). Eğitim,
  # ≥1000 örnek olunca bulut-GPU'da, açık --run ile yapılır (CLAUDE.md kural 8).
  # Detay: docs/RAG_EGITIM_YENIDEN_TASARIM.md
  # NOT (2026-06-14): 'lora-dataset' KALDIRILDI — her turda data/lora_sft/lora_sft.jsonl'i
  # (Stage 2 birleşik dataset, lora-cloud-prep üretir) kart-only veriyle EZİYORDU. Veri
  # synthetic_qa.jsonl'de birikir; birleşik dataset gerektiğinde lora-cloud-prep üretir.
  log "4) Sentetik QA üret (bu turun yeni makaleleri)"
  # En yeni 2 makaleyi sentetik QA'ya çevir; dosyaya BİRİKİR (append). CPU'da
  # ~100sn/chunk; 2 makale × 6 chunk × 3 QA ~ 20-25dk → 3600sn timeout'a rahat sığar.
  timeout 3600 uv run achilles synth-qa --per-chunk 3 --max-chunks 6 --max-papers 2 \
    >> "$LOG" 2>&1 || log "   synth-qa HATA/timeout (devam)"

  # Tur özeti (rag-mastery LLM'siz)
  uv run achilles rag-mastery >> "$LOG" 2>&1
  log "──── TUR $round bitti — 120sn dinlenme ────"
  for i in $(seq 1 24); do [ -f "$STOP" ] && break; sleep 5; done
done

log "SÜREKLİ ÖĞRENME DURDU ($round tur; STOP veya süre doldu)."
rm -f "$STOP"
