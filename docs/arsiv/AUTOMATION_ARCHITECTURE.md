# Achilles — Automation Architecture (Phase 0/1)

_Durum: Phase 0 (belge + manifest) + Phase 1 (runtime gözlemci) tamamlandı.
Phase 2 (task queue + approvals + supervisor) HENÜZ YOK._

Bu belge, Achilles'in mevcut **otonomi yüzeyini** ve **güvenlik sınırlarını** tanımlar.
Tek, bildirimsel kaynak: [`automation_manifest.yaml`](../automation_manifest.yaml)
(`app/agents/runtime/registry.py` ile okunur).

---

## 1. Mevcut agent'lar (runtime)

15 runtime-agent-benzeri bileşen denetimde (audit) bulundu ve manifest'e işlendi.
Tam alanlar için manifest'e bakın; özet:

| agent_id | otonomi | tehlikeli | onay gerekir | varsayılan açık |
|----------|---------|-----------|--------------|-----------------|
| auto-lora-pipeline | requires_approval | ✅ | ✅ | ❌ |
| rag-learning-loop | autonomous | ❌ | ❌ | ❌ (kullanıcı açar) |
| research-orchestrator | semi_auto | ❌ | ❌ | ❌ |
| rag-trend-scanner | semi_auto | ❌ | ❌ | ❌ |
| reflection-agent | manual | ❌ | ❌ | ❌ |
| paper-mastery-agent | semi_auto | ❌ | ❌ | ❌ |
| status-manager | manual | ❌ | ❌ | ❌ |
| lora-control-plane | semi_auto | ❌ | ❌ | ❌ |
| adapter-eval | semi_auto | ❌ | ❌ | ❌ |
| dataset-quality-gate | semi_auto | ❌ | ❌ | ❌ |
| tool-use-trainer | semi_auto | ❌ | ❌ | ❌ |
| auto-researcher | semi_auto | ❌ | ❌ | ❌ |
| arxiv-fetcher | autonomous | ❌ | ❌ | ❌ |
| rules-updater | requires_approval | ❌ | ✅ | ❌ |
| model-advisor | autonomous | ❌ | ❌ | ❌ |

`agent_id` listesi `uv run achilles agents-list` ile de görülebilir.

## 2. Mevcut döngüler (loops)

İki gerçek arka plan döngüsü (web sunucusu açılışında asyncio task olarak başlar ama
**içeride varsayılan KAPALI**):

- **rag-learning-loop** (`app/research/rag_learning_loop.py`) — 15s heartbeat;
  `interval_min` dolunca bir tur çalışır: arXiv çek → indeksle → kart → skor → ustalık.
  **LoRA eğitimi sürerken kendini DURAKLATIR** (`_training_running` → `paused_training`).
  Web'den `/api/rag-loop/enable` ile açılır. Durum: `storage/rag_learning_state.json`.
- **auto-lora-pipeline** (`app/lora/auto_pipeline.py`) — `check_interval_min`'de onaylı
  kart sayısını kontrol eder; eşik geçilirse Gate 0-8 çalışır → `READY_TO_TRAIN`.
  **Eğitim ve terfi insan onayı bekler.** Durum: `storage/auto_lora_state.json`.

Kabuk (shell) döngüleri (uygulama dışı, Windows Task Scheduler / elle): bkz. §6.

## 3. Mevcut güvenlik kapıları (safety gates)

- **CLAUDE.md sert kuralları** koda gömülü (denetimle doğrulandı): maliyet (commission+slippage),
  look-ahead `shift(1)`, `eval`/`exec` yasağı (whitelist AST), seed determinizmi,
  boş-retrieval'da dürüst "kaynak yok", `train --run` varsayılan KAPALI.
- **Gate 0-8** (`lora/control_plane.py`): Gate 7 safety scanner = BLOCKER.
- **pretrain-gate** (`training/dataset_quality.py`): garanti-vaadi / açılış-ezberi → NO-GO.
- **adapter-eval** (`training/adapter_eval.py`): gerçek base-vs-adapter; regresyon → reject; TERFİ ETMEZ.
- **rag-learning-loop**: eğitim sırasında duraklar; `is_substantive_card` içerik kapısı.
- **Phase 1 gözlemci**: ajan koşuları artık `agent_runs`/`agent_events` + JSONL'e kaydedilir
  (gözlem **davranışı değiştirmez**, hata fırlatmaz).

## 4. Şu an otomatik olan

- Makale çekme/indeksleme (idempotent), kart üretimi, comprehension/mastery skorlama,
  RAG öğrenme turu (kullanıcı açarsa), arXiv trend tarama (zamanlanırsa),
  haftalık **rapor-only** bug-scan, CI (ruff+mypy+pytest offline).
- Auto-LoRA **denetim** (Gate 0-8) otomatik; **eğitim/terfi DEĞİL**.

## 5. Şu an otomatikleştirilmesi YASAK olan

- Gerçek LoRA eğitiminin gözetimsiz başlatılması.
- Adapter'ın production'a otomatik terfisi.
- `data/`, `storage/`, `vector_db/`, `models/adapters/` üzerinde otomatik ajan değişikliği.
- `main`'e otomatik push / auto-merge.

## 6. 🔒 GÜVENLİK DONDURMA (Safety Freeze)

Aşağıdakiler **Phase 2 supervisor + approval gelene kadar GÖZETİMSİZ çalıştırılmamalıdır**:

| Öğe | Neden | Şimdilik kural |
|-----|-------|----------------|
| `uv run achilles train --run` | gerçek LoRA eğitimi (geri alınması pahalı; v5 regresyonu) | her koşu **elle, ayrı onay** |
| `scripts/train-loop.ps1` | 24s döngüde tekrar tekrar `train --run` | elle başlat, denetimli; loop'ta bırakma |
| `scripts/mac-loop.sh` | MLX `train --run` her turda | elle, denetimli |
| `scripts/auto-chain.sh` | zincir sonunda 24s eğitim döngüsü | elle, denetimli |
| adapter promotion (`/api/auto-lora/promote`) | base'i değiştirir | yalnız EVAL_PASSED + **insan onayı** |

Phase 1 `train --run` çalıştırıldığında konsola bu uyarıyı basar (davranışı değiştirmez).

### Neden `train --run` manuel approval gerektirmeli?

1. **Geri alınması pahalı.** Eğitim saatler sürer (v5: 46.75 saat CPU) ve çıktı
   `models/adapters/` altına yazılır; kötü bir koşu zaman + disk + (terfi edilirse) kaliteyi yer.
2. **v5 dersi.** Otomatik kalite sinyalleri yanılabilir — v5 "başarıyla eğitildi" ama
   disiplinde GERİLEDİ; eval harness adapter'ı yüklemiyordu. Otomatik "geçti" güvenilmez.
3. **CLAUDE.md Kural 2 + 8.** "Test edilmeden başarılı deme" ve "otomatik ağır eğitim yok".
   CI offline testleri eğitim kalitesini KANITLAYAMAZ; bu yüzden insan kapısı şart.
4. **Kaynak çakışması.** Eğitim ~7GB RAM ister; LLM/RAG ile çakışır. Zamanlamayı insan görmeli.

Phase 2'de bu, supervisor + `approval_requests` ile **zorunlu** hale geldi (aşağıya bakın);
Phase 1'de yalnız **belge + uyarı** düzeyindeydi.

---

## 7. Phase 2 — otomasyon freni (task queue + approvals + supervisor)

Phase 2 "gerçek otomasyon frenini" kurar. **Tam otonomi HÂLÂ aktif değildir**; tehlikeli
her aksiyon insan onayı arkasındadır. Bu fazda web dashboard UI YOK (Phase 3), GitHub
Claude PR otomasyonu KAPALI (Phase 4). Yalnız backend + CLI + API + test.

### 7.1 Task queue (`app/agents/runtime/task_queue.py`)
- SQLite `automation_tasks` tablosu. Durumlar: pending, claimed, running, completed,
  failed, cancelled, **blocked_approval**, **blocked_stop_all**.
- Fonksiyonlar: `create_task / list_tasks / get_task / claim_task / complete_task /
  fail_task / cancel_task` (+ supervisor için `mark_blocked`).
- **Windows Task Scheduler DIŞ cron olarak KALIR** (kullanıcı kararı): bu kuyruk görevleri
  yalnız KAYIT + DURUM olarak izler; zamanlama henüz app içine taşınmadı. (Mevcut dış
  görevler: `Achilles-WeeklyBugScan`, `AchillesWeb`, `AchillesUpdate`, `Achilles-RAG-*`.)

### 7.2 Approval system (`app/agents/runtime/approvals.py`)
- SQLite `approval_requests` tablosu (+ `consumed_at`). Risk: low/medium/high/critical.
  Durum: pending/approved/rejected/expired/cancelled.
- **TEK KULLANIMLIK taze onay** — `require_fresh_approval`: onaylı + tüketilmemiş onay
  varsa tüketir ve yetki verir; yoksa yeni pending oluşturur ve yetki VERMEZ.
  **Standing / kalıcı yetki YOK** — onaylanan istek bir aksiyonda tüketilince biter.

### 7.3 Supervisor (`app/agents/runtime/supervisor.py`)
- Tehlikeli ajan çalıştırma için TEK kapı: registry → görev iptal → STOP_ALL → zaten
  çalışıyor → taze onay. `can_run_agent` (salt kontrol) + `run_with_supervision` (onay
  tüketir, tracker koşusu içinde çalıştırır).

### 7.4 STOP_ALL küresel kill-switch
- `storage/STOP_ALL` dosyası varken **hiçbir tehlikeli aksiyon** çalışmaz; **salt-okuma
  ajanlar (örn. model-advisor) STOP_ALL altında bile çalışır**. CLI: `stop-all` /
  `clear-stop-all`; web: `/api/supervisor/{stop-all,clear-stop-all,status}`.

### 7.5 Detached training stop fix
- Eski açık: `/api/training/stop` yalnız in-process `TrainingManager`'ı durduruyordu;
  detached koşu sürüyordu. Düzeltme: `detached_launch.request_stop_detached_training` —
  `storage/train_status.json`'dan **pid** okur (artık `launch` pid yazar), süreci (+çocuk
  süreçleri) `psutil` ile sonlandırır, `storage/STOP_TRAINING` bırakır; süreç yoksa hata
  vermez, `stop_requested` döner. Windows/Linux/macOS uyumlu. `/api/training/stop` artık
  hem in-process hem detached'i durdurur.

### 7.6 Startup sweep
- Web açılışında `cancel_stale_running_agent_runs()` — crash sonrası `status='running'`
  kalan ESKİ (varsayılan >6 saat) koşuları `cancelled` yapar (canlı eşzamanlı koşuları
  yanlışlıkla iptal etmemek için yaş eşiği). Phase 1'deki orphan-run açığını kapatır.

### 7.7 Tehlikeli aksiyon politikası (ŞİMDİ ZORLANIYOR)
| Aksiyon | Kapı | Davranış |
|---------|------|----------|
| `achilles train --run` (manuel) | STOP_ALL + taze onay (`lora-trainer`/`train_run`, critical) | Onay yoksa eğitim BAŞLAMAZ; approval_id basılır, exit 3 |
| auto-lora `start_training` | STOP_ALL + taze onay (`auto-lora-pipeline`/`auto_lora_start_training`, critical) | Onay yoksa `needs_approval` döner |
| auto-lora `promote_to_production` | STOP_ALL + taze onay (`.../auto_lora_promote_adapter`, high) | Onay yoksa `needs_approval` döner |
| `rules-update --approve` | STOP_ALL + taze onay (`rules-updater`/`rules_apply`, medium) | Onay yoksa uygulanmaz, exit 3 |
| **`POST /api/training/run` (web)** | STOP_ALL + taze onay (`lora-trainer`/`train_run`, critical) | **[Phase 4D-1]** Onay yoksa `needs_approval` + `approval_id` + onay komutu döner; eğitim BAŞLAMAZ |

`launch()`'ın taze onayı **ÜST katmanda** alınır: CLI manuel yol kendi kapısından,
auto-lora `start_training` kendi onayından **ve web `/api/training/run` (Phase 4D-1) artık
`require_fresh_approval` ile** geçer. Onaylı çağrı `launch()`'a girer; spawn edilen iç
`achilles train --run`'a `ACHILLES_TRAIN_SUPERVISED=1` verilir → **çift onay olmaz**; ama
STOP_ALL iç komutta da geçerlidir. Manuel `achilles train --run` bu env'i ALMAZ → onay ister.

> **Web training start is now protected by the same fresh manual approval model as CLI training.**
> (Phase 4D-1) Eskiden `/api/training/run` doğrudan `launch()` çağırıp onayı atlıyordu;
> artık STOP_ALL + tek-kullanımlık taze onay zorunlu. EĞİTİM tabındaki başlat butonu da
> `confirm()` ister ve `needs_approval` yanıtında CLI onay komutunu gösterir.

### 7.8 Net durum (Phase 2 sonrası)
- **Tam otomasyon HÂLÂ aktif değil.** Her gerçek eğitim AYRI manuel onay ister.
  Her adapter terfisi AYRI manuel onay ister. Standing yetki yok.
- **GitHub Claude PR otomasyonu Phase 4'e kadar KAPALI.**
- **Web dashboard Phase 3'te** yapılacak (bu fazda yalnız backend + CLI + API).
- Eğitim kabuk döngüleri (`train-loop.ps1`, `auto-chain.sh`, `mac-loop.sh`) artık iç
  `train --run` onay kapısına takılır → gözetimsiz tekrar eğitim KENDİLİĞİNDEN olmaz.

---

## 8. Phase 3 — Agent/Otomasyon Dashboard

Mevcut Achilles Web UI'na yeni bir sekme (**10 · AGENTS / OTOMASYON**) eklendi —
ayrı uygulama DEĞİL, mevcut `app/web/static/` (vanilla JS) içine. **Yeni backend yetenek
EKLENMEDİ**; yalnız Phase 1/2 endpoint'leri tüketilir.

**Dashboard ne gösterir:**
- **Supervisor/Sağlık** — STOP_ALL durumu, danger gate (her zaman aktif), bekleyen onay
  sayısı, çalışan koşu sayısı, son olay zamanı, `/healthz` durumu + büyük kırmızı STOP_ALL
  kill-switch.
- **Onay İstekleri** — pending önce; Approve/Reject (confirm zorunlu).
- **Agents** — 15 ajan; tehlikeli olanlar uyarı rozetli; tıkla → koşu filtrele.
- **Agent Koşuları** — run listesi; run_id → metadata + outputs + error + **olay zaman
  çizelgesi**.
- **Otomasyon Görevleri** — liste + iptal (confirm) + basit görev oluşturma.
- **Genel Olay Akışı** — son 100 olay; error/warning görsel ayrışır.

**Kullandığı endpoint'ler:** `GET /api/agents`, `/api/agents/runs`,
`/api/agents/runs/{id}`, `/api/automation/tasks` (GET/POST) + `/cancel`,
`/api/approvals` (GET) + `/approve` + `/reject`, `/api/events`, `/api/healthz`,
`/api/supervisor/status` + `/stop-all` + `/clear-stop-all`.

**Dashboard NE YAPMAZ:**
- Eğitim/terfi başlatmaz; yalnız görünürlük + onay + STOP_ALL kontrol yüzeyidir.
- Yeni tehlikeli backend yeteneği eklemez.
- **GitHub Claude PR otomasyonu Phase 4'e kadar KAPALI.**
- training/promotion HÂLÂ manuel TAZE onay ister (dashboard'dan approve dahi tek
  kullanımlıktır; standing yetki yok).

**Güvenlik:** mevcut `api_auth` token akışı korunur; tüm dinamik içerik `esc()` ile
kaçırılır (XSS); STOP_ALL/approve/reject/cancel `confirm()` olmadan çalışmaz; satır
butonları CSP-safe olay-delegasyonuyla bağlanır (inline `onclick` yok).

**Bilinen sınır (raporlandı, backend EKLENMEDİ):** `POST /api/automation/tasks`
query-param tabanlı → create formunda `params_json` yok; `/healthz` kaba (granüler
sqlite/runtime bayrağı yok); approval `decision_note` UI'da girilmiyor. İleride istenirse
küçük read-only zenginleştirme — bu fazda backend büyütülmedi.
