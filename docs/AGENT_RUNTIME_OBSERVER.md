# Achilles — Agent Runtime Observer (tasarım + Phase 1 durumu)

Hedef: "hangi ajan, ne zaman başladı, ne yaptı, nerede düştü, hangi run_id" sorularını
yanıtlayan **gözlemlenebilir ve güvenli** bir temel. Bu belge tüm tasarımı anlatır ve
hangi parçanın **şimdi (Phase 1)** geldiğini, hangisinin **ertelendiğini (Phase 2+)** işaretler.

```
app/agents/runtime/
├── __init__.py     [Phase 1] public API
├── schemas.py      [Phase 1] AgentSpec / AgentRun / AgentEvent + enum'lar
├── registry.py     [Phase 1] automation_manifest.yaml okuyucu/sorgulayıcı
├── tracker.py      [Phase 1] koşu + olay kaydedici (SQLite + JSONL), @tracked
├── task_queue.py   [Phase 2] ❌ henüz yok
├── approvals.py    [Phase 2] ❌ henüz yok
└── supervisor.py   [Phase 2] ❌ henüz yok
```

## 1. Registry  ✅ Phase 1

`automation_manifest.yaml` → `dict[str, AgentSpec]`. Bozuk/eksik YAML'da **açık**
`ManifestError`. API: `load_agent_registry()`, `list_agents()`, `get_agent(id)`,
`dangerous_agents()`, `agents_requiring_approval()`.
Bu, ajanların tek bildirimsel kaynağıdır (kod değil, veri).

## 2. Tracker + run_id  ✅ Phase 1

Her ajan koşusu `start_run → log_event* → finish_run` yaşam döngüsünden geçer.
- `run_id` formatı: `arun_YYYYMMDD_HHMMSS_<8hex>` (benzersiz + okunur).
- Çift yazım: SQLite (`agent_runs`, `agent_events`) **ve** `reports/agent_runs/<run_id>.jsonl`.
- **Davranış değiştirmez, hata fırlatmaz**: tracking hatası sessizce yutulur, ajan akışı sürer.
- `{"ok": False}` dönen ajanlar (rag-loop, auto-lora) `failed` olarak işaretlenir.

Sarmalama yolları:
- `@tracked("agent-id", trigger_type=...)` — async-farkında dekoratör (hafif; mevcut 4 ajanda).
- `with track_agent_run("agent-id") as run:` — elle blok sarmalama.
- `log_step("...")` — mevcut (context) koşuya ara-adım olayı.

**Phase 1'de sarılan ajanlar:** `rag-learning-loop` (run_one_cycle), `auto-lora-pipeline`
(check_and_prepare), `research-orchestrator` (run), `arxiv-fetcher` (fetch_arxiv_papers).

## 3. Event log  ✅ Phase 1

`AgentEvent{event_id, run_id, ts, kind, level, message, payload}`. `kind ∈ {start, step,
info, output, warning, error, finish}`. SQLite + JSONL'e yazılır.
**Retention (kullanıcı kararı): 30 gün VEYA en çok 50.000 son olay** —
`SqliteStore.prune_agent_events` her `finish_run`'da (savunmacı) çağrılır.
JSONL dosyaları şimdilik budanmaz (Phase 2 notu).

## 4. Görüntüleme  ✅ Phase 1 (salt-okuma)

- CLI: `achilles agents-list`, `agents-runs [--limit --agent --status]`, `agents-log <run_id>`.
- Web (salt-okuma): `GET /api/agents`, `GET /api/agents/runs`, `GET /api/agents/runs/{run_id}`.
- Dashboard: mevcut Achilles Web UI'a **ileride** "Agents" sekmesi (ayrı uygulama YOK — kullanıcı kararı).

## 5. Task queue  ❌ Phase 2

`automation_tasks` tablosu + enqueue/claim/complete. Windows Task Scheduler şimdilik
dış cron olarak kalır; Achilles önce bu görevleri **izler**, sonra zamanlamayı içeri taşır.

## 6. Approvals  ❌ Phase 2

`approval_requests` tablosu + `request/approve/reject`. Tehlikeli adımlar (gerçek eğitim,
adapter terfisi, rules-updater uygula) **bloke** edilip insan onayı bekler.
Kullanıcı kararı: **`train --run` her zaman ayrı manuel onay** (otomatik tam-otonomi YOK).

## 7. Supervisor  ❌ Phase 2

Tehlikeli ajanları başlatabilen TEK nokta; `approval_requests` + global `STOP_ALL`
kill-switch'i uygular. Ayrıca **detached eğitim durdurma** açığını giderir
(`/api/training/stop` şu an detached koşuyu durduramıyor).

## 8. GitHub + Claude Code automation  ❌ sonraki faz (Phase 4)

Şimdilik KAPALI (kullanıcı kararı: önce runtime'ı lokalde kanıtla). İleride:
etiketli issue → Claude Code branch → CI kapısı → PR (asla `main`'e push / auto-merge yok,
`data/storage/models` asla dokunulmaz). CI (`ci.yml`) zaten mevcut ve test kapısı olarak kullanılır.

---

### Phase 1 kapsam özeti

| Bileşen | Durum |
|---------|-------|
| schemas / registry / tracker | ✅ yapıldı (Phase 1) |
| SQLite `agent_runs` + `agent_events` (+ retention) | ✅ yapıldı (Phase 1) |
| 4 ajan hafif sarmalama (davranış değişmeden) | ✅ yapıldı (Phase 1) |
| CLI agents-list/runs/log | ✅ yapıldı (Phase 1) |
| Web GET /api/agents[/runs[/{id}]] | ✅ yapıldı (Phase 1) |
| task_queue (`automation_tasks`) | ✅ yapıldı (Phase 2) |
| approvals (`approval_requests`, tek kullanımlık taze onay) | ✅ yapıldı (Phase 2) |
| supervisor + STOP_ALL kill-switch | ✅ yapıldı (Phase 2) |
| detached training stop fix (pid + STOP_TRAINING) | ✅ yapıldı (Phase 2) |
| startup sweep (bayat running → cancelled) | ✅ yapıldı (Phase 2) |
| tehlikeli aksiyon entegrasyonu (train/promote/rules) | ✅ yapıldı (Phase 2) |
| CLI task/approval/stop-all + web /automation,/approvals,/events,/healthz,/supervisor | ✅ yapıldı (Phase 2) |
| dashboard sekmesi (mevcut Web UI içinde) | ✅ yapıldı (Phase 3) |
| GitHub Claude Code PR automation | ❌ Phase 4 |

## Phase 2 API yüzeyi (özet)

**Yeni SQLite tabloları:** `automation_tasks`, `approval_requests`.

**Yeni CLI:** `task-create`, `tasks-list`, `task-cancel`, `approvals-list`,
`approval-approve`, `approval-reject`, `stop-all`, `clear-stop-all`.

**Yeni web (api_auth korumalı):** `GET/POST /api/automation/tasks`,
`POST /api/automation/tasks/{id}/cancel`, `GET /api/approvals`,
`POST /api/approvals/{id}/approve|reject`, `GET /api/events`, `GET /api/healthz`,
`POST /api/supervisor/stop-all`, `POST /api/supervisor/clear-stop-all`,
`GET /api/supervisor/status` + `/api/training/stop` artık detached'i de durdurur.

**Politika:** her gerçek eğitim ve her adapter terfisi AYRI manuel onay ister
(tek kullanımlık taze onay; standing yetki yok). STOP_ALL tüm tehlikeli aksiyonları bloklar.
Tam otomasyon hâlâ kapalı; GitHub PR otomasyonu Phase 4'e kadar kapalı.

## Phase 3 — Agent/Otomasyon Dashboard (özet)

Mevcut Web UI'ye **10 · AGENTS / OTOMASYON** sekmesi eklendi (vanilla JS; ayrı uygulama
değil). Paneller: Supervisor/Sağlık (+ büyük kırmızı STOP_ALL), Onay İstekleri
(Approve/Reject), Agents (tehlikeli rozet + filtre), Agent Koşuları (+ olay zaman çizelgesi),
Otomasyon Görevleri (+ iptal/oluştur), Genel Olay Akışı.

**Yalnız Phase 1/2 endpoint'lerini tüketir; yeni backend yetenek EKLENMEDİ.** Mevcut
`api_auth` token akışı korunur; tüm dinamik içerik `esc()` ile kaçırılır; tehlikeli
butonlar `confirm()` ister; satır butonları CSP-safe olay-delegasyonuyla bağlanır.

**Değişen dosyalar:** `app/web/static/index.html`, `app/web/static/assets/app.js`,
`app/web/static/assets/app.css`, `tests/test_agent_dashboard_static.py`.

**Dashboard NE YAPMAZ:** eğitim/terfi başlatmaz; yeni tehlikeli yetenek eklemez;
GitHub PR otomasyonu (Phase 4) kapalı; training/promotion hâlâ manuel taze onay ister.
Dashboard yalnız **görünürlük + onay + STOP_ALL kontrol yüzeyidir**.

**Bilinen sınır:** create form query-param tabanlı (`params_json` yok); `/healthz` kaba;
approval `decision_note` UI'da yok. İleride küçük read-only zenginleştirme.

## Phase 4D-1 — Web training gate fix

**Web training start is now protected by the same fresh manual approval model as CLI training.**

4D-0 denetimi bir bypass buldu: `POST /api/training/run` doğrudan `launch()` çağırıyor,
`launch()` ise spawn ettiği `train --run`'a `ACHILLES_TRAIN_SUPERVISED=1` veriyordu →
CLI'daki tek-kullanımlık taze onay kapısı atlanıyordu (STOP_ALL yine de geçerliydi ama
fresh manual approval yoktu; EĞİTİM tabında confirm de yoktu).

Düzeltme: endpoint artık CLI ile **aynı anahtarı** (`lora-trainer`/`train_run`, critical)
kullanarak `require_fresh_approval` çağırır. Onay yoksa eğitim BAŞLAMAZ; `needs_approval`
+ `approval_id` + `uv run achilles approval-approve <id>` döner. Onaylanıp istek
tekrarlanınca taze onay TÜKETİLİR ve `launch()` çağrılır. EĞİTİM tabındaki başlat butonu
`confirm()` ister ve `needs_approval` yanıtında onay komutunu gösterir. STOP_ALL hâlâ
hem endpoint'te hem spawn edilen alt süreçte geçerlidir.

**Değişen dosyalar:** `app/web/server.py`, `app/web/schemas.py`,
`app/web/static/assets/app.js`, `tests/test_web_training_gate.py`,
`tests/test_agent_dashboard_static.py`.

## CI mypy parity note (Phase 3.5)

`mypy app`, eğitim backend'i (`torch`/`peft`/`transformers` = opsiyonel `train-cpu`
extra'sı) KURULU OLMASA BİLE temiz geçmelidir — CI `uv sync --extra dev` ile çalışır.

Önceden `peft_lora_train.py` ve `adapter_eval.py` içindeki `# type: ignore` satırları
yalnız torch KURULUYKEN "kullanılıyordu"; torch'suz ortamda mypy bunları
`Unused "type: ignore"` (`warn_unused_ignores`) olarak işaretliyordu → CI mypy kırılırdı.

**Çözüm (yalnız typing):** PEFT ile yeniden-atanan `model` değişkeni ilk bağlandığı yerde
`model: Any` olarak işaretlendi (yerel anotasyon — runtime'da değerlendirilmez,
`from __future__ import annotations`). Böylece `get_peft_model` / `print_trainable_parameters`
/ `model=model` satırları torch'lu ve torch'suz ortamların İKİSİNDE de hata vermez ve
hiçbir `# type: ignore` gerekmez. **Runtime davranışı ve eğitim mantığı DEĞİŞMEDİ;
eğitim backend'i opsiyonel bağımlılık olarak kalır.** Phase 4 GitHub automation buna dayanır.
