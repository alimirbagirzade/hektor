# Achilles — Eğitim Protokolü (Windows / yerel)

_Bu makineye özel ölçülmüş değerlerle. Son güncelleme: 2026-06-14._

Bu belge **bu bilgisayarda** Achilles LoRA eğitiminin nasıl çalıştığını,
donanım/model özelliklerini ve **ölçülmüş eğitim sürelerini** içerir.

> İlke (CLAUDE.md): Gerçek ağır eğitim yalnızca açık `--run` ile başlar.
> Çıktı `verdict != pass` ise "aday"dır, "hazır" değildir.

---

## 1. Bilgisayar özellikleri (bu makine)

| Bileşen | Değer |
|---|---|
| CPU | Intel Core **i7-1165G7** (11. nesil, Tiger Lake) |
| Çekirdek | **4 fiziksel / 8 mantıksal**, ~2.8 GHz baz |
| SIMD | AVX-512 (CPU çıkarım/eğitimi hızlandırır) |
| GPU | **YOK** — Intel Iris Xe (entegre); CUDA yok → her şey CPU'da |
| RAM | **31.7 GB** |
| Disk (C:) | ~209 GB boş |
| OS | Windows 10 Pro 19045 |
| Python / torch | 3.12 · **torch 2.12.0+cpu** · transformers 5.12.0 |

**Sonuç:** Bu bir laptop CPU'su. Çıkarım rahat; **eğitim CPU'da yavaştır**
(adım başına ~dakika). GPU olmadığı için 4B üstü model lokal eğitilemez.

---

## 2. Model — tek beyin: **Qwen3-4B**

RAG çıkarımı ve LoRA eğitimi **aynı** modeldir (tek beyin mimarisi).

| Kullanım | Model | Kaynak | Boyut |
|---|---|---|---|
| RAG / çıkarım | `qwen3:4b` | Ollama (GGUF) | ~2.5 GB |
| LoRA eğitim base | `Qwen/Qwen3-4B` | HuggingFace (safetensors) | ~8 GB |

- LoRA adapter **base-model'e özeldir** → eğitim ve çıkarım aynı model olmalı.
- Daha büyük beyin (9B / 30B / 120B) → **Colab/GPU**'da eğitilir, aynı pipeline.
- Ayar: `.env → ACHILLES_LLM_MODEL=qwen3:4b`, `ACHILLES_PEFT_BASE_MODEL=Qwen/Qwen3-4B`.

### LoRA hiperparametreleri (varsayılan)
`r=8`, `alpha=16`, `dropout=0.05`, `lr=2e-4`, `max_seq_length=512`, `batch_size=1`.
Eğitilebilir parametre: **~2.95M / 4.03B (%0.073)**.

---

## 3. Eğitim süreçleri (pipeline)

```
Onaylı kartlar → Gate 0-8 denetim → dataset + train/valid split
   → LoRA eğitim (PEFT/CPU) → eval → adapter registry → (onayla) promote
```

| Gate | Ad | Kontrol |
|---|---|---|
| 0 | source | onaylı RAG verisi mi |
| 1 | schema | JSONL format |
| 2 | curriculum | seviye atanmış mı |
| 3 | domain | en az bir alan |
| 4 | quality | kısa/tekrar/duplicate |
| 5 | math | hesap doğruluğu |
| 6 | philosophy | mantık tutarlılığı |
| 7 | safety | gizli veri (BLOCKER) |
| 8 | split | train/valid/test sızıntısı |

**Backend otomatik seçilir** (`detect_lora_backend`): macOS ARM64 → MLX,
Windows/Linux → **PEFT** (torch+peft).

---

## 3.5 Eğitim BAŞLATMA — DETACHED + tek-tık (2026-06-14)

**Eğitim artık web/terminal kapansa da sürer** (DETACHED süreç; PC + oturum açık
kaldıkça). Eski in-process yol (`training_manager`) web çökünce/yeniden başlayınca
eğitimi öldürüyordu (`auto_lora_state`'teki "Eğitim COMPLETED olmadı" bundandı).
Yeni yol: **`app/training/detached_launch.py`** (`launch()` / `training_status()` /
`readiness()`).

### Veri akışı — TEK doğru kaynak
```
data/lora_sft/lora_sft.jsonl   (synth-qa + kart birleşik, ~1266 örnek; lora-cloud-prep üretir)
        │   achilles lora-split   (seed=42, valid_ratio=0.05 → train≈1203 + valid≈63)
        ▼
data/training/jsonl/{train,valid}.jsonl   ← "achilles train --run" BUNU okur
```
- ⚠️ **CLOBBER tuzağı (kapatıldı):** `DatasetBuilder.build()` (kart-DB tabanlı, cılız
  `{prompt,completion}`) aynı `train.jsonl`'e yazıp zengin birleşik veriyi ezerdi
  (DB'de uygun örnek yoksa **0/cılız satıra**). Web uçları (`/api/training/dataset`,
  `dry-run`, `colab-notebook`) artık `detached_launch.build_training_split()` ile
  KANONİK `lora_sft.jsonl`'den böler → CLI ile aynı format/sayı (iki-hat drifti yok).
  `DatasetBuilder` yalnız manuel `achilles dataset` (SQLite inceleme) için kaldı.
- **Clobber-proof:** hem `launch()` hem `start-train.ps1`, başlatmadan ÖNCE `lora-split`
  çalıştırır → boş/eski `train.jsonl` otomatik onarılır.

### Başlatma yolları (üçü de DETACHED + clobber-proof)
1. **Web tek-tık (önerilen):** veri hazır olunca üst-bar'da **▶ EĞİTİME HAZIR (N örnek)
   — BAŞLAT** rozeti belirir → tıkla / Tab+Enter → onayla. Kural 8 korunur (yalnız açık
   kullanıcı eylemi; otomatik başlama YOK).
2. **Web · 05 EĞİTİM sekmesi:** "▶ EĞİTİMİ BAŞLAT" butonu (`POST /api/training/run`).
3. **Terminal:** `.\scripts\start-train.ps1` — durdur `-Stop`, durum `-Status`.

`--iterations` verilmezse **1 epoch** (= train örnek sayısı; 1266 veri → 1203 adım).
İlerleme: üst-bar rozeti (15 sn) · `GET /api/training/live` · `logs/train-full-err.log`.
**dtype varsayılan bf16** (~8 GB): web/Ollama ile BİRLİKTE çalışabilsin diye (fp32 ~16 GB
ve daha hızlı ama tek başına; bkz. §4 dtype analizi).

### Güvenlik rayları (2026-06-14 çok-ajanlı düşmanca inceleme sonrası)
- **Atomik kilit** (`storage/.training_launching`, `O_EXCL`): çift-tık / eş-zamanlı istek
  iki eğitim süreci başlatamaz (log yazılmadan önceki yarış penceresi kapatıldı).
- **adapter_name doğrulama** (`^[A-Za-z0-9_-]{1,64}$`): path-traversal / argüman güvenliği.
- `Popen` try/except + mesaj "başlatıldı" (Kural 2: doğrulanmadan "çalışıyor" denmez).
- `auto_pipeline.start_training` (web "Aç" oto-modu) da detached'e taşındı; bitişi
  `training_status()` poll'u ile algılar → eval.

---

## 4. Ölçülmüş eğitim süreleri (bu makine)

Gerçek koşudan ölçülen **adım başına süre** (CPU fp32, batch=1, seq=512):

| Config | Qwen3-4B adım başına | Tam koşu (6 adım) |
|---|---|---|
| İlk (pad→512, eval açık) | ~85 sn/adım | 513.7 sn |
| **Optimize (dinamik padding, eval kapalı)** | **~40.6 sn/adım** | **243.6 sn** |

Hız optimizasyonları (CPU): **dinamik padding** (512'ye doldurma yok → her adım yalnız
gerçek token; batch=1 ile sıfır padding), **eval kapalı** (epoch başına ~35 sn tasarruf),
tek checkpoint, `pin_memory=False`. Sonuç: **~2× hızlanma**.
(Qwen2.5-1.5B referans: ~29 sn/adım.)

### CPU hız analizi — neler işe YARAMADI (ölçüldü, bu makine: i7-1165G7)
Per-adım ~76 sn'nin asıl sebebi: **4B fp32 (16GB) bellek-bağımlı** (örnekler kısa,
medyan 253 token → seq sorun değil). Denenenler:
- **IPEX (Intel oneDNN):** Windows'ta **wheel YOK** (yalnız Linux). Colab/Linux'ta çalışır.
- **bf16 (`ACHILLES_TRAIN_DTYPE=bf16`):** Tiger Lake'te AVX512-BF16 olmadığından **emüle**;
  ölçüldü **95–117 sn/adım — fp32'den YAVAŞ**. (Sapphire Rapids gibi BF16'lı CPU'da işe yarar.)
- **Thread:** torch zaten 4 (fiziksel çekirdek) — optimal.

**İşe YARAYAN:** döngü **iters 40→20** (2× hızlı tur + 15 örnekte daha az overfit; loss zaten
~adım 20'de düşüyor). **Gerçek hız** için: GPU/Colab (IPEX+fp16/bf16 orada çalışır) veya
döngüde küçük model (Qwen2.5-1.5B ~29 sn/adım). Bu Windows CPU'da 4B per-adım tabanı ~76 sn.

**Toplam süre formülü:** `süre ≈ iterations × adım_süresi`
(çünkü toplam adım ≈ `iterations`).

### Qwen3-4B — iterasyona göre tahmini süre (optimize config, ~40 sn/adım)

| `--iterations` | Süre | Not |
|---|---|---|
| 30 | **~20 dk** | hızlı deneme |
| 60 | **~40 dk** | döngü başına önerilen |
| 100 | **~67 dk** | |
| 300 (varsayılan) | **~3.3 saat** | tam koşu |

> **Maksimum pratik süre:** varsayılan `iters=300` → **~3.3 saat** (optimize). Daha büyük
> `iters` orantılı artar ama küçük veri setinde anlamsızdır (overfit). 1.5B ~%30 daha hızlı.
>
> **24 saatlik döngü:** `scripts/train-loop.ps1` (iters=40, 2 dk cooldown) → ~24 saatte
> ~30+ döngü. NOT: veri tavanı (5 örnek) nedeniyle her döngü aynı sonuca yakınsar —
> hız öğrenmeyi artırmaz, sadece overfit'i hızlandırır. Asıl çözüm: veri setini büyütmek.

---

## 5. Periyodik / sürekli eğitim

Sabit "her saat" yerine **makine kapasitesine** göre:

```
[eğitim döngüsü] → [2-3 dk dinlenme] → [eğitim döngüsü] → ...
```

- CPU sürekli %100'de kalmasın diye döngüler arası **2-3 dk cooldown**.
- Yeni onaylı kart geldikçe dataset büyür → her döngü daha anlamlı.
- Auto-pipeline: `.env → ACHILLES_AUTO_LORA_*` (varsayılan kapalı, güvenli).

---

## 5.5 Eğitim ESNASINDA (downtime) — bu süre boşa geçmesin

CPU eğitimi **saatler/günler** sürer. Bu uzun pencerede sistem boş beklememeli;
**eğitimi BOZMADAN** şu kalite/araştırma işleri yapılır:

**Yapılacaklar (öncelik sırasıyla):**
1. **Çok-ajanlı bug avı / kod denetimi** — özellikle **anayasa-kritik** eksenler:
   look-ahead bias (Kural 4), maliyet=komisyon+slippage (Kural 3), determinizm/seed
   (Kural 6), `eval`/`exec` yokluğu (Kural 5). Hem web (`app/web/*`) hem çekirdek
   (`app/trading/*` backtest/strateji/indikatör/risk, `app/memory/*` RAG retrieval).
2. **Bulguları düzelt** → `make format && lint && typecheck` + **hedefli** test.
3. **Doküman/protokol** iyileştir (README, bu dosya, SECURITY.md) — son kullanıcı gözüyle.
4. **Post-training hazırlık**: eval setlerini (`evals/*.jsonl`) gözden geçir, adapter
   değerlendirme/promote protokolünü hazırla (eğitim bitince hemen çalıştırmak için).

**KISITLAR (eğitim sürerken ihlal etme):**
- **RAM dar** (4B eğitimi ~16 GB). Ağır yerel iş YOK: LLM sorgusu, **full `pytest`**,
  embedding, ikinci eğitim. Ajanlar **salt-okunur** (Read/Grep); doğrulama **hedefli**
  (değişen dosyalar) + her adımda boş RAM kontrolü.
- **OOM riski** (boş RAM < 2 GB): Ollama modelini boşalt (`ollama stop <model>` — Ollama'yı
  öldürmez, sorgu gelince yeniden yükler), eğitimi önceliklendir.
- **Eğitim süreçlerine DOKUNMA.** Web yeniden başlatmak güvenlidir (ayrı süreç, detached
  eğitimi öldürmez) ama `achilles train --run` süreçlerini durdurma.
- Değişiklikler küçük commit'lerle; ağır/riskli refactor eğitim bitince.

**Sağlık nöbeti (periyodik):** `logs/train-full-err.log` ilerliyor mu (≤45 dk tazelik),
boş RAM > 2 GB, web (8765) + Ollama ayakta. Müdahale edersen ne yaptığını **logla**.

---

## 6. Komutlar

```bash
uv run achilles lora-status          # genel durum
uv run achilles lora-audit           # Gate 0-7
# SENTETİK veri yolu (önerilen, ~1266 örnek):
uv run achilles lora-cloud-prep      # synth-qa + kart → data/lora_sft/lora_sft.jsonl
uv run achilles lora-split           # lora_sft.jsonl → data/training/jsonl/{train,valid}
# Eğitim (detached başlatıcılar lora-split'i zaten otomatik çalıştırır):
uv run achilles train --run --backend peft \
    --adapter-name <ad> --iterations <n>   # gerçek eğitim (CPU); iters yoksa 1 epoch
.\scripts\start-train.ps1            # DETACHED (önerilen; kapansa da sürer)
uv run achilles lora-registry        # adapter listesi
```

> ⚠️ `lora-dataset` (kart-DB → JSONL) `train.jsonl`'i **EZER**; sentetik veri yolunda
> kullanma — sentetik 1266 örneği `lora-split` köprüler. Detayı §3.5.

Onay gerektirenler: smoke test (200+ örnek), adapter promote, GGUF export,
production adapter değişimi.

---

## 7. SÜREKLİ ÖĞRENME PROTOKOLÜ — "Trader Uzmanı" döngüsü

> Amaç: model yalnız eğitilmez; **sürekli yeni kaynak okur, kavrar, sentezler ve
> eğitilir** — her turda biraz daha "trader gibi düşünen" bir uzmana yaklaşır.
> Çalıştır: `bash scripts/continuous-learning.sh [saat]` · Durdur: `storage/STOP_LEARNING`

```
┌────────────────────────────────────────────────────────────────────┐
│  1) ZENGİNLEŞTİR   arXiv'den 3 makale (dönüşümlü konu kuyruğu)     │
│  2) KAVRA          kart üret (LLM) → içerikli onayla → anlama %    │
│  3) SENTEZLE       (her 3 turda) hipotez + backtest + sentez       │
│                    makalesi → web'den indirilebilir                │
│  4) EĞİT           dataset tazele → LoRA 2×40 iter                 │
│  └─→ tekrar (RAM disiplini: LLM fazı ↔ eğitim fazı asla çakışmaz)  │
└────────────────────────────────────────────────────────────────────┘
```

### Neden bu alanlar? (konu kuyruğu: `scripts/enrichment-topics.txt`)
Trader uzmanlığı yalnız formül değildir; kuyruk bilinçli olarak şu eksenlerde
zengindir:
- **Psikoloji / davranışsal finans** — aşırı güven, kayıp kaçınma, sürü
  davranışı, disposition etkisi → modelin "neden yanılırız"ı öğrenmesi.
- **Belirsizlik** — Knightian belirsizlik, belirsizlik nicemleme, kuyruk riski,
  black swan → emin görünmek yerine belirsizliği DOĞRU ifade etme davranışı.
- **Felsefe / karar kuramı** — olasılık felsefesi, Bayesçi karar, tümevarım
  problemi, sınırlı akılcılık → muhakeme disiplini (LoRA'nın asıl hedefi).
- **Trading / risk** — rejim değişimi (Markov), Kelly, drawdown kontrolü,
  momentum-ortalamaya dönüş etkileşimi.

Kuyruk düz metin dosyasıdır: satır ekle/çıkar, döngü kaldığı yerden devam eder
(`storage/learning_topic_index`).

### Protokol ilkeleri
1. **Kavramadan eğitme yok** — yeni makale önce kart + anlama skoru alır;
   içeriksiz kart asla onaylanmaz/eğitime girmez.
2. **Sentez = yeni bilgi** — hipotezler mevcut kartların ÜZERİNE üretilir,
   backtest edilir (geçemeyen FAIL olarak raporlanır) ve sentez makalesi
   olarak hem insana (web) hem RAG'a geri beslenebilir.
3. **Dürüst metrik** — `rag-mastery` her tur sonunda loglanır; ilerleme
   kapsam/anlama/eğitim bileşenleriyle izlenir.
4. **RAM disiplini** — LLM fazları ve eğitim fazı sıralıdır; düşük RAM'li
   makinede asla çakışmaz.

### RAG + LoRA tek beyin entegrasyonu
Eğitilen adapter'ın RAG ile AYNI modelde çalışması için GGUF dönüşümü ve
RAFT-tarzı veri reçetesi: bkz. **docs/RAG_LORA_ENTEGRASYON.md** (Ollama
ADAPTER yolu, base eşleşme kuralı, ≥1000 örnek hedefi, kaynak-yok reddi
örnekleri, müfredat sırası).

## 8. Güçlü makineye taşıma (ölçekleme rehberi)

Protokol makineden bağımsızdır — yalnız şu düğmeler değişir:

| Düğme | Bu laptop (CPU) | Mac M-serisi | Tek GPU (24GB) | Colab A100 |
|---|---|---|---|---|
| Base model | Qwen3-4B-Instruct-2507 | aynı / 9B | 9B / 14B | 30B+ |
| Backend | PEFT (CPU) | MLX | PEFT (CUDA) | PEFT (CUDA) |
| iters/koşu | 40 | 200+ | 300+ | 300+ |
| max_seq | 1024 | 2048 | 2048 (RAFT) | 4096 |
| batch_size | 1 | 4 | 8+ | 16+ |
| Tur süresi | ~1.5-2 sa | ~20-30 dk | ~10 dk | ~5 dk |
| Anlama skoru | seçili makale | tüm korpus | tüm korpus | tüm korpus + judge kalibrasyon |

Taşıma adımları: repo'yu klonla → `.env`'de `ACHILLES_PEFT_BASE_MODEL` +
`ACHILLES_LLM_MODEL`'i makineye göre seç (base ↔ Ollama tag EŞLEŞMELİ) →
`bash scripts/continuous-learning.sh 72`. Hepsi bu.

## 9. Sınırlar ve uyarılar

- **Veri (2026-06-14):** kart-DB darboğazı (içeriksiz kabuk kartlar) **sentetik veri
  motoruyla** aşıldı → `data/lora_sft/lora_sft.jsonl` **~1266 örnek** (synth-qa + kart).
  Eğitim bunu `lora-split` ile okur (kart-DB `lora-dataset`'i DEĞİL; bkz. §3.5).
  Asıl performans kaldıracı yine **veri miktarı + kalitesi** (anlama %28 hâlâ düşük).
- **CPU yavaşlığı:** ciddi/uzun eğitim için **Colab/GPU** önerilir
  (`peft_lora_train.generate_colab_notebook` notebook üretir).
- **Bellek:** 4B eğitimi ~16 GB RAM kullanır; aynı anda büyük Ollama modeli
  (30B = 18 GB) açık olmasın → OOM riski.
- "Yüksek performans" 4B ile sınırlıdır; üst kalite için 9B/30B (Colab) + RAG.
