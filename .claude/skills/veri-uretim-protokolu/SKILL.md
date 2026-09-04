---
name: veri-uretim-protokolu
description: Stage 1 — lokal sentetik QA veri üretimi. Makale chunk'larından grounded SFT örneği üretir (15→1000+), gece döngüsünü yönetir, Stage 2 eşiğini izler. CPU eğitimi YAPMAZ.
when_to_use: Kullanıcı sentetik eğitim verisi üretmek, üretim döngüsünü başlatmak/izlemek veya bulut-GPU eğitim eşiğine (≥1000 örnek) ulaşılıp ulaşılmadığını kontrol etmek istediğinde.
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

# Stage 1 — Veri Üretim Protokolü

Amaç: lokal CPU'da **eğitim yapmadan** makale chunk'larından grounded sentetik QA
üreterek ≥1000 örneğe çıkmak (Stage 2 / bulut-GPU eğitiminin ön koşulu).
Detay: `docs/PROTOKOL_VERI_URETIM.md` · üst: `docs/PROTOKOL_ASAMALI_EGITIM.md`.

## Sınır (önce bunu doğrula)
- **CPU LoRA eğitimi YASAK** — haftalar sürer + az veride overfit. Bu skill yalnız
  VERİ ÜRETİR. Gerçek eğitim Stage 2'de bulut-GPU + açık onayla (CLAUDE.md kural 8).
- Üretim Ollama (veya API) gerektirir; çıktı her zaman grounded (kural 7).

## Komutlar
| İş | Komut |
|----|-------|
| Tek seferlik üretim | `uv run achilles synth-qa --per-chunk 5 --max-chunks 12 --seed 0` |
| Sürekli döngü (72sa) | `bash scripts/continuous-learning.sh 72` |
| Döngüyü durdur | `New-Item storage/STOP_LEARNING` (Win) / `touch storage/STOP_LEARNING` |
| Eşik durumu | `uv run achilles lora-readiness` |
| RAG panosu | `uv run achilles rag-mastery` |

## İş akışı
1. **Durum kontrol:** `lora-readiness` ile mevcut örnek sayısı + ≥1000 eşiği.
2. **Üret:** döngü çalışmıyorsa başlat; çalışıyorsa logu izle
   (`logs/continuous-learning.log` son 45 dk'da ilerliyor mu?).
3. **Kalite:** üretilen örnekler grounded mı? (sayı-altküme + anchor kapıları otomatik).
4. **Eşik:** ≥1000 olunca kullanıcıya bildir → GATE: `lora-audit` (Gate 0-7) öner →
   onaylanırsa `/bulut-egitim-protokolu` (Stage 2).

## Sağlık kontrolü (döngü için)
- Boş RAM < 2GB → ağır LLM işini beklet, çakışan süreçleri durdur.
- Web (8765) düştüyse: `uv run achilles-web` (arka plan).
- Döngü öldüyse: `bash scripts/continuous-learning.sh 72` ile yeniden başlat.

## Kullanıcı onayı gerektiren
- Stage 2'ye geçiş (gerçek eğitim) — asla otomatik başlatma.
