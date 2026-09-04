# Protokol — LoRA Araştırma & Entegrasyon Ajanı

_Son güncelleme: 2026-06-17 (v1.0)._

> **Tek cümle:** Belirli aralıklarla devreye girip güncel LoRA/SFT literatüründen Achilles'in
> eğitim hattına yarayacak **YENİ, GERÇEK** teknikleri bulur, adversarial doğrular, dokümana
> işler ve kod/reçete entegrasyonunu **PR olarak** önerir. **Eğitim başlatmaz.**

> **🧠 Bağlam:** Bu ajan, `makale-arastirma` ajanının kardeşidir. O **kaynak besler** (arXiv PDF →
> RAG); bu ajan **yöntem besler** (LoRA eğitim reçetesini iyileştirir). Birincil hedef: v5
> catastrophic-forgetting/degenerasyon regresyonunu (bkz. `memory/v5-adapter-regression`) onarmak.

---

## 1. Sıklık (frekans) — iki katman

- **Günlük hafif tarama** — her gün ~09:00 (cron `0 9 * * *`). Ucuz tarama: defterde olmayan
  yeni bir gelişme var mı? Yoksa **no-op** (commit yok). Tam çok-açılı sweep YOK.
- **Haftalık derin tur** — Pazartesi ~10:00 (cron `0 10 * * 1`). Tam çok-açılı araştırma +
  adversarial doğrulama + sentez + entegrasyon.
- **Gerekçe (neden 6 saatte bir DEĞİL):** Gerçekten yeni/değerli LoRA tekniği 6 saatte bir
  çıkmaz (haftada birkaç ilgili arXiv olsa iyi). Sık derin tur = çoğu boş tur + gürültü commit +
  maliyet + doğrulanmamış reçeteyi sık push riski. İki katman: günlük ucuz nöbet + haftalık derin iş.

## 2. İki mod

| Mod | Tetik | Ne yapar | Çıktı |
|-----|-------|----------|-------|
| `daily-light` | günlük cron / elle | sadece tarama + dedup; yeni varsa loga aday | no-op **veya** log+doküman push / kod ise PR |
| `weekly-deep` | haftalık cron / elle | çok-açılı sweep + doğrulama + sentez + entegrasyon | doküman v↑ + PDF; kod ise PR |

## 3. Dedup defteri

`docs/egitim/LORA_ARASTIRMA_LOG.md` — "Kapsanan teknikler" ve "Kapsanan kaynaklar" listeleri
**dedup anahtarıdır**. Burada geçen teknik/arXiv-ID/URL **yeniden derin-araştırılmaz**. Her tur
bu deftere yazılır (insan-okur log + makine-dedup).

## 4. Mutlak kurallar (CLAUDE.md)

- **Kural 2** — test edilmeden "çalışıyor"/"daha iyi" deme. Reçete değişikliği ancak **bulut
  eğitim koşusu + `adapter_eval` gate'i** ile doğrulandıktan sonra "daha iyi" sayılır. Entegre
  edilen ileri teknikler **OPT-IN** (varsayılan davranışı değiştirme).
- **Kural 7** — kaynak uydurma YASAK. Yalnız WebFetch ile **doğrulanmış URL'si** olan tekniği logla.
- **Kural 8** — eğitim BAŞLATMA (gerçek eğitim bulutta, kullanıcı tarafından). Aynı veriyle
  körlemesine retrain önerme.
- **GGUF-güvenli** — embedding/lm_head eğitme (Qwen3 tied-embeddings); PiSSA/OLoRA/CorDA gibi
  base'i değiştiren init'ler için residual dönüşümü uyarısını koru.
- Determinizm (seed). Türkçe yaz.

## 5. Kapı (auto-push vs PR)

- **Doküman/log-yalnız** değişiklik → `main`'e push edilebilir.
- **KOD/REÇETE** değişikliği → main'e **DOĞRUDAN PUSH YOK**. `gh` ile dal + PR aç (başlıkta teknik
  adı; gövdede gerekçe + entegrasyon noktası + "Kural 2: doğrulanmadan terfi yok"). `gh` yoksa
  log'a "kod entegrasyonu için inceleme bekliyor" işaretle, doküman değişikliğini push et.

## 6. Entegrasyon noktaları (kod)

| Dosya | Ne için |
|-------|---------|
| `app/training/peft_lora_train.py` | `PeftTrainConfig` alanları + `build_lora_kwargs`/`build_training_kwargs` saf builder'lar |
| `configs/lora/lora_profiles.yaml` | yeni profil/alan (`load_lora_profile` ile koda bağlı) |
| `app/training/cloud_notebook.py` + `templates/stage2_lora_colab.ipynb` | bulut reçetesi placeholder'ları |
| `app/training/adapter_eval.py` | degenerasyon/eval iyileştirmeleri |

Yeni teknik eklerken: PEFT (`use_rslora`/`use_dora`/`init_lora_weights`) ve transformers/TRL
(`neftune_noise_alpha`) yerel destekli mi kontrol et; saf builder'a alan ekle; profil ekle.

## 7. Akış

### daily-light
1. `LORA_ARASTIRMA_LOG.md` dedup listelerini oku.
2. WebSearch/WebFetch: arXiv (LoRA/PEFT/SFT/catastrophic-forgetting), HF Papers, Unsloth blog/docs,
   PEFT release notes — SON HAFTA. Defterde olmayan + doğrulanmış URL'li öğeleri "yeni" say.
3. Yeni yoksa → no-op, kısa özet, BİTİR.
4. Yeni varsa → loga "## Günlük tarama — <tarih>" + aday(lar). Doküman-yalnızsa push; kod ise PR.

### weekly-deep
1. Dedup oku. CLAUDE.md + mevcut `LORA_EGITIM_DETAYLI_ANLATIM.md` v sürümünü oku.
2. Çok-açılı sweep (her açı ayrı WebSearch): (a) yeni LoRA varyantları, (b) init yöntemleri,
   (c) forgetting/refusal koruma, (d) regularizasyon (NEFTune-vari), (e) Qwen3/Unsloth güncel,
   (f) SFT veri kalitesi, (g) degenerasyon/eval.
3. Her aday: gerçek mi? PEFT 0.19+/Unsloth destekli mi? Achilles'e uygun mu? GGUF-güvenli mi?
   v5'e yardım eder mi? Emin değilsen ELE.
4. Entegrasyon (opt-in) → §6 dosyalar. Test (§8). Kod ise PR.
5. `LORA_ARASTIRMA_LOG.md`'ye "## Tur N — <tarih> (derin tur)" + doğrulanmış kaynaklar tablosu.
6. Anlamlı bulgu varsa `LORA_EGITIM_DETAYLI_ANLATIM.md` sürümünü ARTIR + bölümleri güncelle +
   `.venv\Scripts\python.exe scripts/gen_egitim_pdf.py` (PDF repo + Desktop'a yazılır).

## 8. Doğrulama (kod değiştiyse zorunlu)

`uv run` web sunucusu venv'i kilitlediğinde başarısız olur → **`.venv\Scripts\python.exe`** kullan:
```
.venv\Scripts\python.exe -m ruff format <dosya> ; .venv\Scripts\python.exe -m ruff check app tests
.venv\Scripts\python.exe -m mypy app
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_tmp -m "not ollama and not slow"
```
Testler geçmeden PR'ı "hazır" deme.

## 9. Push prosedürü

`git fetch origin` + `git rebase --autostash origin/main`. Çalışma ağacında başka süreçlerin
değişikliği olabilir → **yalnız KENDİ dosyalarını** `git add <dosya>` ile stage'le (asla `-A`/`-a`).
Commit bash here-doc (`git commit -F - <<'EOF' ... EOF`), Türkçe, sonuna
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Push TAM: `git push origin main`.

## 10. Tetikleme

- **Periyodik:** `mcp__scheduled-tasks` görevleri `lora-arastirma-gunluk-tarama` (daily-light) ve
  `lora-arastirma-haftalik-derin` (weekly-deep) — her ikisi de bu ajanı çağırır. Yalnız Claude
  **uygulaması açıkken** çalışır; kapalıysa sonraki açılışta. İlk kez "Run now" ile araçları ön-onayla.
- **Elle / oturum içi:** `lora-arastirma` ajanını Agent aracıyla başlat (argümanda mod belirt:
  `daily-light` veya `weekly-deep`), ya da bu protokolü doğrudan uygula.

## 11. Çıktı

Türkçe özet: ne tarandı, ne yeni bulundu (ID/URL), ne dokümana/loga eklendi, hangi PR açıldı,
hiçbir şey yoksa "yeni gelişme yok".

---
İlişkili: `.claude/agents/lora-arastirma.md` · `docs/egitim/LORA_EGITIM_DETAYLI_ANLATIM.md` ·
`docs/egitim/LORA_ARASTIRMA_LOG.md` · `memory/v5-adapter-regression` · `PROTOKOL_MAKALE_ARASTIRMA.md`
