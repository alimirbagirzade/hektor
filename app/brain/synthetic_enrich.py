"""Kısa sentetik QA cevaplarını AYNI bağlamdan "açıklamalı" hâle getir (zenginleştirme).

Neden (2026-09-30 ölçümü): `synthetic_qa.jsonl` 1051 satır, asistan cevabı medyanı 166
karakter, 726'sı <200 — çoğu tek cümlelik özet ("Çalışmada X üzerine ilerleme kaydedilmiştir.").
Eğitim setinin ~%60'ı bu satırlar olduğundan adapter KISA cevap vermeyi öğreniyordu; v10–v12
eval'lerinde persona/format setlerinin negatif kalmasının ana adayı ("kısa cevap, eksik bölüm").

Yöntem: soru ve BAĞLAM korunur; yerel LLM aynı bağlamdan 3-5 cümlelik cevap yazar (doğrudan
cevap → bağlamdaki gerekçe/mekanizma → varsayım/sınırlama). Yeni cevap üreticinin TÜM
kapılarından geçmek zorundadır (`_is_grounded` sayı-altküme + anchor, `is_low_value_answer`,
dejenere tekrar, CJK sızıntısı, uzunluk). Geçemezse ORİJİNAL satır aynen kalır — veri asla
kötüleşmez. EĞİTİM BAŞLATMAZ (Kural 8); yalnız veri dosyası üretir.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.brain.local_llm import LLMUnavailable
from app.brain.synthetic_qa_builder import (
    _MAX_PASSAGE_CHARS,
    _coerce_json_list,
    _is_grounded,
    is_low_value_answer,
)

logger = logging.getLogger(__name__)

# Bu uzunluğun altındaki cevaplar zenginleştirilir (medyan 166; disiplin cevapları ~230).
ENRICH_BELOW_CHARS = 200
# Yeni cevap en az bu kadar olmalı VE orijinalden belirgin uzun (yoksa değişikliğe değmez).
_MIN_NEW_CHARS = 220
_MIN_GAIN_CHARS = 60
_MAX_NEW_CHARS = 1400
_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_CONTEXT_RE = re.compile(r"^BAĞLAM:\n(?P<ctx>.*)\n\nSORU: (?P<q>.*)$", re.S)

# Etiketsiz örnek (eski örnekteki "Mekanizma basit:" açılışı modele etiket cümlesi
# öğretiyordu — Kademe-2 C4: 59 cevabın 6'sı "Mekanizma, …" ile açıldı).
_ONESHOT = (
    '{"answer": "ATR, fiyatin tipik gunluk hareket araligini olcer ve bu calismada pozisyon '
    "buyuklugunu volatiliteye gore ayarlamak icin kullaniliyor. ATR yukseldiginde ayni risk "
    "butcesi daha kucuk bir pozisyona karsilik gelir ve stop mesafesi genisler. Calisma bu "
    'ayarin gecmis 14 gunluk volatiliteye dayandigini belirtiyor."}'
)


def build_enrich_prompt(context: str, question: str, answer: str) -> str:
    """Zenginleştirme istemi (ASCII-Türkçe talimat — üreticiyle aynı, canlı test edilmiş biçim)."""
    passage = context[:_MAX_PASSAGE_CHARS]
    return (
        "Gorev: Asagidaki soru icin DAHA ACIKLAYICI bir Turkce cevap yaz (egitim verisi).\n\n"
        f'KAYNAK:\n"""\n{passage}\n"""\n\n'
        f"SORU: {question}\n"
        f"TASLAK CEVAP (kisa; kaynakla celisiyorsa DUZELT, dogruysa genislet): {answer}\n\n"
        "Kurallar:\n"
        "- 2-4 cumle, TURKCE yaz. Once soruya dogrudan cevap ver; sonra kaynagin bu konuda "
        "SOYLEDIGI ek ayrintilari aktar.\n"
        "- Gerekce, mekanizma, varsayim ya da sinirlama YALNIZ kaynakta acikca yaziyorsa "
        "yazilir; kaynakta yoksa kendi yorumunu EKLEME.\n"
        "- Kaynakta olmayan sayi, formul, sonuc, kurum veya olay UYDURMA. Sayilari ve teknik "
        "terimleri kaynaktan aynen kullan.\n"
        "- Genel gecer dolgu ya da ozet cumlesi ('bu durum ... gosterir', 'onemli bir rol "
        "oynar', 'bu nedenle ...') YAZMA; taslak cevabi tekrar etme.\n"
        "- 'Mekanizma:', 'Varsayim:' gibi etiketle cumle baslatma.\n"
        "- Kaynagin soylemedigi seylerden bahsetme; 'pasaj', 'metin', 'kaynakta', 'baglamda' "
        "kelimelerini KULLANMA.\n"
        "- Yatirim tavsiyesi verme; 'yapmali', 'her zaman', 'kesinlikle' gibi kesinlik ya da "
        "yonlendirme dili kurma.\n\n"
        f"Cikti TAM olarak su yapida bir JSON objesi olsun (ornegi DOLDUR, aynen "
        f"kopyalama):\n{_ONESHOT}"
    )


def _is_repetitive(text: str) -> bool:
    """Tekrar döngüsü — eval ile AYNI dedektör (eğitime dejenere cevap girmesin)."""
    from app.training.adapter_eval import _is_degenerate

    return _is_degenerate(text)


# --- Kademe-2 C2-C4 kapıları (2026-09-30) -------------------------------------------------
# Ölçüm (726 gerçek aday satır): eski kapılar "orijinal + sayısız uydurma kuyruk"u 724/726,
# tamamen İngilizce cevabı 412/726, tavsiye kuyruğunu 721/726 kabul ediyordu — tek ortak
# anchor (orijinal cevaptan zaten gelen) bütün cevabı "grounded" sayıyordu.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-zçğıöşü]+")
_TR_STOP = frozenset(
    (
        "ve",
        "bir",
        "bu",
        "için",
        "ile",
        "olarak",
        "da",
        "de",
        "daha",
        "en",
        "gibi",
        "olan",
        "ise",
        "çok",
        "göre",
        "ancak",
        "veya",
        "her",
        "kadar",
        "ya",
        "ki",
        "şu",
        "hem",
        "ne",
        "nasıl",
        "olduğu",
        "olduğunu",
    )
)
_EN_STOP = frozenset(
    (
        "the",
        "and",
        "of",
        "is",
        "to",
        "in",
        "that",
        "for",
        "with",
        "are",
        "this",
        "as",
        "by",
        "be",
        "on",
    )
)
# "metin" + ünlü düşmeli biçimleri (metni/metninde) + kaynak/bağlam atfı (C3).
_SOURCE_REF_RE = re.compile(r"\b(?:metin|metn|pasaj|kaynakta|kaynağında|bağlamda)\w*", re.I)
_LABEL_OPEN_RE = re.compile(r"^(?:mekanizma|varsayım|gerekçe|sınırlama|sonuç)\s*[,:]", re.I)
# Tavsiye / kesinlik dili (orijinalde YOKSA eklenemez).
_ADVICE_RE = re.compile(
    r"\b(?:yap|ayır|yatır|al|sat|kullan|gir|tut|koy|seç|başla|tercih et)(?:ma|me)l[ıi]\w*"
    r"|\b(?:her zaman|her piyasada|kesinlikle|mutlaka)\s+(?:\w+\s+)?(?:kazan|kâr|kar\b|getiri|"
    r"işe yar|iyi sonuç|başarı)\w*"
    r"|\bgaranti(?:li)?\s+(?:kâr|kar\b|kazanç|getiri)\w*"
    r"|\b(?:önerilir|tavsiye ederim|tavsiye edilir|asla kaybet\w*|tamamını ayır\w*|"
    r"iyi sonuç verir)",
    re.I,
)
# Kaba token bütçesi (C7): eğitim max_seq_length 1024; prompt+cevap bunu aşarsa cevap EOS'suz
# kırpılır. İngilizce bağlam ~4, Türkçe ~3 karakter/token; sistem+şablon payı 80 token.
_TOKEN_BUDGET = 1000


def _norm_sentence(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))


def _is_turkish(text: str) -> bool:
    words = _WORD_RE.findall(text.lower())
    if len(words) < 8:
        return True
    tr = sum(w in _TR_STOP for w in words) / len(words)
    en = sum(w in _EN_STOP for w in words) / len(words)
    return tr >= 0.04 and en < 0.03


def _stems(text: str) -> set[str]:
    """Dil-arası kaba kök: tr_fold + ≥5 harfli kelimenin ilk 5 harfi, sayılar aynen.

    İngilizce bağlam / Türkçe cevapta tam-kelime anchor çoğu çeviriyi kaçırır
    ("volatility" ↔ "volatilite"); 5 harflik kök ödünç kelimeleri yakalar.
    """
    from app.lora.safety_scanner import tr_fold

    out: set[str] = set()
    for w in re.findall(r"\w+", tr_fold(text)):
        if any(c.isdigit() for c in w):
            out.add(w)
        elif len(w) >= 5:
            out.add(w[:5])
    return out


def _ungrounded_new_sentences(new: str, old: str, context: str) -> list[str]:
    """Orijinalde olmayan ve bağlamla HİÇ kök paylaşmayan yeni cümleler."""
    ctx_anchors = _stems(context)
    old_norm = _norm_sentence(old)
    out = []
    for sent in _SENT_RE.split(new.strip()):
        if len(sent) < 25 or _norm_sentence(sent) in old_norm:
            continue
        if not (_stems(sent) & ctx_anchors):
            out.append(sent)
    return out


def validate_enriched(new: str, old: str, context: str, *, prompt_chars: int = 0) -> str | None:
    """Yeni cevap kabul edilebilir mi? Kabulse None, değilse red gerekçesi."""
    if len(new) < max(_MIN_NEW_CHARS, len(old) + _MIN_GAIN_CHARS):
        return "kısa"
    if len(new) > _MAX_NEW_CHARS:
        return "uzun"
    if prompt_chars and prompt_chars / 4 + len(new) / 3 + 80 > _TOKEN_BUDGET:
        return "bütçe"
    if _CJK_RE.search(new):
        return "cjk"
    if not _is_turkish(new):
        return "dil"
    if _SOURCE_REF_RE.search(new) or is_low_value_answer(new):
        return "düşük-değer"
    sents = [_norm_sentence(x) for x in _SENT_RE.split(new.strip()) if len(x) > 15]
    if any(_LABEL_OPEN_RE.match(x.strip()) for x in _SENT_RE.split(new.strip())):
        return "etiket"
    if len(sents) != len(set(sents)) or (
        len(_norm_sentence(old)) > 30 and _norm_sentence(new).count(_norm_sentence(old)) > 1
    ):
        return "tekrar"
    if _ADVICE_RE.search(new) and not _ADVICE_RE.search(old):
        return "tavsiye"
    if not _is_grounded(new, context):
        return "grounding"
    # Yeni cümlelerin çoğu bağlamla bağsızsa (çapraz-dil payı: en çok 1 bağsız cümle).
    if len(_ungrounded_new_sentences(new, old, context)) > 1:
        return "grounding"
    if _is_repetitive(new):
        return "tekrar"
    return None


@dataclass
class EnrichOutcome:
    line: str
    status: str  # "enriched" | "skipped" | red gerekçesi ("grounding", "kısa", ...)


def _split_user(content: str) -> tuple[str, str] | None:
    m = _CONTEXT_RE.match(content)
    if not m:
        return None
    return m.group("ctx"), m.group("q")


def enrich_line(
    line: str,
    llm: Any,
    *,
    seed: int = 0,
    below_chars: int = ENRICH_BELOW_CHARS,
) -> EnrichOutcome:
    """Tek JSONL satırını zenginleştir; başarısızlıkta ORİJİNAL satırı döndür."""
    try:
        obj = json.loads(line)
        msgs = obj["messages"]
        answer = msgs[-1]["content"]
        user = next(m["content"] for m in msgs if m.get("role") == "user")
    except (json.JSONDecodeError, KeyError, IndexError, StopIteration, TypeError):
        return EnrichOutcome(line, "skipped")
    meta = obj.get("metadata") or {}
    if msgs[-1].get("role") != "assistant" or len(answer) >= below_chars or meta.get("enriched"):
        return EnrichOutcome(line, "skipped")
    parts = _split_user(user)
    if parts is None:
        return EnrichOutcome(line, "skipped")
    context, question = parts

    try:
        raw = llm.generate(
            build_enrich_prompt(context, question, answer),
            temperature=0.4,
            fmt="json",
            max_tokens=700,
            seed=seed,
        )
    except LLMUnavailable:
        raise
    except Exception as exc:  # ağ/timeout — bu satırı atla, döngü sürsün
        logger.warning("Zenginleştirme hatası: %s", exc)
        return EnrichOutcome(line, "llm-hata")

    items = _coerce_json_list(raw)
    new = str(items[0].get("answer", "")).strip() if items else ""
    reason = validate_enriched(new, answer, context, prompt_chars=len(user))
    if reason:
        return EnrichOutcome(line, reason)

    msgs[-1] = {**msgs[-1], "content": new}
    obj["metadata"] = {**meta, "enriched": True, "orig_answer": answer}
    return EnrichOutcome(json.dumps(obj, ensure_ascii=False), "enriched")


# --------------------------------------------------------------------------- #
# Devam ettirilebilir toplu koşu (CLI: `hektor synth-enrich`)
# --------------------------------------------------------------------------- #
# Çalışma dosyası girdiyle SATIR-SATIR hizalıdır (i. çıktı = i. girdinin sonucu); kesilen koşu
# kaldığı satırdan sürer. Yan dosya girdinin hash'ini tutar: girdi koşu arasında değiştiyse
# hizalama bozulur → devam REDDEDİLİR (yanlış satıra yanlış cevap yazılmasın).


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_adapter_model(name: str) -> bool:
    """Ollama adı Hektor'un KENDİ eğittiği bir adapter mı (hektor-*)?

    2026-09-30 olayı: başka oturum `.env` HEKTOR_LLM_MODEL'i `hektor-v12-30b`'ye çevirdi ve
    zenginleştirme 1051 satırı v12 adapter'ına (bozuk CRLF şablonuyla) ürettirdi — bir
    sonraki eğitimin verisini bir önceki adapter üretirse hatalar kendini besler.
    """
    return name.strip().lower().startswith("hektor")


def work_paths(src: Any) -> tuple[Any, Any]:
    """(çalışma dosyası, meta dosyası) — girdiyle aynı klasörde."""
    return src.with_name(src.stem + ".enriched.jsonl"), src.with_name(src.stem + ".enrich.json")


def run_enrichment(
    src: Any,
    llm: Any,
    *,
    seed: int = 0,
    below_chars: int = ENRICH_BELOW_CHARS,
    limit: int = 0,
    progress: Any = None,
) -> dict:
    """``src`` satırlarını sırayla zenginleştir; çalışma dosyasına ekleyerek yaz (devam eder).

    ``limit>0`` → bu çağrıda en çok N satır işle. Dönen sözlük: toplam/işlenen/durum sayıları.
    Satır başına seed = ``seed + satır_no`` (determinizm, Kural 6).
    """
    from collections import Counter

    src_text = src.read_text(encoding="utf-8")
    lines = [ln for ln in src_text.splitlines() if ln.strip()]
    work, meta_path = work_paths(src)
    src_sha = _sha256_text(src_text)

    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw = work.read_text(encoding="utf-8") if work.exists() else ""
    # Kesilen koşunun yarım son satırı (Kademe-2 C5b): "\n" ile bitmeyen kuyruk atılır;
    # yoksa sonraki satır ona yapışır ve hizalama kayar.
    if raw and not raw.endswith("\n"):
        raw = raw[: raw.rfind("\n") + 1]
        work.write_text(raw, encoding="utf-8")
    done_lines = [ln for ln in raw.splitlines() if ln.strip()]
    model = str(getattr(llm, "model", "") or "")
    if done_lines:
        if meta.get("src_sha256") != src_sha:
            raise ValueError(
                f"{src.name} zenginleştirme başladıktan sonra DEĞİŞMİŞ ya da meta kayıp — satır "
                f"hizası güvenilmez. {work.name} ve {meta_path.name} dosyalarını silip baştan "
                "başla."
            )
        if meta.get("model") != model:
            raise ValueError(
                f"Önceki kısım {meta.get('model')!r} ile üretildi, şimdiki model {model!r} — "
                "tek veri setinde iki üretici karışmasın. Aynı modelle sürdür ya da baştan başla."
            )
    counts: Counter[str] = Counter(meta.get("counts") or {})
    start = len(done_lines)
    end = len(lines) if limit <= 0 else min(len(lines), start + limit)

    def _write_meta(done: int) -> None:
        meta_path.write_text(
            json.dumps(
                {
                    "src_sha256": src_sha,
                    "model": model,
                    "total": len(lines),
                    "done": done,
                    "counts": dict(counts),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Meta döngüden ÖNCE ve her satırdan sonra (C5a: 25'te bir yazmak ilk 25 satırda düşen
    # koşunun devamını "hash değişmiş" diye reddediyordu).
    _write_meta(start)
    with open(work, "a", encoding="utf-8") as fh:
        for i in range(start, end):
            out = enrich_line(lines[i], llm, seed=seed + i, below_chars=below_chars)
            fh.write(out.line + "\n")
            fh.flush()
            counts[out.status] += 1
            _write_meta(i + 1)
            if progress is not None:
                progress(i + 1, len(lines), out.status)
    return {"total": len(lines), "done": end, "counts": dict(counts), "model": model}


def apply_enrichment(src: Any, backup_dir: Any) -> dict:
    """Tamamlanmış çalışma dosyasını ``src``'nin yerine koy (yedek alarak, atomik)."""
    import datetime as _dt
    import os
    import shutil

    work, meta_path = work_paths(src)
    src_text = src.read_text(encoding="utf-8")
    n_src = len([ln for ln in src_text.splitlines() if ln.strip()])
    out_lines = [ln for ln in work.read_text(encoding="utf-8").splitlines() if ln.strip()]
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("src_sha256") != _sha256_text(src_text):
        raise ValueError(f"{src.name} zenginleştirmeden sonra değişmiş — uygulanmadı.")
    if len(out_lines) != n_src:
        raise ValueError(f"Zenginleştirme tamamlanmadı ({len(out_lines)}/{n_src} satır).")
    model = str(meta.get("model") or "")
    if not model or is_adapter_model(model):
        raise ValueError(
            f"Üretici model kaydı yok ya da Hektor adapter'ı ({model!r}) — uygulanmadı."
        )
    # Satır satır hiza (C5b): her çıktı satırı geçerli JSON olmalı ve kaynağıyla AYNI soruyu/
    # bağlamı taşımalı; yarım/kaymış tek satır bile bütün uygulamayı durdurur.
    src_lines = [ln for ln in src_text.splitlines() if ln.strip()]
    for i, (a, b) in enumerate(zip(src_lines, out_lines, strict=True)):
        try:
            sa, sb = json.loads(a), json.loads(b)
            ua = [m["content"] for m in sa["messages"] if m.get("role") != "assistant"]
            ub = [m["content"] for m in sb["messages"] if m.get("role") != "assistant"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Çıktı satırı {i + 1} okunamadı: {exc}") from exc
        if ua != ub:
            raise ValueError(f"Çıktı satırı {i + 1} kaynağıyla hizalı değil — uygulanmadı.")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = backup_dir / f"{src.stem}.bak-{stamp}.jsonl"
    shutil.copy2(src, backup)
    tmp = src.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    os.replace(tmp, src)
    work.unlink()
    meta_path.unlink(missing_ok=True)
    return {
        "applied": n_src,
        "backup": str(backup),
        "counts": meta.get("counts", {}),
        "model": meta.get("model"),
    }
