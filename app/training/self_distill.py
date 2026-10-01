"""Öz-damıtma (self-distillation) SFT örnekleri — base modelin KENDİ uzun/atıflı cevapları.

Neden (v13 LLM-30 2×2, 2026-09-30): eğitim verisinin cevap medyanı ~270 karakterdi ve atıf
taşımıyordu (1499 satırda 6); base modelin kendi cevabı ise ~2300 token. LoRA bu dağılıma
çekildi → cevaplar ~5× kısaldı, RAG'da atıf talimatı tamamen yok sayıldı (D: 0/90). Canlı RAG
istemi (`rag_answer.md` + `SOURCES / KAYNAKLAR`) eğitimde HİÇ görülmemişti.

Yöntem (Self-Distillation Fine-Tuning; Yang ve ark. 2024, arXiv:2402.13669 — hedef cevabı
base modelin kendi dağılımından seçmek ince ayarın genel yetenek kaybını azaltır):
  * ``rag``   — gerçek retrieval + canlı istem (`build_rag_prompt`, bayt-aynı) → base cevabı;
                atıf çiftlerinin HEPSİ getirilenlerde olmalı (Kural 7), en az
                `MIN_UNIQUE_CITATIONS` farklı kaynak.
  * ``plain`` — RAG'sız kavramsal soru, genel Hektor sistem istemi + "bağlam verilmedi"
                notu → base cevabı; atıf, kaynakça, URL, RAG iddiası yasak (Kademe 2 F1-1).
Sorular mevcut sentetik QA havuzundan (korpustan türemiş) seçilir; eval soruları sızıntı
denetimiyle dışlanır. Üretici HER ZAMAN base modeldir (hektor-* reddedilir): önceki adapter'ın
hataları bir sonrakine taşınmasın. Base'in KENDİ hataları da taşınır — öz-damıtma doğruluğu
artırmaz, yalnız uzunluk/biçim/atıf davranışını base'de tutar (hipotez; eval ile ölçülecek).

EĞİTİM BAŞLATMAZ (Kural 8). Determinizm: seçim `random.Random(seed)`, üretim seed'i
`seed + iş_no` (Kural 6; Ollama'nın kendisi bu kurulumda tam deterministik değil — HANDOFF).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DISTILL_FILENAME = "distill_qa.jsonl"
# Eğitim profili `moe30b_attn_long` max_seq_length=6144; chat şablonu + güvenlik payı 96 token.
MAX_TOTAL_TOKENS = 6144 - 96
MAX_ANSWER_TOKENS = 3072
NUM_CTX = 8192
MIN_ANSWER_CHARS = 600
# Farklı kaynak sayısı alt sınırı (Kademe 2 F3-1: eski kapı tekrarları sayıyordu). 3 FARKLI
# kaynak, mevcut kabul edilmiş RAG satırlarının ~%70'ini reddederdi ve top-6 bağlamda alakasız
# kaynağa atıf yapmayı öğretebilirdi → 2 (tasarım kararı, 2026-10-01).
MIN_UNIQUE_CITATIONS = 2
DEFAULT_TEMPERATURE = 0.2  # canlı RagAnswerer ile aynı

_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# Yalnız belirli bir makaleye/pasaja bağlı sorular RAG'sız sorulamaz (uydurmaya davet).
_PAPER_BOUND_RE = re.compile(
    r"\b(?:çalışma|makale|yazar|metin|metn|pasaj|bölüm|şekil|tablo|kaynak|bağlam|denklem|"
    r"örnekte|yukarıda|bu model|önerilen|sunulan|teorem|sayfa|eser|paper|study|author|"
    r"figure|table|section)\w*|\b\d+\.\d+",
    re.I,
)
# Görünmeyen bir pasaja/kaynağa gönderme yapan soru hiçbir kipte kullanılmaz: gerçek kullanıcı
# pasajı görmez (pilot 2026-10-01: "Pasajda…", "Kaynak [250]'deki…" soruları).
_PASSAGE_REF_RE = re.compile(
    r"\b(?:pasaj|metin|metn|yukarı|verilen|belirtilen|bahsedilen|sözü\s+edilen|bu\s+çalışma|"
    r"bu\s+makale|bu\s+bölüm|örnekte|kaynak\s*\[)\w*|\[\d+\]",
    re.I,
)
# Aynı iki kalıp, `tr_fold` edilmiş (aksansız, küçük harf) metin için — Türkçe harfsiz yazılmış
# soruları da yakalar (Kademe 2 F1-7: "Bu calismada", "onerilen", "sekilde" kaçıyordu) + numaralı
# kitap göndermeleri ("Example 7.3.1'deki", "Lemma 9.4", "Eq. (15.65)") — soru kendi başına
# anlaşılmaz (2026-10-01 iş listesi incelemesi).
_PASSAGE_REF_FOLD_RE = re.compile(
    r"\b(?:pasaj|metin|metn|yukari|verilen|belirtilen|bahsedilen|sozu\s+edilen|bu\s+calisma|"
    r"bu\s+makale|bu\s+bolum|ornekte|kaynak\s*\[)\w*|\[\d+\]"
    r"|\b(?:example|ornek|teorem|theorem|lemma|corollary|proposition|eq|denklem|sekil|figure|"
    r"fig|tablo|table|section|bolum|madde|exercise|alistirma|problem|definition|tanim)\w*"
    r"\s*\.?\s*\(?\d+(?:\.\d+)*",
    re.I,
)
_PAPER_BOUND_FOLD_RE = re.compile(
    r"\b(?:calisma|makale|yazar|metin|metn|pasaj|bolum|sekil|tablo|kaynak|baglam|denklem|"
    r"ornekte|yukarida|bu model|onerilen|sunulan|teorem|sayfa|eser|paper|study|author|"
    r"figure|table|section)\w*|\b\d+\.\d+",
    re.I,
)


def question_dependent(q: str, mode: str) -> bool:
    """Soru görünmeyen bir pasaja/numaralı kitap öğesine bağlı mı (aksansız yazım dahil)?"""
    from app.lora.safety_scanner import tr_fold

    f = tr_fold(q)
    return bool(_PASSAGE_REF_FOLD_RE.search(f)) or (
        mode == "plain" and bool(_PAPER_BOUND_FOLD_RE.search(f))
    )


# RAG'sız kip: bağlam verilmediği halde base model genel sistem istemindeki "kaynak temelli…
# RAG bağlamı varsa kullan" ifadesini UYDURMA kaynakça ile karşılıyordu (Kademe 2 F1-1,
# 2026-10-01: ilk 90 plain satırın 87'si — "Kaynaklar (RAG Bağlamı)", arXiv numaraları,
# "et al."). Bu kipte sistem istemine açık talimat eklenir ve uydurma kalıpları reddedilir.
PLAIN_NO_SOURCE_NOTE = (
    "\n\nBu soruda RAG bağlamı VERİLMEDİ. Kaynak listesi, URL, arXiv/DOI numarası ya da "
    "yazar-yıl atıfı YAZMA; cevabın kaynak/RAG temelli olduğunu iddia etme. Genel bilgiyi "
    "kaynaksız olarak ve belirsizliği açıkça belirterek ver."
)
_FABRICATED_SOURCE_RE = re.compile(
    r"(?im)^\W*(?:kaynaklar|kaynakça|references|referanslar|bibliography)\b"
    r"|https?://|www\.|\barxiv\b|\bdoi\b|\bet al\b|\bRAG\b"
    r"|\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+(?:ve|and|&)\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)?,?\s*\((?:19|20)\d{2}\)"
    r"|\bkaynak\s+temelli\b",
)
_FAKE_CITE_RE = re.compile(r"\[(?:paper|card|source)_[^\]]*\]", re.I)
_WORD_RE = re.compile(r"[a-zçğıöşü]+")
_TR_MARKERS = frozenset(
    ("ve", "bir", "bu", "için", "ile", "olarak", "daha", "gibi", "olan", "ise", "değil", "veya")
)
# Yatırım tavsiyesi / kesinlik (Kural 1). Metodoloji emirleri ("OOS test yapılmalı") serbest.
_ADVICE_RE = re.compile(
    r"\bgaranti(?:li)?\s+(?:kâr|kar\b|kazanç|getiri)\w*"
    r"|\b(?:kesin(?:likle)?|mutlaka)\s+(?:kazan|kâr|kar\b|getiri)\w*"
    r"|\b(?:satın\s+al|yatırım\s+yap|pozisyon\s+aç)(?:ın|malı|malısın|malısınız)\b"
    r"|\btavsiye\s+ederim\b|\basla\s+kaybet\w*"
    r"|\bguaranteed\s+(?:profit|return)s?\b",
    re.I,
)


def _dumps(obj: Any) -> str:
    """Tek satırlık JSON. U+2028/2029/0085 kaçışlanır: `ensure_ascii=False` bunları ham bırakır
    ama `str.splitlines()` onları satır sonu sayar (Kademe 2 F1-6; korpusta bu karakterli
    parçalar var ve RAG bağlamıyla satıra girebilir)."""
    from app.training.sft_assembly import _LINE_SEPARATOR_ESCAPES

    out = json.dumps(obj, ensure_ascii=False)
    for ch, esc in _LINE_SEPARATOR_ESCAPES:
        out = out.replace(ch, esc)
    return out


@dataclass(frozen=True)
class DistillJob:
    """Üretilecek tek örnek: kip + soru + kökeni (sentetik QA satırı)."""

    mode: str  # "rag" | "plain"
    question: str
    origin_paper_id: str = ""
    origin_chunk_id: str = ""

    def key(self) -> str:
        return f"{self.mode}\x00{self.question}"


@dataclass
class ChatResult:
    content: str
    done_reason: str
    prompt_tokens: int
    output_tokens: int


# --------------------------------------------------------------------------- #
# Soru seçimi
# --------------------------------------------------------------------------- #


def _norm_q(q: str) -> str:
    """Soru dedup anahtarı. Rakam ve noktalama BİLEREK atılır: yalnız sayıyla ayrışan sorular
    ("GARCH(1,1)…" / "GARCH(2,1)…") yakın-kopya sayılır ve biri tutulur (çeşitlilik)."""
    return " ".join(_WORD_RE.findall(q.lower()))


def _question_ok(q: str, mode: str) -> bool:
    max_len = _MAX_MULTIPART_CHARS if len(_MULTIPART_RE.findall(q)) >= 2 else 300
    if not 25 <= len(q) <= max_len or _YEAR_RE.search(q) or _PASSAGE_REF_RE.search(q):
        return False
    if mode == "plain" and _PAPER_BOUND_RE.search(q):
        return False
    return not question_dependent(q, mode)


def select_jobs(
    synth_lines: list[str],
    *,
    n_rag: int,
    n_plain: int,
    seed: int = 0,
    exclude: Callable[[list[str]], set[str]] | None = None,
    extra: list[dict[str, Any]] | None = None,
) -> list[DistillJob]:
    """Sentetik QA + üretilmiş bağımsız sorulardan deterministik iş listesi (makale dengeli).

    ``extra`` → `load_generated_questions` satırları (question/paper_id/chunk_id).
    ``exclude`` → verilen soru listesinden DIŞLANACAKLARIN kümesi (eval sızıntı denetimi).
    Aynı soru iki kipte kullanılmaz; makaleler round-robin sırayla dolaşılır ki birkaç
    makale seti domine etmesin.
    """
    by_paper: dict[str, list[tuple[str, str]]] = {}
    seen: set[str] = set()

    def _add(q: str, paper_id: str, chunk_id: str) -> None:
        nq = _norm_q(q)
        if not q or nq in seen or not _question_ok(q, "rag"):
            return
        seen.add(nq)
        by_paper.setdefault(paper_id, []).append((q, chunk_id))

    for r in extra or []:
        _add(
            str(r.get("question") or "").strip(),
            str(r.get("paper_id") or ""),
            str(r.get("chunk_id") or ""),
        )
    for ln in synth_lines:
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        meta = obj.get("metadata") or {}
        if meta.get("discipline") or not meta.get("synthetic"):
            continue
        _add(
            str(meta.get("question") or "").strip(),
            str(meta.get("paper_id") or ""),
            str(meta.get("chunk_id") or ""),
        )

    rng = random.Random(seed)
    papers = sorted(by_paper)
    rng.shuffle(papers)
    for p in papers:
        rng.shuffle(by_paper[p])
    ordered: list[tuple[str, str, str]] = []
    depth = 0
    while True:
        layer = [(p, *by_paper[p][depth]) for p in papers if depth < len(by_paper[p])]
        if not layer:
            break
        ordered.extend(layer)
        depth += 1

    banned = exclude([q for _, q, _ in ordered]) if exclude else set()
    jobs: list[DistillJob] = []
    n_r = n_p = 0
    for paper_id, q, chunk_id in ordered:
        if q in banned:
            continue
        # Kavramsal (makaleye bağlı olmayan) sorular önce plain kotasını doldurur.
        if n_p < n_plain and _question_ok(q, "plain"):
            jobs.append(DistillJob("plain", q, paper_id, chunk_id))
            n_p += 1
        elif n_r < n_rag:
            jobs.append(DistillJob("rag", q, paper_id, chunk_id))
            n_r += 1
        if n_r >= n_rag and n_p >= n_plain:
            break
    return jobs


# --------------------------------------------------------------------------- #
# Bağımsız soru üretimi (korpus parçalarından)
# --------------------------------------------------------------------------- #
# Neden: sentetik QA sorularının çoğu görünmeyen pasaja bağlı ("Pasajda…") → filtre sonrası
# yalnız ~360 RAG sorusu kalıyordu (2026-10-01 ölçümü). Kapsamlı RAG için makale başına dengeli
# örneklenen parçalardan base model BAĞIMSIZ soru yazar; her 3 sorudan biri a/b/c alt maddeli
# (v13 2×2: LoRA alt maddeleri atlıyordu, D 51/90).

QUESTIONS_FILENAME = "distill_questions.jsonl"
_MIN_CHUNK_CHARS = 600
_JUNK_CHUNK_RE = re.compile(
    r"\b(?:bibliography|references|index|contents|acknowledg|copyright|isbn|kaynakça)\b", re.I
)
_MULTIPART_RE = re.compile(r"(?m)^\s*[a-c]\)")
_MAX_MULTIPART_CHARS = 900


def _chunk_ok(text: str) -> bool:
    """Soru üretmeye değer parça: yeterince uzun, çoğunluğu harf, kaynakça/dizin değil."""
    if len(text) < _MIN_CHUNK_CHARS or _JUNK_CHUNK_RE.search(text[:400]):
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters / len(text) >= 0.6


def sample_chunks(
    chunks: list[Any],
    *,
    per_paper: int,
    seed: int = 0,
    exclude_papers: frozenset[str] = frozenset(),
) -> list[Any]:
    """Makale başına en çok ``per_paper`` uygun parça (deterministik, parça sırası karışık)."""
    by_paper: dict[str, list[Any]] = {}
    for c in chunks:
        if c.paper_id in exclude_papers or not _chunk_ok(str(c.text or "")):
            continue
        by_paper.setdefault(c.paper_id, []).append(c)
    rng = random.Random(seed)
    out: list[Any] = []
    for pid in sorted(by_paper):
        pool = sorted(by_paper[pid], key=lambda c: c.chunk_id)
        out.extend(rng.sample(pool, min(per_paper, len(pool))))
    rng.shuffle(out)
    return out


def build_question_prompt(text: str, *, multipart: bool) -> str:
    """Parçadan bağımsız soru istemi (ASCII-Türkçe talimat — sentetik QA üreticiyle aynı üslup)."""
    shape = (
        "Soru 2-3 alt maddeli olsun; her alt madde yeni satirda 'a) ', 'b) ', 'c) ' ile baslasin. "
        "En az bir alt madde aciklama, biri hesap/ornek/uygulama istesin."
        if multipart
        else "Tek bir acik uclu soru olsun (neden/nasil/ne zaman/hangi kosulda)."
    )
    return (
        "Gorev: Asagidaki akademik parcayi okuyan bir kantitatif arastirmaci/trader'in "
        "soracagi, bu parcanin cevaplamaya yardim ettigi TURKCE bir soru yaz.\n\n"
        f'PARCA:\n"""\n{text[:2500]}\n"""\n\n'
        "Kurallar:\n"
        "- Soru KENDI BASINA anlasilir olsun: 'pasaj', 'metin', 'yukarida', 'bu calisma', "
        "sekil/tablo/denklem/teorem numarasi, sayfa, yazar-yil GECMESIN.\n"
        "- Parcadaki kavrami, yontemi ya da sonucu parcadaki ADIYLA acikca adlandir.\n"
        f"- {shape}\n"
        "- Yatirim tavsiyesi isteyen soru yazma.\n"
        'Cikti: yalnizca JSON {"question": "..."}'
    )


def validate_question(q: str, *, multipart: bool) -> str | None:
    if multipart and len(_MULTIPART_RE.findall(q)) < 2:
        return "altmadde"
    flat = " ".join(q.split())
    if multipart:
        if not 40 <= len(flat) <= _MAX_MULTIPART_CHARS:
            return "uzunluk"
        if _YEAR_RE.search(flat) or _PASSAGE_REF_RE.search(flat):
            return "bağımlı"
        return None
    return None if _question_ok(flat, "rag") else "bağımlı"


def generate_questions(
    out: Path,
    chunks: list[Any],
    *,
    generate: Callable[[str, int], str],
    seed: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Her parça için bir soru üret; ``out``'a ekle. Kesilirse işlenen parçaları atlar."""
    from app.brain.synthetic_qa_builder import _coerce_json_list

    done: set[str] = set()
    if out.exists():
        for ln in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(str(json.loads(ln)["chunk_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    counts: Counter[str] = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        for i, c in enumerate(chunks):
            if c.chunk_id in done:
                continue
            multipart = i % 3 == 0
            try:
                raw = generate(build_question_prompt(str(c.text), multipart=multipart), seed + i)
            except Exception as exc:  # ağ/timeout — parça atlanır
                logger.warning("Soru üretim hatası: %s", exc)
                counts["llm-hata"] += 1
                continue
            items = _coerce_json_list(raw)
            q = str(items[0].get("question", "")).strip() if items else ""
            reason = validate_question(q, multipart=multipart) if q else "boş"
            status = reason or "kabul"
            counts[status] += 1
            rec = {
                "chunk_id": c.chunk_id,
                "paper_id": c.paper_id,
                "multipart": multipart,
                "status": status,
                "question": q,
            }
            fh.write(_dumps(rec) + "\n")
            fh.flush()
            if progress is not None:
                progress(i + 1, len(chunks), status)
    return {"total": len(chunks), "counts": dict(counts)}


def load_generated_questions(path: Path) -> list[dict[str, Any]]:
    """Kabul edilmiş üretilmiş sorular (dosya yoksa boş)."""
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "kabul" and r.get("question"):
            rows.append(r)
    return rows


def eval_leak_filter(questions: list[str]) -> set[str]:
    """Sızıntı kapısının eval kalemleriyle çakışan soruları döndür (liste: `mix_cli`)."""
    from app.evals.profile.leakage import check_leakage
    from app.lora.mix_cli import leakage_eval_items

    rows = [{"messages": [{"role": "user", "content": q}]} for q in questions]
    rep = check_leakage(leakage_eval_items(), rows)
    return {questions[h.train_index] for h in rep.hits}


# --------------------------------------------------------------------------- #
# Doğrulama kapıları
# --------------------------------------------------------------------------- #


def _has_turkish(text: str) -> bool:
    words = _WORD_RE.findall(text.lower())
    return bool(words) and sum(w in _TR_MARKERS for w in words) / len(words) >= 0.02


def validate_distilled(
    answer: str,
    job_mode: str,
    res: ChatResult,
    chunks: list[Any] | None = None,
    *,
    system: str = "",
) -> str | None:
    """Kabulse None, değilse red gerekçesi. Sıra: ucuz → pahalı."""
    if res.done_reason != "stop":
        return "kesik"
    if res.prompt_tokens + res.output_tokens > MAX_TOTAL_TOKENS:
        return "bütçe"
    if len(answer) < MIN_ANSWER_CHARS:
        return "kısa"
    if _CJK_RE.search(answer):
        return "cjk"
    if not _has_turkish(answer):
        return "dil"
    return content_gate(answer, job_mode, chunks, system)


def content_gate(answer: str, job_mode: str, chunks: list[Any] | None, system: str) -> str | None:
    """Üretim bağımsız içerik kapıları — hem üretimde hem `revalidate_line`'da aynı."""
    from app.training.adapter_eval import _is_degenerate

    if _ADVICE_RE.search(answer):
        return "tavsiye"
    audit = _audit_gate(answer)
    if audit:
        return audit
    if _is_degenerate(_scaffold_free(answer, system)):
        return "tekrar"
    return citation_gate(answer, job_mode, chunks)


def citation_gate(answer: str, job_mode: str, chunks: list[Any] | None) -> str | None:
    """Atıf kapısı (Kademe 2 F3-1/2/3, 2026-10-01).

    rag: her atıfın (paper_id, chunk_id) ÇİFTİ getirilenlerde birebir olmalı (strict), en az
    `MIN_UNIQUE_CITATIONS` FARKLI kaynak (aynı parçayı 4× anmak sayılmaz) ve çözümlenemeyen
    kaynak-benzeri köşeli olmamalı. plain: kaynak verilmediği için HİÇ atıf olmamalı.
    """
    from app.brain.answer_quality import parse_citations, verify_citations

    if job_mode == "rag":
        check = verify_citations(answer, chunks or [], strict=True)
        if check.has_unsupported or check.malformed:
            return "uydurma-atıf"
        if check.n_unique < MIN_UNIQUE_CITATIONS:
            return "atıf-az"
        return None
    cites, malformed = parse_citations(answer)
    if cites or malformed or _FAKE_CITE_RE.search(answer):
        return "uydurma-atıf"
    if _FABRICATED_SOURCE_RE.search(answer):
        return "uydurma-kaynak"
    return None


_HEADING_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s|\d+\.\s|[-*•]\s*\*\*[^*]{1,80}\*\*:?\s*$|\*\*[^*]{1,80}\*\*:?\s*$)"
)
_BRACKET_ANY_RE = re.compile(r"\[[^\[\]\n]{1,400}\]")


def _scaffold_free(answer: str, system: str) -> str:
    """Tekrar denetimi için iskeleti ayıkla (Kademe 2 F4-6 / F1: `_is_degenerate` v14 damıtma
    cevaplarının 8 "tekrar" reddinin ≥7'sinde yanlış pozitifti).

    Atılanlar: başlık satırları (markdown/numaralı/kalın etiket), sistem isteminde harfiyen geçen
    satırlar ("Bu bulgu doğrudan trading kuralına çevrilemez." gibi zorunlu cümleler) ve köşeli
    atıflar (`[p:c, s.9]` içindeki nokta cümleyi bölüp aynı kaynağa tekrarlı atfı "döngü"
    gösteriyordu). İçerik cümlelerinin tekrarı aynen yakalanmaya devam eder.
    """
    sys_norm = " ".join(_WORD_RE.findall(system.lower()))
    kept: list[str] = []
    for line in answer.splitlines():
        if _HEADING_LINE_RE.match(line):
            continue
        norm = " ".join(_WORD_RE.findall(line.lower()))
        if norm and len(norm) >= 12 and norm in sys_norm:
            continue
        kept.append(_BRACKET_ANY_RE.sub("", line))
    return "\n".join(kept)


def _audit_gate(answer: str) -> str | None:
    """pretrain-gate'in satır kapılarıyla aynı tarayıcılar (Kademe 2 F1-4): lora-audit Gate 7
    (sır/PII/finansal yönlendirme, ör. "risk yok") + Gate 5 garanti vaadi. Damıtma burada
    elemezse kanonik birleştirme pretrain-gate'te NO-GO olur."""
    from app.lora.math_verifier import CERTAINTY_ONLY_PHRASES, overconfident_hits
    from app.lora.safety_scanner import scan_for_secrets, tr_fold

    if not scan_for_secrets(answer).passed:
        return "gate7"
    if [h for h in overconfident_hits(tr_fold(answer)) if h not in CERTAINTY_ONLY_PHRASES]:
        return "garanti"
    return None


def _chunks_from_prompt(user_prompt: str) -> list[Any]:
    """RAG isteminin kaynak başlıklarından (paper_id, chunk_id) çiftleri → hafif parça nesneleri.

    `_format_context` her kaynağı `[paper_id:chunk_id(, s.N)]` başlığıyla verir; yeniden
    doğrulama, modele GERÇEKTEN gösterilen çiftlere göre yapılır.
    """
    from types import SimpleNamespace

    from app.brain.answer_quality import parse_citations

    head = user_prompt.split("QUESTION / SORU:", 1)[0]
    pairs, _ = parse_citations(head)
    return [SimpleNamespace(paper_id=p, chunk_id=c) for p, c in dict.fromkeys(pairs)]


def revalidate_line(line: str) -> str | None:
    """Kabul edilmiş öz-damıtma satırını GÜNCEL içerik kapılarından yeniden geçir.

    Birleştirme (`sft_assembly`) bunu çağırır: kapılar sıkılaştırıldığında (Kademe 2 F1/F3)
    eski kapıyla kabul edilmiş satırlar yeniden üretim gerektirmeden elenir. Soru bağımlılığı
    (aksansız yazım + numaralı kitap göndermesi) da burada denetlenir. Kabulse None, değilse
    red gerekçesi.
    """
    try:
        obj = json.loads(line)
        msgs = obj["messages"]
        meta = obj.get("metadata") or {}
        answer = str(msgs[-1]["content"])
        user = next(str(m["content"]) for m in msgs if m.get("role") == "user")
        system = next((str(m["content"]) for m in msgs if m.get("role") == "system"), "")
    except (json.JSONDecodeError, KeyError, IndexError, StopIteration, TypeError):
        return "okunamadı"
    mode = str(meta.get("mode") or "")
    if mode not in {"rag", "plain"}:
        return "kip-yok"
    if question_dependent(str(meta.get("question") or ""), mode):
        return "bağımlı-soru"
    chunks = _chunks_from_prompt(user) if mode == "rag" else None
    return content_gate(answer, mode, chunks, system)


# --------------------------------------------------------------------------- #
# Üretim
# --------------------------------------------------------------------------- #


def ollama_chat(
    host: str, model: str, messages: list[dict[str, str]], *, seed: int, keep_alive: str = "5m"
) -> ChatResult:
    """Tek /api/chat çağrısı (akışsız). `done_reason` + token sayıları kesilme kapısı için."""
    import httpx

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {
            "temperature": DEFAULT_TEMPERATURE,
            "seed": seed,
            "num_predict": MAX_ANSWER_TOKENS,
            "num_ctx": NUM_CTX,
        },
    }
    with httpx.Client(timeout=1800) as client:
        r = client.post(f"{host.rstrip('/')}/api/chat", json=payload)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    return ChatResult(
        content=str((data.get("message") or {}).get("content") or "").strip(),
        done_reason=str(data.get("done_reason") or ""),
        prompt_tokens=int(data.get("prompt_eval_count") or 0),
        output_tokens=int(data.get("eval_count") or 0),
    )


def build_messages(job: DistillJob, chunks: list[Any] | None) -> list[dict[str, str]]:
    """Canlı hatla aynı istem: rag → `build_rag_prompt`; plain → genel Hektor sistem istemi."""
    if job.mode == "rag":
        from app.brain.rag_answerer import build_rag_prompt
        from app.config import get_settings

        system, user = build_rag_prompt(
            job.question, chunks or [], reorder=get_settings().rag_reorder_context
        )
    else:
        from app.lora.dataset_builder import SYSTEM_PROMPT

        system, user = SYSTEM_PROMPT + PLAIN_NO_SOURCE_NOTE, job.question
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def distill_one(
    job: DistillJob,
    *,
    chat: Callable[[list[dict[str, str]], int], ChatResult],
    retrieve: Callable[[str], list[Any]],
    seed: int,
    teacher: str,
    provenance: Callable[[], dict[str, Any]] | None = None,
) -> tuple[str, str | None, str]:
    """(durum, JSONL satırı|None, ham cevap). Kabul edilmeyen iş eğitim satırı üretmez.

    ``provenance`` → son retrieval'ın izi (kullanılan sorgu + çeviri durumu; Kademe 2 F3-8).
    """
    chunks: list[Any] | None = None
    retrieval: dict[str, Any] = {}
    if job.mode == "rag":
        try:
            chunks = retrieve(job.question)
        except Exception as exc:  # retrieval hatası tek işi düşürür, koşuyu değil (F3-6)
            logger.warning("Öz-damıtma retrieval hatası: %s", exc)
            return "retrieval-hata", None, ""
        retrieval = dict(provenance()) if provenance is not None else {}
        if not chunks:
            return "retrieval-boş", None, ""
    messages = build_messages(job, chunks)
    try:
        res = chat(messages, seed)
    except Exception as exc:  # ağ/timeout — iş atlanır, koşu sürer
        logger.warning("Öz-damıtma üretim hatası: %s", exc)
        return "llm-hata", None, ""
    reason = validate_distilled(res.content, job.mode, res, chunks, system=messages[0]["content"])
    if reason:
        return reason, None, res.content
    row = {
        "messages": [*messages, {"role": "assistant", "content": res.content}],
        "metadata": {
            "synthetic": True,
            "distilled": True,
            "mode": job.mode,
            "teacher": teacher,
            "question": job.question,
            "paper_id": job.origin_paper_id,
            "source_id": job.origin_paper_id,
            "chunk_id": job.origin_chunk_id,
            "context_chunk_ids": [c.chunk_id for c in chunks or []],
            "seed": seed,
            "tokens": res.prompt_tokens + res.output_tokens,
            **({"retrieval": retrieval} if retrieval else {}),
        },
    }
    return "kabul", _dumps(row), res.content


def _jobs_sha(jobs: list[DistillJob]) -> str:
    return hashlib.sha256("\n".join(j.key() for j in jobs).encode("utf-8")).hexdigest()


def meta_path_for(out: Path) -> Path:
    return out.with_name(out.stem + ".meta.json")


def distill_files(lora_dir: Path) -> list[Path]:
    """Eğitime giren öz-damıtma dosyaları: `distill_qa.jsonl` + önceki turlar
    (`distill_qa.<tur>.jsonl`, ör. kapılar değişince yeniden başlatılan koşunun ilk kısmı).
    Red günlükleri (`*.rejects.jsonl`) hariç; sıra deterministik (ada göre)."""
    return sorted(
        p
        for p in lora_dir.glob("distill_qa*.jsonl")
        if not p.name.endswith(".rejects.jsonl") and p.is_file()
    )


def norm_question(q: str) -> str:
    """Soru dedup anahtarı (genel ad; bkz. `_norm_q`)."""
    return _norm_q(q)


def progress_done(out: Path) -> int:
    """``out`` koşusunda kaç iş tamamlanmış (meta yoksa 0)."""
    meta_path = meta_path_for(out)
    if not meta_path.exists():
        return 0
    try:
        return int(json.loads(meta_path.read_text(encoding="utf-8")).get("done") or 0)
    except (OSError, ValueError, TypeError):
        return 0


def used_questions(lora_dir: Path, *, skip: Path | None = None) -> set[str]:
    """Öz-damıtma dosyalarındaki soruların dedup anahtarları (yeni turda aynı soru yeniden
    üretilmesin). ``skip`` → koşunun kendi çıktısı (sürdürmede iş listesi sabit kalsın)."""
    out: set[str] = set()
    for p in distill_files(lora_dir):
        if skip is not None and p.resolve() == skip.resolve():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                q = (json.loads(ln).get("metadata") or {}).get("question")
            except (json.JSONDecodeError, AttributeError):
                continue
            if q:
                out.add(_norm_q(str(q)))
    return out


def generation_config() -> dict[str, Any]:
    """Üretim yapılandırması parmak izi (meta'da; sürdürmede eşitlik şartı — F3-8)."""
    from app.brain.rag_answerer import build_rag_prompt
    from app.memory.query_translation import SYSTEM as TRANSLATE_SYSTEM
    from app.memory.rag_version import current_rag_snapshot

    def _h(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    rag_system, _ = build_rag_prompt("", [])
    snap = current_rag_snapshot()
    return {
        "rag_version": snap.rag_version,
        "rag_system": _h(rag_system),
        "plain_note": _h(PLAIN_NO_SOURCE_NOTE),
        "translate_system": _h(TRANSLATE_SYSTEM),
        "max_total_tokens": MAX_TOTAL_TOKENS,
    }


def rejects_path_for(out: Path) -> Path:
    """Red günlüğü (denetim + eşik ayarı için; eğitime GİRMEZ)."""
    return out.with_name(out.stem + ".rejects.jsonl")


def run_distill(
    out: Path,
    jobs: list[DistillJob],
    *,
    chat: Callable[[list[dict[str, str]], int], ChatResult],
    retrieve: Callable[[str], list[Any]],
    teacher: str,
    seed: int = 0,
    limit: int = 0,
    progress: Callable[[int, int, str], None] | None = None,
    provenance: Callable[[], dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """İşleri sırayla üret; kabul edilenleri ``out``'a ekle. Kesilirse kaldığı işten sürer.

    Meta dosyası iş listesinin hash'ini + üretici modeli + üretim yapılandırmasını (``config``:
    RAG sürümü, istem hash'leri) tutar: biri değiştiyse devam REDDEDİLİR (tek dosyada iki
    üretici / iki iş listesi / iki retrieval yapılandırması karışmasın — Kademe 2 F3-8).
    """
    from app.brain.synthetic_enrich import is_adapter_model

    if is_adapter_model(teacher):
        raise ValueError(
            f"Üretici {teacher!r} bir Hektor adapter'ı — öz-damıtma yalnız BASE modelle yapılır."
        )
    meta_path = meta_path_for(out)
    sha = _jobs_sha(jobs)
    meta: dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    done = int(meta.get("done") or 0)
    if done:
        if meta.get("jobs_sha256") != sha or meta.get("teacher") != teacher:
            raise ValueError(
                f"{out.name} başka bir iş listesi/üreticiyle başlamış (meta uyuşmuyor). "
                f"Sürdürmek için aynı parametreleri ver ya da {out.name} + {meta_path.name} "
                "dosyalarını silip baştan başla."
            )
        if (config or {}) != (meta.get("config") or {}):
            raise ValueError(
                f"{out.name} farklı bir üretim yapılandırmasıyla başlamış (RAG/istem): "
                f"{meta.get('config')} ≠ {config}. Aynı ayarlarla sürdür ya da baştan başla."
            )
    elif out.exists() and out.read_text(encoding="utf-8").strip():
        raise ValueError(f"{out} dolu ama meta yok — üzerine yazılmaz; önce taşı/sil.")
    # Yarım son satır (kesilen yazma) atılır.
    if out.exists():
        raw = out.read_text(encoding="utf-8")
        if raw and not raw.endswith("\n"):
            out.write_text(raw[: raw.rfind("\n") + 1], encoding="utf-8")
    counts: Counter[str] = Counter(meta.get("counts") or {})
    end = len(jobs) if limit <= 0 else min(len(jobs), done + limit)

    def _write_meta(n: int) -> None:
        meta_path.write_text(
            json.dumps(
                {
                    "jobs_sha256": sha,
                    "teacher": teacher,
                    "config": config or {},
                    "seed": seed,
                    "total": len(jobs),
                    "done": n,
                    "counts": dict(counts),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    _write_meta(done)
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        out.open("a", encoding="utf-8") as fh,
        rejects_path_for(out).open("a", encoding="utf-8") as rej,
    ):
        for i in range(done, end):
            status, line, raw_answer = distill_one(
                jobs[i],
                chat=chat,
                retrieve=retrieve,
                seed=seed + i,
                teacher=teacher,
                provenance=provenance,
            )
            if line is not None:
                fh.write(line + "\n")
                fh.flush()
            else:
                rec = {"i": i, "mode": jobs[i].mode, "status": status}
                rec |= {"question": jobs[i].question, "answer": raw_answer}
                rej.write(_dumps(rec) + "\n")
                rej.flush()
            counts[f"{jobs[i].mode}:{status}"] += 1
            _write_meta(i + 1)
            if progress is not None:
                progress(i + 1, len(jobs), f"{jobs[i].mode}:{status}")
    return {"total": len(jobs), "done": end, "counts": dict(counts), "teacher": teacher}
