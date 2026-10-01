"""Eğitim ↔ eval sızıntı denetimi (zorunlu).

Eval sorularının/cevaplarının LoRA eğitim verisine girmemesi için beş kontrol:

1. exact duplicate       — metin birebir aynı
2. normalized duplicate  — küçük harf + noktalama/boşluk/aksan normalize edildikten sonra aynı
3. contained             — normalize eval metni (≥ ``_MIN_CONTAIN_LEN`` karakter) daha uzun bir
                           eğitim metninin İÇİNDE birebir geçiyor (ör. ``BAĞLAM: … SORU: <soru>``
                           biçimli kullanıcı mesajı ya da cevaba gömülmüş referans cevap)
4. near-duplicate        — kelime 3-gram Jaccard ≥ eşik (çevrimdışı, deterministik) ve
                           opsiyonel olarak enjekte edilen embedding kosinüsü ≥ eşik ("semantik")
5. source-id overlap     — eval kaydının kaynak kimliği (expected_document_ids /
                           source_provenance'taki paper id) eğitim örneğinin source_id'si ile aynı

Kullanıcı mesajlarındaki ``SORU:`` bölümü ayrıca AYRI bir eğitim metni olarak çıkarılır; böylece
bağlam pasajıyla paketlenmiş bir soru exact/normalized/near-dup denetimlerinden kaçamaz.
Kısa metinler (< ``_MIN_TEXT_LEN``) bilinçli olarak yalnız exact/normalized ile denetlenir
("Evet." gibi ifadeler her yerde geçer; bulanık/içerme eşleşmesi yanlış pozitif üretir).

Eğitim örneği biçimi: ``{"messages": [...], "metadata": {"source_id": ...}}`` (LoRAExample)
veya ``{"question"/"prompt"/"instruction", "answer"/"output"/"response"}``.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.evals.profile.schema import EvalItem

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

DEFAULT_JACCARD = 0.8
DEFAULT_COSINE = 0.95
_MIN_TEXT_LEN = 20  # çok kısa metinler (ör. "Evet.") near-dup için anlamsız
# İçerme (substring) denetimi için normalize eval metninin en kısa uzunluğu: daha kısa ifadeler
# (ör. "standart sapma nedir") uzun pasajlarda doğal olarak geçer → yanlış pozitif.
_MIN_CONTAIN_LEN = 30
# ``BAĞLAM: …\n\nSORU: <soru>`` biçimli kullanıcı mesajında soru bölümünün işareti. Canlı RAG
# biçimi ``QUESTION / SORU:`` da tanınır (Kademe 2 F1-5/F4-4: eskiden '' dönüyordu → soru
# bağlamın içinde kalıyor, yakın-kopya denetimi kör oluyordu).
_QUESTION_MARKER = re.compile(
    r"(?:^|\n)[ \t]*(?:SORU|QUESTION)(?:[ \t]*/[ \t]*(?:SORU|QUESTION))?[ \t]*:[ \t]*",
    re.IGNORECASE,
)
# Çok parçalı eval sorusunun alt maddeleri ("a) …", "b) …") ayrı ayrı denetlenir (F4-3).
_SUBPART_SPLIT = re.compile(r"\n(?=[ \t]*[a-eA-E]\)[ \t])")
_SUBPART_PREFIX = re.compile(r"^[ \t]*[a-eA-E]\)[ \t]*")


@dataclass
class LeakageHit:
    kind: str  # exact | normalized | contained | near_duplicate | semantic | source_id
    eval_id: str
    train_index: int
    detail: str
    split: str = "train"  # isabetin bulunduğu eğitim dosyası (train | valid)


@dataclass
class LeakageReport:
    n_eval: int
    n_train: int
    hits: list[LeakageHit] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.hits

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for h in self.hits:
            out[h.kind] = out.get(h.kind, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_eval": self.n_eval,
            "n_train": self.n_train,
            "clean": self.clean,
            "counts": self.counts(),
            "hits": [asdict(h) for h in self.hits],
        }


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.casefold())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.replace("ı", "i")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = normalize_text(text).split()
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cos(u: Sequence[float], v: Sequence[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return 0.0
    return sum(x * y for x, y in zip(u, v, strict=False)) / (nu * nv)


@dataclass
class TrainText:
    index: int
    texts: list[str]
    source_id: str


def question_segment(text: str) -> str:
    """``… SORU: <soru>`` biçimli mesajda son ``SORU:`` işaretinden sonraki bölüm (yoksa "")."""
    last: re.Match[str] | None = None
    for m in _QUESTION_MARKER.finditer(text):
        last = m
    return text[last.end() :].strip() if last is not None else ""


def extract_train_texts(example: dict[str, Any], index: int) -> TrainText:
    texts: list[str] = []
    for msg in example.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant"):
            content = str(msg.get("content") or "")
            texts.append(content)
            if msg.get("role") == "user":
                q = question_segment(content)
                if q and q != content.strip():
                    texts.append(q)
    for key in ("question", "prompt", "instruction", "input", "answer", "output", "response"):
        if example.get(key):
            texts.append(str(example[key]))
    raw_meta = example.get("metadata")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    source = str(
        meta.get("source_id") or meta.get("paper_id") or example.get("source_id") or ""
    ).strip()
    return TrainText(index=index, texts=[t for t in texts if t.strip()], source_id=source)


def load_train_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _eval_texts(item: EvalItem) -> list[str]:
    texts = [t for t in (item.question, item.reference_answer, *item.accepted_answers) if t.strip()]
    # Alt maddeler: eğitim satırı tek bir alt maddeye eşitse bütün-soru karşılaştırması
    # kaçırıyordu (Kademe 2 F4-3; S12 a) → temiz). Gövde "a) " öneki olmadan eklenir.
    parts = _SUBPART_SPLIT.split(item.question)
    for part in parts[1:]:
        body = _SUBPART_PREFIX.sub("", part).strip()
        if len(body) >= _MIN_TEXT_LEN:
            texts.append(body)
    return texts


def _eval_sources(item: EvalItem) -> set[str]:
    src = {s.strip() for s in item.expected_document_ids if s.strip()}
    m = re.search(r"(?:paper|source)[_ ]?id\s*[:=]\s*([\w.\-]+)", item.source_provenance, re.I)
    if m:
        src.add(m.group(1))
    return src


def check_leakage(
    eval_items: Iterable[EvalItem],
    train_examples: Sequence[dict[str, Any]],
    *,
    jaccard_threshold: float = DEFAULT_JACCARD,
    embed_fn: EmbedFn | None = None,
    cosine_threshold: float = DEFAULT_COSINE,
) -> LeakageReport:
    items = list(eval_items)
    train = [extract_train_texts(ex, i) for i, ex in enumerate(train_examples)]
    report = LeakageReport(n_eval=len(items), n_train=len(train))

    exact_idx: dict[str, int] = {}
    norm_idx: dict[str, int] = {}
    shingle_sets: dict[int, set[tuple[str, ...]]] = {}
    shingle_inv: dict[tuple[str, ...], list[int]] = {}
    source_idx: dict[str, int] = {}
    flat_train: list[tuple[int, str]] = []
    shingle_owner: list[int] = []
    # İçerme denetimi: normalize eğitim metinleri "\n" ile tek dizede birleşir (normalize metin
    # satır sonu içermez → bir eşleşme iki metni aşamaz); ofset → eğitim satırı ikili aramayla.
    corpus_parts: list[str] = []
    corpus_starts: list[int] = []
    corpus_owner: list[int] = []
    corpus_len = 0
    for tt in train:
        if tt.source_id:
            source_idx.setdefault(tt.source_id, tt.index)
        for t in tt.texts:
            exact_idx.setdefault(t.strip(), tt.index)
            nt = normalize_text(t)
            norm_idx.setdefault(nt, tt.index)
            if len(nt) >= _MIN_CONTAIN_LEN:
                corpus_parts.append(nt)
                corpus_starts.append(corpus_len)
                corpus_owner.append(tt.index)
                corpus_len += len(nt) + 1
            if len(t) >= _MIN_TEXT_LEN:
                key = len(shingle_sets)
                shingle_sets[key] = _shingles(t)
                shingle_owner.append(tt.index)
                for gram in shingle_sets[key]:
                    shingle_inv.setdefault(gram, []).append(key)
                flat_train.append((tt.index, t))
    corpus = "\n".join(corpus_parts)

    def contained_in(n_text: str) -> int | None:
        pos = corpus.find(n_text)
        if pos < 0:
            return None
        return corpus_owner[bisect.bisect_right(corpus_starts, pos) - 1]

    for item in items:
        flagged: set[tuple[str, int]] = set()  # (tür, eğitim satırı) kayıt başına bir kez

        def hit(
            kind: str,
            idx: int,
            detail: str,
            _seen: set[tuple[str, int]] = flagged,
            _eval_id: str = item.id,
        ) -> None:
            if (kind, idx) not in _seen:
                _seen.add((kind, idx))
                report.hits.append(LeakageHit(kind, _eval_id, idx, detail))

        for text in _eval_texts(item):
            if text.strip() in exact_idx:
                hit("exact", exact_idx[text.strip()], text[:80])
                continue
            n = normalize_text(text)
            if n and n in norm_idx:
                hit("normalized", norm_idx[n], text[:80])
                continue
            if len(n) >= _MIN_CONTAIN_LEN:
                owner = contained_in(n)
                if owner is not None:
                    hit("contained", owner, f"eğitim metni içinde: {text[:60]}")
                    continue
            if len(text) < _MIN_TEXT_LEN:
                continue
            sh = _shingles(text)
            # Ters indeks: yalnız en az bir 3-gram paylaşan eğitim metinleri aday.
            candidates = sorted({k for g in sh for k in shingle_inv.get(g, ())})
            for key in candidates:
                j = jaccard(sh, shingle_sets[key])
                if j >= jaccard_threshold:
                    hit("near_duplicate", shingle_owner[key], f"jaccard={j:.2f}: {text[:60]}")
                    break
        for src in _eval_sources(item):
            if src in source_idx:
                hit("source_id", source_idx[src], f"source_id={src}")

    if embed_fn is not None and flat_train and items:
        eval_pairs = [
            (it.id, t) for it in items for t in _eval_texts(it) if len(t) >= _MIN_TEXT_LEN
        ]
        if eval_pairs:
            e_vecs = embed_fn([t for _, t in eval_pairs])
            t_vecs = embed_fn([t for _, t in flat_train])
            already = {(h.eval_id, h.train_index) for h in report.hits}
            for (eid, etext), ev in zip(eval_pairs, e_vecs, strict=True):
                for (tidx, _), tv in zip(flat_train, t_vecs, strict=True):
                    c = _cos(ev, tv)
                    if c >= cosine_threshold and (eid, tidx) not in already:
                        already.add((eid, tidx))
                        report.hits.append(
                            LeakageHit("semantic", eid, tidx, f"cos={c:.3f}: {etext[:60]}")
                        )
    return report
