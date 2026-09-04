"""Sentetik QA üretici — chunk'lardan LLM ile çeşitli, grounded eğitim örneği üret.

Amaç (bkz. docs/RAG_EGITIM_YENIDEN_TASARIM.md, Faz A6): 15-50 deterministik şablon
örneğinden, makale chunk'larından üretilen ~1000-2000 çeşitli SFT örneğine geçmek.
LoRA'nın anlamlı olması için pratik eşik ~1000 örnektir; bu modül o "büyüme motoru".

Tasarım ilkeleri:
- **Grounded:** her cevap YALNIZ verilen pasaja dayanır; uydurma yasak (CLAUDE.md 7).
  Prompt katı; ayrıca dil-dayanıklı bir "anchor" backstop'u uydurmayı eler.
- **RAG-stili:** örnek, pasajı BAĞLAM olarak kullanıcı mesajına gömer → model
  bağlamı kullanmayı öğrenir (nihai hibrit önerisiyle uyumlu).
- **Persona çeşitliliği:** aynı pasajdan farklı bakış açılarıyla (kantitatif,
  risk, backtester, şüpheci) sorular → tek-tip olmayan veri.
- **Çevrimdışı test edilebilir:** LLM enjekte edilebilir; gerçek üretim Ollama ister.

NOT: Gerçek üretim LLM çağrısı yapar (ağır değil ama yavaş olabilir). Bu modül
örnek ÜRETİR; eğitimi başlatmaz (CLAUDE.md kural 8).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.lora.dataset_builder import SYSTEM_PROMPT, LoRAExample
from app.lora.quality_filter import QualityFilter

logger = logging.getLogger(__name__)

# Aynı pasajdan farklı bakış açıları — tek-tip olmayan veri için.
PERSONAS: list[tuple[str, str]] = [
    (
        "kantitatif araştırmacı",
        "Formülleri, değişkenleri, parametreleri ve istatistiksel kanıtı vurgula.",
    ),
    (
        "risk yöneticisi",
        "Aşağı yön riski, maliyet, drawdown, kaldıraç ve belirsizliğe odaklan.",
    ),
    (
        "backtester",
        "Test edilebilirlik: piyasa, zaman dilimi, OOS, look-ahead, işlem maliyeti.",
    ),
    (
        "şüpheci denetçi",
        "İddiayı sorgula; overfit, survivorship bias, veri sızıntısı, örneklem yetersizliğini ara.",
    ),
]

# Minimum cevap uzunluğu (quality_filter ile aynı ruh).
_MIN_ANSWER_CHARS = 60
# Backstop grounding: pasajda anchor varsa cevapta en az bu kadarı paylaşılmalı.
_MIN_SHARED_ANCHORS = 1
# Pasajın "anchor say" eşiği — altındaysa grounding'i yalnız uzunlukla geç (kısa pasaj).
_MIN_CHUNK_ANCHORS = 3

_TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+")
_NUM_RE = re.compile(r"\d")
# "Anlamlı" sayı: ondalık (2.3 / 0,75), yüzde (47%) veya ≥2 haneli tam sayı (14, 2021).
# Tek haneli (1-9) sayılar genelde sıralama/sayım amaçlıdır (uydurma metrik değil),
# bu yüzden hariç tutulur — yoksa "2 varsayım" gibi cümleler yanlış elenir.
_SIG_NUM_RE = re.compile(r"\d+[.,]\d+|\d+%|\d{2,}")

# One-shot DOLU örnek — qwen3:4b JSON-mode'da prompt'u "doğrula" sanıp {"error":...}
# döndürebiliyor; somut doldurulmuş örnek modeli ÜRETİME yönlendirir (canlı test edildi).
# Obje formu ({"pairs":[...]}) qwen'de çıplak diziden daha güvenilir.
# v5 SIZINTI DERSİ: eski örnek HER iki cevaba da "Pasaja gore" öneki koyuyordu →
# model bunu KOŞULSUZ açılış olarak ezberledi ve bağlamsız evalde bile "Pasaja gore..."
# dedi (v5 regresyonu). Açılışlar artık ÇEŞİTLİ ve öneksiz (grounding _is_grounded ile
# zaten ayrı garanti edilir; açılış token'ı sabitlenmemeli).
_ONESHOT_EXAMPLE = (
    '{"pairs": [\n'
    '  {"question": "Bu calismada ATR hangi amacla kullanilir?", '
    '"answer": "ATR, volatiliteyi olcmek ve pozisyon buyuklugunu ayarlamak icin '
    'kullaniliyor; yuksek ATR daha genis stop anlamina gelir."},\n'
    '  {"question": "Yontemin temel varsayimi nedir?", '
    '"answer": "Yontem, gecmis volatilitenin gelecekteki riski temsil ettigini '
    'varsayar; bu varsayim rejim degisiminde zayiflayabilir."}\n'
    "]}"
)
# Pasaj prompt'a gömülürken üst sınır (CPU üretim süresini sınırlar).
_MAX_PASSAGE_CHARS = 1800


def _anchor_tokens(text: str) -> set[str]:
    """Dil-değişmez "anchor" token'lar: sayılar + teknik/uzun terimler.

    Makaleler İngilizce, cevaplar Türkçe olabildiğinden ham token örtüşmesi
    güvenilmez. Sayılar (1.5, 14, 0.7), kısaltmalar (ATR, RSI) ve uzun teknik
    terimler (momentum, volatility, kurtosis) çeviriden büyük ölçüde sağ çıkar.
    """
    anchors: set[str] = set()
    for tok in _TOKEN_RE.findall(text):
        is_anchor = (
            bool(_NUM_RE.search(tok))  # sayı içeren → güçlü anchor (14, 0.7)
            or (tok.isupper() and len(tok) >= 2)  # kısaltma (ATR, MACD)
            or len(tok) >= 5  # uzun teknik terim (loanword olasılığı yüksek)
        )
        if is_anchor:
            anchors.add(tok.lower())
    return anchors


_EN_THOUSAND_RE = re.compile(r"^\d{1,3}(,\d{3})+$")


def _significant_numbers(text: str) -> set[str]:
    """Metindeki "anlamlı" sayıları normalize ederek döndür.

    İngilizce binlik ayraç ('1,000', '10,000') ile ondalık virgülü ('1,5') ayırt eder:
    N,DDD kalıbı (binlik) → virgülsüz tam sayı ('1000'), aksi halde virgül → nokta.
    Bu sayede '10,000 iterations' pasajından doğru cevap '10000' reddedilmez.
    """
    out: set[str] = set()
    for m in _SIG_NUM_RE.findall(text):
        if _EN_THOUSAND_RE.match(m):
            out.add(m.replace(",", ""))
        else:
            out.add(m.replace(",", ".").rstrip("%"))
    return out


def _is_grounded(answer: str, chunk_text: str) -> bool:
    """Cevabın pasaja dayandığına dair dil-dayanıklı grounding backstop'u.

    Asıl grounding garantisi katı prompt'tadır; bu, uydurmayı eleyen ek savunmadır.
    Üç kapı (hepsi geçilmeli):
      1) **Sayı-altküme:** cevaptaki HER anlamlı sayı (oran/yüzde/≥2 haneli) pasajda
         da geçmeli. Uydurulan metriklerin (ör. "Sharpe 2.3", "%47 getiri") asıl
         riski budur — pasajda olmayan bir sayı varsa RED (CLAUDE.md kural 7).
      2) **Bilgi-fakir pasaj reddi:** pasajda yeterli anchor yoksa grounded QA
         üretilemez → RED (eskiden gevşetiliyordu; en çok orada halüsinasyon olur).
      3) **Anchor örtüşmesi:** cevap, pasajla en az bir anchor (sayı/kısaltma/uzun
         terim) paylaşmalı. Eşik düşük: İngilizce pasaj + Türkçe cevap çapraz-dil
         örtüşmesi zaten sınırlı (yalnız sayı/özel ad/loanword eşleşir).
    """
    # 1) Sayı-altküme kapısı (uydurulan metriklere karşı en güçlü savunma).
    answer_nums = _significant_numbers(answer)
    if answer_nums and not answer_nums.issubset(_significant_numbers(chunk_text)):
        return False

    chunk_anchors = _anchor_tokens(chunk_text)
    # 2) Bilgi-fakir pasaj → grounded QA üretilemez, RED (artık uzunlukla geçmez).
    if len(chunk_anchors) < _MIN_CHUNK_ANCHORS:
        return False

    # 3) Anchor örtüşmesi.
    answer_anchors = _anchor_tokens(answer)
    shared = chunk_anchors & answer_anchors
    return len(shared) >= _MIN_SHARED_ANCHORS


def _coerce_json_list(raw: str) -> list[dict]:
    """LLM çıktısından QA listesi çıkar — kod çiti / önek-sonek toleranslı."""
    text = raw.strip()
    # ```json ... ``` çitlerini soy
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Doğrudan dene
    for candidate in (text, _extract_bracketed(text)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            # {"qa": [...]} veya {"pairs": [...]} gibi sarmalayıcı
            for key in ("qa", "pairs", "items", "data"):
                if isinstance(data.get(key), list):
                    return [d for d in data[key] if isinstance(d, dict)]
            return [data]
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    return []


def _extract_bracketed(text: str) -> str:
    """İlk '[' ile son ']' arasını döndür (gürültülü çıktı için)."""
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


@dataclass
class _RawQA:
    question: str
    answer: str


def _normalize_pairs(items: list[dict]) -> list[_RawQA]:
    """Esnek anahtar isimlerinden (question/q/soru, answer/a/cevap) QA çıkar."""
    out: list[_RawQA] = []
    for it in items:
        q = str(it.get("question") or it.get("q") or it.get("soru") or "").strip()
        a = str(it.get("answer") or it.get("a") or it.get("cevap") or "").strip()
        if q and a:
            out.append(_RawQA(question=q, answer=a))
    return out


class SyntheticQABuilder:
    """Chunk → çeşitli, grounded LoRAExample üretici.

    Args:
        llm: Üretici LLM (varsayılan `LocalLLM`). Çevrimdışı test için stub enjekte et.
        min_answer_chars: Minimum cevap uzunluğu (altındakiler elenir).
        seed: Determinizm tabanı (CLAUDE.md kural 6). Her chunk'a `seed + persona_index`
            verilir ve backend'e (Ollama/OpenAI/Google) iletilir → örnekleme sabitlenir.
            NOT: CPU LLM çıkarımı çok-threadli FP toplama sırası nedeniyle *bit-bazında*
            deterministik değildir; seed tekrar-üretilebilirliği yaklaşık kılar, garanti
            etmez. (Donanım gerçeği — kod kusuru değil.)
    """

    def __init__(
        self,
        llm: LocalLLM | None = None,
        min_answer_chars: int = _MIN_ANSWER_CHARS,
        seed: int = 0,
    ) -> None:
        self.llm = llm or LocalLLM()
        self.min_answer_chars = min_answer_chars
        self.seed = seed

    # ------------------------------------------------------------------ prompt
    def _build_prompt(
        self, chunk_text: str, persona: tuple[str, str], n: int, title: str | None
    ) -> str:
        # NOT: Talimatlar bilinçli olarak diakritiksiz (ASCII-Türkçe) — bu biçim
        # canlı qwen3:4b'ye karşı test edildi ve düzgün Türkçe çıktı üretti.
        persona_name, persona_hint = persona
        head = f"MAKALE: {title}\n" if title else ""
        passage = chunk_text[:_MAX_PASSAGE_CHARS]
        return (
            f"Gorev: Asagidaki PASAJ'dan egitim verisi URET (dogrulama degil).\n"
            f"Sen bir {persona_name}sin. {persona_hint}\n\n"
            f'{head}PASAJ:\n"""\n{passage}\n"""\n\n'
            f"PASAJ'a DAYANARAK {n} adet Turkce soru-cevap cifti yaz. Kurallar:\n"
            f"- Cevaplar YALNIZ pasajdaki bilgiye dayansin; pasajda olmayan sayi, "
            f"formul, sonuc veya kaynak UYDURMA.\n"
            f"- Sorular farkli yonleri sorgulasin (tanim, mekanizma, varsayim, "
            f"sinirlama, test/uygulama).\n"
            f"- Sayilari ve teknik terimleri pasajdan aynen kullan.\n"
            f"- Pasaj bir trading kuralina cevrilemiyorsa cevapta bunu acikca belirt.\n\n"
            f"Cikti TAM olarak su yapida bir JSON objesi olsun (ornegi DOLDUR, aynen "
            f"kopyalama, kendi sorularini uret):\n{_ONESHOT_EXAMPLE}"
        )

    # ------------------------------------------------------------------ build
    def build_for_chunk(
        self,
        chunk_text: str,
        *,
        paper_id: str,
        chunk_id: str,
        n: int = 5,
        persona_index: int = 0,
        title: str | None = None,
        include_context: bool = True,
        seed: int | None = None,
    ) -> list[LoRAExample]:
        """Tek chunk'tan grounded QA örnekleri üret.

        `include_context=True` ise üretilen örnek pasajı BAĞLAM olarak kullanıcı
        mesajına gömer (model bağlamı kullanmayı öğrensin — RAG-stili SFT).
        `seed=None` → `self.seed`; etkin seed `taban + persona_index` (determinist
        ama chunk'lar arası çeşitli; CLAUDE.md kural 6).
        """
        chunk_text = (chunk_text or "").strip()
        if len(chunk_text) < 40:
            return []

        persona = PERSONAS[persona_index % len(PERSONAS)]
        prompt = self._build_prompt(chunk_text, persona, n, title)
        eff_seed = (self.seed if seed is None else seed) + persona_index

        # num_predict sınırı: n çifte yetecek kadar, ama CPU süresini bağlamak için kapalı.
        max_tokens = min(1600, 300 + n * 200)
        try:
            raw = self.llm.generate(
                prompt, temperature=0.4, fmt="json", max_tokens=max_tokens, seed=eff_seed
            )
        except LLMUnavailable:
            logger.warning("Sentetik QA: LLM kullanılamıyor (chunk %s atlandı).", chunk_id)
            return []
        except Exception as exc:  # ağ/timeout — bu chunk'ı atla, döngü devam etsin
            logger.warning("Sentetik QA üretim hatası (%s): %s", chunk_id, exc)
            return []

        pairs = _normalize_pairs(_coerce_json_list(raw))
        # Boş-parse görünür olsun (model JSON üretemedi / fmt desteklenmeyen backend).
        if raw.strip() and not pairs:
            logger.warning(
                "Sentetik QA: çıktı ayrıştırılamadı (chunk %s, %d karakter ham).",
                chunk_id,
                len(raw),
            )
        examples: list[LoRAExample] = []
        for qa in pairs:
            if len(qa.answer) < self.min_answer_chars:
                continue
            if not _is_grounded(qa.answer, chunk_text):
                continue
            if include_context:
                user_content = f"BAĞLAM:\n{chunk_text}\n\nSORU: {qa.question}"
            else:
                user_content = qa.question
            examples.append(
                LoRAExample(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": qa.answer},
                    ],
                    metadata={
                        "paper_id": paper_id,
                        "chunk_id": chunk_id,
                        "source_id": paper_id,
                        "synthetic": True,
                        "persona": persona[0],
                        # Çıplak soru (bağlam gömülü user mesajından ayrı) — dedup/kalite için.
                        "question": qa.question,
                    },
                )
            )
        return examples

    def build_for_paper(
        self,
        store: object,
        paper_id: str,
        *,
        per_chunk: int = 5,
        max_chunks: int = 12,
        include_context: bool = True,
    ) -> list[LoRAExample]:
        """Bir makalenin chunk'larından örnek üret; personayı chunk'lar arası döndür.

        `store` yalnız `list_chunks(paper_id) -> list[Chunk]` arayüzünü gerektirir
        (SqliteStore uyumlu; test için stub enjekte edilebilir).
        """
        chunks = store.list_chunks(paper_id)  # type: ignore[attr-defined]
        examples: list[LoRAExample] = []
        for i, ch in enumerate(chunks[:max_chunks]):
            examples.extend(
                self.build_for_chunk(
                    getattr(ch, "text", ""),
                    paper_id=paper_id,
                    chunk_id=getattr(ch, "chunk_id", f"{paper_id}_c{i}"),
                    n=per_chunk,
                    persona_index=i,  # chunk'lar arası persona rotasyonu
                    title=None,
                    include_context=include_context,
                )
            )
        return examples


def generate_synthetic_dataset(
    store: object,
    *,
    llm: LocalLLM | None = None,
    per_chunk: int = 5,
    max_chunks_per_paper: int = 12,
    paper_ids: list[str] | None = None,
    max_papers: int | None = None,
    seed: int = 0,
) -> tuple[list[LoRAExample], dict]:
    """Tüm (veya verilen) makalelerden sentetik dataset üret + dedup/kalite uygula.

    Returns:
        (examples, stats) — `stats`: {papers, raw, kept, rejected}. `rejected`
        hem duplicate hem de kalite-eleme (kısa/örtüşen cevap) örneklerini sayar.
    """
    builder = SyntheticQABuilder(llm=llm, seed=seed)
    if paper_ids is None:
        papers = store.list_papers()  # type: ignore[attr-defined]
        paper_ids = [getattr(p, "paper_id", "") for p in papers]
        paper_ids = [p for p in paper_ids if p]
    if max_papers is not None:
        paper_ids = paper_ids[:max_papers]

    raw_examples: list[LoRAExample] = []
    for pid in paper_ids:
        raw_examples.extend(
            builder.build_for_paper(
                store, pid, per_chunk=per_chunk, max_chunks=max_chunks_per_paper
            )
        )

    # Dedup + kalite — quality_filter'ı yeniden kullan. ÇIPLAK soru kullanılır:
    # bağlam-gömülü user mesajı kullanılırsa overlap kontrolü yanlış elerdi.
    qf = QualityFilter()
    cards = [
        {
            "question": ex.metadata.get("question", ""),
            "answer": ex.messages[-1]["content"] if ex.messages else "",
            "_ex": ex,
        }
        for ex in raw_examples
    ]
    passed, rejected = qf.filter_batch(cards)
    kept = [c["_ex"] for c in passed]

    stats = {
        "papers": len(paper_ids),
        "raw": len(raw_examples),
        "kept": len(kept),
        "rejected": len(rejected),  # duplicate + kalite-eleme (kısa/örtüşen)
    }
    logger.info(
        "Sentetik dataset: %d makale → %d ham → %d net (%d elendi: dup+kalite).",
        stats["papers"],
        stats["raw"],
        stats["kept"],
        stats["rejected"],
    )
    return kept, stats


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _line_qa(line: str) -> tuple[str, str] | None:
    """JSONL satırından (çıplak soru, cevap) çıkar. Parse edilemezse None."""
    try:
        msgs = json.loads(line).get("messages", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
    user = next((str(m.get("content", "")) for m in msgs if m.get("role") == "user"), "")
    asst = next((str(m.get("content", "")) for m in msgs if m.get("role") == "assistant"), "")
    q = user.split("SORU:", 1)[-1].strip() if "SORU:" in user else user.strip()
    return q, asst


def dedup_jsonl_lines(lines: list[str], jaccard_threshold: float = 0.9) -> list[str]:
    """JSONL satırlarını dedup et: tam içerik-hash + near-duplicate (token Jaccard).

    Faz A7. LLM benzer chunk'lardan paraphrase-dup üretebilir; tam-hash bunları
    kaçırır. (soru+cevap) token kümeleri Jaccard ≥ eşik ise near-dup sayılır ve
    elenir (eşik muhafazakâr 0.9 → yalnız neredeyse-aynılar; meşru farklı örnekler
    korunur). Sıra korunur; ilk görülen tutulur.
    """
    from app.lora.quality_filter import _content_hash

    seen_hash: set[str] = set()
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    for ln in lines:
        qa = _line_qa(ln)
        if qa is None:  # parse edilemeyen satır → yalnız tam-satır dedup
            if ln in seen_hash:
                continue
            seen_hash.add(ln)
            kept.append(ln)
            kept_tokens.append(set())
            continue
        q, a = qa
        h = _content_hash(q, a)
        if h in seen_hash:
            continue
        toks = set(_WORD_RE.findall((q + " " + a).lower()))
        is_near_dup = False
        for kt in kept_tokens:
            if not kt or not toks:
                continue
            inter = len(toks & kt)
            union = len(toks | kt)
            if union and inter / union >= jaccard_threshold:
                is_near_dup = True
                break
        if is_near_dup:
            continue
        seen_hash.add(h)
        kept.append(ln)
        kept_tokens.append(toks)
    return kept
