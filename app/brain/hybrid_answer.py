"""Hibrit kaynaklı cevap: model TEK kısa cevap yazar, geri kalan bölümler deterministiktir.

Neden hibrit (karar 2026-09-13):
- Yerel CPU'da adapter'lı 4B ~0,4 token/sn üretir; 8 bölümlü tam raporu modele yazdırmak
  soru başına 30-60 dk sürerdi. v8 ise kısa ve bağlama sadık cevap üretmeyi öğrendi.
- Test Planı / Riskler kural uyumu metinleridir (Kural 2-4): modelin "hatırlamasına"
  bırakılmaz, kod garanti eder.
- Kart içeriği ÖZGÜN dilinde (çoğunlukla İngilizce) ve etiketli alıntıdır. Çeviri yapılmaz
  → yanlış çeviri / sayı bozulması / uydurma riski yoktur (Kural 7).

Her bölüm kaynağını taşır: "model" | "kaynak" | "kural" | "kural+kaynak". Arayüz bunu
rozet olarak gösterir; kullanıcı neyin üretildiğini, neyin alıntılandığını ayırt eder.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from app.brain.answer_quality import assess_confidence, is_weak_retrieval
from app.memory.retrieval_service import RetrievedChunk

CardLookup = Callable[[str], dict | None]

MAX_CARD_PAPERS = 2  # Akademik Bulgu / Hipotez / Risk alıntısı için en fazla makale
MAX_QUOTE_ITEMS = 3  # makale başına alıntılanan en fazla madde
QUOTE_LABEL = "(kaynak, çevrilmedi)"
HYPOTHESIS_NOTE = "Makalenin iddiası; test edilmemiş hipotez — sayılar doğrulanmamıştır."

TEST_PLAN_ITEMS = (
    "Veriyi zamana göre böl: in-sample'da geliştir, out-of-sample'da yalnız bir kez doğrula.",
    "Komisyon + slippage her backtest'te dahil.",
    "Pozisyon shift(1) ile gecikmeli (look-ahead yok).",
    "Rastgelelik varsa seed sabitle (determinizm).",
    "Sonucu /backtest-auditor ile denetle (look-ahead + out-of-sample + overfit).",
)
RISK_ITEMS = (
    "Overfit: az örnekle ya da çok parametreyle eğriye uydurma.",
    "Veri sızıntısı / look-ahead bias.",
    "Survivorship bias (yalnız hayatta kalan enstrümanlar).",
    "Spread, slippage ve komisyon kâğıt üstündeki getiriyi silebilir.",
)
NEXT_STEP = (
    "Hipotezi `hektor backtest` ile maliyetli ve out-of-sample doğrulamalı test et, ardından "
    "`/backtest-auditor` ile denetle. Verdict `pass` değilse çıktı yalnızca adaydır."
)


@dataclass(frozen=True)
class AnswerSection:
    """Cevabın tek bölümü; `source` bölümün nereden geldiğini söyler."""

    title: str
    body: str
    source: str
    warning: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _unique_papers(chunks: list[RetrievedChunk]) -> list[str]:
    """Retrieval sırasını koruyarak tekil paper_id listesi."""
    seen: list[str] = []
    for c in chunks:
        if c.paper_id not in seen:
            seen.append(c.paper_id)
    return seen


def _items(value: object) -> list[str]:
    """Kart alanını boş olmayan metin maddelerine çevir (liste ya da tek metin)."""
    raw = value if isinstance(value, list) else [value]
    return [str(v).strip() for v in raw if v is not None and str(v).strip()]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items)


def load_cards(chunks: list[RetrievedChunk], lookup: CardLookup) -> dict[str, dict]:
    """Retrieval sırasındaki ilk MAX_CARD_PAPERS makalenin kartını getir (kartı olmayan atlanır)."""
    cards: dict[str, dict] = {}
    for pid in _unique_papers(chunks)[:MAX_CARD_PAPERS]:
        card = lookup(pid)
        if card:
            cards[pid] = card
    return cards


def _card_header(pid: str, card: dict) -> str:
    title = str(card.get("title") or "").strip()
    return f"[{pid}] {title} {QUOTE_LABEL}" if title else f"[{pid}] {QUOTE_LABEL}"


def sources_section(chunks: list[RetrievedChunk]) -> AnswerSection:
    lines = [f"{c.citation} — {c.title}" if c.title else c.citation for c in chunks]
    return AnswerSection("Kaynaklar", _bullets(lines), "kaynak")


def context_quality_section(
    chunks: list[RetrievedChunk], *, min_similarity: float, min_margin: float
) -> AnswerSection:
    """Retrieval güvenini CRAG-lite eşikleriyle etiketle (abstain kapısıyla AYNI eşikler)."""
    conf = assess_confidence(chunks)
    weak = is_weak_retrieval(conf, min_similarity, min_margin)
    if weak:
        level = "Zayıf"
    elif conf.best_similarity >= min_similarity * 1.5:
        level = "Güçlü"
    else:
        level = "Orta"
    body = (
        f"{level} — en iyi benzerlik {conf.best_similarity:.2f}, "
        f"ilk iki kaynak arası marj {conf.margin:.2f}, {conf.n} parça."
    )
    if weak:
        body += (
            "\nUyarı: getirilen kaynaklar soruyla zayıf ilişkili; kısa cevap zayıf dayanağa bağlı."
        )
    return AnswerSection("Bağlam Kalitesi", body, "kural", warning=weak)


def academic_finding_section(chunks: list[RetrievedChunk], cards: dict[str, dict]) -> AnswerSection:
    blocks = []
    for pid in _unique_papers(chunks):
        card = cards.get(pid)
        claim = _items(card.get("main_claim")) if card else []
        if card and claim:
            blocks.append(f"{_card_header(pid, card)}\n{claim[0]}")
    if not blocks:
        return AnswerSection(
            "Akademik Bulgu", "Getirilen makalelerin kartında ana iddia yok.", "kaynak"
        )
    return AnswerSection("Akademik Bulgu", "\n\n".join(blocks), "kaynak")


def trading_hypothesis_section(
    chunks: list[RetrievedChunk], cards: dict[str, dict]
) -> AnswerSection:
    blocks = []
    for pid in _unique_papers(chunks):
        card = cards.get(pid)
        hyps = _items(card.get("possible_strategy_hypotheses"))[:MAX_QUOTE_ITEMS] if card else []
        if card and hyps:
            blocks.append(f"{_card_header(pid, card)}\n{_bullets(hyps)}")
    if not blocks:
        return AnswerSection(
            "Trading Hipotezi",
            "Kaynak kartlarında strateji hipotezi yok; bu bulgu doğrudan trading kuralına "
            "çevrilemez.",
            "kaynak",
        )
    return AnswerSection(
        "Trading Hipotezi", HYPOTHESIS_NOTE + "\n\n" + "\n\n".join(blocks), "kaynak"
    )


def risks_section(chunks: list[RetrievedChunk], cards: dict[str, dict]) -> AnswerSection:
    body = _bullets(list(RISK_ITEMS))
    blocks = []
    for pid in _unique_papers(chunks):
        card = cards.get(pid)
        warns = _items(card.get("risk_warnings"))[:MAX_QUOTE_ITEMS] if card else []
        if card and warns:
            blocks.append(f"{_card_header(pid, card)}\n{_bullets(warns)}")
    if not blocks:
        return AnswerSection("Riskler", body, "kural")
    return AnswerSection("Riskler", body + "\n\n" + "\n\n".join(blocks), "kural+kaynak")


def build_sections(
    model_answer: str,
    chunks: list[RetrievedChunk],
    cards: dict[str, dict],
    *,
    min_similarity: float,
    min_margin: float,
) -> list[AnswerSection]:
    """8 bölümlü hibrit cevap. Model yalnız "Kısa Cevap"ı yazar; gerisi kaynak/kural."""
    answer = model_answer.strip()
    short = (
        AnswerSection("Kısa Cevap", answer, "model")
        if answer
        else AnswerSection("Kısa Cevap", "(model boş cevap üretti)", "model", warning=True)
    )
    return [
        short,
        sources_section(chunks),
        context_quality_section(chunks, min_similarity=min_similarity, min_margin=min_margin),
        academic_finding_section(chunks, cards),
        trading_hypothesis_section(chunks, cards),
        AnswerSection("Test Planı", _bullets(list(TEST_PLAN_ITEMS)), "kural"),
        risks_section(chunks, cards),
        AnswerSection("Sonraki Adım", NEXT_STEP, "kural"),
    ]
