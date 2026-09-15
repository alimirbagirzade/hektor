"""Bilgi kartı: kitap ön sayfası prompt'a girmez + boş başlık meta veriden dolar (çevrimdışı).

Ölçüm (2026-09-15): López de Prado "Advances in Financial Machine Learning" (256 chunk) için
kart üretimi `full_text[:6000]` ile kapak + içindekiler + şekil/denklem listesini okuyordu;
iki kartın da `title` alanı boştu → `is_substantive_card` eşiğini geçemedi, eğitime giremedi.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

from app.brain.knowledge_card_builder import KnowledgeCardBuilder, _card_source, _meta_title
from app.research.rag_learning_loop import is_substantive_card

_BOOK_TITLE = "Advances in Financial Machine Learning"

_PARAGRAPH = (
    "Fractionally differentiated features preserve memory while making a price series "
    "stationary, which standard integer differentiation destroys. The chapter shows how the "
    "minimum differentiation order that passes the ADF test keeps most of the correlation "
    "with the original series, so that supervised learners receive inputs that are both "
    "stationary and predictive. Backtests must still account for transaction costs and "
    "must be validated out-of-sample with purged cross-validation to avoid leakage."
)

_FRONT_MATTER = [
    "ADVANCES IN FINANCIAL MACHINE LEARNING",
    "Contents " + " ".join(f"Chapter {i} Financial Data Structures {i * 11}" for i in range(40)),
    " ".join(f"Table {i}.1 {i + 10} Table {i}.2 {i + 11} Equation {i} {i + 12}" for i in range(60)),
    "Copyright 2018 by John Wiley & Sons, Inc. All rights reserved. ISBN 978-1-119-48208-6 "
    + "Published simultaneously in Canada. " * 20,
]
_FRONT_MARKERS = ("Table 1.1", "Contents", "All rights reserved", "ISBN", "Equation 26")


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _Store:
    def __init__(self, chunks: list[str], paper: object | None) -> None:
        self._chunks = [_Chunk(t) for t in chunks]
        self._paper = paper
        self.saved: list[dict] = []

    def list_chunks(self, paper_id: str) -> list[_Chunk]:
        return self._chunks

    def get_paper(self, paper_id: str) -> object | None:
        return self._paper

    def save_knowledge_card(
        self, card_id: str, paper_id: str, model: str, card: dict, **kw: object
    ):
        self.saved.append({"paper_id": paper_id, "card": card})


class _PromptLLM:
    """Prompt'ları kaydeder; başlığı BOŞ ama main_claim'i dolu kart döndürür (gerçek vaka)."""

    def __init__(self, title: str = "", main_claim: str | None = None) -> None:
        self.model = "stub-front"
        self.prompts: list[str] = []
        self.title = title
        self.main_claim = (
            "Fractional differentiation keeps memory while achieving stationarity, so ML "
            "features stay predictive; results need purged CV and cost-aware backtests."
            if main_claim is None
            else main_claim
        )

    def generate(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "title": self.title,
                "main_claim": self.main_claim,
                "methods": ["fractional differentiation"],
            }
        )


def _book_chunks() -> list[str]:
    return _FRONT_MATTER + [f"{_PARAGRAPH} BODYMARK{i:03d}" for i in range(60)]


def _builder(tmp_path: Path, store: _Store, llm: _PromptLLM) -> KnowledgeCardBuilder:
    b = KnowledgeCardBuilder(store=store, llm=llm)  # type: ignore[arg-type]
    b.settings = types.SimpleNamespace(extracted_text_dir=tmp_path, reports_dir=tmp_path)
    (tmp_path / "papers").mkdir(exist_ok=True)
    return b


def _article(chunks: list[str]) -> str:
    return "\n\n".join(chunks)


def test_book_front_matter_not_in_prompt_and_title_filled(tmp_path: Path) -> None:
    paper = types.SimpleNamespace(title=_BOOK_TITLE, source_path="kitaplar/afml.pdf")
    store = _Store(_book_chunks(), paper)
    llm = _PromptLLM()

    card = _builder(tmp_path, store, llm).build("paper_4179e6720bae")

    assert llm.prompts
    for prompt in llm.prompts:
        for marker in _FRONT_MARKERS:
            assert marker not in prompt, f"ön sayfa prompt'a girdi: {marker!r}"
    first = llm.prompts[0]
    assert "BODYMARK000" in first  # ilk içerik chunk'ı (giriş) tutulur
    picked = [int(m) for m in re.findall(r"BODYMARK(\d{3})", first)]
    assert max(picked) > 40  # kalan bütçe belgeye yayılır
    assert card.title == _BOOK_TITLE
    assert len(store.saved) == 1 and is_substantive_card(store.saved[0]["card"])


def test_card_source_is_deterministic_and_within_budget() -> None:
    chunks = [_Chunk(t) for t in _book_chunks()]
    full = _article(_book_chunks())
    excerpt, source = _card_source(chunks, full, 6000)
    assert len(excerpt) <= 6000
    assert not any(m in source for m in _FRONT_MARKERS)
    assert _card_source(chunks, full, 6000) == (excerpt, source)


def test_short_article_keeps_abstract_and_intro_prefix(tmp_path: Path) -> None:
    # Temiz arXiv makalesi: kısa başlık/yazar chunk'ı içerik sayılmaz ama ön sayfa değildir →
    # eski davranış aynen: ilk prompt = full_text[:max_chars] (başlık + özet + giriş).
    chunks = ["Momentum Crashes\nKent Daniel, Tobias Moskowitz\nAbstract"] + [
        f"{_PARAGRAPH} INTRO{i:02d}" for i in range(20)
    ]
    store = _Store(chunks, types.SimpleNamespace(title="Momentum Crashes", source_path="x.pdf"))
    llm = _PromptLLM(title="Momentum Crashes in Equity Markets")

    card = _builder(tmp_path, store, llm).build("paper_arxiv")

    assert _article(chunks)[:6000] in llm.prompts[0]
    assert "[...]" not in llm.prompts[0]
    assert card.title == "Momentum Crashes in Equity Markets"  # anlamlı LLM başlığı korunur


def test_title_from_filename_when_paper_title_missing(tmp_path: Path) -> None:
    paper = types.SimpleNamespace(title=None, source_path="C:/pdf/Lopez_de_Prado-AFML_2018.pdf")
    store = _Store(_book_chunks(), paper)

    card = _builder(tmp_path, store, _PromptLLM()).build("paper_book")

    assert card.title == "Lopez de Prado AFML 2018"


def test_empty_card_not_rescued_by_meta_title(tmp_path: Path) -> None:
    # main_claim boşken meta başlık yazılmaz: boş kart "içerikli" görünüp kaydedilmemeli.
    paper = types.SimpleNamespace(title=_BOOK_TITLE, source_path="afml.pdf")
    store = _Store(_book_chunks(), paper)

    card = _builder(tmp_path, store, _PromptLLM(main_claim="")).build("paper_empty")

    assert card.title is None and card.has_content is False
    assert store.saved == []


def test_meta_title_rejects_meaningless_values() -> None:
    assert _meta_title(None) == ""
    assert _meta_title(types.SimpleNamespace(title="", source_path="2101.12345v1.pdf")) == ""
    assert _meta_title(types.SimpleNamespace(title=" Deep  Hedging ", source_path="")) == (
        "Deep Hedging"
    )
