"""CrossEncoderReranker testleri (Faz A8) — enjekte model + graceful fallback.

Çevrimdışı: sentence-transformers kurulu olmasa da çalışır. Model yokken heuristik
`Reranker`'a düştüğü, enjekte modelle skora göre sıraladığı, hata/boşta çökmediği
doğrulanır.
"""

from __future__ import annotations

from app.memory.cross_encoder_reranker import CrossEncoderReranker
from app.memory.reranker import Reranker
from app.memory.retrieval_service import RetrievedChunk


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id="p",
        text=text,
        page_number=1,
        section_name="results",
        title="T",
        distance=0.5,
    )


class _FakeModel:
    """sentence-transformers CrossEncoder.predict arayüzünü taklit eder."""

    def __init__(self, by_text: dict[str, float]) -> None:
        self.by_text = by_text

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self.by_text.get(text, 0.0) for _q, text in pairs]


def test_injected_model_orders_by_score() -> None:
    a, b, c = _chunk("a", "alpha"), _chunk("b", "beta"), _chunk("c", "gamma")
    model = _FakeModel({"alpha": 0.1, "beta": 0.9, "gamma": 0.5})
    cer = CrossEncoderReranker(model=model)
    out = cer.rerank("soru", [a, b, c])
    assert [x.chunk_id for x in out] == ["b", "c", "a"]  # skora göre azalan


def test_falls_back_to_heuristic_when_model_unavailable() -> None:
    # sentence-transformers kurulu değil → model None → heuristik fallback.
    cer = CrossEncoderReranker(model_name="nonexistent/model-xyz", fallback=Reranker())
    plain = _chunk("plain", "Momentum is a known factor in markets.")
    formula = _chunk("formula", r"ATR is $ATR_t = \frac{1}{n}\sum TR_i$ true range.")
    out = cer.rerank("ATR formula", [plain, formula])
    assert out[0].chunk_id == "formula"  # heuristik: formül chunk'ı öne çıkar


def test_predict_error_falls_back_without_crash() -> None:
    class _Bad:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            raise RuntimeError("boom")

    cer = CrossEncoderReranker(model=_Bad(), fallback=Reranker())
    out = cer.rerank("ATR formula", [_chunk("a", "x metni"), _chunk("b", "y metni")])
    assert len(out) == 2  # çökme yok, fallback çalıştı


def test_empty_returns_empty() -> None:
    assert CrossEncoderReranker(model=_FakeModel({})).rerank("soru", []) == []
