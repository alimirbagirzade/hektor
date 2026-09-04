"""KnowledgeCardBuilder tests.

Offline tests inject a stub LLM and a fake store so the JSON parsing + card
assembly is verified without Ollama. One ``@pytest.mark.ollama`` integration
test exercises the real local model end-to-end.
"""

from __future__ import annotations

import json
import types

import pytest

from app.brain.knowledge_card_builder import (
    KnowledgeCard,
    KnowledgeCardBuilder,
    _extract_json,
)
from app.brain.local_llm import LocalLLM


class _FakeChunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeStore:
    """Duck-typed stand-in for SqliteStore (only the methods build() needs)."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = [_FakeChunk(t) for t in chunks]
        self.saved: list[dict] = []

    def list_chunks(self, paper_id: str) -> list[_FakeChunk]:
        return self._chunks

    def save_knowledge_card(
        self,
        card_id: str,
        paper_id: str,
        model: str,
        card: dict,
        *,
        trust_level: str = "draft",
        review_status: str = "pending",
        lora_eligible: int = 0,
        difficulty: float = 0.0,
        stage: str = "",
    ) -> None:
        self.saved.append(
            {
                "card_id": card_id,
                "paper_id": paper_id,
                "model": model,
                "card": card,
                "trust_level": trust_level,
                "review_status": review_status,
                "lora_eligible": lora_eligible,
                "difficulty": difficulty,
                "stage": stage,
            }
        )


class _StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.model = "stub-model"

    def generate(self, prompt: str, *, system: str | None = None, **_: object) -> str:
        return self.response


def _builder(tmp_path, store: _FakeStore, llm: _StubLLM) -> KnowledgeCardBuilder:
    b = KnowledgeCardBuilder(store=store, llm=llm)
    # Isolate disk writes (build() touches extracted_text_dir + reports_dir only).
    b.settings = types.SimpleNamespace(extracted_text_dir=tmp_path, reports_dir=tmp_path)
    (tmp_path / "papers").mkdir(exist_ok=True)
    return b


_VALID_CARD_JSON = """Here is the card:
```json
{
  "paper_id": "WILL_BE_OVERWRITTEN",
  "title": "Vol Clustering",
  "year": "2026",
  "domain": "market microstructure",
  "main_claim": "Volatilite kümelenmesi momentum kalıcılığını artırır.",
  "methods": ["GARCH", "EMA crossover"],
  "datasets": ["XAUUSD 15m"],
  "trading_relevance": "ATR yüksekken trend filtreleri daha kalıcı.",
  "limitations": ["tek piyasa"],
  "possible_strategy_hypotheses": ["ATR>p60 iken trend-following"],
  "risk_warnings": ["overfit"],
  "implementation_notes": ["out-of-sample test şart"]
}
```
"""


def test_extract_json_strips_fences_and_prose():
    data = _extract_json(_VALID_CARD_JSON)
    assert data["title"] == "Vol Clustering"
    assert data["methods"] == ["GARCH", "EMA crossover"]


def test_extract_json_repairs_smart_quotes_and_trailing_commas():
    # küçük modellerin tipik bozulmaları: akıllı tırnak + sondaki virgül
    raw = "{“main_claim”: “x”, “methods”: [“a”, “b”,],}"
    data = _extract_json(raw)
    assert data["main_claim"] == "x"
    assert data["methods"] == ["a", "b"]


def test_build_parses_card_and_persists(tmp_path):
    # Builder kaynak metni <1500 karakterse LLM'e göndermez (commit 79619ea,
    # _MIN_SOURCE_CHARS) → parse/persist davranışını sınamak için yeterince uzun kaynak ver.
    long_chunk = "Volatilite kümelenmesi ve momentum kalıcılığı üzerine analiz cümlesi. " * 30
    store = _FakeStore([long_chunk, "chunk metni iki"])
    builder = _builder(tmp_path, store, _StubLLM(_VALID_CARD_JSON))

    card = builder.build("paper_abc")

    assert isinstance(card, KnowledgeCard)
    assert card.paper_id == "paper_abc"  # forced, overrides JSON value
    assert card.main_claim.startswith("Volatilite")
    assert "GARCH" in card.methods
    assert card.possible_strategy_hypotheses  # non-empty
    # persisted to fake store + written to disk
    assert len(store.saved) == 1
    assert store.saved[0]["paper_id"] == "paper_abc"
    written = tmp_path / "papers" / "paper_abc_card.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["paper_id"] == "paper_abc"


def test_build_sanitizes_paper_id_crlf(tmp_path):
    # Regresyon: Windows'ta python stdout CRLF yüzünden paper_id sonuna \r takılınca
    # dosya adı 'paper_crlf\r_card.json' olup OSError [Errno 22] veriyordu. build()
    # girişte strip() ile temizlemeli; hem DB kaydı hem dosya adı temiz olmalı.
    store = _FakeStore(["chunk metni"])
    builder = _builder(tmp_path, store, _StubLLM(_VALID_CARD_JSON))

    card = builder.build("paper_crlf\r")  # kirli id (trailing CR)

    assert card.paper_id == "paper_crlf"
    assert store.saved[0]["paper_id"] == "paper_crlf"
    written = tmp_path / "papers" / "paper_crlf_card.json"
    assert written.exists()  # \r olsaydı Errno 22 ile patlardı


def test_build_handles_non_json_gracefully(tmp_path):
    store = _FakeStore(["bir metin"])
    builder = _builder(tmp_path, store, _StubLLM("Bu JSON değil, sadece düz metin."))

    card = builder.build("paper_garbage")

    # falls back to an empty-but-valid card, no crash
    assert card.paper_id == "paper_garbage"
    assert card.main_claim == ""
    assert card.methods == []


class _MiddleAwareLLM:
    """Yalnız prompt'ta 'REAL_CLAIM' geçince (belge ORTASI) dolu kart döndürür."""

    def __init__(self, full_card: str) -> None:
        self.full_card = full_card
        self.model = "stub-mid"

    def generate(self, prompt: str, *, system: str | None = None, **_: object) -> str:
        return self.full_card if "REAL_CLAIM" in prompt else "{}"


def test_build_middle_slice_rescues_front_matter_books(tmp_path):
    # Büyük kitap: ilk 6000+ krk kapak/ön-madde (claim yok); gerçek içerik ortada.
    # Builder ilk iki deneme boş kalınca ORTA-KESİTİ denemeli → claim yakalanır.
    front = "x" * 7000  # [:6000] ve [:3000] yalnız bunu görür → claim yok
    middle = "REAL_CLAIM " * 3000  # belge gövdesi; orantılı (%25/%55) kesitler buna düşer
    store = _FakeStore([front, middle])
    builder = _builder(tmp_path, store, _MiddleAwareLLM(_VALID_CARD_JSON))

    card = builder.build("paper_book")

    assert card.main_claim.startswith("Volatilite")  # orantılı kesitten kurtarıldı


def test_build_no_middle_slice_when_small(tmp_path):
    # Küçük belge: orta-kesit denemesi tetiklenmez → ilk denemeler boşsa boş kalır.
    store = _FakeStore(["kısa metin, marker yok"])
    builder = _builder(tmp_path, store, _MiddleAwareLLM(_VALID_CARD_JSON))

    card = builder.build("paper_small")

    assert card.main_claim == ""  # belge küçük; orta-kesit yok


# --------------------------------------------------------------------------
# Integration: real local model (skipped unless Ollama + model are ready)
# --------------------------------------------------------------------------
def _ollama_model_ready(model: str) -> bool:
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        return any(model.split(":")[0] in n for n in names)
    except Exception:
        return False


@pytest.mark.ollama
def test_knowledge_card_real_llm(tmp_path):
    llm = LocalLLM()
    if not _ollama_model_ready(llm.model):
        pytest.skip(f"Ollama model not ready: {llm.model}")

    store = _FakeStore(
        [
            "Bu çalışma volatilite kümelenmesinin momentum sinyallerinin "
            "kalıcılığını nasıl etkilediğini XAUUSD 15m verisinde inceler."
        ]
    )
    builder = KnowledgeCardBuilder(store=store, llm=llm)
    builder.settings = types.SimpleNamespace(extracted_text_dir=tmp_path, reports_dir=tmp_path)
    (tmp_path / "papers").mkdir(exist_ok=True)

    card = builder.build("paper_real")
    assert card.paper_id == "paper_real"
    assert len(store.saved) == 1
