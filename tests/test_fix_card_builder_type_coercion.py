"""KnowledgeCardBuilder LLM tip-sapması toleransı (tamamen çevrimdışı).

Küçük modeller (qwen3:4b) şemayı sık ihlal eder: `"year": 2021` (int), `"methods":
"GARCH"` (tek string), `"limitations": null`. Eskiden bu, `model_validate`'te
ValidationError fırlatıp İÇERİKLİ kartı kaydetmeden `build()`'i çökertiyordu.

Bu dosya iki sözleşmeyi birlikte kilitler:
1. Tip sapmaları tolere edilir ve içerik KORUNUR (uydurma yok — Kural 7).
2. Boş/anlamsız yanıt hâlâ "boş kart" sayılır ve KAYDEDİLMEZ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.brain import knowledge_card_builder as kcb


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeStore:
    """Kaydedilen kartları toplar — 'boş kart KAYDEDİLMEZ' sözleşmesini ölçmek için."""

    def __init__(self, chunk_text: str) -> None:
        self.chunk_text = chunk_text
        self.saved: list[dict[str, Any]] = []

    def list_chunks(self, paper_id: str) -> list[_Chunk]:
        return [_Chunk(self.chunk_text)]

    def save_knowledge_card(self, **kwargs: Any) -> None:
        self.saved.append(kwargs)


class _FakeLLM:
    model = "test"

    def generate(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("Bu testlerde LLM çağrılmamalı; _card_json yamalanıyor.")


def _make_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: Any
) -> tuple[kcb.KnowledgeCardBuilder, _FakeStore]:
    """`_card_json`'u sahte JSON döndürecek şekilde yamalar (ağ/Ollama YOK)."""
    # reports_dir/extracted_text_dir PROJECT_ROOT'tan türer → tmp'ye yönlendir.
    monkeypatch.setattr("app.config.settings.PROJECT_ROOT", tmp_path)
    store = _FakeStore("kelime " * 1000)  # _MIN_SOURCE_CHARS'ın çok üstünde
    builder = kcb.KnowledgeCardBuilder(store=store, llm=_FakeLLM())
    (builder.settings.reports_dir / "papers").mkdir(parents=True, exist_ok=True)

    def fake_card_json(*args: object, **kwargs: object) -> Any:
        return payload

    monkeypatch.setattr(builder, "_card_json", fake_card_json)
    return builder, store


# --- (a) tip sapmaları tolere edilir, içerik korunur -----------------------------------


def test_build_tolerates_llm_type_deviations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "title": "Volatilite Modelleri",
        "year": 2021,  # int → şema `str | None`
        "domain": 3.0,  # float → "3"
        "main_claim": "GARCH getirilerdeki oynaklık kümelenmesini yakalar",
        "methods": "GARCH",  # tek string → list[str]
        "datasets": [{"name": "S&P 500"}, 2020],  # dict + int öğeler
        "limitations": None,  # null → []
        "risk_warnings": {"tip": "aşırı uyum", "not": "örneklem dışı test şart"},
        "trading_relevance": ["sinyal üretimi", "risk ölçümü"],
    }
    builder, store = _make_builder(monkeypatch, tmp_path, payload)

    card = builder.build("paper_types")

    assert card.year == "2021"
    assert card.domain == "3"
    assert card.methods == ["GARCH"]
    assert card.datasets == ["S&P 500", "2020"]
    assert card.limitations == []
    assert card.risk_warnings == ["tip: aşırı uyum; not: örneklem dışı test şart"]
    assert card.trading_relevance == "sinyal üretimi; risk ölçümü"
    # İçerik korundu → kart gerçekten kaydedildi (eskiden ValidationError ile çökerdi).
    assert card.has_content
    assert len(store.saved) == 1
    assert store.saved[0]["paper_id"] == "paper_types"
    out_path = builder.settings.reports_dir / "papers" / "paper_types_card.json"
    assert json.loads(out_path.read_text(encoding="utf-8"))["title"] == "Volatilite Modelleri"


def test_build_does_not_raise_on_deviant_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Doğrulama: aynı veri ham hâliyle pydantic'i GERÇEKTEN kırıyor (bulgu gerçek).
    with pytest.raises(ValidationError):
        kcb.KnowledgeCard.model_validate({"paper_id": "p", "year": 2021, "methods": "GARCH"})
    builder, _ = _make_builder(
        monkeypatch, tmp_path, {"main_claim": "geçerli bir iddia", "year": 2021, "methods": "X"}
    )
    assert builder.build("paper_ok").main_claim == "geçerli bir iddia"


# --- (b) boş-kart sözleşmesi korunur ---------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},  # LLM boş/parse edilemez
        {"title": None, "main_claim": None, "methods": None},  # her şey null
        {"title": True, "main_claim": False},  # anlamsız bool → içerik sayılmaz
        {"title": "", "main_claim": "   ", "methods": ["", "  "]},  # boş metinler
        {"title": {}, "main_claim": []},  # boş kapsayıcılar
    ],
)
def test_empty_card_still_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    builder, store = _make_builder(monkeypatch, tmp_path, payload)

    card = builder.build("paper_empty")

    assert not card.has_content, "normalizasyon boş kartı 'dolu' göstermemeli"
    assert store.saved == [], "boş kart KAYDEDİLMEMELİ"
    assert not (builder.settings.reports_dir / "papers" / "paper_empty_card.json").exists()
    assert card.paper_id == "paper_empty"


def test_normalization_never_invents_values() -> None:
    """Kural 7: çevrilemeyen değer UYDURULMAZ, boş bırakılır."""
    out = kcb._normalize_card_data({"title": True, "year": None, "methods": False})
    assert out["title"] is None  # şema varsayılanı
    assert out["year"] is None
    assert out["methods"] == []
    # Gelmeyen alana dokunulmaz (pydantic varsayılanında kalır).
    assert "main_claim" not in out


def test_nesting_depth_is_bounded() -> None:
    deep: Any = {"a": {"b": {"c": {"d": "çok derin"}}}}
    assert kcb._as_text(deep) == ""  # _MAX_NEST_DEPTH aşıldı → uydurma yok, boş
    # Tek anahtarlı sözlükte anahtar şema etiketidir ({"name": "GARCH"}) → yalnız değer.
    assert kcb._as_text({"name": "GARCH"}) == "GARCH"
    # Çok anahtarlıda bilgi kaybolmasın diye "anahtar: değer" korunur.
    assert kcb._as_text({"a": "1", "b": "2"}) == "a: 1; b: 2"


def test_card_json_reduces_non_object_json_to_empty_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model nesne yerine dizi/skaler döndürürse `_card_json` boş dict verir.

    Aksi hâlde `build()` içindeki `data["paper_id"] = ...` TypeError ile çökerdi.
    """
    monkeypatch.setattr("app.config.settings.PROJECT_ROOT", tmp_path)

    class _ListLLM:
        model = "test"

        def generate(self, *args: object, **kwargs: object) -> str:
            return "[1, 2, 3]"

    builder = kcb.KnowledgeCardBuilder(store=_FakeStore(""), llm=_ListLLM())
    assert builder._card_json("metin", "sistem", max_tokens=10) == {}


def test_normalize_card_data_rejects_non_dict() -> None:
    assert kcb._normalize_card_data(["liste döndü"]) == {}
    assert kcb._normalize_card_data(None) == {}
    assert kcb._normalize_card_data("düz metin") == {}
