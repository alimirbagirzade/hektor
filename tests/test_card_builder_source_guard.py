"""KnowledgeCardBuilder kaynak-yetersizlik guard testi (tamamen çevrimdışı).

Bozuk/0KB kaynaklı makalelere (PDF çıkarımı başarısız) LLM çağrısı YAPILMAZ — yerel
CPU'da boş kart üretip ~5 dk/çağrı boşa harcamak yerine erken çıkılır. Yeterli kaynakta
ise LLM normal çağrılır.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.brain import knowledge_card_builder as kcb


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


def _make_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chunk_text: str,
    gen_counter: dict,
    gen_return: str,
) -> kcb.KnowledgeCardBuilder:
    # reports_dir/extracted_text_dir PROJECT_ROOT'tan türer → tmp'ye yönlendir (repo kirlenmesin).
    monkeypatch.setattr("app.config.settings.PROJECT_ROOT", tmp_path)

    class FakeLLM:
        model = "test"

        def generate(self, *args: object, **kwargs: object) -> str:
            gen_counter["n"] += 1
            return gen_return

    class FakeStore:
        def list_chunks(self, paper_id: str) -> list[_Chunk]:
            return [_Chunk(chunk_text)]

        def save_knowledge_card(self, **kwargs: object) -> None:
            pass

    b = kcb.KnowledgeCardBuilder(store=FakeStore(), llm=FakeLLM())
    (b.settings.reports_dir / "papers").mkdir(parents=True, exist_ok=True)
    return b


def test_build_skips_llm_for_tiny_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gen = {"n": 0}
    b = _make_builder(monkeypatch, tmp_path, "çok kısa metin", gen, "{}")
    card = b.build("paper_tiny")
    assert gen["n"] == 0  # yetersiz kaynak → LLM HİÇ çağrılmaz (boşa CPU yok)
    assert (card.main_claim or "") == ""
    assert card.paper_id == "paper_tiny"


def test_build_calls_llm_for_sufficient_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gen = {"n": 0}
    big = "kelime " * 1000  # _MIN_SOURCE_CHARS'tan çok büyük
    ret = '{"title":"Başlık","main_claim":"yeterince uzun ve anlamlı bir iddia metni buraya"}'
    b = _make_builder(monkeypatch, tmp_path, big, gen, ret)
    card = b.build("paper_big")
    assert gen["n"] >= 1  # yeterli kaynak → LLM çağrılır
    assert card.main_claim.startswith("yeterince")


def test_min_source_chars_constant_sane() -> None:
    # Gerçek makale çıkarımları (~11KB+) eşiğin çok üstünde; bozuk olanlar (0-2KB) altında.
    assert 500 <= kcb._MIN_SOURCE_CHARS <= 5000
