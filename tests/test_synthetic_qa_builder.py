"""SyntheticQABuilder testleri — tamamen çevrimdışı (fake LLM enjekte edilir).

Ollama/ağ gerektirmez. JSON ayrıştırma toleransı, grounding backstop'u, persona
rotasyonu ve dedup doğrulanır.
"""

from __future__ import annotations

import json

from app.brain.synthetic_qa_builder import (
    _ONESHOT_EXAMPLE,
    SyntheticQABuilder,
    _coerce_json_list,
    _is_grounded,
    dedup_jsonl_lines,
    generate_synthetic_dataset,
)


def test_oneshot_example_has_no_leaky_prefix() -> None:
    # v5 sızıntı dersi: one-shot örnek cevapları SABİT "Pasaja gore" öneğiyle
    # başlamamalı (model bunu koşulsuz açılış olarak ezberliyordu). Açılışlar çeşitli olmalı.
    pairs = json.loads(_ONESHOT_EXAMPLE)["pairs"]
    answers = [p["answer"] for p in pairs]
    assert len(answers) >= 2
    for ans in answers:
        assert not ans.lower().startswith("pasaja"), f"sızıntılı önek: {ans[:30]!r}"
    # Açılış token'ları homojen olmasın (ilk kelimeler farklı).
    first_words = {a.split()[0].lower() for a in answers}
    assert len(first_words) == len(answers)


def _jsonl(q: str, a: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": f"BAĞLAM:\nX\n\nSORU: {q}"},
                {"role": "assistant", "content": a},
            ]
        },
        ensure_ascii=False,
    )


def test_dedup_removes_exact_duplicate() -> None:
    ln = _jsonl("ATR nedir", "ATR ortalama gercek araliktir ve volatilite olcer.")
    assert len(dedup_jsonl_lines([ln, ln])) == 1


def test_dedup_removes_near_duplicate() -> None:
    # Tek kelime farkı (~%92 Jaccard) → near-dup elenir.
    a1 = "ATR ortalama gercek araliktir ve volatilite olcer bu yuzden onemli"
    a2 = "ATR ortalama gercek araliktir ve volatilite olcer bu yuzden cok onemli"
    out = dedup_jsonl_lines([_jsonl("ATR nedir", a1), _jsonl("ATR nedir", a2)])
    assert len(out) == 1


def test_dedup_keeps_distinct() -> None:
    out = dedup_jsonl_lines(
        [
            _jsonl("ATR nedir", "ATR ortalama gercek araliktir, volatilite olcer."),
            _jsonl("Sharpe orani nedir", "Sharpe getiriyi riske boler, risk-ayarli olcum."),
        ]
    )
    assert len(out) == 2


# Anchor'lı (sayı + teknik terim) pasaj — grounding kontrolü anlamlı olsun.
_CHUNK = (
    "The ATR (Average True Range) over 14 periods measures volatility. "
    "A momentum filter uses ATR to size positions and reduce drawdown."
)

# 3 QA: (1) grounded+yeterli, (2) çok kısa, (3) uzun ama uydurma (anchor yok).
# Satır uzunluğu için bitişik string birleştirme (implicit concatenation).
_GOOD_PAYLOAD = (
    "[\n"
    '  {"question": "ATR nasil hesaplanir?", "answer": '
    '"Pasaja gore ATR 14 periyot ile hesaplanan ortalama gercek '
    'araliktir ve momentum filtresi olarak kullanilir."},\n'
    '  {"question": "Kisa?", "answer": "Evet."},\n'
    '  {"question": "Alakasiz?", "answer": '
    '"Bu cevap tamamen alakasiz uydurma kelimelerden ibarettir '
    'tipki kirmizi balon ile mor bisiklet gibi seyler."}\n'
    "]"
)


class _FakeLLM:
    """LocalLLM.generate arayüzünü taklit eder; sabit payload döner."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0
        self.last_seed: int | None = None

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        fmt: str | None = None,
        seed: int | None = None,
        **_: object,
    ) -> str:
        self.calls += 1
        self.last_seed = seed
        return self.payload


class _StubChunk:
    def __init__(self, chunk_id: str, text: str) -> None:
        self.chunk_id = chunk_id
        self.text = text


class _StubStore:
    def __init__(self, chunks_by_paper: dict[str, list[_StubChunk]]) -> None:
        self._chunks = chunks_by_paper

    def list_chunks(self, paper_id: str) -> list[_StubChunk]:
        return self._chunks.get(paper_id, [])

    def list_papers(self) -> list[object]:
        return [type("P", (), {"paper_id": pid})() for pid in self._chunks]


# --------------------------------------------------------------------- parsing
def test_coerce_json_plain_array() -> None:
    items = _coerce_json_list('[{"question":"a","answer":"b"}]')
    assert items == [{"question": "a", "answer": "b"}]


def test_coerce_json_code_fenced() -> None:
    raw = '```json\n[{"question":"a","answer":"b"}]\n```'
    assert _coerce_json_list(raw) == [{"question": "a", "answer": "b"}]


def test_coerce_json_wrapper_and_noise() -> None:
    raw = 'İşte sonuç: {"qa": [{"question":"a","answer":"b"}]} bitti.'
    assert _coerce_json_list(raw) == [{"question": "a", "answer": "b"}]


# ------------------------------------------------------------------- grounding
def test_grounded_answer_passes() -> None:
    # "14" pasajda var + anchor örtüşmesi (momentum/atr) → grounded.
    assert _is_grounded("ATR 14 periyot ile momentum olcer.", _CHUNK) is True


def test_hallucinated_answer_fails() -> None:
    assert _is_grounded("Kirmizi balon mor bisiklet ile ucar gider.", _CHUNK) is False


def test_fabricated_number_rejected() -> None:
    # Pasajda olmayan bir metrik (Sharpe 2.3) uydurulmuş → sayı-altküme kapısı REDDEDER.
    assert (
        _is_grounded("Pasaja gore Sharpe orani 2.3 ve momentum filtresi getiriyi artirir.", _CHUNK)
        is False
    )


def test_fabricated_percentage_rejected() -> None:
    assert _is_grounded("Pasaja gore strateji %47 getiri ve momentum saglar.", _CHUNK) is False


def test_single_digit_enumeration_not_blocked() -> None:
    # Tek haneli sayma (2 varsayim) anlamlı sayı değildir → kapı engellememeli;
    # anchor örtüşmesi (momentum) ile grounded.
    assert _is_grounded("Pasajda 2 temel momentum varsayimi vardir ve ATR olcer.", _CHUNK) is True


def test_info_poor_passage_rejected() -> None:
    # Bilgi-fakir pasaj (yeterli anchor yok) → grounded QA üretilemez, RED.
    assert _is_grounded("x" * 70, "ab cd ef") is False


# ----------------------------------------------------------------- build_for_chunk
def test_build_for_chunk_keeps_only_grounded_and_long() -> None:
    builder = SyntheticQABuilder(llm=_FakeLLM(_GOOD_PAYLOAD))
    out = builder.build_for_chunk(_CHUNK, paper_id="p1", chunk_id="p1_c0", n=3)

    assert len(out) == 1  # kısa + uydurma elenir
    ex = out[0]
    assert ex.messages[0]["role"] == "system"
    assert "BAĞLAM:" in ex.messages[1]["content"]  # RAG-stili bağlam gömülü
    assert "SORU: ATR" in ex.messages[1]["content"]
    assert ex.metadata["synthetic"] is True
    assert ex.metadata["persona"] == "kantitatif araştırmacı"
    assert ex.metadata["question"].startswith("ATR")


def test_include_context_false_uses_bare_question() -> None:
    builder = SyntheticQABuilder(llm=_FakeLLM(_GOOD_PAYLOAD))
    out = builder.build_for_chunk(
        _CHUNK, paper_id="p1", chunk_id="p1_c0", n=3, include_context=False
    )
    assert out[0].messages[1]["content"].startswith("ATR")  # bağlam yok


def test_seed_is_forwarded_deterministically() -> None:
    # Etkin seed = taban(seed) + persona_index → determinist ama chunk'lar arası çeşitli.
    llm = _FakeLLM(_GOOD_PAYLOAD)
    builder = SyntheticQABuilder(llm=llm, seed=42)
    builder.build_for_chunk(_CHUNK, paper_id="p", chunk_id="c", n=3, persona_index=2)
    assert llm.last_seed == 44  # 42 + 2


def test_llm_unavailable_returns_empty() -> None:
    from app.brain.local_llm import LLMUnavailable

    class _Dead:
        def generate(self, *a: object, **k: object) -> str:
            raise LLMUnavailable("down")

    builder = SyntheticQABuilder(llm=_Dead())
    assert builder.build_for_chunk(_CHUNK, paper_id="p", chunk_id="c", n=3) == []


# ----------------------------------------------------------------- build_for_paper
def test_build_for_paper_rotates_personas() -> None:
    store = _StubStore({"p1": [_StubChunk("p1_c0", _CHUNK), _StubChunk("p1_c1", _CHUNK)]})
    builder = SyntheticQABuilder(llm=_FakeLLM(_GOOD_PAYLOAD))
    out = builder.build_for_paper(store, "p1", per_chunk=3, max_chunks=12)

    assert len(out) == 2  # her chunk'tan 1 grounded örnek
    personas = {ex.metadata["persona"] for ex in out}
    assert len(personas) == 2  # chunk'lar arası persona rotasyonu


def test_build_for_paper_respects_max_chunks() -> None:
    store = _StubStore({"p1": [_StubChunk(f"p1_c{i}", _CHUNK) for i in range(10)]})
    builder = SyntheticQABuilder(llm=_FakeLLM(_GOOD_PAYLOAD))
    out = builder.build_for_paper(store, "p1", per_chunk=3, max_chunks=3)
    assert len(out) == 3


# --------------------------------------------------------- generate_synthetic_dataset
def test_generate_dataset_dedups_identical_examples() -> None:
    # İki chunk aynı payload → aynı (soru,cevap) → biri duplicate elenmeli.
    store = _StubStore({"p1": [_StubChunk("p1_c0", _CHUNK), _StubChunk("p1_c1", _CHUNK)]})
    kept, stats = generate_synthetic_dataset(store, llm=_FakeLLM(_GOOD_PAYLOAD), per_chunk=3)
    assert stats["raw"] == 2
    assert stats["kept"] == 1
    assert stats["rejected"] == 1
    assert len(kept) == 1
