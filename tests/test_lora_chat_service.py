"""Web LoRA sohbeti — eğitim ↔ çıkarım prompt eşleşmesi (çevrimdışı; model yüklenmez).

Bulgu (2026-09-13): sohbet yalnız çıplak kullanıcı mesajı gönderiyordu. Oysa v8'in
eğitim verisinin %92'sinde SYSTEM_PROMPT, %72'sinde "BAĞLAM: … SORU: …" vardı → model
eğitildiği dağılımın dışında sorgulanıyor, cevabı ezberinden uyduruyordu. Ayrıca adapter
listesi alfabetik olduğu için arayüz v8 varken v7'yi varsayılan seçiyordu.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.lora.dataset_builder import SYSTEM_PROMPT
from app.memory.retrieval_service import RetrievedChunk
from app.training import adapter_eval
from app.web import lora_chat_service as svc


def _chunk(text: str, chunk_id: str = "c0001") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="paper_x",
        text=text,
        page_number=3,
        section_name=None,
        title="Backtest tuzakları",
        distance=0.21,
    )


class _StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int | None]] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.calls.append((query, top_k))
        return list(self.chunks)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Ayarları tmp'ye yönlendir; model yükleme/üretimi kaydedici sahtelerle değiştir."""
    rec: dict[str, Any] = {"loads": 0, "generate": []}
    settings = SimpleNamespace(
        adapters_dir=tmp_path,
        peft_base_model="base-4b",
        rag_abstain_min_similarity=0.55,
        rag_abstain_min_margin=0.02,
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)

    def _load(base: str, adapter_dir: str | None) -> tuple[str, str]:
        rec["loads"] += 1
        return "tok", "model"

    def _gen(tok: Any, model: Any, question: str, max_new_tokens: int = 220, **kw: Any) -> str:
        rec["generate"].append({"question": question, "max_new_tokens": max_new_tokens, **kw})
        return "cevap"

    monkeypatch.setattr(adapter_eval, "_load_model", _load)
    monkeypatch.setattr(adapter_eval, "_generate", _gen)
    monkeypatch.setitem(svc._CACHE, "key", None)
    return rec


def test_build_messages_system_verilirse_basta() -> None:
    assert adapter_eval._build_messages("soru", "sistem") == [
        {"role": "system", "content": "sistem"},
        {"role": "user", "content": "soru"},
    ]


def test_build_messages_system_yoksa_yalniz_user() -> None:
    # Eval bilerek system'siz çağırır; bu davranış değişmemeli.
    assert adapter_eval._build_messages("soru") == [{"role": "user", "content": "soru"}]


def test_sohbet_egitimdeki_system_promptu_gonderir(fake_model: dict[str, Any]) -> None:
    out = svc.chat("Look-ahead bias nedir?", None, max_tokens=64)

    (call,) = fake_model["generate"]
    assert call["system"] == SYSTEM_PROMPT
    assert call["question"] == "Look-ahead bias nedir?"
    assert call["max_new_tokens"] == 64
    assert out["answer"] == "cevap"
    assert out["llm_used"] is True
    assert out["used_context"] is False
    assert out["sources"] == []
    assert out["base_model"] == "base-4b"


def test_kaynakli_mod_baglami_egitim_bicimiyle_gomer(fake_model: dict[str, Any]) -> None:
    retriever = _StubRetriever(
        [_chunk("Pozisyon shift(1) ile gecikmeli."), _chunk("İkinci.", "c2")]
    )
    looked: list[str] = []

    def _cards(pid: str) -> dict | None:
        looked.append(pid)
        return {"title": "Backtest tuzakları", "main_claim": "Lag positions by one bar."}

    out = svc.chat(
        "Nasıl önlenir?",
        None,
        use_context=True,
        top_k=4,
        retriever=retriever,
        card_lookup=_cards,
    )

    assert retriever.calls == [("Nasıl önlenir?", 4)]
    (call,) = fake_model["generate"]
    # synthetic_qa_builder / discipline_dataset ile BİREBİR aynı biçim
    assert call["question"] == (
        "BAĞLAM:\nPozisyon shift(1) ile gecikmeli.\n\nİkinci.\n\nSORU: Nasıl önlenir?"
    )
    assert call["system"] == SYSTEM_PROMPT
    assert out["used_context"] is True
    assert [s["chunk_id"] for s in out["sources"]] == ["c0001", "c2"]
    assert out["sources"][0] == {
        "paper_id": "paper_x",
        "chunk_id": "c0001",
        "title": "Backtest tuzakları",
        "page": 3,
        "distance": 0.21,
    }
    # Hibrit cevap: model yalnız Kısa Cevap; kart tek makale için bir kez çekildi
    assert looked == ["paper_x"]
    titles = [s["title"] for s in out["sections"]]
    assert len(titles) == 8 and titles[0] == "Kısa Cevap"
    assert out["sections"][0] == {
        "title": "Kısa Cevap",
        "body": "cevap",
        "source": "model",
        "warning": False,
    }
    assert "Lag positions by one bar." in out["sections"][3]["body"]


def test_kaynaksiz_modda_bolum_sablonu_yok(fake_model: dict[str, Any]) -> None:
    out = svc.chat("soru", None)
    assert out["sections"] == []


def test_kaynak_yoksa_model_hic_cagrilmaz(fake_model: dict[str, Any]) -> None:
    retriever = _StubRetriever([_chunk("   ")])  # yalnız boş metinli parça = kaynak yok

    def _no_cards(pid: str) -> dict | None:
        raise AssertionError("kaynak yokken kart çekilmemeli")

    out = svc.chat(
        "Nasıl önlenir?", None, use_context=True, retriever=retriever, card_lookup=_no_cards
    )

    assert fake_model["loads"] == 0
    assert fake_model["generate"] == []
    assert out["llm_used"] is False
    assert out["answer"] == svc.NO_SOURCE_ANSWER
    assert out["sources"] == []


def test_baglam_butcesi_asilmaz() -> None:
    chunks = [_chunk("a" * 1500, "c1"), _chunk("b" * 1500, "c2"), _chunk("c" * 1500, "c3")]
    content = svc.build_user_content("soru", chunks, budget=2400)
    context = content.removeprefix("BAĞLAM:\n").removesuffix("\n\nSORU: soru")
    assert len(context.replace("\n\n", "")) == 2400
    assert "c" not in context


def test_olmayan_adapter_404(fake_model: dict[str, Any]) -> None:
    with pytest.raises(FileNotFoundError):
        svc.chat("soru", "yok_boyle_adapter")


def test_adapter_listesi_en_yeni_once_ve_yarim_olanlari_atlar(
    fake_model: dict[str, Any], tmp_path: Path
) -> None:
    def _make(name: str, mtime: float, weights: bool = True) -> None:
        d = tmp_path / name
        d.mkdir()
        (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        if weights:
            w = d / "adapter_model.safetensors"
            w.write_bytes(b"x")
            os.utime(w, (mtime, mtime))

    _make("hektor_lora_v7_4b", 1_000)
    _make("hektor_lora_v8_4b", 3_000)
    _make("hektor_smoke_olcum", 2_000)
    _make("hektor_lora_v6_local", 9_000, weights=False)  # yalnız checkpoint'li yarım koşu

    assert svc.list_adapters() == ["hektor_lora_v8_4b", "hektor_smoke_olcum", "hektor_lora_v7_4b"]


# ---------- API katmanı ----------
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    from app.web.server import app

    return TestClient(app)


def test_api_kaynakli_parametreleri_servise_iletir(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def _chat(question: str, adapter: str | None, **kw: Any) -> dict:
        seen.update(kw)
        return {
            "answer": "cevap",
            "adapter": adapter or "(base)",
            "base_model": "base-4b",
            "used_context": kw["use_context"],
            "llm_used": True,
            "sources": [{"paper_id": "p", "chunk_id": "c1", "title": None, "page": 2}],
        }

    monkeypatch.setattr(svc, "chat", _chat)
    r = client.post(
        "/api/lora-chat",
        json={"question": "soru?", "adapter": None, "use_context": True, "top_k": 3},
    )
    assert r.status_code == 200, r.text
    assert seen == {"max_tokens": 256, "use_context": True, "top_k": 3}
    body = r.json()
    assert body["used_context"] is True
    assert body["sources"][0]["chunk_id"] == "c1"


def test_api_import_hatasinda_surum_catismasini_da_soyler(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _chat(*a: Any, **kw: Any) -> dict:
        raise ImportError("tokenizers>=0.23.1,<0.24.0 is required ... found tokenizers==0.22.2")

    monkeypatch.setattr(svc, "chat", _chat)
    r = client.post("/api/lora-chat", json={"question": "soru?"})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "tokenizers==0.22.2" in detail
    assert "uv sync --extra dev --inexact" in detail
