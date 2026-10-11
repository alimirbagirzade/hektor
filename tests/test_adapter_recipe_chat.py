"""Adapter başına istem biçimi + bağlam bütçesi (çevrimdışı; sahte tokenizer, model yüklenmez).

Bulgu (2026-10-11): web LoRA sohbeti her adapter'a v11–v13 reçetesini (1024 token, sentetik QA
"BAĞLAM: … SORU: …", 2400 karakter bağlam) uyguluyordu. `hektor_lora_v14_30b` ise
`moe30b_attn_long` (max_seq_length 6144) ile canlı RAG istemiyle (`build_rag_prompt`) eğitildi
→ v14 sohbette eğitim dağılımı dışında sorgulanıyordu.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.brain.rag_answerer import build_rag_prompt
from app.lora.dataset_builder import SYSTEM_PROMPT
from app.memory.retrieval_service import RetrievedChunk
from app.training import adapter_eval
from app.training.adapter_recipe import (
    LIVE_RAG,
    SYNTHETIC_QA,
    count_prompt_formats,
    detect_adapter_recipe,
)
from app.training.self_distill import PLAIN_NO_SOURCE_NOTE
from app.web import lora_chat_service as svc


class FakeTokenizer:
    """Boşlukla ayrılmış her parça = 1 token; şablon sabit 3 token ekler."""

    def apply_chat_template(
        self, msgs: list[dict], tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        body = " ".join(f"<{m['role']}> {m['content']}" for m in msgs)
        return f"{body} <assistant>"

    def __call__(self, text: str) -> dict:
        return {"input_ids": list(range(len(text.split())))}


def _tokens(system: str, user: str) -> int:
    return svc.count_prompt_tokens(FakeTokenizer(), system, user)


def _chunk(cid: str, words: int, paper: str = "paper_x") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        paper_id=paper,
        text=" ".join(f"{cid}w{i}" for i in range(words)),
        page_number=2,
        section_name=None,
        title="Başlık",
        distance=0.2,
    )


def _adapter(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    return d


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


# ---------- Reçete tespiti ----------


def test_v14_eski_run_plan_arsivlenmis_durumdan_profil_bulur(tmp_path: Path) -> None:
    # v14'ün klasöründe FİİLEN yazılan: adım alanlı run_plan.json + train_status.final.json.
    d = _adapter(tmp_path, "hektor_lora_v14_30b")
    _write(d / "run_plan.json", {"max_steps": 187, "planned_iterations": 1489})
    _write(
        d / "train_status.final.json",
        {"adapter": "hektor_lora_v14_30b", "profile": "moe30b_attn_long", "finished_at": "x"},
    )
    r = detect_adapter_recipe(d)
    assert (r.prompt_format, r.max_seq_length, r.profile) == (LIVE_RAG, 6144, "moe30b_attn_long")
    assert r.source == "train_status.final.json" and r.known


def test_yeni_run_plan_olculen_bicimi_kullanir(tmp_path: Path) -> None:
    d = _adapter(tmp_path, "hektor_lora_v15")
    _write(
        d / "run_plan.json",
        {
            "max_steps": 10,
            "profile": "ozel",  # YAML'da olmayan profil — ölçüm yeterli
            "max_seq_length": 4096,
            "prompt_formats": {"live_rag": 3, "synthetic_qa": 9, "plain": 1},
        },
    )
    r = detect_adapter_recipe(d)
    assert (r.prompt_format, r.max_seq_length, r.source) == (LIVE_RAG, 4096, "run_plan.json")


def test_kayit_defteri_notundan_profil(tmp_path: Path) -> None:
    d = _adapter(tmp_path / "adapters", "hektor_lora_v12_30b")
    reg = tmp_path / "registry.jsonl"
    rows = [
        {"adapter_name": "baska", "notes": "profile=moe30b_attn_long"},
        {"adapter_name": "hektor_lora_v12_30b", "notes": "train --run (max_steps=5, profile=-)"},
        {
            "adapter_name": "hektor_lora_v12_30b",
            "notes": "denetimli train --run --backend peft (max_steps=187, "
            "profile=moe30b_attn_local) mix[wd_1]=trading_analysis_v1",
        },
    ]
    reg.write_text("\n".join(json.dumps(x) for x in rows) + "\nbozuk{\n", encoding="utf-8")
    r = detect_adapter_recipe(d, registry_path=reg)
    assert (r.prompt_format, r.max_seq_length, r.source) == (SYNTHETIC_QA, 1024, "registry")


def test_baska_adapterin_durum_kaydi_sayilmaz_reçete_bilinmez(tmp_path: Path) -> None:
    d = _adapter(tmp_path, "hektor_lora_v13_30b")
    _write(d / "train_status.final.json", {"adapter": "hektor_lora_v14_30b", "profile": "x"})
    r = detect_adapter_recipe(d, registry_path=tmp_path / "yok.jsonl")
    assert not r.known
    assert r.prompt_format == SYNTHETIC_QA and r.max_seq_length is None
    assert "hektor_lora_v13_30b" in (r.warning or "")


def test_bicim_sayimi() -> None:
    rag_user = build_rag_prompt("soru?", [_chunk("c1", 3)])[1]
    rows = [
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": rag_user}]},
        {"messages": [{"role": "user", "content": "BAĞLAM:\nx\n\nSORU: y"}]},
        {"messages": [{"role": "user", "content": "Çıplak soru?"}]},
        {"bozuk": True},
    ]
    assert count_prompt_formats(rows) == {"live_rag": 1, "synthetic_qa": 1, "plain": 1}


def test_sentetik_butce_1024te_eski_degerle_ayni() -> None:
    assert svc.synthetic_char_budget(None) == svc.CONTEXT_CHAR_BUDGET == 2400
    assert svc.synthetic_char_budget(1024) == 2400
    assert svc.synthetic_char_budget(2048) == (2048 - 424) * 4


# ---------- Canlı RAG istemini bütçeye sığdırma ----------


def test_sigdirma_dusuk_sirali_parcayi_butun_dusurur_istem_bayt_ayni() -> None:
    chunks = [_chunk("c1", 50), _chunk("c2", 50), _chunk("c3", 50)]
    full = build_rag_prompt("soru?", chunks)
    two = build_rag_prompt("soru?", chunks[:2])
    budget = _tokens(*two)  # tam 2 parça sığar, 3'ü sığmaz
    assert _tokens(*full) > budget

    system, user, used = svc.fit_live_rag_prompt(
        "soru?", chunks, FakeTokenizer(), max_prompt_tokens=budget
    )
    assert [c.chunk_id for c in used] == ["c1", "c2"]
    assert (system, user) == two  # kırpma yok, canlı istemle bayt-aynı
    assert user.startswith("SOURCES / KAYNAKLAR:") and "QUESTION / SORU: soru?" in user


def test_tek_parca_sigmazsa_kisaltilir_hic_sigmazsa_hata() -> None:
    one = [_chunk("c1", 400)]
    empty_budget = _tokens(*build_rag_prompt("soru?", [_chunk("c1", 0)]))
    _, _user, used = svc.fit_live_rag_prompt(
        "soru?", one, FakeTokenizer(), max_prompt_tokens=empty_budget + 20
    )
    assert len(used) == 1 and 0 < len(used[0].text) < len(one[0].text)
    assert one[0].text.startswith(used[0].text)  # önek; orijinal parça değişmedi
    with pytest.raises(svc.ContextBudgetError):
        svc.fit_live_rag_prompt("soru?", one, FakeTokenizer(), max_prompt_tokens=empty_budget - 1)


# ---------- Sohbet uçtan uca (sahte model) ----------


class _Retriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return list(self.chunks)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"generate": []}
    adapters = tmp_path / "models" / "adapters"
    adapters.mkdir(parents=True)
    settings = SimpleNamespace(
        root=tmp_path,
        adapters_dir=adapters,
        peft_base_model="base-30b",
        rag_abstain_min_similarity=0.55,
        rag_abstain_min_margin=0.02,
        rag_reorder_context=True,
    )
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr(adapter_eval, "_load_model", lambda base, ad: (FakeTokenizer(), "model"))

    def _gen(tok: Any, model: Any, question: str, max_new_tokens: int = 0, **kw: Any) -> str:
        rec["generate"].append({"question": question, "max_new_tokens": max_new_tokens, **kw})
        return "cevap [paper_x:c1, s.2]"

    monkeypatch.setattr(adapter_eval, "_generate", _gen)
    monkeypatch.setitem(svc._CACHE, "key", None)
    rec["adapters"] = adapters
    return rec


def _v14(adapters: Path) -> None:
    d = _adapter(adapters, "hektor_lora_v14_30b")
    _write(d / "run_plan.json", {"max_steps": 187})
    _write(
        d / "train_status.final.json",
        {"adapter": "hektor_lora_v14_30b", "profile": "moe30b_attn_long"},
    )


def test_v14_kaynakli_sohbet_canli_rag_istemi_ve_token_butcesi(env: dict[str, Any]) -> None:
    _v14(env["adapters"])
    chunks = [_chunk("c1", 1500), _chunk("c2", 1500), _chunk("c3", 3000, "paper_y")]
    out = svc.chat(
        "Look-ahead nasıl önlenir?",
        "hektor_lora_v14_30b",
        max_tokens=1024,
        use_context=True,
        retriever=_Retriever(chunks),
        card_lookup=lambda pid: None,
    )
    (call,) = env["generate"]
    system, user = build_rag_prompt("Look-ahead nasıl önlenir?", chunks[:2])
    assert (call["system"], call["question"]) == (system, user)  # eğitimle bayt-aynı biçim
    assert "BAĞLAM:" not in call["question"]
    assert _tokens(system, user) <= 6144 - 1024
    assert out["prompt_format"] == "live_rag"
    assert out["recipe_source"] == "train_status.final.json"
    assert out["recipe_warning"] is None
    assert out["context_budget"] == "5120 token"
    # 3. parça düşüldü → kaynaklar yalnız modele GERÇEKTEN gidenler
    assert [s["chunk_id"] for s in out["sources"]] == ["c1", "c2"]


def test_v14_kaynaksiz_sohbet_plain_sistem_istemi(env: dict[str, Any]) -> None:
    _v14(env["adapters"])
    out = svc.chat("EMA nedir?", "hektor_lora_v14_30b")
    (call,) = env["generate"]
    assert call["system"] == SYSTEM_PROMPT + PLAIN_NO_SOURCE_NOTE
    assert call["question"] == "EMA nedir?"
    assert out["prompt_format"] == "live_rag" and out["context_budget"] is None


def test_v14_max_tokens_verilmezse_uzun_cevap_payi(env: dict[str, Any]) -> None:
    _v14(env["adapters"])
    chunks = [_chunk("c1", 1500), _chunk("c2", 1500), _chunk("c3", 1500)]
    out = svc.chat(
        "soru?",
        "hektor_lora_v14_30b",
        use_context=True,
        retriever=_Retriever(chunks),
        card_lookup=lambda pid: None,
    )
    (call,) = env["generate"]
    # Bağlam en az 2048 token cevap payı bırakacak şekilde sığdırıldı (3. parça düştü) ...
    assert out["context_budget"] == f"{6144 - svc.LIVE_RAG_MIN_ANSWER_TOKENS} token"
    assert [s["chunk_id"] for s in out["sources"]] == ["c1", "c2"]
    # ... ve üretim sınırı kalan yere genişletildi (eğitimdeki ≤3072 cevap, toplam ≤6144).
    used = _tokens(call["system"], call["question"])
    expected = min(3072, 6144 - used)
    assert expected >= svc.LIVE_RAG_MIN_ANSWER_TOKENS
    assert call["max_new_tokens"] == out["max_new_tokens"] == expected
    assert used + call["max_new_tokens"] <= 6144


def test_v14_kisa_baglamda_cevap_siniri_3072(env: dict[str, Any]) -> None:
    _v14(env["adapters"])
    out = svc.chat("EMA nedir?", "hektor_lora_v14_30b")
    assert env["generate"][0]["max_new_tokens"] == out["max_new_tokens"] == 3072


def test_acik_max_tokens_her_zaman_kazanir(env: dict[str, Any]) -> None:
    _v14(env["adapters"])
    svc.chat("EMA nedir?", "hektor_lora_v14_30b", max_tokens=128)
    svc.chat("EMA nedir?", None)  # base / sentetik varsayılan: eski 256
    assert [c["max_new_tokens"] for c in env["generate"]] == [128, svc.DEFAULT_MAX_TOKENS]


def test_receteli_eski_adapter_sentetik_bicimde_kalir(env: dict[str, Any]) -> None:
    d = _adapter(env["adapters"], "hektor_lora_v13_30b")
    _write(
        d / "train_status.final.json",
        {"adapter": "hektor_lora_v13_30b", "profile": "moe30b_attn_local"},
    )
    out = svc.chat(
        "soru",
        "hektor_lora_v13_30b",
        use_context=True,
        retriever=_Retriever([_chunk("c1", 3)]),
        card_lookup=lambda pid: None,
    )
    (call,) = env["generate"]
    assert call["system"] == SYSTEM_PROMPT
    assert call["question"].startswith("BAĞLAM:\n") and call["question"].endswith("SORU: soru")
    assert out["recipe_warning"] is None and out["context_budget"] == "2400 karakter"


def test_sentetik_yolda_kaynaklar_yalniz_modelin_gordugu_parcalar(env: dict[str, Any]) -> None:
    # Kademe 2 F3-1: varsayılan top_k=6 × 1200 karakter, bütçe 2400 → model ~2 parça görür.
    chunks = [
        RetrievedChunk(f"c{i}", f"paper_{i}", "x" * 1200, 1, None, f"T{i}", 0.2) for i in range(6)
    ]
    looked: list[str] = []

    def _cards(pid: str) -> dict | None:
        looked.append(pid)
        return {"title": pid, "main_claim": f"iddia {pid}"}

    out = svc.chat("soru", None, use_context=True, retriever=_Retriever(chunks), card_lookup=_cards)
    (call,) = env["generate"]
    assert call["question"].count("x") == 2400
    assert [s["chunk_id"] for s in out["sources"]] == ["c0", "c1"]
    body = "\n".join(sec["body"] for sec in out["sections"])
    assert "paper_0" in body and "paper_2" not in body and "paper_5" not in body


def test_recetesi_bilinmeyen_adapter_eski_davranis_ve_gorunur_uyari(
    env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    _adapter(env["adapters"], "hektor_lora_v11_30b")
    with caplog.at_level("WARNING"):
        out = svc.chat(
            "soru",
            "hektor_lora_v11_30b",
            use_context=True,
            retriever=_Retriever([_chunk("c1", 3)]),
            card_lookup=lambda pid: None,
        )
    (call,) = env["generate"]
    assert call["system"] == SYSTEM_PROMPT
    assert call["question"] == svc.build_user_content("soru", [_chunk("c1", 3)])
    assert out["prompt_format"] == "synthetic_qa"
    assert out["recipe_source"] == "bilinmiyor"
    assert "belirlenemedi" in (out["recipe_warning"] or "")
    assert any("belirlenemedi" in r.message for r in caplog.records)


def test_base_model_uyarisiz_eski_davranis(env: dict[str, Any]) -> None:
    out = svc.chat("soru", None)
    (call,) = env["generate"]
    assert call["system"] == SYSTEM_PROMPT and call["question"] == "soru"
    assert out["recipe_source"] == "base" and out["recipe_warning"] is None


# ---------- API ----------
fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    from app.web.server import app

    return TestClient(app)


def test_api_recete_alanlari_ve_butce_hatasi(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.training.detached_launch.is_running", lambda: False)

    def _ok(question: str, adapter: str | None, **kw: Any) -> dict:
        return {
            "answer": "cevap",
            "adapter": "x",
            "base_model": "b",
            "prompt_format": "synthetic_qa",
            "recipe_source": "bilinmiyor",
            "recipe_warning": "reçete belirlenemedi",
            "context_budget": "2400 karakter",
        }

    monkeypatch.setattr(svc, "chat", _ok)
    body = client.post("/api/lora-chat", json={"question": "soru?"}).json()
    assert body["recipe_warning"] == "reçete belirlenemedi"
    assert body["context_budget"] == "2400 karakter"

    def _too_long(*a: Any, **kw: Any) -> dict:
        raise svc.ContextBudgetError("sığmıyor")

    monkeypatch.setattr(svc, "chat", _too_long)
    r = client.post("/api/lora-chat", json={"question": "soru?"})
    assert r.status_code == 422 and "sığmıyor" in r.json()["detail"]
