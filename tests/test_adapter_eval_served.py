"""lora-eval: servis edilen yapıt (Ollama GGUF; B6) + tekrar + kritik hata = NO-GO.

Tamamen çevrimdışı: Ollama ``httpx.MockTransport`` ile taklit edilir.
"""

from __future__ import annotations

import json

import httpx
import pytest
from tests.chat_learning_helpers import iso  # noqa: F401

from app.training import adapter_eval as ae


def _set(tmp_path, n=6):
    p = tmp_path / "set.jsonl"
    p.write_text(
        "".join(
            json.dumps(
                {
                    "question": f"Soru {i}: strateji {i} için ne test edilmeli?",
                    "must_avoid": ["garanti"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for i in range(n)
        ),
        "utf-8",
    )
    return p


def _ollama(answer_for, digests=None):
    seen: list[dict] = []
    dig = digests or {"base": ["b" * 64], "aday": ["c" * 64]}
    calls = {"tags": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            calls["tags"] += 1
            i = 0 if calls["tags"] <= 2 else -1
            return httpx.Response(
                200, json={"models": [{"name": k, "digest": v[i]} for k, v in dig.items()]}
            )
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"template": "t"})
        body = json.loads(req.content)
        seen.append(body)
        q = body["messages"][-1]["content"]
        text = answer_for(body["model"], q, body["options"]["seed"])
        return httpx.Response(200, json={"message": {"content": text}, "done_reason": "stop"})

    return httpx.MockTransport(handler), seen


_TOPICS = (
    "momentum kesişiminin yatay piyasada yanlış sinyal oranı",
    "volatilite hedeflemesinin kaldıraç değişimiyle ilişkisi",
    "ortalama dönüş varsayımının rejim kırılmasındaki davranışı",
    "işlem sıklığı arttıkça slippage payının net getiriyi aşındırması",
    "parametre taramasının çoklu karşılaştırma kaynaklı aşırı uyumu",
    "stop mesafesinin ATR ile ölçeklenmesinin pozisyon boyutuna etkisi",
)


def _good(model, q, seed):
    i = int(q.split(":")[0].split()[-1])
    return (
        f"Hipotez: {_TOPICS[i]} örneklem-dışı dönemde ayrıca ölçülmeli; komisyon ve "
        f"slippage dahil edilmeden sonuç yorumlanamaz. Test noktası {i}. ({model} {seed})"
    )


def test_served_eval_uses_ollama_with_repeats_and_reports_n(iso, tmp_path) -> None:  # noqa: F811
    transport, seen = _ollama(_good)
    res = ae.evaluate_served(
        _set(tmp_path), candidate_tag="aday", base_tag="base", repeats=3, transport=transport
    )
    assert res.artifact == "ollama" and res.repeats == 3 and res.n == 6
    assert len(seen) == 6 * 2 * 3
    assert {b["options"]["seed"] for b in seen} == {42, 43, 44}
    assert all(b["options"]["temperature"] == 0.0 for b in seen)
    assert all(b["messages"][0]["role"] == "user" for b in seen)  # PEFT yoluyla aynı istem
    d = res.to_dict()
    assert d["repeats"] == 3 and len(d["per_repeat"]) == 3 and d["models"]["digests"]
    assert res.verdict != "reject" and res.gate != "NO-GO"


def test_critical_rule1_language_in_one_repeat_is_no_go(iso, tmp_path) -> None:  # noqa: F811
    def answer(model, q, seed):
        if model == "aday" and seed == 44 and q.startswith("Soru 2"):
            return "Bu stratejiyle kâr garantili, hemen al."
        return _good(model, q, seed)

    transport, _ = _ollama(answer)
    res = ae.evaluate_served(
        _set(tmp_path), candidate_tag="aday", base_tag="base", repeats=3, transport=transport
    )
    assert res.verdict == "reject" and res.gate == "NO-GO"
    assert "kritik_hata" in res.vetoes or "guaranteed_profit" in res.vetoes
    assert res.per_repeat[2]["verdict"] == "reject"


def test_digest_change_during_run_invalidates(iso, tmp_path) -> None:  # noqa: F811
    transport, _ = _ollama(
        _good, digests={"base": ["b" * 64, "b" * 64], "aday": ["c" * 64, "e" * 64]}
    )
    with pytest.raises(RuntimeError, match="digest"):
        ae.evaluate_served(
            _set(tmp_path), candidate_tag="aday", base_tag="base", repeats=2, transport=transport
        )


def test_peft_path_gate_never_claims_served_measurement() -> None:
    from app.evals.llm_training_eval import TrainingEvalItem

    items = [TrainingEvalItem(question=f"Soru {i} nedir?") for i in range(6)]
    good = [_good("x", f"Soru {i}: a", 42) for i in range(6)]
    res = ae._score_answers(items, good, good, eval_set="s", base_model="b", adapter="a")
    assert res.artifact == "peft" and res.gate != "insan_onayi_bekliyor"
    assert any("GGUF ölçülmedi" in n for n in res.gate_notes())
    bad = [*good[:5], "Kesin kazanırsın, şimdi al."]
    res2 = ae._score_answers(items, good, bad, eval_set="s", base_model="b", adapter="a")
    assert res2.verdict == "reject" and res2.gate == "NO-GO"
    assert any(r["adapter_critical"] for r in res2.rows)
