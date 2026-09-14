"""LocalLLM düşünme (thinking) kipi + num_predict tavanı testleri — sahte HTTP, çevrimdışı.

Kök neden (2026-09-13, Ollama 0.34.0): `qwen3:4b` = Qwen3-4B-Thinking-2507; şablonu
`<think>` açar ve `.Think` dalı yoktur → `think: false` etkisiz, düşünme düz metin olarak
`response`'a sızar. İstemci `/api/show` ile kipi seçmeli ve `thinking` alanını ASLA cevap
olarak döndürmemeli.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.brain.local_llm import (
    LLMUnavailable,
    LocalLLM,
    classify_think_support,
    strip_thinking,
)
from app.config import get_settings

# Gerçek qwen3:4b (thinking-2507) şablonunun ilgili parçaları: `.IsThinkSet` ve
# `.Thinking` var, `.Think` yok; asistan turu koşulsuz `<think>` ile açılıyor.
_FORCED_TEMPLATE = (
    '{{ else if eq .Role "assistant" }}<|im_start|>assistant\n'
    "{{ if (and $.IsThinkSet (and .Thinking $last)) -}}<think>{{ .Thinking }}</think>{{ end -}}"
    '{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant\n<think>\n{{ end }}'
)
# Hibrit qwen3 şablonu: düşünme kapalıyken boş blok ekleyen `.Think` dalı var.
_TOGGLE_TEMPLATE = (
    "<|im_start|>assistant\n{{ if and $.IsThinkSet (not $.Think) }}<think>\n\n</think>\n\n{{ end }}"
)

# Gerçek qwen3:4b-instruct-2507-q4_K_M şablonlarının ilgili parçaları (2026-09-13).
_INSTRUCT_JINJA_TEMPLATE = (
    "{%- if '</think>' in content %}"
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content"
    " + '\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)
_INSTRUCT_GO_TEMPLATE = (
    "{{ if (and $.IsThinkSet (and .Thinking $last)) -}}<think>{{ .Thinking }}</think>{{ end -}}"
    '{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant\n{{ end }}'
)

_THINKING_CAPS = ["completion", "tools", "thinking"]


class _FakeOllama:
    """Kaydeden sahte Ollama: /api/tags, /api/show, /api/generate."""

    def __init__(
        self,
        *,
        show: dict[str, Any] | None = None,
        show_status: int = 200,
        generate: dict[str, Any] | None = None,
    ) -> None:
        self.show = show if show is not None else {"capabilities": ["completion"]}
        self.show_status = show_status
        self.generate = generate or {"response": "tamam", "done_reason": "stop"}
        self.show_calls = 0
        self.payloads: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": []})
        if path == "/api/show":
            self.show_calls += 1
            return httpx.Response(self.show_status, json=self.show)
        if path == "/api/generate":
            self.payloads.append(json.loads(request.content))
            return httpx.Response(200, json=self.generate)
        return httpx.Response(404)

    def llm(self) -> LocalLLM:
        return LocalLLM(
            model="qwen3:4b",
            host="http://ollama.test",
            transport=httpx.MockTransport(self.handler),
        )


# ---------------------------------------------------------------- sınıflandırma


@pytest.mark.parametrize(
    ("show", "beklenen"),
    [
        ({"capabilities": ["completion", "tools"], "template": "<|im_start|>"}, "none"),
        ({"capabilities": _THINKING_CAPS, "template": _FORCED_TEMPLATE}, "forced"),
        ({"capabilities": _THINKING_CAPS, "template": _TOGGLE_TEMPLATE}, "toggle"),
        # Yerleşik renderer: şablonda `<think>` yok → kapatmayı Ollama yönetir.
        ({"capabilities": _THINKING_CAPS, "template": "{{ .Prompt }}"}, "toggle"),
        # Düşünmesiz Instruct-2507: `thinking` ilan eder ama `<think>` yalnız kapalı geçmiş
        # blokta; üretim prompt'u düz `assistant\n` (Jinja ve Go biçimleri, gerçek şablon).
        ({"capabilities": _THINKING_CAPS, "template": _INSTRUCT_JINJA_TEMPLATE}, "toggle"),
        ({"capabilities": _THINKING_CAPS, "template": _INSTRUCT_GO_TEMPLATE}, "toggle"),
        ({}, "none"),
    ],
)
def test_classify_think_support(show: dict[str, Any], beklenen: str) -> None:
    assert classify_think_support(show) == beklenen


@pytest.mark.parametrize(
    ("ham", "temiz"),
    [
        ("<think>\nuzun düşünme\n</think>\n\n4", "4"),
        ("Okay, let me think...\n</think>\nDört", "Dört"),  # açılış şablonda kalmış
        ("Cevap: 4 <think>yarım kalan", "Cevap: 4"),
        ("<think>hiç bitmedi", ""),
        ("  düz cevap, %5-10 aralığı  ", "düz cevap, %5-10 aralığı"),
    ],
)
def test_strip_thinking(ham: str, temiz: str) -> None:
    assert strip_thinking(ham) == temiz


# ---------------------------------------------------------------- payload kipi


def test_forced_model_serbest_metin_think_true_ve_ek_butce() -> None:
    fake = _FakeOllama(
        show={"capabilities": _THINKING_CAPS, "template": _FORCED_TEMPLATE},
        generate={"response": "4", "thinking": "Okay, so 2+2...", "done_reason": "stop"},
    )
    s = get_settings()
    out = fake.llm().generate("2+2 kaç eder?", max_tokens=64)
    assert out == "4"
    p = fake.payloads[0]
    assert p["think"] is True
    assert p["options"]["num_predict"] == 64 + s.llm_thinking_extra_tokens


def test_forced_model_json_formatinda_think_false_kalir() -> None:
    # think=true + format → Ollama JSON'u `thinking`'e yazıp cevabı boşaltıyor (ölçüldü).
    fake = _FakeOllama(
        show={"capabilities": _THINKING_CAPS, "template": _FORCED_TEMPLATE},
        generate={"response": '{"answer": "4"}', "done_reason": "stop"},
    )
    out = fake.llm().generate("json ver", fmt="json", max_tokens=200)
    assert out == '{"answer": "4"}'
    p = fake.payloads[0]
    assert p["think"] is False
    assert p["format"] == "json"
    assert p["options"]["num_predict"] == 200


def test_toggle_model_think_false() -> None:
    fake = _FakeOllama(show={"capabilities": _THINKING_CAPS, "template": _TOGGLE_TEMPLATE})
    fake.llm().generate("merhaba")
    assert fake.payloads[0]["think"] is False


def test_dusunmesiz_model_think_alani_gonderilmez() -> None:
    fake = _FakeOllama(show={"capabilities": ["completion", "tools"], "template": "x"})
    fake.llm().generate("merhaba")
    assert "think" not in fake.payloads[0]


def test_show_hatasinda_eski_davranis_ve_yeniden_deneme() -> None:
    fake = _FakeOllama(show_status=500)
    llm = fake.llm()
    llm.generate("a")
    llm.generate("b")
    assert fake.payloads[0]["think"] is False
    assert fake.show_calls == 2  # "unknown" önbelleğe alınmaz


def test_show_basarili_ise_onbellege_alinir() -> None:
    fake = _FakeOllama(show={"capabilities": ["completion"]})
    llm = fake.llm()
    llm.generate("a")
    llm.generate("b")
    assert fake.show_calls == 1


# ---------------------------------------------------------------- num_predict tavanı


def test_max_tokens_verilmezse_varsayilan_butce() -> None:
    fake = _FakeOllama()
    fake.llm().generate("merhaba")
    assert fake.payloads[0]["options"]["num_predict"] == get_settings().llm_default_max_tokens


def test_max_tokens_tavani_asamaz() -> None:
    fake = _FakeOllama()
    fake.llm().generate("merhaba", max_tokens=10_000_000)
    assert fake.payloads[0]["options"]["num_predict"] == get_settings().llm_max_tokens_cap


# ---------------------------------------------------------------- cevap temizliği


def test_thinking_alani_asla_cevap_olarak_donmez() -> None:
    fake = _FakeOllama(
        generate={"response": "", "thinking": "Okay, let's tackle this...", "done_reason": "stop"}
    )
    assert fake.llm().generate("merhaba") == ""


def test_butce_cevaptan_once_biterse_llm_unavailable() -> None:
    fake = _FakeOllama(
        show={"capabilities": _THINKING_CAPS, "template": _FORCED_TEMPLATE},
        generate={"response": "", "thinking": "Okay, so I need to...", "done_reason": "length"},
    )
    with pytest.raises(LLMUnavailable, match="bütçesini"):
        fake.llm().generate("çevir", max_tokens=16)


def test_cevaptaki_think_etiketi_temizlenir() -> None:
    fake = _FakeOllama(generate={"response": "<think>hmm</think>\n\nDört", "done_reason": "stop"})
    assert fake.llm().generate("2+2?") == "Dört"


# ---------------------------------------------------------------- canlı (Ollama gerekli)


@pytest.mark.ollama
def test_canli_ollama_dusunme_cevaba_sizmaz() -> None:
    llm = LocalLLM()
    out = llm.generate(
        "Tek kelimeyle cevap ver: 2+2 kaç eder?", temperature=0.0, max_tokens=64, seed=42
    )
    assert "4" in out or "dört" in out.lower()
    assert "<think>" not in out and "</think>" not in out
    assert len(out) < 80, f"düşünme metni sızmış olabilir: {out[:120]!r}"


@pytest.mark.ollama
def test_canli_ollama_json_hizli_ve_gecerli() -> None:
    llm = LocalLLM()
    out = llm.generate(
        'Return JSON {"answer": string}: what is 2+2?',
        fmt="json",
        temperature=0.0,
        max_tokens=64,
        seed=42,
        # Paylaşımlı Ollama tek slot: istek başka işin (ör. read-all) arkasında kuyrukta
        # bekleyebilir → süre değil doğruluk ölçülür (120 sn kuyrukta düştü, 2026-09-14).
        timeout=600,
    )
    assert "4" in json.loads(out)["answer"]
