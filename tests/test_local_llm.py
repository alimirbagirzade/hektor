"""LocalLLM testleri — ağ/zaman aşımı hataları LLMUnavailable'a çevrilmeli.

Aksi halde ham httpx hatası çağıranlara sızar (örn. sınav harness'i sadece
LLMUnavailable yakalar) ve /api/understanding-score gibi uçlar 500 olur.
"""

from __future__ import annotations

import httpx
import pytest

from app.brain.local_llm import LLMUnavailable, LocalLLM


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timed out"),
    ],
)
def test_ollama_ag_hatasi_llm_unavailable_olur(monkeypatch, exc: Exception) -> None:
    llm = LocalLLM()
    monkeypatch.setattr(LocalLLM, "active_backend", lambda self: "ollama")

    def _boom(self, *a, **k):
        raise exc

    monkeypatch.setattr(httpx.Client, "post", _boom)
    with pytest.raises(LLMUnavailable):
        llm.generate("merhaba", timeout=1)


def test_ollama_http_durum_hatasi_llm_unavailable_olur(monkeypatch) -> None:
    # Ollama 500/404 dönerse de ham HTTPStatusError sızmamalı → LLMUnavailable.
    llm = LocalLLM()
    monkeypatch.setattr(LocalLLM, "active_backend", lambda self: "ollama")

    class _Resp:
        status_code = 500

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "server error",
                request=httpx.Request("POST", "http://x"),
                response=self,  # type: ignore[arg-type]
            )

    monkeypatch.setattr(httpx.Client, "post", lambda self, *a, **k: _Resp())
    with pytest.raises(LLMUnavailable):
        llm.generate("merhaba", timeout=1)
