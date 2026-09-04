"""LLM istemcisi — YALNIZ yerel Ollama.

Bu proje **lokal-öncelikliktir**: çalışma-zamanı LLM hattı tamamen yereldir.
Pay-per-token bulut API'si (OpenAI / Anthropic / Google) KULLANILMAZ ve istemci
kodu bilinçli olarak yoktur — kalıcı proje kısıtı. Geliştirme yardımı aylık
abonelikli CLI araçlarıyla (Claude Code / Codex) yapılır, API anahtarıyla değil.

Ollama erişilemezse ``LLMUnavailable`` yükselir; çağıranlar bunu ele alır
(cevap üretmek yerine kaynakları göstermek / sınavı `skipped` saymak gibi).
"""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        s = get_settings()
        self._ollama_host = (host or s.ollama_host).rstrip("/")
        self._ollama_model = model or s.llm_model
        self.model = self._ollama_model  # backward compat
        self._ollama_keep_alive = s.ollama_keep_alive  # RAM darsa "0" → sorgu sonrası boşalt

    # ---------------------------------------------------------------- probes

    def _ollama_alive(self) -> bool:
        try:
            r = httpx.get(f"{self._ollama_host}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def available(self) -> bool:
        return self._ollama_alive()

    def active_backend(self) -> str:
        """'ollama' (canlı) veya 'none'. Başka backend yoktur."""
        return "ollama" if self._ollama_alive() else "none"

    # ---------------------------------------------------------------- generate

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        fmt: str | None = None,
        timeout: int = 600,
        seed: int | None = None,
    ) -> str:
        # seed: determinizm (CLAUDE.md kural 6) — Ollama `options.seed` ile destekler.
        if not self._ollama_alive():
            raise LLMUnavailable(
                "Ollama'ya ulaşılamıyor. Çözüm: `ollama serve` çalıştırın ve "
                f"modeli çekin (`ollama pull {self._ollama_model}`). Host: {self._ollama_host}"
            )
        return self._generate_ollama(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            fmt=fmt,
            timeout=timeout,
            seed=seed,
        )

    # ---------------------------------------------------------------- backend

    def _generate_ollama(
        self,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
        fmt: str | None,
        timeout: int,
        seed: int | None = None,
    ) -> str:
        options: dict = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        if seed is not None:
            options["seed"] = seed
        payload: dict = {
            "model": self._ollama_model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "think": False,
            "keep_alive": self._ollama_keep_alive,
            "options": options,
        }
        if fmt:
            payload["format"] = fmt
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(f"{self._ollama_host}/api/generate", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            # Zaman aşımı / bağlantı / HTTP hatası → yerel model bu çağrıda yanıt
            # üretemedi. Ham httpx hatasını, çağıranların ZATEN ele aldığı
            # LLMUnavailable'a çeviriyoruz; aksi halde (ör. yavaş CPU'da
            # ReadTimeout) sınav harness'i ve /api/understanding-score 500 olur.
            raise LLMUnavailable(f"Ollama yanıt vermedi ({type(exc).__name__}): {exc}") from exc
        data = r.json()
        # qwen3 gibi thinking-mode modeller response bos birakilip thinking'e yazar
        response = data.get("response", "").strip()
        if not response:
            response = data.get("thinking", "").strip()
        return response
