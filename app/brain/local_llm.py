"""LLM istemcisi — YALNIZ yerel Ollama.

Bu proje **lokal-öncelikliktir**: çalışma-zamanı LLM hattı tamamen yereldir.
Pay-per-token bulut API'si (OpenAI / Anthropic / Google) KULLANILMAZ ve istemci
kodu bilinçli olarak yoktur — kalıcı proje kısıtı. Geliştirme yardımı aylık
abonelikli CLI araçlarıyla (Claude Code / Codex) yapılır, API anahtarıyla değil.

Ollama erişilemezse ``LLMUnavailable`` yükselir; çağıranlar bunu ele alır
(cevap üretmek yerine kaynakları göstermek / sınavı `skipped` saymak gibi).

Düşünme (thinking) davranışı — 2026-09-13 ölçümü (Ollama 0.34.0):
``qwen3:4b`` etiketi manifest özeti olarak ``qwen3:4b-thinking-2507-q4_K_M`` ile
AYNIDIR; yani hibrit değil, yalnız-düşünen Qwen3-4B-Thinking-2507'dir. Şablonu
asistan turunu koşulsuz ``<think>`` ile açar ve ``.Think`` dalı yoktur; bu yüzden
``think: false``, ``/no_think``, ``/api/chat`` ve boş ``<think></think>`` ön-dolgusu
ETKİSİZDİR — model düşünmeyi etiketsiz düz metin olarak ``response``'a yazar.
Bu istemci modeli ``/api/show`` ile sınıflandırır:

- ``none``   : düşünme yeteneği yok → ``think`` alanı hiç gönderilmez.
- ``toggle`` : düşünme kapatılabilir → ``think: false``.
- ``forced`` : kapatılamaz → serbest metinde ``think: true`` (Ollama düşünmeyi ayrı
  ``thinking`` alanına ayırır, cevap temiz kalır) + ek düşünme bütçesi. ``format``
  verilen çağrıda ``think: false`` kalır: JSON grameri ilk token'dan kısıtlar, düşünme
  metni sızamaz (``think: true`` + format ise JSON'u ``thinking``'e yazıp cevabı boşaltır).

``thinking`` alanı ASLA cevap olarak döndürülmez. Hızlı, düşünmesiz yol için
``HEKTOR_LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M`` önerilir.
"""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar, Literal

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

ThinkMode = Literal["none", "toggle", "forced", "unknown"]

# `.Think` (bool) şablonda düşünmeyi aç/kapa dalı demektir; `.Thinking` (içerik) ve
# `.IsThinkSet` tek başına kapatma sağlamaz. `\b` sayesinde `.Thinking` eşleşmez.
_THINK_TOGGLE_RE = re.compile(r"\.Think\b")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMUnavailable(RuntimeError):
    pass


def classify_think_support(show: dict[str, Any]) -> ThinkMode:
    """``/api/show`` çıktısından modelin düşünme davranışını sınıflandır."""
    capabilities = show.get("capabilities") or []
    if "thinking" not in capabilities:
        return "none"
    template = str(show.get("template") or "")
    if _THINK_TOGGLE_RE.search(template):
        return "toggle"
    # Yalnız-düşünen model üretim prompt'unu `<think>` ile AÇIK bırakır: şablondaki son
    # `<think>`'ten sonra `</think>` gelmez. Düşünmesiz modeller (ör. Instruct-2507) de
    # `thinking` yeteneği ilan edip geçmiş turlar için kapalı `<think>…</think>` taşıyabilir
    # — onlar `forced` DEĞİLDİR (2026-09-13: prompt token'larında `<think>` yok, ölçüldü).
    # Şablonda `<think>` yoksa (yerleşik renderer/parser) kapatmayı Ollama yönetir.
    last_open = template.rfind("<think>")
    if last_open != -1 and "</think>" not in template[last_open:]:
        return "forced"
    return "toggle"


def strip_thinking(text: str) -> str:
    """Cevaptan ``<think>`` bloklarını temizle.

    - Tam ``<think>…</think>`` blokları silinir.
    - Açılışı şablonda kalmış (yalnız ``</think>`` görünen) blokta son kapanıştan
      sonrası alınır.
    - Kapanmamış ``<think>`` → cevap hiç başlamamıştır; öncesi alınır.

    Etiketsiz düz-metin düşünme güvenilir ayırt edilemez; o durum ``think`` kipi
    doğru seçilerek (``classify_think_support``) önlenir.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


class LocalLLM:
    # Aynı model için "düşünme kapatılamıyor" uyarısını süreç başına bir kez logla.
    _warned_forced: ClassVar[set[str]] = set()

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        s = get_settings()
        self._ollama_host = (host or s.ollama_host).rstrip("/")
        self._ollama_model = model or s.llm_model
        self.model = self._ollama_model  # backward compat
        self._ollama_keep_alive = s.ollama_keep_alive  # RAM darsa "0" → sorgu sonrası boşalt
        self._default_max_tokens = max(1, int(s.llm_default_max_tokens))
        self._max_tokens_cap = max(1, int(s.llm_max_tokens_cap))
        self._thinking_extra_tokens = max(0, int(s.llm_thinking_extra_tokens))
        # Çevrimdışı testler için sahte HTTP taşıyıcı (httpx.MockTransport) enjekte edilebilir.
        self._transport = transport
        self._think_mode: ThinkMode | None = None

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(timeout=timeout, transport=self._transport)

    # ---------------------------------------------------------------- probes

    def _ollama_alive(self) -> bool:
        try:
            with self._client(5.0) as client:
                r = client.get(f"{self._ollama_host}/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    def available(self) -> bool:
        return self._ollama_alive()

    def active_backend(self) -> str:
        """'ollama' (canlı) veya 'none'. Başka backend yoktur."""
        return "ollama" if self._ollama_alive() else "none"

    def think_mode(self) -> ThinkMode:
        """Modelin düşünme kipi (``/api/show`` ile bir kez sorulur, önbelleğe alınır).

        Sorgu başarısızsa ``unknown`` döner ve önbelleğe ALINMAZ (sonraki çağrı
        yeniden dener); bu durumda eski davranış (``think: false``) uygulanır.
        """
        if self._think_mode is not None:
            return self._think_mode
        try:
            with self._client(10.0) as client:
                r = client.post(f"{self._ollama_host}/api/show", json={"model": self._ollama_model})
            r.raise_for_status()
            mode = classify_think_support(r.json())
        except Exception as exc:  # eski Ollama / geçici hata — kararı sonraki çağrıya bırak
            logger.debug("Ollama /api/show okunamadı (%s): %s", self._ollama_model, exc)
            return "unknown"
        self._think_mode = mode
        if mode == "forced" and self._ollama_model not in LocalLLM._warned_forced:
            LocalLLM._warned_forced.add(self._ollama_model)
            logger.warning(
                "'%s' modelinde düşünme KAPATILAMIYOR (şablon <think> açıyor, .Think dalı yok). "
                "Serbest metin çağrıları think=true + %d ek token bütçesiyle yavaş çalışır. "
                "Hızlı yol: HEKTOR_LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M "
                "(önce `ollama pull qwen3:4b-instruct-2507-q4_K_M`).",
                self._ollama_model,
                self._thinking_extra_tokens,
            )
        return mode

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

    def _num_predict(self, max_tokens: int | None) -> int:
        """Cevap token bütçesi: verilmezse varsayılan, verilse de tavanı aşamaz.

        Sınırsız üretimde qwen3:4b "2+2" sorusunu 240 sn'de bitiremedi (2026-09-13).
        """
        budget = max_tokens if max_tokens and max_tokens > 0 else self._default_max_tokens
        return min(budget, self._max_tokens_cap)

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
        num_predict = self._num_predict(max_tokens)
        mode = self.think_mode()
        think: bool | None
        if mode == "none":
            think = None  # düşünmesiz modele `think` alanı gönderilmez
        elif mode == "forced" and not fmt:
            # Kapatılamayan düşünmeyi ayrı alana yönlendir; düşünme cevaptan ÖNCE
            # harcandığı için bütçeye ek pay eklenir (yoksa cevap hiç üretilmez).
            think = True
            num_predict += self._thinking_extra_tokens
        else:
            think = False

        options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": self._ollama_model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "keep_alive": self._ollama_keep_alive,
            "options": options,
        }
        if think is not None:
            payload["think"] = think
        if fmt:
            payload["format"] = fmt
        try:
            with self._client(timeout) as client:
                r = client.post(f"{self._ollama_host}/api/generate", json=payload)
            r.raise_for_status()
        except httpx.HTTPError as exc:
            # Zaman aşımı / bağlantı / HTTP hatası → yerel model bu çağrıda yanıt
            # üretemedi. Ham httpx hatasını, çağıranların ZATEN ele aldığı
            # LLMUnavailable'a çeviriyoruz; aksi halde (ör. yavaş CPU'da
            # ReadTimeout) sınav harness'i ve /api/understanding-score 500 olur.
            raise LLMUnavailable(f"Ollama yanıt vermedi ({type(exc).__name__}): {exc}") from exc
        data = r.json()
        raw = str(data.get("response") or "")
        # JSON grameri etiket üretimini zaten engeller; JSON içeriğine dokunmayız.
        response = raw.strip() if fmt else strip_thinking(raw)
        if not response and data.get("done_reason") == "length":
            # Eskiden `thinking` alanı cevap diye döndürülüyordu → düşünme metni cevaba
            # sızıyordu. Bütçe cevaba geçmeden bittiyse bu çağrıda cevap YOKTUR.
            raise LLMUnavailable(
                f"Model cevaba geçmeden num_predict={num_predict} bütçesini tüketti "
                f"(think={think}). Daha büyük max_tokens verin veya düşünmesiz model seçin."
            )
        return response
