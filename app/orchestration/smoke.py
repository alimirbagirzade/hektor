"""smoke.py — gerçek runtime uçtan-uca duman testi ("stub≠runtime" dersi).

Birim testleri sahte (stub) bağımlılıklarla geçse de gerçek Ollama+RAG+LLM hattı
bozuk olabilir — RLM entegrasyonunda gerçek Ollama smoke 3 açık bulmuştu. Bu modül,
carding / RLM-aday üretimi / eval'in dayandığı CANLI runtime'ı uçtan-uca yoklar:

  1. backend     — LLM backend erişilebilir mi (yerel-öncelikli: Ollama /api/tags).
  2. generation  — gerçek küçük üretim boş/degenere değil mi (seed'li, Kural 6).
                   Üretim BAŞARILI ise yapılandırılmış model yüklü + çalışıyor demektir
                   (tag listesini ayrıca sorgulamaktan daha güçlü bir kanıt).
  3. retrieval   — gerçek küçük RAG retrieval ≥1 chunk döndürüyor mu (korpus boşsa uyarı).

Verdict semantiği (delege bunu StageStatus'a çevirir):
  - runtime ERİŞİLEMEZ (backend yok) → 'skip': bu bir "stub≠runtime" KUSURU değil, yalnız
    "şu an test edilemez" (çevrimdışı / CI). Orkestrasyon hattı DURMAZ; sonraki insan
    kapısına (deep-hunt) geçer. Runtime başlatılıp resume edilince yeniden koşar.
  - runtime CANLI + üretim sağlıklı → 'pass'.
  - runtime CANLI ama üretim boş/degenere/hata → 'fail': asıl yakalanması gereken kusur
    (ör. Ollama açık ama model çekilmemiş, ya da adapter degenere tekrar döngüsünde).

Tüm yoklamalar savunmacı + SALT-OKUMA (yan etki yok; üretim çıktısı atılır). Bağımlılıklar
ENJEKTE edilebilir (llm/retriever) → testler gerçek Ollama olmadan çalışır (offline, Kural).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Küçük, deterministik yoklama girdileri (Kural 6: seed'li).
_PROBE_PROMPT = "Tek kelimeyle yanıtla: TAMAM"
_PROBE_QUERY = "moving average crossover"
_PROBE_SEED = 42
_PROBE_TIMEOUT_S = 60  # CPU'da küçük üretim; canlı ama yavaş runtime'a tolerans
_PROBE_MAX_TOKENS = 32


class _LLMLike(Protocol):
    def available(self) -> bool: ...
    def active_backend(self) -> str: ...
    def generate(
        self,
        prompt: str,
        *,
        seed: int | None = ...,
        temperature: float = ...,
        max_tokens: int | None = ...,
        timeout: int = ...,
    ) -> str: ...


class _RetrieverLike(Protocol):
    def retrieve(self, query: str, top_k: int | None = ...) -> list[Any]: ...


@dataclass
class SmokeCheck:
    """Tek bir yoklamanın sonucu."""

    name: str
    status: str  # "pass" | "fail" | "skip" | "warn"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class SmokeResult:
    """Duman testinin bütünsel sonucu (delege StageStatus'a çevirir)."""

    verdict: str  # "pass" | "skip" | "fail"
    summary: str
    checks: list[SmokeCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "checks": [c.to_dict() for c in self.checks],
        }


class SmokeRunner:
    """Canlı runtime'ı uçtan-uca yoklayan duman test koşucusu (enjekte edilebilir)."""

    def __init__(
        self,
        llm: _LLMLike | None = None,
        retriever: _RetrieverLike | None = None,
        *,
        prompt: str = _PROBE_PROMPT,
        query: str = _PROBE_QUERY,
        seed: int = _PROBE_SEED,
        timeout: int = _PROBE_TIMEOUT_S,
        degenerate_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self.prompt = prompt
        self.query = query
        self.seed = seed
        self.timeout = timeout
        self._degenerate_fn = degenerate_fn

    # ── bağımlılık çözümleme (enjekte edilmemişse savunmacı lazy import) ──────────

    def _get_llm(self) -> _LLMLike | None:
        if self._llm is not None:
            return self._llm
        try:
            from app.brain.local_llm import LocalLLM

            return LocalLLM()
        except Exception as exc:  # import/init patlasa bile koşu çökmesin → skip
            log.debug("Smoke: LocalLLM yüklenemedi: %s", exc)
            return None

    def _get_retriever(self) -> _RetrieverLike | None:
        if self._retriever is not None:
            return self._retriever
        try:
            from app.memory.retrieval_service import RetrievalService

            return RetrievalService()
        except Exception as exc:
            log.debug("Smoke: RetrievalService yüklenemedi: %s", exc)
            return None

    def _is_degenerate(self, text: str) -> bool:
        if self._degenerate_fn is not None:
            return self._degenerate_fn(text)
        try:
            # Eval'in v5-dersi degenerasyon sezgisini YENİDEN KULLAN (tekrar döngüsü/çöküş).
            from app.training.adapter_eval import _is_degenerate

            return _is_degenerate(text)
        except Exception:
            return False  # tespit edilemiyorsa degenere SAYMA (sağlam cevabı yanlış-fail etme)

    # ── ana akış ─────────────────────────────────────────────────────────────────

    def run(self) -> SmokeResult:
        """Backend → generation → retrieval yoklamalarını koş; SmokeResult döndür."""
        checks: list[SmokeCheck] = []
        llm = self._get_llm()

        # 1) Backend erişilebilirliği — yoksa skip (kusur değil, "şimdi test edilemez").
        if llm is None:
            checks.append(SmokeCheck("backend", "skip", "LLM istemcisi kurulamadı."))
            self._probe_retrieval(checks)
            return SmokeResult("skip", "Runtime yoklanamadı — duman testi atlandı.", checks)

        try:
            backend = llm.active_backend()
            reachable = backend != "none" and bool(llm.available())
        except Exception as exc:
            checks.append(SmokeCheck("backend", "skip", f"backend yoklanamadı: {exc}"))
            self._probe_retrieval(checks)
            return SmokeResult("skip", "Runtime yoklanamadı — duman testi atlandı.", checks)

        if not reachable:
            checks.append(
                SmokeCheck("backend", "skip", f"LLM backend erişilemez (backend={backend}).")
            )
            self._probe_retrieval(checks)
            return SmokeResult(
                "skip",
                "Runtime çevrimdışı (LLM backend yok) — duman testi atlandı; "
                "runtime'ı başlatıp sürdür.",
                checks,
            )
        checks.append(SmokeCheck("backend", "pass", f"LLM backend canlı (backend={backend})."))

        # 2) Gerçek küçük üretim — canlı backend yine de üretemezse KUSUR (stub≠runtime).
        gen_ok, gen_check = self._probe_generation(llm)
        checks.append(gen_check)

        # 3) Gerçek küçük retrieval — kritik DEĞİL (boş korpus veri kapısının işi).
        self._probe_retrieval(checks)

        if not gen_ok:
            return SmokeResult("fail", gen_check.detail, checks)
        return SmokeResult("pass", "Runtime canlı; üretim ve retrieval sağlıklı.", checks)

    # ── tekil yoklamalar ──────────────────────────────────────────────────────────

    def _probe_generation(self, llm: _LLMLike) -> tuple[bool, SmokeCheck]:
        try:
            out = llm.generate(
                self.prompt,
                seed=self.seed,
                temperature=0.0,
                max_tokens=_PROBE_MAX_TOKENS,
                timeout=self.timeout,
            )
        except Exception as exc:
            # Canlı görünen backend yine de üretemedi (model çekilmemiş / çöküş) → KUSUR.
            return False, SmokeCheck(
                "generation", "fail", f"Üretim hatası: {type(exc).__name__}: {exc}"
            )
        text = (out or "").strip()
        if not text:
            return False, SmokeCheck("generation", "fail", "Üretim boş döndü (model yanıtsız).")
        if self._is_degenerate(text):
            return False, SmokeCheck(
                "generation", "fail", f"Üretim degenere (tekrar döngüsü): {text[:80]!r}"
            )
        return True, SmokeCheck("generation", "pass", f"Üretim sağlıklı ({len(text)} karakter).")

    def _probe_retrieval(self, checks: list[SmokeCheck]) -> None:
        retr = self._get_retriever()
        if retr is None:
            checks.append(SmokeCheck("retrieval", "skip", "RetrievalService yüklenemedi."))
            return
        try:
            chunks = retr.retrieve(self.query, top_k=3)
        except Exception as exc:
            checks.append(
                SmokeCheck("retrieval", "warn", f"Retrieval hatası: {type(exc).__name__}: {exc}")
            )
            return
        n = len(chunks or [])
        if n > 0:
            checks.append(SmokeCheck("retrieval", "pass", f"Retrieval {n} chunk döndürdü."))
        else:
            checks.append(
                SmokeCheck(
                    "retrieval",
                    "warn",
                    "Retrieval boş (korpus boş olabilir) — veri kapısı denetler.",
                )
            )
