"""Embedding generation.

Primary: a local Ollama embedding model (e.g. ``nomic-embed-text``).
Fallback: a deterministic hash-based embedder so tests and offline development
work without a running model. The fallback is NOT semantically meaningful — it
exists only so the pipeline is runnable end-to-end without Ollama.
"""

from __future__ import annotations

import hashlib
import logging
import struct

from app.config import get_settings

logger = logging.getLogger(__name__)

_FAKE_DIM = 256


class EmbeddingService:
    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        allow_fake: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.embed_model
        self.host = (host or settings.ollama_host).rstrip("/")
        self.allow_fake = settings.allow_fake_embeddings if allow_fake is None else allow_fake
        self._mode: str | None = None  # "ollama" | "fake"

    # --- public API -------------------------------------------------------
    _BATCH_SIZE = 64  # Ollama'ya gönderilecek maksimum chunk sayısı (bellek/zaman sınırı)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._mode is None:
            self._mode = "ollama" if self._ollama_available() else self._fallback_mode()
        if self._mode == "ollama":
            return self._embed_in_batches(texts)
        return [self._embed_fake(t) for t in texts]

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Büyük listeleri _BATCH_SIZE'lık alt gruplara bölerek toplu embed eder."""
        results: list[list[float]] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            try:
                results.extend(self._embed_ollama_batch(batch))
            except Exception:
                logger.debug(
                    "Toplu embedding basarisiz (%d metin), tekli moda geciliyor", len(batch)
                )
                results.extend(self._embed_ollama_single(t) for t in batch)
        return results

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def mode(self) -> str:
        if self._mode is None:
            self._mode = "ollama" if self._ollama_available() else self._fallback_mode()
        return self._mode

    # --- ollama -----------------------------------------------------------
    def _ollama_available(self) -> bool:
        try:
            import requests

            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _embed_ollama_batch(self, texts: list[str]) -> list[list[float]]:
        """Ollama /api/embed ile tek istekte toplu embedding (Ollama >= 0.1.26)."""
        import requests

        r = requests.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def _embed_ollama_single(self, text: str) -> list[float]:
        """Eski Ollama sürümleri için tekli yedek endpoint."""
        import requests

        r = requests.post(
            f"{self.host}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    # --- fallback ---------------------------------------------------------
    def _fallback_mode(self) -> str:
        if not self.allow_fake:
            raise RuntimeError(
                "Ollama'ya ulaşılamadı ve ACHILLES_ALLOW_FAKE_EMBEDDINGS=false. "
                "Ollama'yı başlatın (ollama serve) veya fake embedding'e izin verin."
            )
        logger.warning(
            "Ollama bulunamadı; deterministik FAKE embedding kullanılıyor "
            "(yalnızca geliştirme/test içindir, semantik değildir)."
        )
        return "fake"

    @staticmethod
    def _embed_fake(text: str, dim: int = _FAKE_DIM) -> list[float]:
        vec: list[float] = []
        seed = text.encode("utf-8")
        counter = 0
        while len(vec) < dim:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
            for i in range(0, len(digest), 4):
                (val,) = struct.unpack("<I", digest[i : i + 4])
                vec.append((val / 2**32) * 2 - 1)  # in [-1, 1]
                if len(vec) >= dim:
                    break
            counter += 1
        # L2 normalize
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
