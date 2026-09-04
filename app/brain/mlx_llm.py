"""MLX-LM tabanlı çıkarım motoru (Apple Silicon, LoRA adapter desteği).

Ollama ile aynı arayüzü paylaşır ama modeli MLX formatında çalıştırır.
Bu sayede eğitilmiş adapter'ı doğrudan inference'da kullanmak mümkün olur.

Her çağrıda subprocess açılır — yüksek eş zamanlılık için tasarlanmamıştır,
araştırma ve karşılaştırma için uygundur.

8GB kısıtı: MLX model belleği (~0.8 GB 1.5B-4bit) + adapter yükü küçük;
Ollama aynı anda çalışıyorsa toplam ~2 GB — kabul edilebilir.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class MlxLLMUnavailable(RuntimeError):
    pass


class MlxLLM:
    """MLX-LM generate ile LoRA adapter destekli çıkarım."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str | Path | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.max_tokens = max_tokens

    def available(self) -> bool:
        """mlx_lm modülü kurulu mu?"""
        try:
            import importlib.util

            return importlib.util.find_spec("mlx_lm") is not None
        except Exception:
            return False

    def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Prompt'a yanıt üret. Hata durumunda MlxLLMUnavailable fırlatır."""
        if not self.available():
            raise MlxLLMUnavailable("mlx_lm kurulu değil. Kurmak için: uv sync --extra train")

        cmd = [
            sys.executable,
            "-m",
            "mlx_lm",
            "generate",
            "--model",
            self.base_model,
            "--prompt",
            prompt,
            "--max-tokens",
            str(max_tokens or self.max_tokens),
            "--temp",
            "0.2",
        ]
        if self.adapter_path and self.adapter_path.exists():
            cmd += ["--adapter-path", str(self.adapter_path)]

        logger.debug("MLX generate: %s", " ".join(cmd[:6]) + " …")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MlxLLMUnavailable("MLX generate zaman aşımı (300s).") from exc

        if result.returncode != 0:
            raise MlxLLMUnavailable(
                f"MLX generate başarısız (exit {result.returncode}): {result.stderr[:200]}"
            )

        # mlx_lm generate çıktısı "==========\n<yanıt>\n==========" şeklinde
        output = result.stdout
        sep = "=" * 10
        if sep in output:
            parts = output.split(sep)
            # genellikle 3 parça: prompt echo, yanıt, boş
            if len(parts) >= 2:
                return parts[1].strip()
        return output.strip()
