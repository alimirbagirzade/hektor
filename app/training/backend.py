"""Platform tespiti: hangi LoRA backend kullanılacak."""

from __future__ import annotations

import platform
import sys


def detect_lora_backend() -> str:
    """Apple Silicon → 'mlx', diğer her şey → 'peft'."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return "peft"
