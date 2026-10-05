"""Kesilebilir yerel Ollama plan incelemesi; araç çağrısı ve eğitim yetkisi yoktur."""

from __future__ import annotations

import sys

from app.brain.local_llm import LocalLLM
from app.orchestration.research_package import BASE_MODELS


def main() -> None:
    model, seed, prompt = sys.argv[1:]
    if model not in BASE_MODELS:
        raise ValueError("İnceleyici yalnız izin verilen temel modeli kullanabilir.")
    print(LocalLLM(model=model).generate(prompt, temperature=0, max_tokens=700, seed=int(seed)))


if __name__ == "__main__":
    main()
