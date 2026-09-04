"""PEFT adapter'ını LocalLLM arayüzüne saran basit shim.

Anlama merdiveni (L3/L4 sınavları) LocalLLM.generate() arayüzünü kullanır.
Bu shim, eğitilmiş bir adapter'ı o arayüze sarmalar → auto_pipeline._run_eval
şunu yapabilir: base LLM vs. adapter LLM anlama skoru kıyası.

Kullanım örneği (auto_pipeline içi):
    shim = PeftAdapterLLMShim.load(adapter_dir)
    if shim is not None:
        adapter_score = score_full_ladder(llm=shim, ...)

Bağımlılıklar (torch/transformers/peft) kurulu değilse ``load()`` None döner —
çağıran kod ImportError ile çökmez, atlama yolu açık kalır.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class PeftAdapterLLMShim:
    """Eğitilmiş PEFT adapter'ı LocalLLM.generate() arayüzüyle sar.

    ``auto_pipeline._run_eval`` içinde kullanılır — base model yerine adapter
    üzerinden L3/L4 sınavlarını koşmak için.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self.model = "peft-adapter"  # LocalLLM uyumluluğu için

    # ---------------------------------------------------------------- interface

    def available(self) -> bool:
        return self._model is not None

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 512,
        fmt: str | None = None,
        timeout: int = 600,
        seed: int | None = None,
    ) -> str:
        """Adapter ile metin üret — LocalLLM.generate() imzasıyla uyumlu."""
        import torch

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens or 512,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        # Yalnız üretilen kısmı al (girdi token'larını çıkar)
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    # ---------------------------------------------------------------- factory

    @classmethod
    def load(cls, adapter_dir: Path | str) -> PeftAdapterLLMShim | None:
        """Adapter dizininden base+adapter model yükle; bağımlılık yoksa None döner."""
        return cls._load(adapter_dir, with_adapter=True)

    @classmethod
    def load_base_of(cls, adapter_dir: Path | str) -> PeftAdapterLLMShim | None:
        """Adapter'ın KENDİ base modelini (adapter YÜKLEMEDEN) shim olarak döndür.

        Anlama-merdiveni base bacağı için ZORUNLU: base vs adapter kıyası AYNI model
        ailesi / boyut / runtime ile yapılmalı. Aksi halde base bacağı `llm=None` →
        LocalLLM (Ollama `settings.llm_model`, örn. qwen3:4b) olur ve 4B base, 1.5B+adapter
        ile kıyaslanır (elmayla-armut); L3/L4 sayısal sınavlarda 1.5B sistematik kaybeder →
        kalıcı SAHTE-gerileme → iyi adapter bile terfi bloklanır (Kademe-2 av bulgusu).
        Bu factory ile base bacağı adapter'ın 1.5B base'ini adapter'sız koşar → elmayla-elma.
        """
        return cls._load(adapter_dir, with_adapter=False)

    @classmethod
    def _load(cls, adapter_dir: Path | str, *, with_adapter: bool) -> PeftAdapterLLMShim | None:
        """Ortak yükleyici: base modeli çöz+yükle, `with_adapter` ise PEFT adapter'ı da bindir."""
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            log.debug("PeftAdapterLLMShim: bağımlılık eksik → atlıyorum (%s)", exc)
            return None

        adapter_dir = Path(adapter_dir)
        if not adapter_dir.exists():
            log.warning("PeftAdapterLLMShim: adapter dizini yok: %s", adapter_dir)
            return None

        try:
            from app.config import get_settings
            from app.training.adapter_eval import _resolve_base_model

            settings = get_settings()
            # Adapter'ın KENDİ base'ini adapter_config.json'dan çöz; settings.peft_base_model
            # yalnız fallback. Aksi halde küçük-model adapter (1.5B) settings'teki 4B base'e
            # yüklenmeye çalışıp boyut-uyuşmazlığıyla patlar → except None döner → auto_pipeline
            # anlama-merdiveni kıyası (v5 regresyon savunması) SESSİZCE atlanır. adapter_eval
            # ile aynı çözüm (tutarlı base çözümleme).
            base_model_id = _resolve_base_model(adapter_dir) or settings.peft_base_model

            what = "base+adapter" if with_adapter else "base-only"
            log.info(
                "PeftAdapterLLMShim(%s): %s üzerine %s yükleniyor…",
                what,
                base_model_id,
                adapter_dir,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            model = PeftModel.from_pretrained(base, str(adapter_dir)) if with_adapter else base
            model.eval()
            return cls(model, tokenizer)
        except Exception as exc:
            log.warning("PeftAdapterLLMShim: yükleme başarısız → %s: %s", type(exc).__name__, exc)
            return None
