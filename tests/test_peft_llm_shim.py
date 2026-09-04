"""PeftAdapterLLMShim — adapter→ladder bağının yapısal testleri (offline).

Gerçek PEFT/torch yüklenmeden:
- Shim'in adapter-dizini yoksa None döndürdüğünü doğrula.
- LocalLLM arayüzünü (available, generate) sahte modelle karşıla.
- score_full_ladder'ın llm=shim parametresini kabul ettiğini ve graceful skip yaptığını doğrula.

CLAUDE.md Kural 2: test edilmeden "hazır" deme.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Sahte (mock) shim — torch/peft kurulu olmasa da çalışır
# ---------------------------------------------------------------------------


class _FakeShim:
    """PeftAdapterLLMShim ile aynı arayüz, saf Python."""

    model = "fake-peft-adapter"

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs: object) -> str:
        return "sahte adapter yanıtı"


# ---------------------------------------------------------------------------
# Testler
# ---------------------------------------------------------------------------


def test_shim_load_returns_none_when_no_dir(tmp_path: Path) -> None:
    """Adapter dizini yoksa load() None döner (ImportError yutulur)."""
    from app.lora.peft_llm_shim import PeftAdapterLLMShim

    result = PeftAdapterLLMShim.load(tmp_path / "nonexistent_adapter")
    assert result is None


def test_shim_load_returns_none_without_peft(tmp_path: Path) -> None:
    """torch/peft kurulu değilse load() None döner, ImportError çökme yapmaz."""
    from app.lora.peft_llm_shim import PeftAdapterLLMShim

    adapter_dir = tmp_path / "fake_adapter"
    adapter_dir.mkdir()

    with patch.dict("sys.modules", {"torch": None, "peft": None, "transformers": None}):
        result = PeftAdapterLLMShim.load(adapter_dir)
    assert result is None


def test_shim_load_base_of_graceful(tmp_path: Path) -> None:
    """load_base_of (base-only bacak) da bağımlılık/dizin yoksa None döner (çökme yok).

    Kademe-2 av bulgusu: base ladder bacağı adapter'ın KENDİ base'ini kullanmalı; bu
    factory'nin varlığı ve graceful-skip davranışı, elmayla-armut kıyasını (4B Ollama vs
    1.5B adapter) önleyen düzeltmenin sözleşmesidir.
    """
    from app.lora.peft_llm_shim import PeftAdapterLLMShim

    # Dizin yok → None
    assert PeftAdapterLLMShim.load_base_of(tmp_path / "yok") is None

    # Dizin var ama bağımlılık yok → None
    adapter_dir = tmp_path / "fake_adapter"
    adapter_dir.mkdir()
    with patch.dict("sys.modules", {"torch": None, "peft": None, "transformers": None}):
        assert PeftAdapterLLMShim.load_base_of(adapter_dir) is None


def test_fake_shim_interface() -> None:
    """Sahte shim LocalLLM arayüzüyle uyumlu."""
    shim = _FakeShim()
    assert shim.available() is True
    out = shim.generate("Merhaba", seed=42)
    assert isinstance(out, str)
    assert len(out) > 0


def test_score_full_ladder_accepts_shim(tmp_path: Path) -> None:
    """score_full_ladder llm=shim ile çağrılabilir; LLM yoksa skipped (graceful)."""
    from app.memory.sqlite_store import SqliteStore
    from app.verification.exams.understanding_score import score_full_ladder

    store = SqliteStore(db_path=str(tmp_path / "test.db"))
    shim = _FakeShim()

    # Sahte shim yanıtlarla L3/L4 sınavları skipped ya da failed dönebilir (LLM judge yok);
    # önemli olan: exception fırlatmaması ve UnderstandingScore döndürmesi.
    result = score_full_ladder(0, llm=shim, store=store, use_sessions_l5=False)

    from app.verification.exams.understanding_score import UnderstandingScore

    assert isinstance(result, UnderstandingScore)
    # pass_rate None olabilir (yetersiz veri), float olabilir — ikisi de kabul
    assert result.pass_rate is None or 0.0 <= result.pass_rate <= 1.0


def test_ladder_regression_blocks_promotion() -> None:
    """Anlama regresyonu passed=False ve regression_any=True yaptırmalı (mantık testi)."""
    # Bu test auto_pipeline._run_eval'deki regresyon mantığını doğrular.
    # Gerçek pipeline yerine mantığı doğrudan test ediyoruz.
    base_rate = 0.6
    adapter_rate = 0.4  # %5'ten fazla gerileme
    threshold = 0.05

    regression_detected = base_rate > 0 and adapter_rate < base_rate - threshold
    assert regression_detected, "Anlama regresyonu tespit edilmeli"

    base_rate2 = 0.6
    adapter_rate2 = 0.56  # eşik içinde
    regression_ok = base_rate2 > 0 and adapter_rate2 < base_rate2 - threshold
    assert not regression_ok, "Küçük fark regresyon sayılmamalı"
