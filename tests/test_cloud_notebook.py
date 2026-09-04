"""Stage 2 bulut notebook üretici testleri — şablon doldurma + Modelfile.

Çevrimdışı: yalnız dosya üretir (eğitim/ağ yok). Şablonun geçerli JSON kaldığını,
placeholder'ların tamamen dolduğunu ve değerlerin enjekte edildiğini doğrular.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.training.cloud_notebook import build_stage2_notebook, write_modelfile


def _build(tmp_path: Path) -> Path:
    out = tmp_path / "nb.ipynb"
    return build_stage2_notebook(
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        adapter_name="achilles_test",
        hf_dataset_repo="user/achilles-lora-sft",
        max_seq_length=2048,
        lora_r=16,
        learning_rate=2e-4,
        num_epochs=2,
        out_path=out,
    )


def test_notebook_is_valid_json_nbformat4(tmp_path: Path) -> None:
    out = _build(tmp_path)
    nb = json.loads(out.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10
    assert all("cell_type" in c for c in nb["cells"])


def test_all_placeholders_filled(tmp_path: Path) -> None:
    out = _build(tmp_path)
    text = out.read_text(encoding="utf-8")
    for ph in (
        "{BASE_MODEL}",
        "{ADAPTER_NAME}",
        "{HF_DATASET_REPO}",
        "{MAX_SEQ_LEN}",
        "{LORA_R}",
        "{LEARNING_RATE}",
        "{NUM_EPOCHS}",
    ):
        assert ph not in text, f"doldurulmamış placeholder: {ph}"


def test_injected_values_present(tmp_path: Path) -> None:
    out = _build(tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "Qwen/Qwen3-4B-Instruct-2507" in text  # base model
    assert "achilles_test" in text  # adapter name
    assert "user/achilles-lora-sft" in text  # hf repo


def test_base_model_pinned_to_2507(tmp_path: Path) -> None:
    # Adapter'ın Ollama'da çalışması için base BİREBİR Instruct-2507 olmalı.
    out = _build(tmp_path)
    nb = json.loads(out.read_text(encoding="utf-8"))
    cfg_cell = next(c for c in nb["cells"] if "BASE_MODEL" in c["source"] and "=" in c["source"])
    assert "Qwen3-4B-Instruct-2507" in cfg_cell["source"]


def test_write_modelfile(tmp_path: Path) -> None:
    mf = write_modelfile(tmp_path)
    assert mf.name == "Modelfile"
    content = mf.read_text(encoding="utf-8")
    assert content.startswith("FROM ./achilles-Q4_K_M.gguf")
    assert "<|im_end|>" in content  # stop token şart
