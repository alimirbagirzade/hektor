"""assistant_only_loss maskeleme birim testleri (çevrimdışı, model/ağ gerektirmez).

Degenerate-tekrar onarımının çekirdeği: prompt token'ları -100 ile maskelenmeli,
yalnız asistan cevabı loss'a girmeli. Bu test o maskelemeyi saf seviyede doğrular.
"""

from __future__ import annotations

import pytest

from app.training.peft_lora_train import build_masked_labels, sample_rows


def test_normal_masking_prompt_minus_100() -> None:
    """prompt token'ları -100, asistan token'ları korunur."""
    prompt = [1, 2, 3]
    full = [1, 2, 3, 10, 11, 12]
    labels = build_masked_labels(prompt, full)
    assert labels == [-100, -100, -100, 10, 11, 12]
    # En az bir öğrenilebilir token olmalı (boş-loss guard).
    assert any(label != -100 for label in labels)


def test_labels_length_matches_full() -> None:
    """labels uzunluğu input (full) ile birebir eşleşmeli — collator için şart."""
    prompt = [5, 6]
    full = [5, 6, 7, 8]
    labels = build_masked_labels(prompt, full)
    assert labels is not None
    assert len(labels) == len(full)


def test_prefix_mismatch_returns_none() -> None:
    """full prompt ile başlamıyorsa maskeleme güvenli değil → None (örnek atılır)."""
    assert build_masked_labels([1, 2, 3], [9, 2, 3, 4]) is None


def test_empty_prompt_returns_none() -> None:
    assert build_masked_labels([], [1, 2, 3]) is None


def test_prompt_covers_whole_full_returns_none() -> None:
    """Öğrenilecek asistan token'ı kalmıyorsa None (sıfır-loss önlenir)."""
    assert build_masked_labels([1, 2, 3], [1, 2, 3]) is None
    assert build_masked_labels([1, 2, 3, 4], [1, 2, 3]) is None


def test_sample_rows_caps_to_max_examples() -> None:
    rows = [{"i": i} for i in range(100)]
    out = sample_rows(rows, max_examples=10, seed=42)
    assert len(out) == 10
    # Hepsi orijinal havuzdan gelmeli (uydurma yok).
    assert all(r in rows for r in out)


def test_sample_rows_deterministic_same_seed() -> None:
    rows = [{"i": i} for i in range(100)]
    assert sample_rows(rows, 10, seed=42) == sample_rows(rows, 10, seed=42)


def test_sample_rows_zero_or_excess_returns_all() -> None:
    rows = [{"i": i} for i in range(5)]
    assert sample_rows(rows, 0, seed=42) == rows  # 0 = hepsi
    assert sample_rows(rows, 999, seed=42) == rows  # yetersiz havuz → hepsi
    assert sample_rows(rows, -1, seed=42) == rows  # negatif güvenli


def test_chat_input_ids_handles_list_and_batchencoding() -> None:
    """apply_chat_template list[int] VEYA BatchEncoding(dict) dönebilir → her ikisi de
    düz token listesine inmeli (yoksa maskeleme Encoding kıyaslar, tüm örnek atılır)."""
    from app.training.peft_lora_train import _chat_input_ids

    class FakeTokList:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return [1, 2, 3]

    class FakeTokDict:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
            return {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]}

    assert _chat_input_ids(FakeTokList(), [], add_generation_prompt=True) == [1, 2, 3]
    assert _chat_input_ids(FakeTokDict(), [], add_generation_prompt=False) == [4, 5, 6]


def test_masked_collator_pads_with_minus_100() -> None:
    """_MaskedDataCollator: kısa örneği sağdan doldurur; labels dolgusu -100 olmalı."""
    torch = pytest.importorskip("torch")
    from app.training.peft_lora_train import _MaskedDataCollator

    collator = _MaskedDataCollator(pad_token_id=0)
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
        {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
    ]
    batch = collator(features)
    assert batch["input_ids"].shape == (2, 3)
    # Kısa örnek pad_token (0) ile dolduruldu; attention 0; labels -100.
    assert batch["input_ids"][1].tolist() == [4, 5, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["labels"][1].tolist() == [-100, 5, -100]
    assert batch["labels"].dtype == torch.long
