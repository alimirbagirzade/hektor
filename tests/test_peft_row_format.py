"""PEFT trainer satır→mesaj dönüşümü testleri (format uyumsuzluğu regresyonu).

DatasetBuilder {prompt, completion} üretir; PEFT eskiden bunu tanımayıp tüm
örnekleri sessizce atıyordu (boş eğitim verisi → çökme). Bu testler köprüyü korur.
"""

from __future__ import annotations

from app.training.peft_lora_train import _row_to_messages, _row_to_text


def test_prompt_completion_format_supported() -> None:
    row = {"prompt": "Soru?", "completion": "Cevap."}
    assert _row_to_messages(row) == [
        {"role": "user", "content": "Soru?"},
        {"role": "assistant", "content": "Cevap."},
    ]
    assert _row_to_text(row).strip() != ""


def test_messages_format_still_supported() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Yo"},
        ]
    }
    assert _row_to_messages(row) == row["messages"]


def test_text_format_still_supported() -> None:
    assert _row_to_messages({"text": "düz metin"}) == [{"role": "user", "content": "düz metin"}]


def test_unknown_format_returns_empty() -> None:
    assert _row_to_messages({"foo": "bar"}) == []
