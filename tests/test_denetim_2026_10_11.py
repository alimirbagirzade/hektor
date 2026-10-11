"""Denetim 2026-10-11 temizlik regresyonları.

- `hektor lora-split` eğitimin kendi KAYNAK-GRUPLU bölücüsünü kullanır (satır karıştırma değil).
- MLX backend'i `--profile`'ı sessizce yok saymaz; onay/kapılardan önce reddeder.
- PEFT `train()` uzunluk kaybı kapısı model YÜKLENMEDEN çalışır (yalnız tokenizer).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from app.main import app
from app.training import detached_launch as dl
from app.training import peft_lora_train as plt


def _row(paper_id: str, i: int, answer: str = "") -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": f"soru {paper_id}-{i}"},
                {"role": "assistant", "content": answer or f"cevap {paper_id}-{i}"},
            ],
            "metadata": {"paper_id": paper_id, "source_id": paper_id},
        },
        ensure_ascii=False,
    )


def test_lora_split_kaynak_gruplu_ve_egitim_bolmesiyle_ayni(tmp_path, monkeypatch) -> None:
    lines = [_row(f"paper_{p:03d}", i) for p in range(30) for i in range(8)]
    src = tmp_path / "lora_sft.jsonl"
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    jd = tmp_path / "jsonl"
    monkeypatch.setattr(
        "app.main.get_settings", lambda: SimpleNamespace(root=tmp_path, jsonl_dir=jd)
    )

    res = CliRunner().invoke(app, ["lora-split", "--source", str(src)])
    assert res.exit_code == 0, res.output

    train = (jd / "train.jsonl").read_text(encoding="utf-8").splitlines()
    valid = (jd / "valid.jsonl").read_text(encoding="utf-8").splitlines()
    exp_train, exp_valid = dl.split_lines_by_source(lines)
    assert (train, valid) == (exp_train, exp_valid)

    def pids(rows: list[str]) -> set[str]:
        return {json.loads(r)["metadata"]["paper_id"] for r in rows}

    assert valid and not pids(train) & pids(valid)


def test_mlx_backend_profili_reddeder() -> None:
    res = CliRunner().invoke(app, ["train", "--backend", "mlx", "--profile", "moe30b_attn_long"])
    assert res.exit_code == 1
    assert "MLX" in res.output and "--backend peft" in res.output


class _ModelYuklendi(RuntimeError):
    """Test nöbetçisi: model yüklemesine ulaşıldı."""


class _SahteTokenizer:
    chat_template = "sahte"
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        n = sum(len(str(m["content"]).split()) + 2 for m in msgs)
        return list(range(n + (1 if add_generation_prompt else 0)))


def _sahte_agir_moduller(monkeypatch) -> None:
    torch = types.ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    torch.bfloat16 = "bf16"  # type: ignore[attr-defined]
    torch.float32 = "fp32"  # type: ignore[attr-defined]
    torch.float16 = "fp16"  # type: ignore[attr-defined]
    peft = types.ModuleType("peft")
    peft.LoraConfig = object  # type: ignore[attr-defined]
    peft.get_peft_model = lambda *a, **k: None  # type: ignore[attr-defined]
    tr = types.ModuleType("transformers")

    def _model_yukle(*a, **k):
        raise _ModelYuklendi("model yüklendi")

    tr.AutoModelForCausalLM = SimpleNamespace(from_pretrained=_model_yukle)  # type: ignore[attr-defined]
    tr.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: _SahteTokenizer())  # type: ignore[attr-defined]
    tr.DataCollatorForLanguageModeling = object  # type: ignore[attr-defined]
    tr.TrainingArguments = object  # type: ignore[attr-defined]
    for name, mod in (("torch", torch), ("peft", peft), ("transformers", tr)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(plt, "_check_deps", lambda: [])
    monkeypatch.setattr(plt, "check_cpu_ram", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(plt, "precheck_target_modules", lambda *a, **k: None)


def test_uzunluk_kaybi_model_yuklenmeden_reddedilir(tmp_path: Path, monkeypatch) -> None:
    _sahte_agir_moduller(monkeypatch)
    uzun = " ".join(["kelime"] * 400)
    tr = tmp_path / "train.jsonl"
    tr.write_text("\n".join(_row("p", i, uzun) for i in range(10)) + "\n", encoding="utf-8")
    cfg = plt.PeftTrainConfig(
        base_model="yok/model",
        train_jsonl=tr,
        valid_jsonl=tmp_path / "valid.jsonl",
        adapter_output_path=tmp_path / "ad",
        assistant_only_loss=True,
        max_seq_length=64,
    )
    out = plt.train(cfg)  # _ModelYuklendi yükselirse kapı model yüklemesinden SONRA demektir
    assert out["ok"] is False and "Uzunluk kaybı" in out["error"]
