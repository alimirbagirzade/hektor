"""v15 aktarımında sayısal sınırları ölç; model kaydetmez, kabul kararı vermez."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.merge_adapter import PROBES, distribution_metrics, gate_failures  # noqa: E402


def main() -> None:
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import Linear as LoraLinear
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = ROOT / "models/adapters/hektor_lora_v15_localrev_pilot"
    report = ROOT / "reports/v15/repair_r1/precision_diagnosis.json"
    if report.exists():
        raise FileExistsError(report)
    config = json.loads((adapter / "adapter_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            config["base_model_name_or_path"], dtype=torch.bfloat16
        ),
        str(adapter),
    ).eval()
    inputs = []
    for messages in PROBES:
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        inputs.append(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)

    def logits():
        with torch.no_grad():
            return [model(input_ids=ids, use_cache=False).logits[0].float() for ids in inputs]

    canonical = logits()
    with model.disable_adapter():
        canonical_base = logits()
    targets = []

    def enter(_module, args):
        return (args[0].float(), *args[1:])

    def leave(_module, _args, output):
        return output.to(torch.bfloat16)

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            targets.append(name.removeprefix("base_model.model."))
            module.base_layer.float()
            module.lora_A.float()
            module.lora_B.float()
            handles += [
                module.register_forward_pre_hook(enter),
                module.register_forward_hook(leave),
            ]
    island = logits()
    with model.disable_adapter():
        island_base = logits()
    for handle in handles:
        handle.remove()
    merged = model.merge_and_unload()
    for name in targets:
        module = merged.get_submodule(name)
        assert module.weight.dtype == torch.float32
        module.register_forward_pre_hook(enter)
        module.register_forward_hook(leave)
    with torch.no_grad():
        after = [merged(input_ids=ids, use_cache=False).logits[0].float() for ids in inputs]
    drift = [
        distribution_metrics(a, b, c)
        for a, b, c in zip(canonical, island, canonical_base, strict=True)
    ]
    parity = [
        distribution_metrics(a, b, c) for a, b, c in zip(island, after, island_base, strict=True)
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "diagnostic_only": True,
                "model_saved": False,
                "target_count": len(targets),
                "targets": targets,
                "canonical_to_island": drift,
                "canonical_to_island_failures": gate_failures(drift),
                "island_to_merged": parity,
                "island_to_merged_failures": gate_failures(parity),
                "threshold": 0.01,
                "scope": "all_prompt_positions",
                "routing_causation": "not_measured",
                "export_reload": "not_tested",
            },
            stream,
            indent=2,
        )
    print(report, flush=True)
    print(json.dumps({"drift": drift, "parity": parity}), flush=True)


if __name__ == "__main__":
    main()
