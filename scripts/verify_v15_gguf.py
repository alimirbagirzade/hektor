"""HF→bf16 GGUF attention bayt eşliği ve Q8 attention tür/kayıp kanıtı."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "C:/HP/tools/llama.cpp/gguf-py")


def main():
    import numpy as np
    import torch
    from gguf import GGMLQuantizationType, GGUFReader
    from gguf.quants import dequantize
    from safetensors import safe_open

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--quant", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    full = {t.name: t for t in GGUFReader(str(args.bf16)).tensors}
    quant = {t.name: t for t in GGUFReader(str(args.quant)).tensors}
    records = []
    mapping = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output"}
    for layer in range(48):
        for source, target in mapping.items():
            key = f"model.layers.{layer}.self_attn.{source}.weight"
            gguf_key = f"blk.{layer}.{target}.weight"
            with safe_open(str(args.model / index["weight_map"][key]), framework="pt") as stream:
                tensor = stream.get_tensor(key)
            original_bytes = tensor.view(torch.uint8).numpy().reshape(-1)
            exact = full[gguf_key].tensor_type == GGMLQuantizationType.BF16 and np.array_equal(
                original_bytes, full[gguf_key].data.reshape(-1)
            )
            encoded = quant[gguf_key]
            if encoded.tensor_type != GGMLQuantizationType.Q8_0:
                raise ValueError(f"{gguf_key}: Q8_0 attention bekleniyor")
            restored = dequantize(encoded.data, encoded.tensor_type).reshape(tensor.shape)
            error = restored - tensor.float().numpy()
            records.append(
                {
                    "tensor": gguf_key,
                    "bf16_bytes_exact": bool(exact),
                    "quant_type": encoded.tensor_type.name,
                    "quant_max_absolute_error": float(np.max(np.abs(error))),
                    "quant_rmse": float(np.sqrt(np.mean(error**2))),
                }
            )
    result = {
        "passed_structural_conversion": all(r["bf16_bytes_exact"] for r in records),
        "adapted_tensor_count": len(records),
        "scope": "192 attention projection tensors only; not full model logits",
        "quantization_equivalence_claim": False,
        "records": records,
    }
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return 0 if result["passed_structural_conversion"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
