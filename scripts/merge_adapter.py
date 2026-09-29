"""LoRA adapter'ı base modelle birleştir (merge_and_unload) — GGUF/Ollama dönüşümü için.

Neden: Ollama 0.34.x çalışma-anı ``ADAPTER`` desteğini kaldırdı; adapter'ı Ollama'da
kullanmanın yolu base ile birleştirip GGUF'a çevirmektir (bkz. HANDOFF 2026-09-28 (3)).
HF önbelleğindeki base DEĞİŞMEZ; birleşik model AYRI klasöre yazılır.

Doğrulama (Kural 2): birleştirmeden ÖNCE (PEFT) ve SONRA (birleşik) aynı kısa girdide
logit farkı ölçülür; eşik aşılırsa çıkış kodu 2 (dönüşüme geçilmez).

Kullanım:
    uv run --no-sync python scripts/merge_adapter.py hektor_lora_v12_30b \
        --out models/merged/hektor_lora_v12_30b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Doğrulama ölçütleri (mutlak logit eşiği DEĞİL): bf16 birleştirme ağırlıkları bf16'ya
# yuvarlar; 30B'de logitler ~20-30 mertebesinde olduğundan mutlak fark (ölçülen 0.42) tek
# başına anlamsızdır — ilk sürümdeki 0.25 eşiği tahmindi ve yanlış alarm verdi (2026-09-29).
# Esas soru: birleştirme hatası, ADAPTER'IN KENDİ ETKİSİNE (PEFT − base) göre küçük mü,
# dağılım korunuyor mu? Sonraki Q4_K_M nicemlemesi bundan çok daha büyük sapma getirir.
MAX_MERGE_TO_EFFECT = 0.5  # birleştirme |fark|max ≤ 0.5 × adapter etkisi |fark|max
MAX_KL = 0.01  # KL(PEFT ‖ birleşik), nat
MIN_TOP10_OVERLAP = 8  # ilk-10 token kümesinin örtüşmesi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter", help="models/adapters/<ad> ya da tam yol")
    ap.add_argument("--out", required=True, help="Birleşik modelin yazılacağı klasör")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        adapter_dir = Path("models/adapters") / args.adapter
    cfg_path = adapter_dir / "adapter_config.json"
    if not (cfg_path.is_file() and (adapter_dir / "adapter_model.safetensors").is_file()):
        print(f"HATA: adapter eksik ({adapter_dir}) — adapter_config/safetensors yok.")
        return 1
    base = json.loads(cfg_path.read_text(encoding="utf-8"))["base_model_name_or_path"]
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"HATA: çıktı klasörü dolu: {out} (üzerine yazılmaz).")
        return 1
    print(f"base={base}\nadapter={adapter_dir}\nout={out}", flush=True)

    tok = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    probe = tok.apply_chat_template(
        [{"role": "user", "content": "Bir backtest'te komisyon neden hesaba katılmalı?"}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    ids = probe["input_ids"] if isinstance(probe, dict) or hasattr(probe, "keys") else probe
    with torch.no_grad():
        before = model(input_ids=ids).logits[0, -1].float()
        with model.disable_adapter():
            base_logits = model(input_ids=ids).logits[0, -1].float()
    merged = model.merge_and_unload()
    with torch.no_grad():
        after = merged(input_ids=ids).logits[0, -1].float()
    diff = float((before - after).abs().max())
    effect = float((before - base_logits).abs().max())
    kl = float(
        torch.nn.functional.kl_div(
            after.log_softmax(-1), before.log_softmax(-1), log_target=True, reduction="sum"
        )
    )
    top10 = len(set(before.topk(10).indices.tolist()) & set(after.topk(10).indices.tolist()))
    same_top = int(before.argmax()) == int(after.argmax())
    print(
        f"birleştirme |fark|max={diff:.4g} · adapter etkisi |fark|max={effect:.4g} "
        f"(oran {diff / max(effect, 1e-9):.3f}) · KL={kl:.3g} · top10 örtüşme={top10}/10 · "
        f"en olası token aynı: {same_top}",
        flush=True,
    )
    ok = (
        same_top
        and top10 >= MIN_TOP10_OVERLAP
        and kl <= MAX_KL
        and diff <= MAX_MERGE_TO_EFFECT * effect
    )
    if not ok:
        print("HATA: birleşik model PEFT çıktısından anlamlı sapıyor — dönüşüme geçilmez.")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    tmpl = adapter_dir / "chat_template.jinja"
    if tmpl.is_file() and not (out / "chat_template.jinja").exists():
        shutil.copy2(tmpl, out / "chat_template.jinja")
    (out / "merge_info.json").write_text(
        json.dumps(
            {
                "base": base,
                "adapter": str(adapter_dir),
                "merge_logit_max_diff": diff,
                "adapter_effect_logit_max_diff": effect,
                "kl_peft_vs_merged": kl,
                "top10_overlap": top10,
                "top_same": same_top,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"TAMAM: birleşik model yazıldı → {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
