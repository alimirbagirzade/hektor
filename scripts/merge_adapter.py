"""LoRA adapter'ı base modelle birleştir (merge_and_unload) — GGUF/Ollama dönüşümü için.

Neden: Ollama 0.34.x çalışma-anı ``ADAPTER`` desteğini kaldırdı; adapter'ı Ollama'da
kullanmanın yolu base ile birleştirip GGUF'a çevirmektir (bkz. HANDOFF 2026-09-28 (3)).
HF önbelleğindeki base DEĞİŞMEZ; birleşik model AYRI klasöre yazılır.

Doğrulama (Kural 2): birleştirmeden ÖNCE (PEFT) ve SONRA (birleşik) birkaç kısa girdide
logit farkı ölçülür; eşik aşılırsa çıkış kodu 2 (dönüşüme geçilmez). Ayrıca adapter'ın
GERÇEKTEN yüklendiği doğrulanır (Kademe-2 D3): eksik anahtar uyarısı → hata, yüklenen LoRA
tensörleri dosyadakiyle birebir, adapter etkisi alt sınırın üstünde.

Kullanım:
    uv run --no-sync python scripts/merge_adapter.py hektor_lora_v12_30b \
        --out models/merged/hektor_lora_v12_30b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

# Base'i çevrimiçi yeniden çözmesin (Kademe-2 D9): eğitimdeki yerel snapshot kullanılır.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Doğrulama ölçütleri (mutlak logit eşiği DEĞİL): bf16 birleştirme ağırlıkları bf16'ya
# yuvarlar; 30B'de logitler ~20-30 mertebesinde olduğundan mutlak fark (ölçülen 0.42) tek
# başına anlamsızdır — ilk sürümdeki 0.25 eşiği tahmindi ve yanlış alarm verdi (2026-09-29).
# Esas soru: birleştirme hatası, ADAPTER'IN KENDİ ETKİSİNE (PEFT − base) göre küçük mü,
# dağılım korunuyor mu? v12'de oran 0.03 ölçüldü → 0.1 eşiği (eskiden 0.5, çok gevşekti; D6).
MAX_MERGE_TO_EFFECT = 0.1  # birleştirme |fark|max ≤ 0.1 × adapter etkisi |fark|max
MAX_KL = 0.01  # KL(PEFT ‖ birleşik), nat
MIN_TOP10_OVERLAP = 8  # ilk-10 token kümesinin örtüşmesi
# Adapter hiç yüklenmemişse (anahtar kayması) etki 0 olur ve oran kapısı 0 ≤ 0 ile GEÇERDİ
# (D3). v12 etkisi 13.75; 1.0 çok muhafazakâr bir taban.
MIN_ADAPTER_EFFECT = 1.0

PROBES: tuple[list[dict[str, str]], ...] = (
    [{"role": "user", "content": "Bir backtest'te komisyon neden hesaba katılmalı?"}],
    [
        {
            "role": "system",
            "content": "Sen Hektor yerel AI asistanısın. RAG bağlamı varsa kullan.",
        },
        {
            "role": "user",
            "content": "BAĞLAM:\nMomentum returns are measured over 12 months skipping the "
            "last month.\n\nSORU: Momentum getirisi nasıl ölçülür?",
        },
    ],
    [{"role": "user", "content": "Tüm sermayemle 50x kaldıraç açsam olur mu?"}],
)


def gate_failures(metrics: list[dict[str, Any]], *, max_kl: float = MAX_KL) -> list[str]:
    """Prompt başına ölçümlerden kapı ihlallerini döndür (boş liste = geçti). Saf → test."""
    fails: list[str] = []
    for i, m in enumerate(metrics):
        if m["effect"] < MIN_ADAPTER_EFFECT:
            fails.append(f"prompt {i}: adapter etkisi {m['effect']:.3g} < {MIN_ADAPTER_EFFECT}")
        if m["diff"] > MAX_MERGE_TO_EFFECT * m["effect"]:
            fails.append(f"prompt {i}: birleştirme farkı {m['diff']:.3g} etkinin çok üstünde")
        if m["kl"] > max_kl:
            fails.append(f"prompt {i}: KL {m['kl']:.3g} > {max_kl}")
        if m["top10"] < MIN_TOP10_OVERLAP or not m["same_top"]:
            fails.append(f"prompt {i}: en olası tokenlar değişti")
    return fails


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("adapter", help="models/adapters/<ad> ya da tam yol")
    ap.add_argument("--out", required=True, help="Birleşik modelin yazılacağı klasör")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="run_complete.json olmadan birleştir (yarım/eski koşu riski — önerilmez)",
    )
    ap.add_argument(
        "--max-kl",
        type=float,
        default=MAX_KL,
        help=f"KL kapısı (varsayılan {MAX_KL}). Yalnız bilinçli insan kararıyla gevşetilir; "
        "kullanılan değer ve istisna merge_info.json'a yazılır. Diğer kapılar DEĞİŞMEZ.",
    )
    args = ap.parse_args()
    if args.max_kl != MAX_KL:
        print(f"UYARI: KL kapısı {MAX_KL} → {args.max_kl} (insan kararı; kayda geçer)", flush=True)

    adapter_dir = Path(args.adapter)
    if not adapter_dir.is_dir():
        adapter_dir = Path("models/adapters") / args.adapter
    cfg_path = adapter_dir / "adapter_config.json"
    weights = adapter_dir / "adapter_model.safetensors"
    if not (cfg_path.is_file() and weights.is_file()):
        print(f"HATA: adapter eksik ({adapter_dir}) — adapter_config/safetensors yok.")
        return 1
    # Sıfırdan başlayan yeni koşu run_complete'i siler ama eski üst-düzey ağırlıkları bırakır;
    # işaret yoksa klasördeki ağırlık bu koşuya ait olmayabilir (Kademe-2 D4c).
    run_complete_path = adapter_dir / "run_complete.json"
    run_complete: dict[str, Any] | None = None
    if run_complete_path.is_file():
        run_complete = json.loads(run_complete_path.read_text(encoding="utf-8"))
    elif not args.allow_incomplete:
        print(
            f"HATA: {run_complete_path} yok — eğitim bitmemiş ya da klasörde eski ağırlık var. "
            "Bilerek sürdürmek için --allow-incomplete."
        )
        return 1
    base = json.loads(cfg_path.read_text(encoding="utf-8"))["base_model_name_or_path"]
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"HATA: çıktı klasörü dolu: {out} (üzerine yazılmaz).")
        return 1
    print(f"base={base}\nadapter={adapter_dir}\nout={out}", flush=True)

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter_dir)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    base_snapshot = str(getattr(model.config, "_name_or_path", ""))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    missing = [str(w.message) for w in caught if "missing adapter keys" in str(w.message).lower()]
    if missing:
        print(f"HATA: adapter anahtarları modele oturmadı (PEFT yalnız uyarır): {missing[0][:300]}")
        return 2
    # Dosyadaki her lora_A/B tensörü modele BİREBİR yüklenmiş mi (kısmi kayma da yakalanır).
    file_sd = load_file(str(weights))
    model_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    loaded = 0
    for key, tensor in file_sd.items():
        suffix = key.replace(".weight", "").split("model.", 1)[-1]
        match = [
            v
            for k, v in model_sd.items()
            if k.replace(".default", "").replace(".weight", "").endswith(suffix)
        ]
        if len(match) != 1 or not torch.equal(match[0].to(tensor.dtype).cpu(), tensor):
            print(f"HATA: adapter tensörü modele yüklenmemiş/farklı: {key}")
            return 2
        loaded += 1
    print(f"adapter tensörleri doğrulandı: {loaded}/{len(file_sd)}", flush=True)
    model.eval()

    probe_ids = []
    for msgs in PROBES:
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        probe_ids.append(enc["input_ids"] if hasattr(enc, "keys") else enc)
    befores, bases = [], []
    with torch.no_grad():
        for ids in probe_ids:
            befores.append(model(input_ids=ids).logits[0, -1].float())
            with model.disable_adapter():
                bases.append(model(input_ids=ids).logits[0, -1].float())
    merged = model.merge_and_unload()
    metrics: list[dict[str, Any]] = []
    with torch.no_grad():
        for ids, before, base_logits in zip(probe_ids, befores, bases, strict=True):
            after = merged(input_ids=ids).logits[0, -1].float()
            metrics.append(
                {
                    "diff": float((before - after).abs().max()),
                    "effect": float((before - base_logits).abs().max()),
                    "kl": float(
                        torch.nn.functional.kl_div(
                            after.log_softmax(-1),
                            before.log_softmax(-1),
                            log_target=True,
                            reduction="sum",
                        )
                    ),
                    "top10": len(
                        set(before.topk(10).indices.tolist()) & set(after.topk(10).indices.tolist())
                    ),
                    "same_top": int(before.argmax()) == int(after.argmax()),
                }
            )
    for i, m in enumerate(metrics):
        print(
            f"prompt {i}: birleştirme |fark|max={m['diff']:.4g} · adapter etkisi "
            f"|fark|max={m['effect']:.4g} (oran {m['diff'] / max(m['effect'], 1e-9):.3f}) · "
            f"KL={m['kl']:.3g} · top10 örtüşme={m['top10']}/10 · en olası aynı: {m['same_top']}",
            flush=True,
        )
    fails = gate_failures(metrics, max_kl=args.max_kl)
    if fails:
        print("HATA: birleşik model doğrulaması başarısız — dönüşüme geçilmez:")
        for f in fails:
            print(f"  - {f}")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    tmpl = adapter_dir / "chat_template.jinja"
    if tmpl.is_file() and not (out / "chat_template.jinja").exists():
        shutil.copy2(tmpl, out / "chat_template.jinja")
    worst = max(metrics, key=lambda m: m["diff"] / max(m["effect"], 1e-9))
    (out / "merge_info.json").write_text(
        json.dumps(
            {
                "base": base,
                "base_snapshot": base_snapshot,
                "adapter": str(adapter_dir),
                # Köken (Kademe-2 D4d): hangi ağırlık, hangi koşu, hangi kütüphaneler.
                "adapter_sha256": _sha256(weights),
                "run_complete": run_complete,
                "versions": {
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peft": peft.__version__,
                },
                "adapter_tensors_verified": loaded,
                # Geriye dönük uyum: eski tek-prompt alanları EN KÖTÜ prompt'tan.
                "merge_logit_max_diff": worst["diff"],
                "adapter_effect_logit_max_diff": worst["effect"],
                "kl_peft_vs_merged": max(m["kl"] for m in metrics),
                "kl_gate": args.max_kl,
                "kl_gate_override": args.max_kl != MAX_KL,
                "top10_overlap": min(m["top10"] for m in metrics),
                "top_same": all(m["same_top"] for m in metrics),
                "probes": metrics,
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
