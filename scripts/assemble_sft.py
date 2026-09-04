"""Birleşik SFT veri setini (synth-qa + onaylı kart + ~%25 disiplin) lora_sft.jsonl'e yaz.

Kanonik `app.training.sft_assembly.assemble_sft_lines` yolunu kullanır — `lora-cloud-prep`
ve `pretrain-gate` ile AYNI birleştirme mantığı (drift yok). `lora-dataset` (sadece kart)
yerine train-loop'un çağırdığı adım: CPU eğitiminin de disiplin örnekleri + sentetik QA ile
eğitilmesini sağlar (v5 regresyon dersi; bkz. docs + memory v5-adapter-regression).

EĞİTİM BAŞLATMAZ (CLAUDE.md kural 8). Determinizm: seed=0 (kural 6).
Kullanım:
    uv run python scripts/assemble_sft.py            # disiplin %25
    uv run python scripts/assemble_sft.py --no-discipline
"""

from __future__ import annotations

import argparse

from app.config.settings import get_settings
from app.training.sft_assembly import assemble_sft_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Birleşik SFT (synth+kart+disiplin) → lora_sft.jsonl")
    ap.add_argument("--no-discipline", action="store_true", help="Disiplin karışımını kapat")
    ap.add_argument("--ratio", type=float, default=0.25, help="Disiplin payı (v5 dersi ~0.25)")
    ap.add_argument("--seed", type=int, default=0, help="Determinizm tabanı (kural 6)")
    args = ap.parse_args()

    settings = get_settings()
    res = assemble_sft_lines(
        settings,
        discipline=not args.no_discipline,
        discipline_ratio=args.ratio,
        seed=args.seed,
    )
    out = settings.root / "data" / "lora_sft" / "lora_sft.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(res.lines) + ("\n" if res.lines else ""), encoding="utf-8")

    # Disiplin, dedup'tan SONRA karıştığı için eklenen = toplam - dedup'lu taban (sağlam).
    disc_added = res.total - res.deduped
    print(
        f"✓ {res.total} örnek → {out}\n"
        f"  synth={res.synth_n} kart={res.card_n} dedup_sonrası={res.deduped} "
        f"disiplin_eklenen={disc_added}"
    )


if __name__ == "__main__":
    main()
