"""Birleşik SFT veri setini (synth-qa ≤400 + öz-damıtma + onaylı kart + ~%25 disiplin) yaz.

Kanonik `app.training.sft_assembly.assemble_sft_lines` yolunu kullanır — `lora-cloud-prep`
ve `pretrain-gate` ile AYNI birleştirme mantığı (drift yok). `lora-dataset` (sadece kart)
yerine train-loop'un çağırdığı adım: CPU eğitiminin de disiplin örnekleri + sentetik QA ile
eğitilmesini sağlar (v5 regresyon dersi; bkz. docs + memory v5-adapter-regression).

EĞİTİM BAŞLATMAZ (CLAUDE.md kural 8). Determinizm: seed=0 (kural 6).
Kullanım:
    uv run python scripts/assemble_sft.py            # disiplin %25
    uv run python scripts/assemble_sft.py --no-discipline
    uv run python scripts/assemble_sft.py --chat-dataset chat_v3   # sohbet sürümünü SEÇ + birleştir
    uv run python scripts/assemble_sft.py --no-chat                # sohbet seçimini kaldır

Sohbet verisi yalnız `--chat-dataset` ile AÇIKÇA seçildiğinde girer (seçim
`data/lora_sft/chat_selection.json`'da kalır; pretrain-gate ve tazelik denetimi aynı seçimi
kullanır). Yalnız train satırları; satır ve hedef token payı üst sınırı aile düzeyinde.
"""

from __future__ import annotations

import argparse

from app.config.settings import get_settings
from app.training.sft_assembly import CANONICAL_SYNTH_CAP, assemble_sft_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Birleşik SFT (synth+kart+disiplin) → lora_sft.jsonl")
    ap.add_argument("--no-discipline", action="store_true", help="Disiplin karışımını kapat")
    ap.add_argument("--ratio", type=float, default=0.25, help="Disiplin payı (v5 dersi ~0.25)")
    ap.add_argument("--seed", type=int, default=0, help="Determinizm tabanı (kural 6)")
    chat = ap.add_mutually_exclusive_group()
    chat.add_argument(
        "--chat-dataset", default="", help="Sohbet veri sürümünü seç (ör. chat_v3) ve birleştir"
    )
    chat.add_argument("--no-chat", action="store_true", help="Sohbet veri seçimini kaldır")
    args = ap.parse_args()

    from app.feedback.chat_dataset import clear_selection, note_assembly, select_version

    if args.no_chat and clear_selection():
        print("Sohbet veri seçimi kaldırıldı.")
    if args.chat_dataset:
        try:
            sel = select_version(args.chat_dataset)
        except (KeyError, ValueError) as exc:
            raise SystemExit(f"Sohbet veri sürümü seçilemedi: {exc}") from exc
        print(f"Sohbet veri sürümü seçildi: {sel['version_id']} ({sel['train_path']})")

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

    # Eklenen disiplin: karıştırma istatistiğinden (şablon inceltme toplamı değiştirir).
    disc_added = (res.discipline or {}).get("discipline_used", res.total - res.deduped)
    print(
        f"✓ {res.total} örnek → {out}\n"
        f"  synth={res.synth_n} (sınır {CANONICAL_SYNTH_CAP}) öz-damıtma={res.distill_n} "
        f"kart={res.card_n} dedup_sonrası={res.deduped} disiplin_eklenen={disc_added} "
        f"şablon_inceltme={res.template_thinned}"
    )
    if res.chat:
        note_assembly(out, res.chat)
        c = res.chat
        print(
            f"  sohbet {c['version_id']}: {c['used_rows']}/{c['offered_rows']} satır "
            f"({c['used_families']}/{c['offered_families']} aile) · satır payı "
            f"%{c['row_share'] * 100:.2f} · hedef token payı %{c['token_share'] * 100:.2f} "
            f"(üst sınır %{c['max_share'] * 100:.0f}; {c['token_method']})"
        )
        if c["dropped_families"]:
            print(f"  sınır nedeniyle alınmayan aile: {len(c['dropped_families'])}")


if __name__ == "__main__":
    main()
