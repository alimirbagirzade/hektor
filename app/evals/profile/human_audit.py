"""İnsan denetimi — tabakalı rastgele örnek, KÖR inceleme, evaluator↔insan uyumu.

İki aşama:
1. ``export`` (KÖR): reviewer'a soru, gold cevap, RAG chunk'ları, model cevabı gösterilir;
   otomatik karar/skor GÖSTERİLMEZ (çıpalama önyargısı olmasın). Otomatik karar ayrı
   ``_key.jsonl`` dosyasında saklanır.
2. ``score`` (AÇIK): reviewer PASS / PARTIAL / FAIL + hata kategorisi doldurduktan sonra
   otomatik kararla birleştirilir → uyum oranı, Cohen kappa, **false PASS** (otomatik PASS,
   insan FAIL/PARTIAL) ayrıca raporlanır — yanlışı doğru sayan evaluator en tehlikelisidir.

CLI::

    python -m app.evals.profile.human_audit --run-id RUN_ID --sample-size 100
    python -m app.evals.profile.human_audit --run-id RUN_ID --score
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evals.profile.dataset_loader import load_split
from app.evals.profile.eval_registry import EvalRegistry
from app.evals.profile.failure_diagnosis import FAILURE_CATEGORIES
from app.evals.profile.schema import EvalItem, ItemScore, Split
from app.lora.mix_common import read_jsonl, write_jsonl

HUMAN_DECISIONS: tuple[str, ...] = ("PASS", "PARTIAL", "FAIL")


def audit_dir(run_id: str) -> Path:
    return get_settings().root / "reports" / "eval" / "audit" / run_id


def stratified_sample(scores: list[ItemScore], sample_size: int, seed: int) -> list[ItemScore]:
    """Domain'ler arasında dengeli (round-robin) deterministik örnek; ölçülemeyenler hariç."""
    pool = [s for s in scores if s.correct is not None]
    by_dom: dict[str, list[ItemScore]] = defaultdict(list)
    for s in sorted(pool, key=lambda x: (x.domain, x.system, x.item_id)):
        by_dom[s.domain].append(s)
    rng = random.Random(seed)
    for lst in by_dom.values():
        rng.shuffle(lst)
    out: list[ItemScore] = []
    domains = sorted(by_dom)
    while len(out) < sample_size and any(by_dom[d] for d in domains):
        for d in domains:
            if by_dom[d] and len(out) < sample_size:
                out.append(by_dom[d].pop())
    return out


def _auto_decision(s: ItemScore) -> str:
    return "PASS" if s.correct else "FAIL"


def export_audit(
    run_id: str,
    scores: list[ItemScore],
    items_by_id: dict[str, EvalItem],
    *,
    sample_size: int,
    seed: int,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    sample = stratified_sample(scores, sample_size, seed)
    d = out_dir or audit_dir(run_id)
    blind_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for n, s in enumerate(sample, 1):
        item = items_by_id.get(s.item_id)
        audit_id = f"aud_{n:04d}"
        blind_rows.append(
            {
                "audit_id": audit_id,
                "domain": s.domain,
                "question": item.question if item else "",
                "gold_answer": (item.reference_answer if item else "")
                or ("; ".join(item.accepted_answers) if item else ""),
                "rag_chunk_ids": (s.retrieval or {}).get("retrieved_chunk_ids", []),
                "model_answer": s.answer,
                "human_decision": "",  # PASS | PARTIAL | FAIL
                "error_category": "",  # FAILURE_CATEGORIES'den biri (FAIL/PARTIAL ise)
                "reviewer_note": "",
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "item_id": s.item_id,
                "system": s.system,
                "automatic_decision": _auto_decision(s),
                "automatic_score": s.score,
                "automatic_failure_categories": s.failure_categories,
            }
        )
    blind = d / "audit_blind.jsonl"
    key = d / "audit_key.jsonl"
    write_jsonl(blind, blind_rows)
    write_jsonl(key, key_rows)
    readme = d / "README.md"
    readme.write_text(
        "# İnsan denetimi (KÖR)\n\n"
        f"Koşu: `{run_id}` · örnek: {len(sample)} (domain-dengeli, seed={seed})\n\n"
        "`audit_blind.jsonl` içindeki her satır için `human_decision` alanına PASS / PARTIAL / "
        "FAIL yaz; FAIL/PARTIAL ise `error_category` alanına şunlardan birini yaz:\n\n"
        + "".join(f"- `{c}`\n" for c in FAILURE_CATEGORIES)
        + "\n`audit_key.jsonl` dosyasını inceleme bitene kadar AÇMA (otomatik karar oradadır).\n"
        f"Bitince: `python -m app.evals.profile.human_audit --run-id {run_id} --score`\n",
        encoding="utf-8",
    )
    return {"blind": blind, "key": key, "readme": readme}


@dataclass
class AgreementReport:
    n_reviewed: int
    agreement: float | None
    cohen_kappa: float | None
    false_pass: int
    false_pass_rate: float | None  # otomatik PASS'lerin kaçı insan tarafından reddedildi
    false_fail: int
    partial_count: int
    false_pass_items: list[dict[str, Any]] = field(default_factory=list)
    error_categories: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    labels = {x for p in pairs for x in p}
    pe = sum(
        (sum(a == lab for a, _ in pairs) / n) * (sum(b == lab for _, b in pairs) / n)
        for lab in labels
    )
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def compute_agreement(
    blind_rows: list[dict[str, Any]], key_rows: list[dict[str, Any]]
) -> AgreementReport:
    """PARTIAL ikili karşılaştırmada FAIL sayılır (katı: kısmen doğru ≠ doğru)."""
    keys = {r["audit_id"]: r for r in key_rows}
    pairs: list[tuple[str, str]] = []
    false_pass_items: list[dict[str, Any]] = []
    false_fail = partial = 0
    cats: dict[str, int] = {}
    for row in blind_rows:
        human = str(row.get("human_decision", "")).strip().upper()
        if human not in HUMAN_DECISIONS:
            continue
        k = keys.get(row["audit_id"])
        if k is None:
            continue
        auto = k["automatic_decision"]
        human_bin = "PASS" if human == "PASS" else "FAIL"
        partial += human == "PARTIAL"
        pairs.append((auto, human_bin))
        if auto == "PASS" and human_bin == "FAIL":
            false_pass_items.append(
                {
                    "audit_id": row["audit_id"],
                    "item_id": k["item_id"],
                    "system": k["system"],
                    "human_decision": human,
                    "error_category": row.get("error_category", ""),
                }
            )
        if auto == "FAIL" and human_bin == "PASS":
            false_fail += 1
        cat = str(row.get("error_category") or "").strip()
        if cat:
            cats[cat] = cats.get(cat, 0) + 1
    n = len(pairs)
    auto_pass = sum(a == "PASS" for a, _ in pairs)
    return AgreementReport(
        n_reviewed=n,
        agreement=round(sum(a == b for a, b in pairs) / n, 4) if n else None,
        cohen_kappa=_kappa(pairs),
        false_pass=len(false_pass_items),
        false_pass_rate=round(len(false_pass_items) / auto_pass, 4) if auto_pass else None,
        false_fail=false_fail,
        partial_count=partial,
        false_pass_items=false_pass_items,
        error_categories=cats,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval koşusu için insan denetimi")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score", action="store_true", help="doldurulmuş denetimi puanla")
    args = parser.parse_args(argv)

    d = audit_dir(args.run_id)
    if args.score:
        blind, key = read_jsonl(d / "audit_blind.jsonl"), read_jsonl(d / "audit_key.jsonl")
        if not blind:
            print(f"HATA: denetim dosyası yok: {d}", file=sys.stderr)
            return 2
        rep = compute_agreement(blind, key)
        (d / "agreement.json").write_text(
            json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
        return 0

    run = EvalRegistry().load(args.run_id)
    label = str(run.manifest.get("eval_dataset", "validation@"))
    split = Split(label.split("@", 1)[0])
    purpose = "audit"
    items = {it.id: it for it in load_split(split, purpose=purpose, verify_hash=False)}
    paths = export_audit(
        args.run_id, run.items, items, sample_size=args.sample_size, seed=args.seed
    )
    print(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
