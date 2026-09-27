"""Deney matrisi / profil karşılaştırması — CSV + JSON + insan-okunur Markdown.

Kayıtlı eval koşularını okur (LLM çağırmaz). Her profil için en son koşudaki tüm sistemler
(Base, Base+RAG, tekil LoRA'lar, birleşik profil, RAG+profil) tek tabloda gösterilir;
koşulamayan sistemler nedeniyle birlikte listelenir (sessizce atlanmaz).

CLI::

    python -m app.evals.profile.comparison --profiles balanced_v1 statistics_v1
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evals.profile.eval_registry import EvalRegistry, StoredRun

COLUMNS: tuple[tuple[str, str], ...] = (
    ("math_accuracy", "Math"),
    ("statistics_accuracy", "Stats"),
    ("reasoning_accuracy", "Reasoning"),
    ("trading_accuracy", "Trading"),
    ("coding_accuracy", "Coding"),
    ("grounding", "Grounding"),
    ("hallucination_rate", "Hallucination"),
    ("abstention_accuracy", "Abstention"),
    ("citation_accuracy", "Citation"),
    ("retrieval_recall_at_k", "Recall@k"),
    ("latency", "Latency(s)"),
    ("tokens_per_second", "Tok/s"),
)


def matrix_rows(run: StoredRun) -> list[dict[str, Any]]:
    labels = run.manifest.get("system_labels") or {}
    rows: list[dict[str, Any]] = []
    for system in run.manifest.get("systems") or []:
        m = run.metrics.get(system)
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "profile": run.manifest.get("profile"),
            "system": system,
            "configuration": labels.get(system, system),
            "status": "ok"
            if m
            else f"koşulamadı: {run.manifest.get('not_run', {}).get(system, '?')}",
        }
        for key, _ in COLUMNS:
            row[key] = (m or {}).get(key)
        row["overall"] = ((m or {}).get("overall") or {}).get("overall")
        row["n_items"] = (m or {}).get("n_items")
        rows.append(row)
    return rows


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_markdown(rows: list[dict[str, Any]], title: str) -> str:
    head = "| Configuration | " + " | ".join(lbl for _, lbl in COLUMNS) + " | n | Overall* |"
    sep = "|" + "---|" * (len(COLUMNS) + 3)
    lines = [f"# {title}", "", head, sep]
    skipped: list[str] = []
    for r in rows:
        if r["status"] != "ok":
            skipped.append(f"- `{r['configuration']}` — {r['status']}")
            continue
        cells = " | ".join(_fmt(r[k]) for k, _ in COLUMNS)
        lines.append(
            f"| {r['configuration']} | {cells} | {_fmt(r['n_items'])} | {_fmt(r['overall'])} |"
        )
    lines += [
        "",
        "\\* Overall yalnız raporlama kolaylığıdır (eval_weights); terfi kararı domain skorları "
        "+ regression gate ile verilir. Hallucination düşük = iyi. Eval ağırlıkları LoRA profil "
        "ağırlıklarıyla aynı kavram değildir.",
    ]
    if skipped:
        lines += ["", "## Koşulamayan konfigürasyonlar", "", *skipped]
    return "\n".join(lines) + "\n"


def write_reports(
    rows: list[dict[str, Any]], out_dir: Path, stem: str, title: str
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / f"{stem}.csv",
        "json": out_dir / f"{stem}.json",
        "md": out_dir / f"{stem}.md",
    }
    fields = ["run_id", "profile", "system", "configuration", "status"]
    fields += [k for k, _ in COLUMNS] + ["n_items", "overall"]
    with paths["csv"].open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    paths["json"].write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["md"].write_text(render_markdown(rows, title), encoding="utf-8")
    return paths


def compare_profiles(
    profiles: list[str],
    *,
    registry: EvalRegistry | None = None,
    out_dir: Path | None = None,
    dataset_prefix: str = "",
) -> dict[str, Any]:
    reg = registry or EvalRegistry()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for p in profiles:
        run = reg.latest_for_profile(p, dataset_prefix)
        if run is None:
            missing.append(p)
            continue
        rows.extend(matrix_rows(run))
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    stem = f"comparison_{'_'.join(profiles)}_{stamp}"
    target = out_dir or get_settings().root / "reports" / "eval"
    paths = write_reports(rows, target, stem, f"Profil karşılaştırması: {', '.join(profiles)}")
    return {
        "rows": len(rows),
        "missing_profiles": missing,
        "paths": {k: str(v) for k, v in paths.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profil eval koşularını karşılaştır")
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--dataset", default="", help="ör. validation veya golden_test")
    args = parser.parse_args(argv)
    res = compare_profiles(args.profiles, dataset_prefix=args.dataset)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if res["missing_profiles"]:
        print(f"UYARI: koşusu olmayan profiller: {res['missing_profiles']}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
