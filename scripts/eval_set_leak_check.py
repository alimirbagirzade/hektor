"""Karşılaştırma soru seti ↔ eğitim/geliştirme verisi sızıntı denetimi (AİLE düzeyinde).

Bir ailenin HERHANGİ bir sorusu (ya da kanıt metni dışındaki soru metni) eğitim verisinde veya
geliştirme setlerinde exact / normalize / içerme / yakın-kopya (3-gram Jaccard) olarak geçerse
o ailenin TAMAMI sızmış sayılır. Ayrıca eğitim/geliştirme setlerinin sorusu bu setin
sorularıyla eşleşiyorsa (ters yön) yine sızıntıdır.

Kullanım:
    uv run python scripts/eval_set_leak_check.py evals/candidate_compare/broad_v1.jsonl \
        --out reports/eval_sets/broad_v1_leak.json

Taranan kaynaklar (varsa): data/lora_sft/*.jsonl (eğitim + damıtma + reddedilenler),
data/jsonl/{train,valid}.jsonl, data/learning/chat_datasets/*/{train,eval}.jsonl ve
GELİŞTİRME setleri evals/*.jsonl, evals/llm30/*.jsonl, evals/candidate_compare/ altındaki
DİĞER setler. Çevrimdışı ve deterministiktir; eğitim başlatmaz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.evals.profile.leakage import check_leakage, load_train_jsonl  # noqa: E402
from app.evals.profile.schema import EvalItem  # noqa: E402


def _sources(root: Path, target: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for p in sorted((root / "data" / "lora_sft").glob("*.jsonl")):
        out.append(("egitim", p))
    for name in ("train.jsonl", "valid.jsonl"):
        p = root / "data" / "jsonl" / name
        if p.is_file():
            out.append(("egitim", p))
    for p in sorted((root / "data" / "learning" / "chat_datasets").glob("*/*.jsonl")):
        out.append(("egitim", p))
    for p in [
        *sorted((root / "evals").glob("*.jsonl")),
        *sorted((root / "evals" / "llm30").glob("*.jsonl")),
        *sorted((root / "evals" / "candidate_compare").glob("*.jsonl")),
    ]:
        if p.resolve() != target.resolve():
            out.append(("gelistirme", p))
    return out


def _as_train(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eval satırlarını (question/answer) eğitim biçimine çevir (check_leakage tanır)."""
    return [
        {"question": str(r.get("question") or r.get("prompt") or ""), "answer": ""}
        if "messages" not in r
        else r
        for r in rows
    ]


def _row_refs(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Eğitim satırının (parça kimlikleri, makale kimlikleri)."""
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    chunks = {str(meta.get("chunk_id") or "")} | {
        str(c) for c in meta.get("context_chunk_ids") or []
    }
    papers = {str(meta.get("source_id") or meta.get("paper_id") or "")}
    return {c for c in chunks if c}, {p for p in papers if p}


def evidence_overlap(rows: list[dict[str, Any]], root: Path, target: Path) -> dict[str, Any]:
    """Kaynaklı soruların KANIT parçası/makalesi eğitimde geçiyor mu (bilgi amaçlı).

    Kademe 2 P-6/E-9 (2026-10-10): sözcüksel kapı ``evidence`` kimliklerini görmüyordu. Makale
    düzeyi örtüşme bilinçli kabul edildi (kanıt istemde verilir → "kanıta sadakat" ölçülür);
    birebir AYNI parçadan eğitim satırı ise ezber riskidir ve burada ayrıca listelenir.
    Sızıntı kararını (``clean``) DEĞİŞTİRMEZ — karar insanındır.
    """
    want = {
        r["id"]: {str(e) for e in r.get("evidence_ids") or []}
        for r in rows
        if r.get("evidence_ids")
    }
    out: dict[str, Any] = {}
    for kind, path in _sources(root, target):
        if kind != "egitim":
            continue
        try:
            train = load_train_jsonl(path)
        except (OSError, ValueError):
            continue
        for i, row in enumerate(train):
            chunks, papers = _row_refs(row)
            for qid, ids in want.items():
                rec = out.setdefault(qid, {"ayni_parca": [], "ayni_makale": 0})
                same = ids & chunks
                if same:
                    rec["ayni_parca"].append(f"{path.relative_to(root)}:{i}")
                elif {c.rsplit("_c", 1)[0] for c in ids} & papers:
                    rec["ayni_makale"] += 1
    return out


def run(set_path: Path, root: Path) -> dict[str, Any]:
    rows = [json.loads(x) for x in set_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    items = [
        EvalItem(id=r["id"], domain="trading", question=r["question"], source_provenance="")
        for r in rows
    ]
    fam_of = {r["id"]: r["family"] for r in rows}
    scanned: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for kind, path in _sources(root, set_path):
        try:
            train = _as_train(load_train_jsonl(path))
        except (OSError, ValueError) as exc:
            scanned.append({"path": str(path.relative_to(root)), "error": str(exc)[:200]})
            continue
        rep = check_leakage(items, train)
        scanned.append(
            {
                "path": str(path.relative_to(root)),
                "kind": kind,
                "n": len(train),
                "hits": len(rep.hits),
            }
        )
        for h in rep.hits:
            hits.append(
                {
                    "eval_id": h.eval_id,
                    "family": fam_of.get(h.eval_id, "?"),
                    "kind": h.kind,
                    "source": str(path.relative_to(root)),
                    "source_kind": kind,
                    "train_index": h.train_index,
                    "detail": h.detail[:200],
                }
            )
    leaked = sorted({h["family"] for h in hits})
    return {
        "set": str(set_path),
        "set_sha256": hashlib.sha256(set_path.read_bytes()).hexdigest(),
        "n_questions": len(rows),
        "n_families": len(set(fam_of.values())),
        "leaked_families": leaked,
        "clean": not leaked,
        "hits": hits,
        "scanned": scanned,
        "kanit_ortusmesi": evidence_overlap(rows, root, set_path),
        "method": "exact + normalize + içerme + 3-gram Jaccard ≥ 0.8 (app.evals.profile.leakage); "
        "aile düzeyinde: bir soru sızarsa aile sızmış sayılır",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("set_path", type=Path)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rep = run(a.set_path, a.root)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
    print(
        f"{rep['n_questions']} soru / {rep['n_families']} aile · taranan {len(rep['scanned'])} "
        f"dosya · sızan aile: {rep['leaked_families'] or 'YOK'}"
    )
    same = {q: len(v["ayni_parca"]) for q, v in rep["kanit_ortusmesi"].items() if v["ayni_parca"]}
    print(f"kanıt parçası eğitimde (bilgi; karar insanın): {same or 'YOK'}")
    return 0 if rep["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
