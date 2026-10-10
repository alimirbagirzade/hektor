"""Geniş karşılaştırma setinin aile düzeyinde sızıntı denetimi (çevrimdışı)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "eval_set_leak_check", ROOT / "scripts" / "eval_set_leak_check.py"
    )
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def test_planted_training_leak_flags_whole_family(tmp_path) -> None:
    q = "Hedef yıllık volatilite yüzde on iken portföy ağırlığı yüzde kaç olmalıdır?"
    target = tmp_path / "evals" / "candidate_compare" / "set.jsonl"
    _write(
        target,
        [
            {"id": "a", "family": "vol", "question": q},
            {"id": "b", "family": "temiz", "question": "Bambaşka ve benzersiz bir soru metni?"},
        ],
    )
    _write(
        tmp_path / "data" / "lora_sft" / "lora_sft.jsonl",
        [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": "x"}]}],
    )
    rep = _mod().run(target, tmp_path)
    assert rep["leaked_families"] == ["vol"] and not rep["clean"]


def test_broad_v1_is_clean_against_repo_dev_sets() -> None:
    target = ROOT / "evals" / "candidate_compare" / "broad_v1.jsonl"
    rep = _mod().run(target, ROOT)
    assert rep["clean"], rep["hits"][:3]
    assert rep["n_families"] == rep["n_questions"] == 27
    rows = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert {r["dimension"] for r in rows} == {"matematik", "kaynak", "talimat", "strateji"}
    assert all(r["answer_key"] is not None for r in rows if r["type"] == "math")
    assert all(r.get("evidence") for r in rows if r["dimension"] == "kaynak")


def test_evidence_chunk_overlap_is_reported_not_gating(tmp_path) -> None:
    """Kademe 2 P-6/E-9: kanıt parçası eğitimde → raporlanır; makale düzeyi ayrı sayılır."""
    target = tmp_path / "evals" / "candidate_compare" / "s.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "id": "k1",
                "family": "f",
                "question": "Verilen kaynağa göre nedir?",
                "evidence_ids": ["paper_aa_c0003"],
            }
        )
        + "\n",
        "utf-8",
    )
    train = tmp_path / "data" / "lora_sft" / "lora_sft.jsonl"
    train.parent.mkdir(parents=True)

    def row(chunk: str) -> str:
        return json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "başka bir soru metni burada"},
                    {"role": "assistant", "content": "cevap"},
                ],
                "metadata": {"source_id": "paper_aa", "chunk_id": chunk},
            }
        )

    train.write_text(row("paper_aa_c0003") + "\n" + row("paper_aa_c0009") + "\n", "utf-8")
    rep = _mod().run(target, tmp_path)
    ov = rep["kanit_ortusmesi"]["k1"]
    assert len(ov["ayni_parca"]) == 1 and ov["ayni_makale"] == 1
    assert rep["clean"]  # karar kapısı değişmedi
