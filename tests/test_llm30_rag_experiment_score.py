"""RAG deneyi puanlaması: P@k, nDCG@k, havuz-recall@k — elle hesaplanmış küçük örnek."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from app.evals.llm30_rag_experiment import score


def _row(cid: str, internal: bool = False) -> dict:
    return {"chunk_id": cid, "paper_id": "p", "title": "t", "distance": 0.1, "internal": internal}


def test_score_hand_computed(tmp_path: Path) -> None:
    results = {
        "V0": {"q1": [_row("a", internal=True), _row("b")]},
        "V1": {"q1": [_row("c"), _row("b")]},
    }
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    rows = [("a", 0), ("b", 1), ("c", 2)]
    labels.write_text(
        "\n".join(
            json.dumps({"question_id": "q1", "chunk_id": c, "relevance": r}) for c, r in rows
        ),
        encoding="utf-8",
    )
    rep = score(tmp_path, labels, top_k=2)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    # V0: [0, 1] → P=0.5, DCG = 1/log2(3), havuz-recall = 1/2 ilgili
    assert rep["V0"]["P@k"] == 0.5
    assert rep["V0"]["nDCG@k"] == pytest.approx(round((1 / math.log2(3)) / idcg, 3))
    assert rep["V0"]["havuz_recall@k"] == 0.5
    assert rep["V0"]["ic_dokuman"] == "1/2"
    # V1: [2, 1] → ideal sıralama → nDCG 1, P=1, recall=1
    assert (rep["V1"]["P@k"], rep["V1"]["nDCG@k"], rep["V1"]["havuz_recall@k"]) == (1.0, 1.0, 1.0)
    assert rep["V1"]["hic_ilgili_yok_soru"] == 0 and rep["V1"]["etiketsiz"] == 0
