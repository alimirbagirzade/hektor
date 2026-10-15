"""LLM-30 cevaplarını YEREL hakemle rubrik üzerinden puanla (sürekli döngü kapısı).

Kullanıcı kararı (2026-10-15): eğitimler arası kapıyı yerel Ollama hakemi otomatik puanlar;
ölçüt "sorular ve doğru cevaplar". Protokol (``docs/PROTOKOL_LORA_RAG_IYILESTIRME.md``):
hakem LLM TEK otorite değildir — bu puanlar yalnız döngünün devam/dur kararını verir; terfi
insanda ve kilitli final takımıyla yapılır.

Girdi: ``llm30_run`` ``raw.jsonl``. Çıktı: aynı klasörde ``raw_scored.jsonl`` (``raw.jsonl``
DEĞİŞMEZ) — her satıra ``score_by_dimension`` (``rubric_gate.rubric_dims``), ``critical_errors``
(yalnız o sorunun kritik kodlarından) ve ``judge`` (model, tohum, gerekçe) yazılır. Hakem
cevabı şemaya uymazsa bir kez yeniden sorulur; yine uymazsa satır PUANSIZ kalır (kapı onu
puansız sayar → geçemez; sessiz varsayılan puan yok).

Hakeme verilen: soru, (varsa) referans cevap ve o soruya ait deterministik sayısal anahtarlar,
rubrik, kritik hata listesi, model cevabı. Eval soruları/anahtarları yalnız hakem istemine
girer — eğitime ve retrieval indeksine GİRMEZ (değişmeyen kural).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

SEED = 42
SCORED_NAME = "raw_scored.jsonl"

#: (istem, sistem) → hakem metni. Testlerde sahte üreticiyle değiştirilir.
Generate = Callable[[str, str], str]

JUDGE_SYSTEM = (
    "Sen titiz bir sınav hakemisin. Bir trading-araştırma modelinin cevabını verilen rubriğe "
    "göre puanlarsın. Yalnız istenen JSON nesnesini yaz; başka metin yazma. Referans cevap ya "
    "da sayısal anahtar verildiyse doğruluğu ONA göre değerlendir; verilmediyse alan bilgine "
    "göre değerlendir ve belirsizsen düşük puan ver. Uzunluk ya da özgüven puan kazandırmaz."
)


def judge_model() -> str:
    from app.config import get_settings

    s = get_settings()
    return (getattr(s, "continuous_loop_judge_model", "") or "").strip() or s.llm_model


def _numeric_for(question_id: str) -> dict[str, object]:
    from app.evals.llm30 import numeric_keys

    prefix = question_id.removeprefix("llm30-") + "_"
    return {k: v for k, v in numeric_keys().items() if k.startswith(prefix)}


def build_prompt(item: Any, row: dict[str, Any], dims: dict[str, int], codes: list[str]) -> str:
    from app.evals.llm30 import CRITICAL_ERRORS

    ref = (getattr(item, "reference_answer", "") or "").strip()
    nums = _numeric_for(row["question_id"])
    parts = [f"SORU:\n{item.question}"]
    if ref:
        parts.append(f"REFERANS CEVAP (doğru kabul edilen):\n{ref}")
    if nums:
        parts.append(
            "DETERMİNİSTİK SAYISAL ANAHTARLAR (hesapla üretildi; doğruluk ölçütü):\n"
            + "\n".join(f"- {k}: {v}" for k, v in nums.items())
        )
    if not ref and not nums:
        parts.append("REFERANS: yok (alan bilgine göre değerlendir).")
    rubric = "\n".join(f"- {d}: 0–{m} tam sayı" for d, m in dims.items())
    crit = (
        "\n".join(f"- {c}: {CRITICAL_ERRORS.get(c, c)}" for c in codes)
        if codes
        else "- (bu soru için tanımlı kritik hata yok)"
    )
    shape = ", ".join(f'"{d}": <0-{m}>' for d, m in dims.items())
    parts += [
        f"RUBRİK:\n{rubric}\n(dogruluk 4 = tüm alt maddeler doğru; alt_madde = her alt madde "
        "cevaplandı mı; tutarlilik = hesap/açıklama çelişkisiz mi; dogrulama = uygulanabilir "
        "doğrulama/test önerisi var mı; kaynak_destegi = kaynak kimlikleri iddiayı destekliyor mu)",
        f"KRİTİK HATALAR (cevapta varsa kodunu listele):\n{crit}",
        f"MODEL CEVABI:\n{row.get('raw_answer') or ''}",
        "YALNIZ şu biçimde JSON yaz: {" + shape + ', "kritik": [<kod>...], "gerekce": "<kısa>"}',
    ]
    return "\n\n".join(parts)


def parse_judgement(
    text: str, dims: dict[str, int], codes: list[str]
) -> tuple[dict[str, int], list[str], str] | None:
    """Hakem metnini doğrula; şemaya uymuyorsa None (uydurma düzeltme yok)."""
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    scores: dict[str, int] = {}
    for dim, top in dims.items():
        v = obj.get(dim)
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= top:
            return None
        scores[dim] = v
    crit = obj.get("kritik", [])
    if not isinstance(crit, list) or any(not isinstance(c, str) for c in crit):
        return None
    if any(c not in codes for c in crit):
        return None  # bu soruya ait olmayan kod → şema dışı (sessizce atılmaz)
    return scores, sorted(set(crit)), str(obj.get("gerekce", ""))[:500]


def _ollama(model: str) -> Generate:
    from app.brain.local_llm import LocalLLM

    llm = LocalLLM(model=model)

    def gen(prompt: str, system: str) -> str:
        return llm.generate(
            prompt,
            system=system,
            temperature=0.0,
            max_tokens=400,
            fmt="json",
            seed=SEED,
            timeout=900,
        )

    return gen


def score_run(
    run_dir: Path, *, generate: Generate | None = None, model: str | None = None
) -> dict[str, Any]:
    """``run_dir/raw.jsonl`` → ``raw_scored.jsonl``. Sürdürülebilir (puanlı satır atlanır)."""
    from app.evals.llm30 import CRITICAL_BY_QUESTION, load_llm30
    from app.evals.rubric_gate import rubric_dims
    from app.orchestration.local_judge import model_blocker

    model = model or judge_model()
    blocker = model_blocker(model)
    if blocker:
        raise ValueError(blocker)
    gen = generate or _ollama(model)
    items = {it.id: it for it in load_llm30(purpose="selection")}
    raw = [json.loads(x) for x in (run_dir / "raw.jsonl").read_text("utf-8").split("\n") if x]
    out_path = run_dir / SCORED_NAME
    done: dict[tuple[str, str, Any], dict[str, Any]] = {}
    if out_path.exists():
        for x in out_path.read_text("utf-8").split("\n"):
            if x:
                r = json.loads(x)
                if r.get("score_by_dimension") is not None:
                    done[(r["condition"], r["question_id"], r.get("seed"))] = r
    rows: list[dict[str, Any]] = []
    n_new = n_fail = 0
    for row in raw:
        key = (row["condition"], row["question_id"], row.get("seed"))
        if key in done:
            rows.append(done[key])
            continue
        dims = rubric_dims(requires_source=bool(row.get("rag_enabled")))
        codes = list(CRITICAL_BY_QUESTION.get(row["question_id"], ()))
        prompt = build_prompt(items[row["question_id"]], row, dims, codes)
        parsed = None
        for _ in range(2):
            parsed = parse_judgement(gen(prompt, JUDGE_SYSTEM), dims, codes)
            if parsed is not None:
                break
        new = dict(row)
        if parsed is None:
            n_fail += 1
            new["score_by_dimension"] = None
            new["critical_errors"] = None
            new["judge"] = {"model": model, "seed": SEED, "error": "şema dışı cevap (2 deneme)"}
        else:
            n_new += 1
            new["score_by_dimension"], new["critical_errors"], why = parsed
            new["judge"] = {"model": model, "seed": SEED, "gerekce": why}
        rows.append(new)
        out_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return {"path": str(out_path), "rows": len(rows), "new": n_new, "unscored": n_fail}


def all_correct(scored: Path, condition: str = "B") -> dict[str, Any]:
    """ "Tüm sorular doğru mu?" — döngünün DURMA (insan) ölçütü.

    Doğru = o koşuldaki HER satırda hakem ``dogruluk`` tam puan (4), kritik hata yok ve satır
    puanlı. Tek puansız/eksik satır → doğru DEĞİL (kanıt yetersiz).
    """
    rows = [
        json.loads(x)
        for x in scored.read_text("utf-8").split("\n")
        if x and json.loads(x).get("condition") == condition
    ]
    wrong = sorted(
        {
            r["question_id"]
            for r in rows
            if not isinstance(r.get("score_by_dimension"), dict)
            or r["score_by_dimension"].get("dogruluk") != 4
            or r.get("critical_errors")
        }
    )
    qs = {r["question_id"] for r in rows}
    return {
        "ok": bool(rows) and not wrong,
        "n_questions": len(qs),
        "n_rows": len(rows),
        "wrong": wrong,
        "n_correct": len(qs) - len(wrong),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="LLM-30 raw.jsonl → yerel hakem rubrik puanı")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    print(json.dumps(score_run(args.run_dir, model=args.model), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
