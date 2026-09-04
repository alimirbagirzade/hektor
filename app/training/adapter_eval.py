"""Gerçek PEFT adapter değerlendirmesi: base vs adapter (Kural 2 — dürüst gate).

`ModelEvaluator` base Ollama'yı kullanır, eğitilen PEFT adapter'ı YÜKLEMEZ — bu yüzden
adapter'ı gerçekte ölçmez. Bu modül adapter'ı transformers/PEFT ile GERÇEKTEN yükler,
eval sorularına cevap ürettirir, red-flag sezgileriyle (evaluate_model.check_flags) +
dejenerasyon (tekrar döngüsü) cezasıyla puanlar ve **base ile yan yana** kıyaslar.

AĞIR: CPU'da 4B inference (her soru dakikalar). Eğitim bittikten sonra çalıştır (RAM serbest).
Verdict: adapter base'den iyi → accept, kötü → reject (terfi etme), eşit → inconclusive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.training.evaluate_model import check_flags, load_eval_set


def _max_ngram_repeat(answer: str, n: int = 3) -> int:
    """En çok tekrar eden kelime n-gram'ının görülme sayısı (token-düzeyi döngü sezgisi)."""
    from collections import Counter

    toks = answer.split()
    if len(toks) < n * 2:
        return 1
    grams = Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values()) if grams else 1


def _is_degenerate(answer: str) -> bool:
    """Tekrar döngüsü / overfit-çöküş sezgisi (v5 dersi: token-düzeyi döngüyü de yakala).

    Üç sinyalden herhangi biri: (1) aynı CÜMLE tekrarı, (2) aynı 3-gram'ın ≥4 kez
    tekrarı (cümle ayıracı olmasa da; v5 adapter aynı ifadeyi 5 kez yazdı), (3) aynı
    SATIRın (madde/liste) tekrarı. Eşikler muhafazakâr — sağlam cevabı yanlış-flag'lemez.
    """
    sents = [s.strip() for s in answer.split(".") if len(s.strip()) > 15]
    sent_dup = len(sents) >= 3 and len(set(sents)) <= max(1, len(sents) // 2)
    ngram_loop = _max_ngram_repeat(answer, 3) >= 4
    lines = [ln.strip() for ln in answer.splitlines() if len(ln.strip()) > 15]
    line_dup = len(lines) >= 4 and len(set(lines)) <= max(1, len(lines) // 2)
    return sent_dup or ngram_loop or line_dup


def _flags_for(answer: str, must_avoid: list[str]) -> list[str]:
    flags = check_flags(answer, must_avoid)
    # Boş/whitespace cevap = çökmüş adapter. check_flags yalnız red-flag DESENİ arar;
    # boş cevapta hiç desen olmadığından 0 bayrak → skor 1.0 → çalışan base'i geçip 'accept'
    # alır (v5-sınıfı SAHTE-KABUL: eval adapter'ın gerçek kalitesini ölçmüyor). Boş cevabı
    # açıkça bayrakla. NOT: meşru çekimserlik ('Bilmiyorum'/'kaynakta yok') non-empty olduğu
    # için bayraklanMAZ (Kural 7 abstain korunur) — yalnız GERÇEKTEN boş çıktı cezalanır.
    if not answer.strip():
        flags.append("empty_answer")
    if _is_degenerate(answer):
        flags.append("degenerate_repetition")
    return flags


@dataclass
class AdapterEvalResult:
    eval_set: str
    base_model: str
    adapter: str
    n: int
    base_score: float
    adapter_score: float
    base_flags: int
    adapter_flags: int
    regression: bool
    verdict: str  # accept | reject | inconclusive
    rows: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "eval_set": self.eval_set,
            "base_model": self.base_model,
            "adapter": self.adapter,
            "n": self.n,
            "base_score": self.base_score,
            "adapter_score": self.adapter_score,
            "base_flags": self.base_flags,
            "adapter_flags": self.adapter_flags,
            "regression": self.regression,
            "verdict": self.verdict,
            "rows": self.rows,
        }


def _resolve_base_model(adapter_dir: str | Path) -> str | None:
    """adapter_config.json'dan base_model_name_or_path oku (adapter kendi base'ini bilir).

    Küçük-model adapter'ı (örn. Qwen2.5-1.5B) settings.peft_base_model (4B) ile yüklenirse
    LoRA boyutları uyuşmaz ve yükleme çöker. Bu yüzden base ÖNCE adapter'ın kendi
    config'inden alınır; yoksa çağıran settings'e düşer.
    """
    cfg = Path(adapter_dir) / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    base = data.get("base_model_name_or_path")
    return str(base) if base else None


def _load_model(base_model: str, adapter_dir: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    # model: Any → adapter (PeftModel) yeniden-atamasında torch'lu/torch'suz ortamların
    # ikisinde de "unused type: ignore" / assignment hatası olmasın. Runtime davranışı aynı.
    model: Any = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def _generate(tok, model, question: str, max_new_tokens: int = 220) -> str:
    import torch

    msgs = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False
        )  # greedy=determinist
    return tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()


_MIN_EVAL_N = 5  # bu sayının altında 'accept' YASAK (v5 dersi: n=1 ile sahte accept)


def _decide_verdict(
    base_score: float,
    adapter_score: float,
    *,
    n: int,
    min_n: int = _MIN_EVAL_N,
    adapter_degenerate: bool = False,
) -> str:
    """Eval verdict'i — küçük-n'de 'accept'i bloklar (v5 disiplin-regresyon dersi).

    v5'te eval n=1 örnekle 'accept' demişti; tek soruda base bir bayrak alıp adapter almayınca
    istatistiksel temeli olmayan bir 'kabul' üretiliyordu. Kurallar:
      * adapter_degenerate → 'reject' (tekrar döngüsü/çöküş KATEGORİK başarısızlık; v5 adapter
        dejenere tekrar yapmıştı ama eski kod bunu yalnız skora ekliyordu → base de kötüyse
        'accept' kaçabiliyordu. Degenerasyon skordan BAĞIMSIZ veto).
      * adapter < base  → 'reject' (regresyon, HER n'de — güvenli yön).
      * n < min_n        → 'inconclusive' (az örnek; accept'e güvenme).
      * adapter > base  → 'accept'.
      * eşitlik          → 'inconclusive'.
    """
    if adapter_degenerate:
        return "reject"
    if adapter_score < base_score:
        return "reject"
    if n < min_n:
        return "inconclusive"
    if adapter_score > base_score:
        return "accept"
    return "inconclusive"


def evaluate_adapter(
    adapter_dir: str | Path,
    eval_set: str | Path,
    *,
    base_model: str | None = None,
    n: int | None = None,
    min_n: int = _MIN_EVAL_N,
) -> AdapterEvalResult:
    """Base vs adapter karşılaştırmalı eval. (Kural 6: greedy determinist üretim.)

    `min_n`: bu sayının altında örnekle 'accept' verilmez ('inconclusive'). v5 regresyonunun
    (n=1 ile sahte accept) doğrudan koruması; CLI `--n 1` veya küçük eval set artık terfi
    sinyali üretemez (regresyon yine her n'de raporlanır).
    """
    s = get_settings()
    # Base önceliği: açık argüman → adapter'ın kendi config'i → settings (4B). Küçük-model
    # adapter'ını 4B base ile yüklememek için config'ten okumak ŞART (boyut uyuşmazlığı).
    base_model = base_model or _resolve_base_model(adapter_dir) or s.peft_base_model
    items = load_eval_set(eval_set)
    if n:
        items = items[:n]

    # 1) BASE (adapter yok) — tek tek üret, sonra belleği boşalt
    tok, model = _load_model(base_model, None)
    base_ans = [_generate(tok, model, it.question) for it in items]
    del model

    # 2) ADAPTER (base + PEFT)
    tok, model = _load_model(base_model, str(adapter_dir))
    adapt_ans = [_generate(tok, model, it.question) for it in items]
    del model

    base_flag_total = 0
    adapt_flag_total = 0
    rows: list[dict] = []
    for it, b, a in zip(items, base_ans, adapt_ans, strict=True):
        bf = _flags_for(b, it.must_avoid)
        af = _flags_for(a, it.must_avoid)
        base_flag_total += len(bf)
        adapt_flag_total += len(af)
        rows.append(
            {"q": it.question, "base": b, "adapter": a, "base_flags": bf, "adapter_flags": af}
        )

    denom = max(1, len(items))
    base_score = round(1.0 - base_flag_total / denom, 4)
    adapter_score = round(1.0 - adapt_flag_total / denom, 4)
    regression = adapter_score < base_score
    # Çöküş (tekrar döngüsü VEYA boş çıktı) skordan BAĞIMSIZ veto — v5 adapter dejenere
    # tekrar yaptı ama eski kod bunu yalnız flag/skora ekliyordu, base de kötüyse 'accept'
    # kaçabiliyordu. Boş çıktı da aynı sınıf çöküştür (kısmi boş kalırsa skor yine base'i
    # geçebilir). Adapter herhangi bir soruda dejenere/boş olduysa kategorik reddet.
    adapter_degenerate = any(
        "degenerate_repetition" in r["adapter_flags"] or "empty_answer" in r["adapter_flags"]
        for r in rows
    )
    verdict = _decide_verdict(
        base_score,
        adapter_score,
        n=len(items),
        min_n=min_n,
        adapter_degenerate=adapter_degenerate,
    )

    result = AdapterEvalResult(
        eval_set=Path(eval_set).stem,
        base_model=base_model,
        adapter=str(adapter_dir),
        n=len(items),
        base_score=base_score,
        adapter_score=adapter_score,
        base_flags=base_flag_total,
        adapter_flags=adapt_flag_total,
        regression=regression,
        verdict=verdict,
        rows=rows,
    )
    out = s.reports_dir / "evals" / f"adapter_eval_{Path(adapter_dir).name}_{result.eval_set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result
