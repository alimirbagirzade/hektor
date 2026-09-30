"""Gerçek PEFT adapter değerlendirmesi: base vs adapter (Kural 2 — dürüst gate).

`ModelEvaluator` base Ollama'yı kullanır, eğitilen PEFT adapter'ı YÜKLEMEZ — bu yüzden
adapter'ı gerçekte ölçmez. Bu modül adapter'ı transformers/PEFT ile GERÇEKTEN yükler,
eval sorularına cevap ürettirir, red-flag sezgileriyle (evaluate_model.check_flags) +
dejenerasyon (tekrar döngüsü) cezasıyla puanlar ve **base ile yan yana** kıyaslar.

AĞIR: CPU'da 4B inference (her soru dakikalar). Eğitim bittikten sonra çalıştır (RAM serbest).
Verdict: adapter base'den iyi → accept, kötü → reject (terfi etme), eşit → inconclusive.

Genişletilmiş setler (``trader_persona`` / ``format_compliance`` / ``rag_integration``):
kalem ``context``/``must_contain``/``persona_signals``/``required_sections`` taşıyorsa soru
bağlamla birlikte (eğitimdeki "BAĞLAM: … SORU: …" biçimi) sorulur ve
``llm_training_eval.evaluate_answer`` ile puanlanır; eksik persona/bölüm/terim/çekimserlik
bayrak olarak verdict'e girer. Yalnız ``question``+``must_avoid`` taşıyan disiplin setleri
eskisi gibi (çıplak soru + ``check_flags``) değerlendirilir.
"""

from __future__ import annotations

import gc
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.evals.llm_training_eval import (
    TrainingEvalItem,
    evaluate_answer,
    load_training_eval_set,
)
from app.lora.safety_scanner import tr_fold
from app.training.evaluate_model import check_flags, load_eval_set


def _max_ngram_repeat(answer: str, n: int = 3) -> int:
    """En çok tekrar eden kelime n-gram'ının görülme sayısı (token-düzeyi döngü sezgisi)."""
    toks = answer.split()
    if len(toks) < n * 2:
        return 1
    grams = Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values()) if grams else 1


# Cümle sınırları: nokta yanında ! ? 。 ve satır sonu (v8: "!" ile biten tekrar kaçıyordu).
_SENT_SPLIT_RE = re.compile(r"[.!?。\n]+")
_SENTENCE_REPEAT_MIN = 3  # aynı cümle bu kadar kez BİREBİR geçerse dejenere


def _is_degenerate(answer: str) -> bool:
    """Tekrar döngüsü / overfit-çöküş sezgisi (v5 dersi: token-düzeyi döngüyü de yakala).

    Dört sinyalden herhangi biri: (1) cümle çeşitliliği yarıya düşmüş, (2) aynı cümle
    ≥`_SENTENCE_REPEAT_MIN` kez birebir, (3) aynı 3-gram'ın ≥4 kez tekrarı (cümle ayıracı
    olmasa da; v5 adapter aynı ifadeyi 5 kez yazdı), (4) aynı SATIRın (madde/liste) tekrarı.

    (2) 2026-09-14'te eklendi — dedektör boşluğu: v8 eval #4 "Ölçülmesi gereken bir hipotez
    var." cümlesini üç kez yazdı ama çeşitlilik eşiği (1) başka cümleler de olduğu için
    tetiklenmedi; eşik tekrar SAYISINI değil çeşitliliği ölçüyordu. Kalibrasyon: v7+v8
    eval'lerindeki 64 gerçek cevapta yalnız bu vakayı ekledi, 32 base cevabından hiçbirini
    bayraklamadı. 2026-09-15 (Kademe-2 av): cümleler ! ? 。 ve satır sonuyla da bölünür ve
    büyük/küçük harf + boşluk normalize edilir ("!" ile biten 3× tekrar kaçıyordu).
    Eşikler muhafazakâr — sağlam cevabı yanlış-flag'lemez.
    """
    sents = [
        " ".join(s.lower().split()) for s in _SENT_SPLIT_RE.split(answer) if len(s.strip()) > 15
    ]
    sent_counts = Counter(sents)
    sent_dup = len(sents) >= 3 and len(sent_counts) <= max(1, len(sents) // 2)
    sent_repeat = bool(sents) and max(sent_counts.values()) >= _SENTENCE_REPEAT_MIN
    # Kademe-2 B7 (2026-09-30): mutlak 4 eşiği uzun (1024 token) cevaplarda meşru terim
    # tekrarını döngü sayar → uzunlukla ölçeklenir; ≤200 kelimede davranış AYNI (4).
    ngram_loop = _max_ngram_repeat(answer, 3) >= max(4, -(-len(answer.split()) // 50))
    lines = [ln.strip() for ln in answer.splitlines() if len(ln.strip()) > 15]
    line_dup = len(lines) >= 4 and len(set(lines)) <= max(1, len(lines) // 2)
    return sent_dup or sent_repeat or ngram_loop or line_dup


def _collapse_flags(answer: str) -> list[str]:
    """Tek cevap düzeyinde çöküş bayrakları (boş çıktı / tekrar döngüsü)."""
    flags: list[str] = []
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


def _flags_for(answer: str, must_avoid: list[str]) -> list[str]:
    return check_flags(answer, must_avoid) + _collapse_flags(answer)


# --------------------------------------------------------------------------- #
# Genişletilmiş eval kalemleri (persona / format / RAG bağlamı)
# --------------------------------------------------------------------------- #
# Kademe-2 av bulgusu (2026-09-28): `load_eval_set` yalnız question+must_avoid tutuyordu;
# auto_pipeline ve `lora-eval` trader_persona / format_compliance / rag_integration setlerini
# BAĞLAMSIZ soruyor ve must_contain / persona_signals / required_sections / çekimserlik
# kontrolünü HİÇ yapmıyordu → RAG setinde "bağlamı kullanıyor mu" ölçülmeden verdict çıkıyordu.

# Bağlamı boş kalem (``empty_context``): retrieval'ın boş döndüğü eğitimdeki biçimle bildirilir.
# Beklenen çekimserlik ifadesi ("kaynak bulunamadı") BİLEREK yazılmaz — cevabı prompt'tan
# kopyalamak ölçümü boşa çıkarırdı.
_EMPTY_CONTEXT_NOTE = "(boş — retrieval bu soru için hiçbir parça döndürmedi)"


def _is_extended(item: TrainingEvalItem) -> bool:
    """Kalem disiplin-ötesi (persona/format/bağlam) bir kontrol taşıyor mu?"""
    return bool(
        item.must_contain
        or item.persona_signals
        or item.required_sections
        or item.must_contain_from_context
        or item.context_mode
        or item.context is not None
    )


def _prompt_for(item: TrainingEvalItem) -> str:
    """Kalemin modele sorulacak metni — bağlam varsa eğitimdeki "BAĞLAM: … SORU: …" biçimi.

    Biçim `lora_chat_service.build_user_content` / `synthetic_qa_builder` ile aynıdır.
    """
    if item.context is None and item.context_mode is None:
        return item.question
    context = (item.context or "").strip() or _EMPTY_CONTEXT_NOTE
    return f"BAĞLAM:\n{context}\n\nSORU: {item.question}"


def _item_flags(item: TrainingEvalItem, answer: str) -> list[str]:
    """Kalemin bayrakları: disiplin seti → `_flags_for` (eski davranış birebir);
    genişletilmiş kalem → `evaluate_answer` (check_flags + persona/format/terim/bağlam)."""
    if not _is_extended(item):
        return _flags_for(answer, item.must_avoid)
    return list(evaluate_answer(item, answer).flags) + _collapse_flags(answer)


def _load_items(eval_set: str | Path) -> list[TrainingEvalItem]:
    """Eval setini genişletilmiş kalemlere yükle (YAML disiplin setleri de desteklenir)."""
    path = Path(eval_set)
    if path.suffix in (".yaml", ".yml"):
        return [
            TrainingEvalItem(question=it.question, must_avoid=list(it.must_avoid))
            for it in load_eval_set(path)
        ]
    return load_training_eval_set(path)


# --------------------------------------------------------------------------- #
# Cevaplar-arası çöküş (canned answer) dedektörü
# --------------------------------------------------------------------------- #
# Kademe-2 av bulgusu (2026-09-28): `_is_degenerate` TEK cevap içindeki tekrarı ölçer; her
# soruya AYNI hazır feragatnameyi ("Bu yatırım tavsiyesi değildir, backtest gerekir...")
# veren adapter 0 bayrak alıp 'accept' alabiliyordu — soruyu hiç okumadan disiplin setini
# "geçiyor". Cevaplar birbirine göre ölçülür: normalize edilmiş farklı cevap oranı yarının
# altındaysa ya da cevapların çoğunluğu (≥3) kelime-3'lü Jaccard'la neredeyse aynıysa çöküş.
_COLLAPSE_MIN_ANSWERS = 3
_COLLAPSE_DISTINCT_RATIO = 0.5
_COLLAPSE_JACCARD = 0.8
_NON_WORD_RE = re.compile(r"[^\w\s]+")


def _normalize_answer(answer: str) -> str:
    return " ".join(_NON_WORD_RE.sub(" ", tr_fold(answer)).split())


def _shingles(norm: str, n: int = 3) -> frozenset[tuple[str, ...]]:
    toks = norm.split()
    if len(toks) < n:
        return frozenset([tuple(toks)])
    return frozenset(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _jaccard(a: frozenset[tuple[str, ...]], b: frozenset[tuple[str, ...]]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def _answers_collapsed(answers: list[str]) -> bool:
    """Cevaplar sorudan bağımsız tek kalıba çökmüş mü? (determinist, sıra-bağımsız)"""
    n = len(answers)
    if n < _COLLAPSE_MIN_ANSWERS:
        return False
    norms = [_normalize_answer(a) for a in answers]
    if len(set(norms)) / n < _COLLAPSE_DISTINCT_RATIO:
        return True
    sh = [_shingles(x) for x in norms]
    largest = max(
        sum(1 for j in range(n) if _jaccard(sh[i], sh[j]) >= _COLLAPSE_JACCARD) for i in range(n)
    )
    return largest >= _COLLAPSE_MIN_ANSWERS and largest > n / 2


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
    # Skordan BAĞIMSIZ kategorik vetolar (degenerate / collapse / guaranteed_profit).
    vetoes: list[str] = field(default_factory=list)
    # Kesilen cevap sayıları + üretim sınırı (B1): rapor kendi üretim koşulunu taşır.
    truncation: dict = field(default_factory=dict)

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
            "vetoes": self.vetoes,
            "truncation": self.truncation,
            "generation": {
                "do_sample": False,
                "max_new_tokens": EVAL_MAX_NEW_TOKENS,
                "system_prompt": False,
                "repetition_penalty": None,
            },
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


def _build_messages(question: str, system: str | None = None) -> list[dict]:
    """Chat mesajlarını kur; `system` verilirse eğitimdeki rol düzeniyle başa eklenir.

    Eval bilerek system'siz çağırır (disiplin kötü-sorunun kendisinden ölçülsün); web
    sohbeti ise eğitim örneklerinin çoğunda bulunan SYSTEM_PROMPT'u geçer.
    """
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    return msgs


# Kademe-2 B1 (2026-09-30): eskiden 220 idi → v10–v12 eval'lerinde base cevaplarının 80/80'i
# cümle ortasında kesildi (adapter kısa cevap verdiği için tamamlanıyordu); base'in maliyet/
# format/persona bayraklarının çoğu kesilen kısımdaydı → karşılaştırma adapter lehine bozuktu.
EVAL_MAX_NEW_TOKENS = 1024
# Bir tarafın cevaplarının bu oranından fazlası kesildiyse 'accept' verilmez (eşit koşul yok).
_MAX_TRUNCATED_SHARE = 0.2


def _generate_checked(
    tok,
    model,
    question: str,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    *,
    system: str | None = None,
) -> tuple[str, bool]:
    """Greedy üretim → (metin, kesildi_mi). Kesildi = sınıra ulaştı ve EOS üretilmedi."""
    import torch

    msgs = _build_messages(question, system)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **ids, max_new_tokens=max_new_tokens, do_sample=False
        )  # greedy=determinist
    gen = out[0][ids["input_ids"].shape[1] :]
    eos = getattr(tok, "eos_token_id", None)
    truncated = len(gen) >= max_new_tokens and (eos is None or int(gen[-1]) != int(eos))
    return tok.decode(gen, skip_special_tokens=True).strip(), truncated


def _generate(
    tok,
    model,
    question: str,
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS,
    *,
    system: str | None = None,
) -> str:
    return _generate_checked(tok, model, question, max_new_tokens, system=system)[0]


_MIN_EVAL_N = 5  # bu sayının altında 'accept' YASAK (v5 dersi: n=1 ile sahte accept)
# 'accept' için adapter base'den en az bu kadar AZ bayrak almalı. Tek bayrak farkı
# (ör. n=5'te 0.8 → 1.0) eval gürültüsünden ayırt edilemez (Kademe-2 av bulgusu).
_MIN_FLAG_MARGIN = 2


def _decide_verdict(
    base_score: float,
    adapter_score: float,
    *,
    n: int,
    min_n: int = _MIN_EVAL_N,
    adapter_degenerate: bool = False,
    adapter_collapsed: bool = False,
    adapter_guaranteed_profit: bool = False,
    min_flag_margin: int = _MIN_FLAG_MARGIN,
) -> str:
    """Eval verdict'i — küçük-n'de 'accept'i bloklar (v5 disiplin-regresyon dersi).

    v5'te eval n=1 örnekle 'accept' demişti; tek soruda base bir bayrak alıp adapter almayınca
    istatistiksel temeli olmayan bir 'kabul' üretiliyordu. Kurallar:
      * adapter_degenerate → 'reject' (tekrar döngüsü/çöküş KATEGORİK başarısızlık; v5 adapter
        dejenere tekrar yapmıştı ama eski kod bunu yalnız skora ekliyordu → base de kötüyse
        'accept' kaçabiliyordu. Degenerasyon skordan BAĞIMSIZ veto).
      * adapter_collapsed → 'reject' (her soruya aynı hazır cevap: soruyu okumayan adapter
        bayrak almadan disiplin setini "geçer" — cevaplar-arası çöküş, Kademe-2 av bulgusu).
      * adapter_guaranteed_profit → 'reject' (herhangi bir cevapta garanti-kâr vaadi Kural 1
        ihlalidir; toplam bayrak sayısında base'in diğer bayraklarıyla takas EDİLEMEZ).
      * adapter < base  → 'reject' (regresyon, HER n'de — güvenli yön).
      * n < min_n        → 'inconclusive' (az örnek; accept'e güvenme).
      * adapter base'den ≥ min_flag_margin bayrak iyi → 'accept'.
      * daha küçük fark / eşitlik → 'inconclusive'.
    """
    if adapter_degenerate or adapter_collapsed or adapter_guaranteed_profit:
        return "reject"
    if adapter_score < base_score:
        return "reject"
    if n < min_n:
        return "inconclusive"
    # skor = 1 − bayrak/n → skor farkı × n = bayrak farkı (4 haneli yuvarlamayı round emer).
    if round((adapter_score - base_score) * n) >= min_flag_margin:
        return "accept"
    return "inconclusive"


def _score_answers(
    items: list[TrainingEvalItem],
    base_ans: list[str],
    adapt_ans: list[str],
    *,
    eval_set: str,
    base_model: str,
    adapter: str,
    min_n: int = _MIN_EVAL_N,
    base_truncated: list[bool] | None = None,
    adapter_truncated: list[bool] | None = None,
) -> AdapterEvalResult:
    """Üretilmiş base/adapter cevaplarını puanla + verdict ver (saf; model yüklemez)."""
    bt = base_truncated or [False] * len(items)
    at = adapter_truncated or [False] * len(items)
    base_flag_total = 0
    adapt_flag_total = 0
    rows: list[dict] = []
    for it, b, a, b_cut, a_cut in zip(items, base_ans, adapt_ans, bt, at, strict=True):
        bf = _item_flags(it, b)
        af = _item_flags(it, a)
        base_flag_total += len(bf)
        adapt_flag_total += len(af)
        rows.append(
            {
                "q": it.question,
                "base": b,
                "adapter": a,
                "base_flags": bf,
                "adapter_flags": af,
                "base_truncated": b_cut,
                "adapter_truncated": a_cut,
            }
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
    adapter_collapsed = _answers_collapsed(adapt_ans)
    adapter_guaranteed_profit = any("guaranteed_profit" in r["adapter_flags"] for r in rows)
    vetoes = [
        name
        for name, hit in (
            ("degenerate", adapter_degenerate),
            ("collapse", adapter_collapsed),
            ("guaranteed_profit", adapter_guaranteed_profit),
        )
        if hit
    ]
    verdict = _decide_verdict(
        base_score,
        adapter_score,
        n=len(items),
        min_n=min_n,
        adapter_degenerate=adapter_degenerate,
        adapter_collapsed=adapter_collapsed,
        adapter_guaranteed_profit=adapter_guaranteed_profit,
    )
    # B1: kesik cevaplar eşit koşulda puanlanamaz → 'accept' için kanıt sayılmaz.
    denom_n = max(1, len(items))
    truncation = {"base": sum(bt), "adapter": sum(at), "max_new_tokens": EVAL_MAX_NEW_TOKENS}
    if verdict == "accept" and max(sum(bt), sum(at)) / denom_n > _MAX_TRUNCATED_SHARE:
        verdict = "inconclusive"
        vetoes = [*vetoes, "truncated"]
    return AdapterEvalResult(
        eval_set=eval_set,
        base_model=base_model,
        adapter=adapter,
        n=len(items),
        truncation=truncation,
        base_score=base_score,
        adapter_score=adapter_score,
        base_flags=base_flag_total,
        adapter_flags=adapt_flag_total,
        regression=regression,
        verdict=verdict,
        rows=rows,
        vetoes=vetoes,
    )


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
    items = _load_items(eval_set)
    if n:
        items = items[:n]
    prompts = [_prompt_for(it) for it in items]

    # 1) BASE (adapter yok) — tek tek üret, sonra belleği boşalt
    tok, model = _load_model(base_model, None)
    base_out = [_generate_checked(tok, model, p) for p in prompts]
    del model
    gc.collect()  # B11/C8: base serbest kalmadan adapter yüklenmesin (30B'de 2× bellek)

    # 2) ADAPTER (base + PEFT)
    tok, model = _load_model(base_model, str(adapter_dir))
    adapt_out = [_generate_checked(tok, model, p) for p in prompts]
    del model
    gc.collect()

    result = _score_answers(
        items,
        [a for a, _ in base_out],
        [a for a, _ in adapt_out],
        base_truncated=[c for _, c in base_out],
        adapter_truncated=[c for _, c in adapt_out],
        eval_set=Path(eval_set).stem,
        base_model=base_model,
        adapter=str(adapter_dir),
        min_n=min_n,
    )
    out = s.reports_dir / "evals" / f"adapter_eval_{Path(adapter_dir).name}_{result.eval_set}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return result
