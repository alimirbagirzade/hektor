"""Doğrulama kapıları (Gate 0-8) — LoRA dataset denetim hattı.

Her kapı bir saf fonksiyondur ve bir `GateResult` döndürür. Kapılar
sıralı çalışır; bir kapının çıktısı sonrakine girdi olabilir. Kapılar
veri reddeder veya inceleme işaretler; ağır eğitim başlatmaz.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.lora.dataset_splitter import DatasetSplit, check_leakage, split_dataset
from app.lora.domain_classifier import classify_domains
from app.lora.math_verifier import verify_math_content
from app.lora.quality_filter import QualityFilter
from app.lora.safety_scanner import scan_for_secrets

# Felsefe/mantık tutarsızlık işaretleri (Gate 6).
CONTRADICTION_MARKERS: list[str] = [
    "çelişki",
    "contradiction",
    "tutarsız",
    "inconsistent",
    "hem doğru hem yanlış",
    "both true and false",
]
# Korelasyon-nedensellik karışımı işareti.
CAUSALITY_MARKERS: list[str] = [
    "korelasyon nedenselliktir",
    "correlation implies causation",
    "korelasyon = nedensellik",
]

# Tavsiye / yönlendirme dili ve falsifiye-edilemez üstünlük iddiaları (Gate 6).
# CLAUDE.md kural 1: çıktı _hipotez_ + _test noktası_ olmalı, tavsiye değil.
# "can be directly applied", "Traders should…", "superior performance" gibi
# ifadeler v5 disiplin-gerilemesini besleyen kart sınıfıdır. BLOK DEĞİL; insan
# incelemesine yönlendirilir (bağlam önemli). Desenler tr_fold'lanmış metne
# (küçük harf + aksansız) karşı çalışır ve olumsuzlama-bilinçlidir
# ("not directly applicable" / "doğrudan uygulanabilir değildir" işaretlenMEZ).
ADVICE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "doğrudan-uygulama tavsiyesi",
        re.compile(r"can be directly applied|directly applicable|dogrudan uygulan\w*"),
    ),
    (
        "yönlendirme dili (… should)",
        re.compile(r"\b(?:traders?|investors?|you)\s+should\b"),
    ),
    (
        "falsifiye-edilemez üstünlük iddiası",
        re.compile(r"superior performance|demonstrated superior|ustun performans"),
    ),
]
# Eşleşmeden hemen ÖNCE/SONRA olumsuzlama → meşru (alçakgönüllü) ifade, atla.
_NEG_BEFORE_RE = re.compile(r"\b(?:not|non|cannot|can't|isn't|aren't|hardly|never)\b[\s\-]*$")
_NEG_AFTER_RE = re.compile(r"^\s*\w*\s*(?:degil|olmaz|olmad)")
_NEG_WINDOW: int = 14

# Gate 6 yumuşak-blok: incelemeli kart oranı bu eşiği aşarsa eğitim-öncesi kapı
# BAŞARISIZ olur (disiplin dili yaygınsa eğitimi durdur). 0.25 = kartların dörtte
# birinden fazlası tavsiye/aşırı-iddia dili taşıyorsa pervasif kabul edilir.
DEFAULT_PHIL_REVIEW_RATIO: float = 0.25
# Bu sayıdan az kartlı küçük batch'lerde oran gürültülü → yumuşak-blok uygulanmaz.
DEFAULT_PHIL_MIN_CARDS: int = 20


@dataclass
class GateResult:
    """Tek bir kapının sonucu."""

    gate_id: int
    name: str
    passed: bool
    skipped_count: int = 0
    rejected_count: int = 0
    review_count: int = 0
    details: list[str] = field(default_factory=list)


def _card_text(card: dict) -> str:
    """Bir karttan denetlenebilir serbest metni topla."""
    card_json = card.get("card_json")
    if isinstance(card_json, dict):
        parts = [
            str(card_json.get("title") or ""),
            str(card_json.get("summary") or ""),
            str(card_json.get("main_claim") or ""),
            str(card_json.get("trading_relevance") or ""),
            str(card_json.get("domain") or ""),
        ]
        # limitations/datasets/risk_warnings de KnowledgeCard'ın serbest-metin
        # alanları (knowledge_card_builder şeması); eskiden toplanmıyordu →
        # bu alanlardaki sır/PII Gate 7 (BLOCKER) taramasını ATLIYORDU. Şimdi
        # dahil edildi: hem güvenlik kapsamı hem domain/math/felsefe denetimi
        # tüm denetlenebilir metni görür.
        for field in (
            "methods",
            "possible_strategy_hypotheses",
            "implementation_notes",
            "limitations",
            "datasets",
            "risk_warnings",
        ):
            items = card_json.get(field) or []
            if isinstance(items, list):
                parts.extend(str(x) for x in items if x)
        formulas = card_json.get("formulas") or []
        if isinstance(formulas, list):
            for f in formulas:
                if isinstance(f, dict):
                    parts.append(str(f.get("description") or ""))
                    parts.append(str(f.get("plain") or ""))
        return "\n".join(p for p in parts if p)
    # düz örnek (messages) durumunda
    msgs = card.get("messages")
    if isinstance(msgs, list):
        return "\n".join(str(m.get("content") or "") for m in msgs)
    return ""


def gate_0_source(cards: list[dict], valid_paper_ids: set[str] | None = None) -> GateResult:
    """Kaynak bütünlüğü: approved + paper_id papers'ta VAR + domain + created_at.

    ``valid_paper_ids`` verilirse (control_plane'den ``papers`` tablosundaki gerçek
    id'ler), ``paper_id`` bu sette OLMAYAN kart 'orphan' sayılıp reddedilir: dayandığı
    makale tabloda yok → iddiası (ör. '%92 doğruluk') hiçbir kaynağa bağlanamaz
    (CLAUDE.md kural 7 — kaynak uydurma yasak). Denetim DETERMİNİSTİK (küme üyeliği,
    fuzzy değil). Set ``None`` ise (geriye-dönük uyum) varlık denetimi atlanır —
    yalnız review_status/domain/created_at bakılır.
    """
    rejected = 0
    orphan = 0
    details: list[str] = []
    for card in cards:
        if card.get("review_status") != "approved":
            rejected += 1
            details.append(f"{card.get('card_id', '?')}: review_status != approved")
            continue
        if valid_paper_ids is not None:
            pid = str(card.get("paper_id") or "")
            if pid not in valid_paper_ids:
                rejected += 1
                orphan += 1
                details.append(
                    f"{card.get('card_id', '?')}: paper_id '{pid}' papers tablosunda yok (orphan)"
                )
                continue
        text = _card_text(card)
        if not classify_domains(text):
            rejected += 1
            details.append(f"{card.get('card_id', '?')}: domain atanamadı")
            continue
        if not card.get("created_at"):
            rejected += 1
            details.append(f"{card.get('card_id', '?')}: created_at yok")
    if orphan:
        details.insert(0, f"ÖZET: {orphan} orphan kart (paper_id papers tablosunda yok)")
    return GateResult(
        gate_id=0,
        name="source",
        passed=rejected == 0,
        rejected_count=rejected,
        details=details[:20],
    )


def gate_1_schema(examples: list[dict]) -> GateResult:
    """Şema: messages formatı (system→user→assistant) ve metadata varlığı."""
    rejected = 0
    details: list[str] = []
    for i, ex in enumerate(examples):
        messages = ex.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            rejected += 1
            details.append(f"#{i}: messages eksik veya < 3")
            continue
        roles = [m.get("role") for m in messages]
        if roles[:3] != ["system", "user", "assistant"]:
            rejected += 1
            details.append(f"#{i}: rol sırası system→user→assistant değil")
            continue
        if "metadata" not in ex:
            rejected += 1
            details.append(f"#{i}: metadata yok")
    return GateResult(
        gate_id=1,
        name="schema",
        passed=rejected == 0,
        rejected_count=rejected,
        details=details[:20],
    )


def gate_2_curriculum(cards: list[dict]) -> GateResult:
    """Müfredat: difficulty 0.0-1.0 arasında atanmış mı?"""
    rejected = 0
    details: list[str] = []
    for card in cards:
        difficulty = card.get("difficulty")
        if difficulty is None or not (0.0 <= float(difficulty) <= 1.0):
            rejected += 1
            details.append(f"{card.get('card_id', '?')}: geçersiz difficulty={difficulty}")
    return GateResult(
        gate_id=2,
        name="curriculum",
        passed=rejected == 0,
        rejected_count=rejected,
        details=details[:20],
    )


def gate_3_domain(cards: list[dict]) -> GateResult:
    """Domain: her kart en az bir domaine ait mi?"""
    rejected = 0
    details: list[str] = []
    for card in cards:
        if not classify_domains(_card_text(card)):
            rejected += 1
            details.append(f"{card.get('card_id', '?')}: domain bulunamadı")
    return GateResult(
        gate_id=3,
        name="domain",
        passed=rejected == 0,
        rejected_count=rejected,
        details=details[:20],
    )


def gate_4_quality(cards: list[dict]) -> tuple[GateResult, list[dict]]:
    """Kalite: kısa/tekrarlı/duplicate içerikleri ele; temiz listeyi döndür."""
    quality_inputs = []
    for card in cards:
        card_json = card.get("card_json")
        question = ""
        answer = ""
        if isinstance(card_json, dict):
            question = f"{card_json.get('title', '')} konusunu açıkla."
            answer_parts = [
                str(card_json.get("main_claim") or ""),
                str(card_json.get("summary") or ""),
                str(card_json.get("trading_relevance") or ""),
            ]
            hypotheses = card_json.get("possible_strategy_hypotheses") or []
            if isinstance(hypotheses, list):
                answer_parts.extend(str(h) for h in hypotheses if h)
            answer = " ".join(p for p in answer_parts if p)
        quality_inputs.append({**card, "question": question, "answer": answer})

    passed, rejected = QualityFilter().filter_batch(quality_inputs)
    details = [f"{c.get('card_id', '?')}: {c.get('_quality_reason')}" for c in rejected]
    # Temiz çıktı: orijinal kart alanlarını koru (eklenen question/answer'ı at).
    clean = [{k: v for k, v in c.items() if k not in ("question", "answer")} for c in passed]
    result = GateResult(
        gate_id=4,
        name="quality",
        passed=len(rejected) == 0,
        rejected_count=len(rejected),
        details=details[:20],
    )
    return result, clean


def gate_5_math(cards: list[dict]) -> GateResult:
    """Matematik/istatistik: hesap tutarlılığı ve kırmızı bayraklar."""
    rejected = 0
    review = 0
    details: list[str] = []
    for card in cards:
        result = verify_math_content(_card_text(card))
        card_json = card.get("card_json")
        requires_check = isinstance(card_json, dict) and bool(card_json.get("requires_math_check"))
        if not result.passed:
            rejected += 1
            details.append(f"{card.get('card_id', '?')}: {'; '.join(result.issues)}")
        elif result.requires_review or requires_check:
            review += 1
            reason = "; ".join(result.issues) or "requires_math_check"
            details.append(f"{card.get('card_id', '?')}: inceleme — {reason}")
    return GateResult(
        gate_id=5,
        name="math",
        passed=rejected == 0,
        rejected_count=rejected,
        review_count=review,
        details=details[:20],
    )


def _scan_advice_language(folded: str) -> list[str]:
    """Tavsiye/yönlendirme + üstünlük iddialarını olumsuzlama-bilinçli işaretle.

    `folded`: tr_fold'lanmış metin. Her örüntü için eşleşmenin hemen öncesinde
    ya da sonrasında olumsuzlama varsa (örn. "not directly applicable") atlanır.
    Aynı örüntüden tek etiket yeter.
    """
    flags: list[str] = []
    for label, pattern in ADVICE_PATTERNS:
        for match in pattern.finditer(folded):
            before = folded[max(0, match.start() - _NEG_WINDOW) : match.start()]
            after = folded[match.end() : match.end() + _NEG_WINDOW]
            if _NEG_BEFORE_RE.search(before) or _NEG_AFTER_RE.search(after):
                continue  # olumsuzlanmış → meşru (alçakgönüllü) ifade
            flags.append(label)
            break
    return flags


def gate_6_philosophy(
    cards: list[dict],
    *,
    review_ratio_threshold: float = DEFAULT_PHIL_REVIEW_RATIO,
    min_cards_for_block: int = DEFAULT_PHIL_MIN_CARDS,
) -> GateResult:
    """Mantık/felsefe + disiplin dili: çelişki, korelasyon-nedensellik karışımı,
    tavsiye/yönlendirme ve doğrulanmamış üstünlük iddialarını işaretle.

    Bireysel kartlar BLOKLANMAZ (işaretler insan incelemesine yönlendirir).
    Ancak eğitim-öncesi YUMUŞAK-BLOK olarak: incelemeli kart oranı eşiği aşar VE
    yeterli kart varsa (`min_cards_for_block`) kapı BAŞARISIZ olur — disiplin dili
    yaygın bir korpus eğitime girmemelidir (v5 disiplin-gerilemesi dersi).
    """
    from app.lora.safety_scanner import tr_fold

    review = 0
    details: list[str] = []
    for card in cards:
        # tr_fold: büyük harf/aksanlı Türkçe işaretler (ÇELİŞKİ, TUTARSIZ) str.lower()
        # ile (İ→i+nokta) kaçıyordu — safety_scanner/math_verifier ile aynı desen.
        folded = tr_fold(_card_text(card))
        flags = [m for m in CONTRADICTION_MARKERS + CAUSALITY_MARKERS if tr_fold(m) in folded]
        flags.extend(_scan_advice_language(folded))
        if flags:
            review += 1
            details.append(f"{card.get('card_id', '?')}: {', '.join(flags)}")

    total = len(cards)
    # Küçük batch (< min_cards) → oran gürültülü, yumuşak-blok uygulanmaz (passed=True).
    passed = (review / total <= review_ratio_threshold) if total >= min_cards_for_block else True
    if not passed:
        details.insert(
            0,
            f"YUMUŞAK-BLOK: incelemeli oran {review}/{total}={review / total:.2f} > "
            f"{review_ratio_threshold:.2f} — disiplin dili yaygın, eğitim-öncesi kapı kapalı",
        )
    return GateResult(
        gate_id=6,
        name="philosophy",
        passed=passed,
        review_count=review,
        details=details[:20],
    )


def gate_7_safety(cards: list[dict]) -> GateResult:
    """Güvenlik (BLOCKER): sır, kişisel veri, finansal yönlendirme."""
    rejected = 0
    details: list[str] = []
    for card in cards:
        result = scan_for_secrets(_card_text(card))
        if not result.passed:
            rejected += 1
            details.append(f"{card.get('card_id', '?')}: {'; '.join(result.violations)}")
    return GateResult(
        gate_id=7,
        name="safety",
        passed=rejected == 0,
        rejected_count=rejected,
        details=details[:20],
    )


def gate_8_split(examples: list[dict]) -> tuple[GateResult, DatasetSplit]:
    """Dataset bölme ve sızıntı denetimi.

    Sızıntı (aynı source_id birden çok bölmede) YOK + valid/test BOŞ DEĞİL şartı.
    Boş valid/test, az sayıda benzersiz kaynakta (n_groups≤5) oluşur ve OOS garantisini
    yok eder; eski hâlde Gate 8 bunu PASS verip 'eğitime hazır + sahte OOS' raporluyordu
    (CLAUDE.md kural 2). Artık boş valid/test BLOKLAR.
    """
    split = split_dataset(examples)
    leaks = check_leakage(split)
    empty_issues: list[str] = []
    if not split.valid:
        empty_issues.append("boş valid kümesi — OOS doğrulanamaz (yetersiz benzersiz kaynak)")
    if not split.test:
        empty_issues.append("boş test kümesi — out-of-sample garantisi yok")
    details = empty_issues + list(leaks)
    details.append(f"train={len(split.train)} valid={len(split.valid)} test={len(split.test)}")
    result = GateResult(
        gate_id=8,
        name="split",
        passed=len(leaks) == 0 and not empty_issues,
        rejected_count=len(leaks) + len(empty_issues),
        details=details[:20],
    )
    return result, split
