"""UnderstandingScore — objektif anlama-skoru toplayıcı.

Kaba "anlama %"nin (öz-değerlendirme/keyword) yerine, L3/L4/L5 sınavlarının
OBJEKTİF sonuçlarını tek bir geçme-oranına toplar:

    pass_rate = passed / (passed + failed)

'skipped' (LLM yok) ve 'no_data' (belirgin yön/veri yok) paydaya KATILMAZ —
ayrı raporlanır (şeffaflık; sahte yüksek skor üretmeyiz). Hiç notlanan sınav
yoksa skor 'insufficient_data' bayrağıyla döner (CLAUDE.md Kural 2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from app.verification.exams.l3_application import ExamResult
from app.verification.exams.l5_composition import CompositionResult

__all__ = [
    "RagAnswerLike",
    "UnderstandingScore",
    "aggregate",
    "composition_to_result",
    "l5_example_result",
    "l5_results_from_sessions",
    "rag_answers_to_results",
    "run_rag_ladder_answers",
    "score_full_ladder",
    "score_indicator_exams",
]


class RagAnswerLike(Protocol):
    """RagExamRunner.ExamAnswer'ın yapısal arabirimi (ağır import'tan kaçınmak için)."""

    question_type: str
    requires_abstention: bool
    answer_text: str
    citation_score: float
    grounding_score: float
    abstention_correct: bool
    hallucination_detected: bool


@dataclass
class UnderstandingScore:
    total: int
    passed: int
    failed: int
    skipped: int
    no_data: int
    graded: int  # passed + failed (paydanın tamamı)
    pass_rate: float | None  # passed/graded (0-1); graded < min eşik ise None
    status: str  # "scored" | "insufficient_data"
    by_level: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Backtest'in ANLAMA başarısızlığı değil "TEST EDİLEMEDİ" olduğunu gösteren işaretler:
# veri yok / çok az işlem (örneklem yetersiz) / belirsiz verdict. Bunlar 'skipped'
# olmalı (paydaya girmez) — yoksa sentetik veride trade üretemeyen sabit örnek
# haksızca 'failed' sayılıp pass_rate'i yapay olarak %0'a çeker (bilinen zayıflık).
_L5_UNTESTABLE_MARKERS = ("veri yok", "az işlem", "yetersiz", "inconclusive", "belirsiz")


def _backtest_untestable(comp: CompositionResult) -> bool:
    return any(
        g.gate == "backtest"
        and not g.passed
        and any(any(m in d.lower() for m in _L5_UNTESTABLE_MARKERS) for d in g.details)
        for g in comp.gates
    )


def composition_to_result(comp: CompositionResult) -> ExamResult:
    """L5 CompositionResult'ı ortak ExamResult biçimine indirger.

    aday → passed; backtest SADECE test-edilemediği için (veri yok / çok az işlem /
    belirsiz) reddedildiyse → skipped (paydaya girmez, "test edilmeden hazır deme");
    math/novelty düştüyse veya backtest substantif disiplin/metrik sebebiyle düştüyse
    → failed (gerçek anlama/üretim başarısızlığı).
    """
    if comp.candidate:
        status = "passed"
    else:
        non_backtest_ok = all(g.passed for g in comp.gates if g.gate != "backtest")
        status = "skipped" if (non_backtest_ok and _backtest_untestable(comp)) else "failed"
    return ExamResult(
        level="L5",
        name=comp.name,
        passed=comp.candidate,
        status=status,
        seed=0,
        detail={"gates": [g.gate for g in comp.gates if not g.passed]},
    )


def rag_answers_to_results(answers: list[RagAnswerLike]) -> list[ExamResult]:
    """RAG sınavı (citation/grounding/abstention) cevaplarını merdiven sonuçlarına çevirir.

    - abstention soruları → "Taban" (Dürüstlük): doğru çekimserlik mi.
    - diğerleri → L1 (Çıkarım/retrieval: doğru kaynağı buldu/alıntıladı mı) + L2
      (Sadakat: cevap dayanaklı mı, halüsinasyon yok mu).
    Cevap yoksa (retrieval boş) → L1/L2 'no_data' (test edilemedi, paydaya girmez).
    """
    out: list[ExamResult] = []
    for a in answers:
        no_answer = not (a.answer_text or "").strip()
        if a.requires_abstention:
            p = bool(a.abstention_correct)
            out.append(_mk("Taban", a.question_type, p, {"abstention_correct": p}))
            continue
        if no_answer:
            out.append(_mk_status("L1", a.question_type, "no_data"))
            out.append(_mk_status("L2", a.question_type, "no_data"))
            continue
        l1 = a.citation_score >= 0.3
        out.append(_mk("L1", a.question_type, l1, {"citation_score": a.citation_score}))
        l2 = a.grounding_score >= 0.4 and not a.hallucination_detected
        out.append(
            _mk(
                "L2",
                a.question_type,
                l2,
                {"grounding_score": a.grounding_score, "hallucination": a.hallucination_detected},
            )
        )
    return out


def _mk(level: str, name: str, passed: bool, detail: dict[str, Any]) -> ExamResult:
    return ExamResult(
        level=level,
        name=name,
        passed=passed,
        status="passed" if passed else "failed",
        seed=0,
        detail=detail,
    )


def _mk_status(level: str, name: str, status: str) -> ExamResult:
    return ExamResult(level=level, name=name, passed=False, status=status, seed=0, detail={})


def _indicator_exam_results(seed: int = 0, *, llm: Any = None) -> list[ExamResult]:
    """L3+L4 sınavlarını tüm registry spec'lerinde koşar → ham ExamResult listesi.

    LLM gerektirir; çevrimdışıysa sınavlar 'skipped' olur (sahte pass üretilmez,
    CLAUDE.md Kural 2). ``llm`` enjekte edilirse O model ölçülür (base yerine ADAPTER
    ölçümü için DİKİŞ — v5-savunması: base-vs-adapter anlama kıyası); ``None`` → LocalLLM
    (base). ``score_indicator_exams`` ve ``score_full_ladder`` bunu kullanır.
    """
    from app.brain.local_llm import LLMUnavailable
    from app.verification.exams.l3_application import ApplicationExam
    from app.verification.exams.l4_counterfactual import CounterfactualExam
    from app.verification.exams.registry import list_specs

    l3 = ApplicationExam(llm=llm)
    l4 = CounterfactualExam(llm=llm)
    results: list[ExamResult] = []
    # ÖN-KONTROL: LLM hiç erişilebilir değilse (gerçekten offline) boşuna timeout
    # bekleme — hepsini hızlıca 'skipped'. (available() = hızlı /api/tags.)
    from app.brain.local_llm import LocalLLM

    if not (llm or LocalLLM()).available():
        for spec in list_specs():
            for level in ("L3", "L4"):
                results.append(_mk_status(level, spec.name, "skipped"))
        return results
    # LLM ERİŞİLEBİLİR: tek bir zaman aşımı TÜM sınavları öldürmesin (eski cascade hatası:
    # ilk timeout → llm_down → 10 sınav skip → pass_rate hep 0). Yalnız 2 ARDIŞIK
    # başarısızlıkta (gerçekten asılı) bail. Yavaş CPU'da her sınava şans tanı.
    consecutive_skips = 0
    for spec in list_specs():
        for level, exam in (("L3", l3), ("L4", l4)):
            if consecutive_skips >= 2:
                results.append(
                    ExamResult(
                        level,
                        spec.name,
                        False,
                        "skipped",
                        seed,
                        {"reason": "LLM ardışık 2 başarısızlık — kalan sınavlar atlandı"},
                    )
                )
                continue
            try:
                res = exam.run(spec, seed=seed)
            except LLMUnavailable:
                res = ExamResult(
                    level, spec.name, False, "skipped", seed, {"reason": "LLMUnavailable"}
                )
            except Exception as exc:  # beklenmedik hata tek sınavı atlasın, uç çökmesin
                res = ExamResult(
                    level,
                    spec.name,
                    False,
                    "skipped",
                    seed,
                    {"reason": f"beklenmedik hata: {type(exc).__name__}: {exc}"[:200]},
                )
            results.append(res)
            consecutive_skips = consecutive_skips + 1 if res.status == "skipped" else 0
    return results


def score_indicator_exams(seed: int = 0) -> UnderstandingScore:
    """L3+L4 gösterge sınav-geçme-oranı (geri-uyum; web /api/understanding-score varsayılanı)."""
    return aggregate(_indicator_exam_results(seed))


def l5_example_result(seed: int = 42) -> ExamResult:
    """L5 KOMPOZİSYON — örnek IR'ı sentetik veride math+novelty+backtest kapısından geçir.

    Deterministik ve LLM'siz (sentetik OHLCV seed'li) → çevrimdışı bile NOTLANIR;
    merdivenin LLM gerektirmeyen tek objektif sinyalidir. Hata olursa 'skipped' döner.
    """
    from app.trading.market_data_loader import generate_synthetic_ohlcv
    from app.trading.strategy_ir import example_ir
    from app.verification.exams.l5_composition import CompositionGate

    df = generate_synthetic_ohlcv(n=2000, seed=seed)
    comp = CompositionGate().evaluate_composition(example_ir(), df)
    return composition_to_result(comp)


def l5_results_from_sessions(store: Any = None, *, limit: int = 5) -> list[ExamResult]:
    """L5 — sistemin KENDİ ürettiği kompozisyonların (research_sessions) GERÇEK verdict'leri.

    Çekirdek fikri doğrudan ölçer: "anladı = test edilebilir YENİ bir şey üretebildi".
    Yalnız bir kompozisyon (strategy_ir/proposed_indicator) ÖNERİLMİŞ oturumlar sayılır:
    verdict=pass → passed; fail → failed; inconclusive/boş → skipped (test edilemedi,
    paydaya girmez). Hiç uygun oturum yoksa boş döner (çağıran örnek-kompozisyona düşebilir).
    Sabit ``example_ir`` yerine zamanla DEĞİŞEN gerçek bir sinyal sağlar.
    """
    from app.memory.sqlite_store import SqliteStore

    store = store or SqliteStore()
    try:
        sessions = store.list_research_sessions(limit=limit)
    except Exception:
        return []
    out: list[ExamResult] = []
    for s in sessions:
        if not (s.get("strategy_ir") or s.get("proposed_indicator")):
            continue  # kompozisyon önerilmemiş oturum → L5 sinyali değil
        verdict = (s.get("verdict") or "").lower()
        if verdict == "pass":
            status, passed = "passed", True
        elif verdict == "fail":
            status, passed = "failed", False
        else:  # inconclusive / boş → test edilemedi (paydaya girmez)
            status, passed = "skipped", False
        name = str(s.get("question") or s.get("session_id") or "kompozisyon")[:60]
        out.append(
            ExamResult(
                "L5",
                name,
                passed,
                status,
                0,
                {"session_id": s.get("session_id"), "verdict": verdict or None},
            )
        )
    return out


def run_rag_ladder_answers(
    store: Any = None, *, max_papers: int = 1, count: int = 8
) -> list[RagAnswerLike]:
    """Taban/L1/L2 için canlı RAG sınavı koştur (kartı olan makaleler üzerinde).

    Soru üretimi LLM'siz (şablon); cevap + doğrulama RagAnswerer/verifier'larla. LLM
    çevrimdışıysa cevaplar boş → L1/L2 'no_data' (paydaya girmez), Taban değerlendirilir.
    """
    from app.learning.question_generator import QuestionGenerator
    from app.learning.rag_exam_runner import RagExamRunner
    from app.memory.sqlite_store import SqliteStore

    store = store or SqliteStore()
    qg = QuestionGenerator(store)
    runner = RagExamRunner(store)
    answers: list[RagAnswerLike] = []
    carded = [p for p in store.list_papers() if store.has_knowledge_card(p.paper_id)]
    for paper in carded[:max_papers]:
        questions = qg.generate(paper.paper_id, test_id="ladder", count=count)
        answers.extend(runner.run(questions, paper.paper_id))
    return answers


def score_full_ladder(
    seed: int = 0,
    *,
    store: Any = None,
    with_rag: bool = False,
    rag_answers: list[RagAnswerLike] | None = None,
    l5_seed: int = 42,
    llm: Any = None,
    use_sessions_l5: bool = True,
) -> UnderstandingScore:
    """Tam merdiven: L5 + L3/L4 (LLM) + opsiyonel Taban/L1/L2 (RAG).

    - **L5 önceliği:** sistemin GERÇEK ürettiği kompozisyonlar (research_sessions);
      yoksa deterministik örnek-kompozisyon (fallback). Sabit örnek artık tek sinyal değil.
    - ``llm`` enjekte edilirse L3/L4 O modeli ölçer (base yerine ADAPTER — v5-savunması dikişi).
    - ``rag_answers`` verilirse onlar; ``with_rag=True`` ise canlı RAG (başarısızsa skipped).
    Hata yutulur → skipped (uç çökmez, Kural 2). Çevrimdışı bile L5 notlanır.
    """
    results: list[ExamResult] = []
    # L5 — önce sistemin gerçek kompozisyonları, yoksa örnek (deterministik fallback)
    l5: list[ExamResult] = []
    if use_sessions_l5:
        try:
            l5 = l5_results_from_sessions(store)
        except Exception:
            l5 = []
    if not l5:
        try:
            l5 = [l5_example_result(seed=l5_seed)]
        except Exception as exc:  # L5 hatası tüm merdiveni çökermesin
            l5 = [
                ExamResult(
                    "L5",
                    "ornek_kompozisyon",
                    False,
                    "skipped",
                    l5_seed,
                    {"reason": f"L5 koşulamadı: {type(exc).__name__}: {exc}"[:200]},
                )
            ]
    results.extend(l5)
    results.extend(_indicator_exam_results(seed, llm=llm))
    if rag_answers is not None:
        results.extend(rag_answers_to_results(rag_answers))
    elif with_rag:
        try:
            results.extend(rag_answers_to_results(run_rag_ladder_answers(store)))
        except Exception as exc:  # corpus/LLM yoksa merdiven yine de döner
            reason = f"RAG sınavı koşulamadı: {type(exc).__name__}"
            for lvl in ("Taban", "L1", "L2"):
                results.append(ExamResult(lvl, "rag", False, "skipped", seed, {"reason": reason}))
    return aggregate(results)


# Notlanan (passed+failed) sınav sayısı bu eşiğin ALTINDAysa skor güvenilmez sayılır:
# pass_rate=None + status='insufficient_data'. Neden: çevrimdışı tam-merdivende L3/L4
# 'skipped' olur, yalnız deterministik L5 örneği notlanır (graded=1) → tek geçen sınav
# pass_rate=%100 + 'scored' verip auto_pipeline base-vs-adapter TERFİ kapısını şişirirdi
# (v5-gerileme sınıfı küçük-n sahte-güveni; adapter eval min_n=5 ile aynı ilke). Eşik=3,
# mevcut 'graded=3 → scored' sözleşmesini korur (test_pass_rate_skipped_haric). 7 registry
# spec'i online L3/L4'te 14'e kadar graded üretir → meşru skor bastırılmaz.
_MIN_GRADED_FOR_SCORE = 3


def aggregate(
    results: list[ExamResult], *, min_graded: int = _MIN_GRADED_FOR_SCORE
) -> UnderstandingScore:
    by_level: dict[str, dict[str, int]] = {}
    counts = {"passed": 0, "failed": 0, "skipped": 0, "no_data": 0}

    for r in results:
        status = r.status if r.status in counts else "no_data"
        counts[status] += 1
        lvl = by_level.setdefault(r.level, {"passed": 0, "failed": 0, "skipped": 0, "no_data": 0})
        lvl[status] += 1

    graded = counts["passed"] + counts["failed"]
    scored = graded >= min_graded
    pass_rate = (counts["passed"] / graded) if scored else None
    return UnderstandingScore(
        total=len(results),
        passed=counts["passed"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        no_data=counts["no_data"],
        graded=graded,
        pass_rate=pass_rate,
        status="scored" if scored else "insufficient_data",
        by_level=by_level,
    )
