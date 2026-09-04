"""S3 regresyon: RagExamRunner.context_precision GÖZLEMSEL alan.

RAGAS context_precision (2309.15217) retrieval-gürültü sinyali olarak eklendi; AMA
`passed` kararına KATILMAZ — korele token-proxy'yi geçme-kapısına eklemek sınır cevapları
haksız eler (CLAUDE.md Kural 2; v5-sınıfı over-tightening). Bu testler tam bunu kanıtlar.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.learning.question_generator import MasteryQuestion
from app.learning.rag_exam_runner import ExamAnswer, RagExamRunner
from app.verification.grounding_verifier import GroundingLevel


def _runner(
    *,
    answer,
    chunks,
    llm_used=True,
    cit_exists=True,
    level=GroundingLevel.SUPPORTED,
    can_answer=True,
):
    """__init__'i atla (Ollama/DB'ye dokunma); bileşenleri deterministik stub'la."""
    r = RagExamRunner.__new__(RagExamRunner)
    r._store = None
    r._answerer = SimpleNamespace(
        answer=lambda q: SimpleNamespace(answer=answer, sources=chunks, llm_used=llm_used)
    )
    r._citation_v = SimpleNamespace(verify=lambda a, c: [SimpleNamespace(exists=cit_exists)])
    r._grounding_v = SimpleNamespace(verify=lambda a, c: [SimpleNamespace(level=level)])
    r._sufficiency = SimpleNamespace(classify=lambda q, c: SimpleNamespace(can_answer=can_answer))
    return r


def _q(requires_abstention=False):
    return MasteryQuestion(
        question_id="q1",
        test_id="t1",
        paper_id="paperA",
        question_text="RSI nedir?",
        question_type="factual",
        requires_abstention=requires_abstention,
    )


def _chunks(texts, paper_id="paperA"):
    return [SimpleNamespace(paper_id=paper_id, text=t) for t in texts]


def test_context_precision_to_dict_alani_var():
    a = ExamAnswer(
        answer_id="a",
        question_id="q",
        test_id="t",
        paper_id="p",
        question_text="x",
        question_type="factual",
        requires_abstention=False,
        answer_text="y",
    )
    d = a.to_dict()
    assert "context_precision" in d
    assert d["context_precision"] == 0.0


def test_dusuk_context_precision_passed_etkilemez():
    # Cevap güçlü (atıf + grounding + yeterlilik + llm_used) → passed=True olmalı.
    answer = "RSI 14 momentum osilatorudur asiri alim satim seviyesini olcer"
    relevant = "RSI 14 momentum osilatorudur asiri alim satim seviyesini olcer deger"
    noise = "tamamen alakasiz baska konu kelimeler buraya farkli icerik dolgu metni"

    # 1 alakalı + 1 gürültü chunk → context_precision < 1.0 (gözlemsel olarak görünür)
    a1 = _runner(answer=answer, chunks=_chunks([relevant, noise]))._run_one(_q(), "paperA")
    assert a1.passed is True
    assert 0.0 < a1.context_precision < 1.0

    # Her iki chunk da gürültü → context_precision ≈ 0; AMA grounding/atıf/yeterlilik hâlâ
    # geçer durumda → passed YİNE True. context_precision'ın kapıya KATILMADIĞININ kanıtı.
    a2 = _runner(answer=answer, chunks=_chunks([noise, noise]))._run_one(_q(), "paperA")
    assert a2.passed is True
    assert a2.context_precision < 0.5


def test_yuksek_context_precision_zayif_cevabi_kurtarmaz():
    # Tüm chunk'lar cevapla birebir örtüşür → context_precision yüksek. AMA grounding
    # UNSUPPORTED (halüsinasyon) → passed=False kalmalı; yüksek precision kapıyı AÇMAZ.
    answer = "RSI 14 momentum osilatorudur asiri alim satim seviyesini olcer"
    r = _runner(answer=answer, chunks=_chunks([answer, answer]), level=GroundingLevel.UNSUPPORTED)
    a = r._run_one(_q(), "paperA")
    assert a.context_precision > 0.5  # yüksek
    assert a.passed is False  # ama halüsinasyon → geçemez (precision kurtarmadı)
