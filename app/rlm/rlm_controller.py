"""RLM Controller — çok-adımlı, kaynaklı, denetimli cevap orkestratörü.

Mevcut RAG retrieval + doğrulama modüllerini bir reasoning kontrol katmanında
birleştirir (talimat §11). Yeni bilgi deposu DEĞİLDİR.

Akış:
    classify → plan → (retrieval ⇄ reformulation)* → evidence gate → draft (LLM)
    → claim extraction → citation/grounding verify → contradiction → confidence
    → abstention → yapısal nihai cevap → run logları + JSON rapor

Mutlak kurallar (CLAUDE.md):
  1. Canlı trading sinyali / yatırım tavsiyesi ASLA. Trading-içerikli her çıktıda
     (soru VEYA cevap trading dili taşıyorsa) zorunlu uyarı bloğu eklenir — karar
     classifier görev tipine değil, çıktı içeriğine bağlıdır (_apply_trading_guard).
  4. Desteklenmeyen iddia nihai cevaba KONMAZ.
  6. Determinizm: LLM çağrıları sabit seed ile (Ollama/OpenAI/Google). Anthropic
     API seed desteklemez → o backend'de en yakın determinizm için temperature=0.0.
  7. Eksik bağlamda uydurma YOK → "yeterli kaynak yok" denir.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.brain.prompt_loader import load_prompt
from app.config import get_settings
from app.memory.reranking_retriever import RerankingRetriever
from app.memory.retrieval_service import RetrievedChunk, Retriever
from app.rlm.claim_extractor import Claim, extract_claims
from app.rlm.evidence_builder import EvidenceReport, EvidenceSufficiencyScorer
from app.rlm.rlm_store import RlmStore
from app.rlm.task_classifier import ReasoningPlan, TaskClassifier
from app.verification.abstention_policy import AbstentionPolicy
from app.verification.citation_verifier import CitationVerifier
from app.verification.confidence_scorer import ConfidenceScorer
from app.verification.context_sufficiency import ContextSufficiencyClassifier
from app.verification.contradiction_detector import Contradiction, ContradictionDetector
from app.verification.grounding_verifier import GroundingVerifier

log = logging.getLogger(__name__)

_FALLBACK_SYSTEM = (
    "Yalnızca verilen KAYNAKLAR'a dayan; kaynak yoksa 'kaynak bulunamadı' de, uydurma. "
    "Her iddiadan sonra [paper_id:chunk_id] satır-içi atıf ver. Yatırım tavsiyesi verme."
)

# Zorunlu uyarı bloğunun başlığı — idempotens kontrolü bunu VE gövdeyi birlikte arar.
_TRADING_GUARD_HEADER = "Trading uyarıları:"

_TRADING_DISCLAIMER = (
    "- Bu yatırım tavsiyesi değildir.\n"
    "- Bu canlı sinyal değildir.\n"
    "- Hipotez backtest gerektirir (OOS + komisyon + slippage + look-ahead yok).\n"
    "- Kullanılan makalelerin veri seti ve zaman aralığı kontrol edilmelidir."
)

# Trading-içerik tespiti (kural 1). Soru VEYA nihai cevap bunlardan birini taşıyorsa
# zorunlu uyarı bloğu eklenir. Görev sınıflandırıcıdan BAĞIMSIZ — classifier trading
# sorusunu MATH/MULTI/UNCERTAINTY'ye düşürse bile uyarı kaçmaz.
# Terimler SİNYAL/STRATEJİ-özgü tutuldu; jenerik İngilizce kelimeler (return/long/short/
# position) BİLEREK çıkarıldı — yoksa "return" geçen her akademik cevap yanlış-bağlam
# yatırım uyarısı taşır ve uyarı anlamını yitirir (sürekli-uyarı = göz ardı edilen uyarı).
_TRADING_SIGNAL_RE = re.compile(
    r"\b(trading|strateji\w*|strategy|sinyal\w*|signal\w*|al-?sat|buy|sell|"
    r"momentum|mean.?reversion|volatilit\w*|backtest\w*|getiri|sharpe|sortino|"
    r"pozisyon|portföy|portfolio|drawdown|kelly)\b",
    re.IGNORECASE,
)

# Nihai cevap GÖVDESİNE giren ham LLM cümlesindeki satır-içi [paper_id:chunk_id] atıfları:
# getirilen chunk kümesinde OLMAYAN (uydurma) chunk_id'leri ÇIKAR (kural 7 — uydurma kaynak
# yasak). Geçerli atıflar korunur; gövde böylece "Kaynak dayanakları" bloğuyla tutarlı kalır.
_INLINE_CITATION_RE = re.compile(r"\s*\[([^:\]]+):([^\]]+)\]")
_CIT_PAGE_SUFFIX_RE = re.compile(r",\s*s\.\d+\s*$")


def _sanitize_citations(text: str, valid_chunk_ids: set[str]) -> str:
    """Metindeki geçersiz (getirilen sette olmayan) satır-içi atıf işaretlerini çıkar."""

    def _repl(m: re.Match[str]) -> str:
        chunk_id = _CIT_PAGE_SUFFIX_RE.sub("", m.group(2).strip()).strip()
        return m.group(0) if chunk_id in valid_chunk_ids else ""

    return _INLINE_CITATION_RE.sub(_repl, text).strip()


def apply_trading_guard(answer: str, query: str, *, allow_live_signal: bool = False) -> str:
    """Kural 1 MOTOR-BAĞIMSIZ tek-nokta yaptırımı: canlı sinyal/yatırım tavsiyesi ASLA.

    Soru VEYA cevap trading dili taşıyorsa ve uyarı henüz yoksa zorunlu uyarı bloğu eklenir.
    Idempotent (uyarı zaten varsa çiftlemez) → native yol kendi içinde uygulasa bile cevap
    pipeline'da yeniden geçirilebilir. `allow_live_signal` MUTLAK False'tur; biri yanlışlıkla
    True yapsa dahi sistem canlı sinyal üretmez, yalnızca bu (kozmetik olmayan) uyarıyı atlardı.

    `RlmController` bunu `_synthesize` içinde uygular; `insufficient_evidence` erken
    dönüşleri de dahil TÜM çıkış yollarından geçer (eskiden erken-dönüşler atlıyordu
    → kural-1 sızıntısı).
    """
    if allow_live_signal:
        return answer
    # Idempotens kontrolü BLOĞUN KENDİSİNE bakar, jenerik bir alt-dizgeye değil
    # (Kademe-2 av bulgusu, 2026-09-07). Eski kontrol serbest bir cümle arıyordu;
    # o cümle modelin metninde ya da alıntılanan makale parçasında geçebiliyor —
    # o durumda zorunlu uyarının KALAN ÜÇ SATIRI (canlı sinyal değil / backtest
    # gerekir / veri aralığı kontrol edilmeli) hiç eklenmiyor ve Kural 1 yaptırımı
    # sessizce eksik uygulanıyordu.
    if _TRADING_GUARD_HEADER in answer and _TRADING_DISCLAIMER in answer:
        return answer
    if not (_TRADING_SIGNAL_RE.search(query) or _TRADING_SIGNAL_RE.search(answer)):
        return answer
    return f"{answer}\n\n{_TRADING_GUARD_HEADER}\n{_TRADING_DISCLAIMER}"


def _has_content(text: str) -> bool:
    """Atıf temizliği sonrası iddia gerçek içerik (harf/rakam) taşıyor mu?

    Yalnız geçersiz/uydurma satır-içi atıftan ibaret bir iddia sanitize edilince
    geriye sadece noktalama kalır (ör. '.'); bu boş-içerikli iddia gövdeye
    konursa "Kısa cevap" tek bir noktaya çökerdi (kural 7 + degenerate-cevap).
    """
    return any(ch.isalnum() for ch in text)


@dataclass
class RlmResult:
    """RLM koşu sonucu (CLI/API/rapor için)."""

    run_id: str
    query: str
    task_type: str
    status: str  # answered / answered_with_limitation / abstained / no_llm
    final_answer: str
    final_confidence: float
    confidence_level: str  # High / Medium / Low
    evidence_score: float
    retrieval_rounds: int
    n_sources: int
    supported_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    report_path: str | None = None


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        head = c.citation
        if c.title:
            head += f" — {c.title}"
        blocks.append(f"{head}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def _merge_chunks(
    existing: list[RetrievedChunk], new: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """chunk_id'ye göre dedup ederek birleştir (sıra korunur)."""
    seen = {c.chunk_id for c in existing}
    merged = list(existing)
    for c in new:
        if c.chunk_id not in seen:
            merged.append(c)
            seen.add(c.chunk_id)
    return merged


class RlmController:
    """Recursive/Reasoning LM kontrol katmanı."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LocalLLM | None = None,
        store: RlmStore | None = None,
    ) -> None:
        self.settings = get_settings()
        self.retriever = retriever or RerankingRetriever()
        self.llm = llm or LocalLLM()
        self.store = store or RlmStore()
        self.seed = self.settings.rlm_seed

        self.classifier = TaskClassifier()
        self.evidence_scorer = EvidenceSufficiencyScorer()
        self.citation_verifier = CitationVerifier()
        self.grounding_verifier = GroundingVerifier()
        self.context_classifier = ContextSufficiencyClassifier()
        self.contradiction_detector = ContradictionDetector()
        self.confidence_scorer = ConfidenceScorer()
        self.abstention_policy = AbstentionPolicy()

    # ----------------------------------------------------------------- public

    def answer(
        self,
        query: str,
        *,
        paper_ids: list[str] | None = None,
        top_k: int | None = None,
        max_rounds: int | None = None,
        write_report: bool = True,
    ) -> RlmResult:
        top_k = top_k or self.settings.rag_top_k
        max_rounds = max_rounds or self.settings.rlm_max_retrieval_rounds

        task_type = self.classifier.classify(query, paper_ids)
        plan = self.classifier.plan(
            task_type, n_papers=len(paper_ids) if paper_ids else 0, max_rounds=max_rounds
        )
        model_name = self.llm.active_backend()
        # Her yeni run'da dış-kill/crash ile asılı kalmış eski 'running' run'ları temizle.
        self.store.mark_stale_running_failed()
        run_id = self.store.create_run(query, task_type, model_name)

        try:
            return self._execute(
                run_id, query, task_type, plan, paper_ids, top_k, model_name, write_report
            )
        except Exception as exc:
            # Beklenmeyen hata (Chroma/embedding/IO vb.) → run'ı 'running' asılı BIRAKMA:
            # 'failed' işaretle ki rlm-runs/audit gerçeği yansıtsın, sonra yeniden fırlat.
            # KRİTİK: yalnız run HÂLÂ 'running' ise işaretle. finish_run ZATEN terminal bir
            # durum (answered/abstained/no_llm) yazdıysa ve sonrasındaki bir bookkeeping
            # (ör. _log_evidence_rows içinde 'database is locked') patladıysa, gerçek cevabı
            # 'failed' ile EZME (audit bütünlüğü — aksi halde başarılı run sahte 'failed' olurdu).
            # KRİTİK-2: get_run/finish_run'ı kendi try'ına al. Bunlar da AYNI kilide
            # ('database is locked' — guard'ın motive olduğu senaryo) takılabilir; takılırsa
            # orijinal kök-neden hatasını MASKELEME (her durumda `raise`). Asılı 'running'
            # kalırsa reaper (mark_stale_running_failed) backstop'u temizler.
            try:
                current = self.store.get_run(run_id)
                if current is not None and current["status"] == "running":
                    self.store.finish_run(
                        run_id,
                        status="failed",
                        final_answer=f"[RLM hata: {type(exc).__name__}: {exc}]",
                        final_confidence=0.0,
                        evidence_score=0.0,
                    )
            except Exception:
                log.warning(
                    "RLM run 'failed' işaretlenemedi (run=%s) — reaper temizleyecek",
                    run_id,
                    exc_info=True,
                )
            raise

    def _execute(
        self,
        run_id: str,
        query: str,
        task_type: str,
        plan: ReasoningPlan,
        paper_ids: list[str] | None,
        top_k: int,
        model_name: str,
        write_report: bool,
    ) -> RlmResult:
        self.store.add_step(
            run_id,
            1,
            "classify",
            input_text=query,
            output_text=task_type,
            tool_used="TaskClassifier",
        )
        self.store.add_step(
            run_id, 2, "plan", output_text=json.dumps(asdict(plan), ensure_ascii=False)
        )

        # --- Çok-turlu retrieval + kanıt yeterlilik kapısı --------------------
        chunks, evidence, rounds_used = self._gather_evidence(run_id, query, plan, paper_ids, top_k)

        # Kanıt çok zayıf → uydurma; abstain (kural 7).
        if not chunks or evidence.decision == "insufficient":
            return self._finish_insufficient(
                run_id, query, task_type, evidence, rounds_used, chunks
            )

        # --- Taslak cevap (LLM) ----------------------------------------------
        draft, llm_used = self._draft(query, chunks, plan, backend=model_name)
        self.store.add_step(
            run_id,
            10,
            "draft",
            output_text=(draft[:2000] if llm_used else "[LLM çevrimdışı]"),
            tool_used=f"LocalLLM({model_name})",
        )

        if not llm_used:
            return self._finish_no_llm(
                run_id, query, task_type, evidence, rounds_used, chunks, write_report
            )

        # --- Doğrulama: atıf + dayanak + çelişki -----------------------------
        citations = self.citation_verifier.verify(draft, chunks)
        groundings = self.grounding_verifier.verify(draft, chunks)
        contradictions = self.contradiction_detector.detect(chunks)
        sufficiency = self.context_classifier.classify(query, chunks)
        confidence = self.confidence_scorer.score(
            sufficiency, citations, groundings, contradictions
        )
        abstention = self.abstention_policy.decide(confidence, sufficiency)

        claims = extract_claims(groundings)
        supported = [c for c in claims if c.is_supported]
        unsupported = [c for c in claims if not c.is_supported]

        self.store.add_step(
            run_id,
            11,
            "verify",
            output_text=(
                f"citation={confidence.citation_score} grounding={confidence.grounding_score} "
                f"supported={len(supported)} unsupported={len(unsupported)} "
                f"contradictions={len(contradictions)}"
            ),
            tool_used="Citation+Grounding+Contradiction",
        )

        # --- Nihai cevap sentezi ---------------------------------------------
        return self._synthesize(
            run_id=run_id,
            query=query,
            task_type=task_type,
            chunks=chunks,
            evidence=evidence,
            rounds_used=rounds_used,
            supported=supported,
            unsupported=unsupported,
            contradictions=contradictions,
            confidence_score=confidence.score,
            citation_score=confidence.citation_score,
            grounding_score=confidence.grounding_score,
            context_score=confidence.context_score,
            decision=confidence.decision,
            abstain=abstention.should_abstain,
            abstain_reason=abstention.reason,
            write_report=write_report,
        )

    # ----------------------------------------------------------------- helpers

    def _gather_evidence(
        self,
        run_id: str,
        query: str,
        plan: ReasoningPlan,
        paper_ids: list[str] | None,
        top_k: int,
    ) -> tuple[list[RetrievedChunk], EvidenceReport, int]:
        chunks: list[RetrievedChunk] = []
        evidence = self.evidence_scorer.score(query, [], [])
        rounds = max(1, plan.retrieval_rounds)
        used = 0
        for r in range(rounds):
            used = r + 1
            q = self._reformulate(query, plan, r)
            new = self.retriever.retrieve(q, top_k=top_k)
            if paper_ids:
                new = [c for c in new if c.paper_id in paper_ids]
            chunks = _merge_chunks(chunks, new)
            contradictions = self.contradiction_detector.detect(chunks)
            evidence = self.evidence_scorer.score(
                query,
                chunks,
                contradictions,
                min_to_answer=self.settings.rlm_min_evidence_to_answer,
                min_to_skip_retry=self.settings.rlm_min_evidence_to_skip_retry,
                min_to_retry=self.settings.rlm_min_evidence_to_retry,
            )
            self.store.add_step(
                run_id,
                3 + r,
                "retrieval",
                input_text=q,
                output_text=f"round={used} chunks={len(chunks)} evidence={evidence.score} "
                f"decision={evidence.decision}",
                tool_used="RerankingRetriever",
            )
            # Yeterli → erken çık. Tekrar gerekmiyorsa veya reformülasyon kapalıysa dur.
            # NOT: "yeni chunk yok → dur" guard'ı KALDIRILDI — _reformulate her tur FARKLI
            # bölüm hedeflediğinden (methodology→findings→…) bir tur yeni chunk getirmese
            # bile sonraki FARKLI-bölüm turu yeni chunk yüzeye çıkarabilir; guard o turu
            # haksız yere kesip section-diversity'yi iptal ederdi. Tur sayısı zaten
            # max_rounds ile sınırlı; farklı sorgular boşa-tur değildir.
            if evidence.decision in ("answer", "answer_with_limitation"):
                break
            if not self.settings.rlm_enable_query_reformulation:
                break
        return chunks, evidence, used

    def _reformulate(self, query: str, plan: ReasoningPlan, round_idx: int) -> str:
        """Deterministik (LLM-free) sorgu yeniden-formülasyonu: ilk tur orijinal; sonraki
        turlarda plan.must_include'dan DÖNÜŞÜMLÜ TEK bölüm anahtarı ekler → her tur farklı
        bölüme (methodology→findings→limitations…) yönelir.

        Eskiden her tur ' '.join(must_include) ile AYNI sorguyu üretiyordu → deterministik
        retriever aynı chunk'ları getirip 3-turlu görevlerde 3. tur boşa gidiyordu (ölü iş +
        yanıltıcı 'round=3' audit). Tur-başına farklı bölüm anahtarı bu redundansı kapatır."""
        if round_idx == 0 or not plan.must_include:
            return query
        section = plan.must_include[(round_idx - 1) % len(plan.must_include)]
        return f"{query} {section}"

    def _draft(
        self, query: str, chunks: list[RetrievedChunk], plan: ReasoningPlan, *, backend: str = ""
    ) -> tuple[str, bool]:
        context = _format_context(chunks)
        try:
            system = load_prompt("rag_answer")
        except FileNotFoundError:
            system = _FALLBACK_SYSTEM
        prompt = f"SOURCES / KAYNAKLAR:\n{context}\n\nQUESTION / SORU: {query}"
        # Determinizm (kural 6): Anthropic API seed desteklemez → orada temperature=0.0
        # ile en yakın tekrarlanabilirlik; seed onurlanan backend'lerde 0.2 + seed.
        temperature = 0.0 if backend == "anthropic" else 0.2
        try:
            # SINIRLI çağrı: max_tokens üretimi, timeout wall-clock'u bağlar. Timeout
            # aşılırsa generate() LLMUnavailable fırlatır → no_llm yolu (graceful degrade,
            # dangling 'running' yok). Yavaş CPU'da sınırsız çağrı 9dk+ sürüyordu.
            text = self.llm.generate(
                prompt,
                system=system,
                temperature=temperature,
                seed=self.seed,
                max_tokens=self.settings.rlm_draft_max_tokens,
                timeout=self.settings.rlm_draft_timeout_s,
            )
            return text, True
        except LLMUnavailable:
            return "", False

    # ---- sonuç yolları -------------------------------------------------------

    def _sources_payload(self, chunks: list[RetrievedChunk]) -> list[dict]:
        return [
            {
                "paper_id": c.paper_id,
                "chunk_id": c.chunk_id,
                "citation": c.citation,
                "section": c.section_name,
                "page": c.page_number,
                "title": c.title,
            }
            for c in chunks
        ]

    def _log_evidence_rows(
        self, run_id: str, chunks: list[RetrievedChunk], used_ids: set[str]
    ) -> None:
        # EN-İYİ-ÇABA: kanıt satırları audit logudur; finish_run SONRASI çalışır. Tek bir
        # add_evidence hatası (ör. 'database is locked') cevabı DÜŞÜRMEMELİ — aksi halde
        # hata answer()'ın except'ine sızıp başarılı run'ı 'failed' ezerdi (regresyon).
        for c in chunks:
            try:
                self.store.add_evidence(
                    run_id,
                    c.paper_id,
                    c.chunk_id,
                    # distance=0.0 MÜKEMMEL eşleşme (relevance 1.0); None = bilinmiyor (0.0).
                    # Eski `if c.distance` truthiness'i 0.0'ı falsy görüp en iyi eşleşmeyi
                    # en kötü skorla loglardı. max(0.0,…) negatif olmayan metrik garantisi.
                    relevance_score=(
                        round(max(0.0, 1.0 - c.distance), 4) if c.distance is not None else 0.0
                    ),
                    used_in_final_answer=c.chunk_id in used_ids,
                )
            except Exception:  # audit logu cevabı bloke etmemeli (best-effort)
                # exc_info: yutulan hata (gerçek programlama hatası olabilir) loglarda izlenebilsin.
                log.warning(
                    "RLM evidence logu yazılamadı (run=%s chunk=%s)",
                    run_id,
                    c.chunk_id,
                    exc_info=True,
                )

    def _finish_insufficient(
        self,
        run_id: str,
        query: str,
        task_type: str,
        evidence: EvidenceReport,
        rounds_used: int,
        chunks: list[RetrievedChunk],
    ) -> RlmResult:
        msg = (
            "Bu makalelerde bu soruya yeterli kaynak yok.\n\n"
            "Kanıt yeterlilik skoru çok düşük (uydurma yapılmadı). "
            "Lütfen ilgili PDF'leri ingest edin veya soruyu yeniden formüle edin."
        )
        msg = self._apply_trading_guard(msg, query)  # trading sorusuysa uyarı taşı (kural 1)
        self.store.set_verification(
            run_id,
            supported_claims=[],
            unsupported_claims=[],
            contradictions=[],
            citation_score=0.0,
            grounding_score=0.0,
            context_sufficiency_score=0.0,
            final_decision="insufficient",
        )
        self.store.finish_run(
            run_id,
            status="abstained",
            final_answer=msg,
            final_confidence=0.0,
            evidence_score=evidence.score,
        )
        # Kaynak getirildi ama yetersiz → kanıt satırlarını yine de logla (eksiksiz audit
        # izi; _finish_no_llm ile tutarlı). chunks boşsa döngü no-op.
        self._log_evidence_rows(run_id, chunks, set())
        return RlmResult(
            run_id=run_id,
            query=query,
            task_type=task_type,
            status="abstained",
            final_answer=msg,
            final_confidence=0.0,
            confidence_level="Low",
            evidence_score=evidence.score,
            retrieval_rounds=rounds_used,
            n_sources=len(chunks),
            sources=self._sources_payload(chunks),
        )

    def _finish_no_llm(
        self,
        run_id: str,
        query: str,
        task_type: str,
        evidence: EvidenceReport,
        rounds_used: int,
        chunks: list[RetrievedChunk],
        write_report: bool,
    ) -> RlmResult:
        cites = "\n".join(f"- {c.citation} {c.title or ''}".strip() for c in chunks)
        msg = (
            "[LLM çevrimdışı — yalnızca retrieval sonuçları döndürüldü; iddia üretilmedi]\n\n"
            "Kaynaklar bulundu ama cevap üretimi için LLM gerekiyor "
            f"(kanıt skoru: {evidence.score}).\n\n"
            "Bulunan kaynaklar:\n" + cites
        )
        msg = self._apply_trading_guard(msg, query)  # trading sorusuysa uyarı taşı (kural 1)
        self.store.set_verification(
            run_id,
            supported_claims=[],
            unsupported_claims=[],
            contradictions=[],
            citation_score=0.0,
            grounding_score=0.0,
            context_sufficiency_score=evidence.score / 100.0,
            final_decision="no_llm",
        )
        report_path = (
            self._write_report(run_id, query, task_type, msg, evidence, chunks, [], [])
            if write_report
            else None
        )
        self.store.finish_run(
            run_id,
            status="no_llm",
            final_answer=msg,
            final_confidence=0.0,
            evidence_score=evidence.score,
            report_path=report_path,
        )
        self._log_evidence_rows(run_id, chunks, set())
        return RlmResult(
            run_id=run_id,
            query=query,
            task_type=task_type,
            status="no_llm",
            final_answer=msg,
            final_confidence=0.0,
            confidence_level="Low",
            evidence_score=evidence.score,
            retrieval_rounds=rounds_used,
            n_sources=len(chunks),
            sources=self._sources_payload(chunks),
            report_path=report_path,
        )

    def _synthesize(
        self,
        *,
        run_id: str,
        query: str,
        task_type: str,
        chunks: list[RetrievedChunk],
        evidence: EvidenceReport,
        rounds_used: int,
        supported: list[Claim],
        unsupported: list[Claim],
        contradictions: list[Contradiction],
        confidence_score: float,
        citation_score: float,
        grounding_score: float,
        context_score: float,
        decision: str,
        abstain: bool,
        abstain_reason: str,
        write_report: bool,
    ) -> RlmResult:
        confidence_level = (
            "High" if decision == "answer" else "Medium" if decision == "warn" else "Low"
        )

        # Sahte/geçersiz atıftan ibaret olup sanitize sonrası içerikten boşalan supported
        # iddialar (kural 7): gövdeye gerçek dayanak katmaz, "Kısa cevap"ı tek bir noktaya
        # ('.') çökertirdi → karar + gövde için anlamlı iddialarla çalış.
        valid_chunk_ids = {c.chunk_id for c in chunks}
        meaningful_supported = [
            c for c in supported if _has_content(_sanitize_citations(c.claim, valid_chunk_ids))
        ]

        # Audit kaydı (set_verification + evidence) nihai cevaba GERÇEKTEN gireni yansıtmalı.
        # Çekimser yolda cevap iddia İÇERMEZ → recorded_supported=[] ve used_ids=∅ (kardeş
        # _finish_insufficient/_finish_no_llm yollarıyla tutarlı). Aksi halde abstain run'ı
        # 'supported' + used_in_final_answer=True yazıp audit'i çelişkiye düşürürdü.
        if abstain or not meaningful_supported:
            reason = abstain_reason or (
                "Destekli iddialar yalnızca geçersiz atıftan ibaretti (gerçek dayanak yok)."
                if supported
                else "Taslak cevaptaki hiçbir iddia kaynaklarla yeterince desteklenmedi."
            )
            final_answer = (
                "Kısa cevap:\n"
                "Bu soruya kaynaklarla desteklenen güvenilir bir cevap üretilemedi.\n\n"
                f"Neden: {reason}\n\n"
                "Kaynak dayanakları (retrieval):\n"
                + "\n".join(f"- {c.citation} {c.title or ''}".strip() for c in chunks)
                + "\n\nGüven seviyesi: Low"
            )
            status = "abstained"
            # Çekimser çıktı 'High' rozeti taşımamalı (gövde 'Low'); final_confidence 0.0
            # (çekimserde cevaba güven yok). Audit: hiç iddia/kullanılan-chunk YOK.
            confidence_level = "Low"
            final_confidence = 0.0
            recorded_supported: list[Claim] = []
            used_ids: set[str] = set()
        else:
            used_ids = {cid for c in meaningful_supported for cid in c.supporting_chunks}
            used_chunks = [c for c in chunks if c.chunk_id in used_ids] or chunks
            final_answer = self._build_envelope(
                meaningful_supported,
                used_chunks,
                contradictions,
                confidence_level,
                valid_chunk_ids,
            )
            status = "answered_with_limitation" if decision != "answer" else "answered"
            final_confidence = round(confidence_score, 4)
            recorded_supported = meaningful_supported

        # Kural 1: trading-içerikli her çıktıya (soru veya cevap) zorunlu uyarı bloğu.
        final_answer = self._apply_trading_guard(final_answer, query)

        self.store.set_verification(
            run_id,
            supported_claims=[c.claim for c in recorded_supported],
            unsupported_claims=[c.claim for c in unsupported],
            contradictions=[f"{ct.chunk_id_a}↔{ct.chunk_id_b}" for ct in contradictions],
            citation_score=citation_score,
            grounding_score=grounding_score,
            context_sufficiency_score=context_score,
            final_decision=status,
        )
        report_path = (
            self._write_report(
                run_id, query, task_type, final_answer, evidence, chunks, supported, unsupported
            )
            if write_report
            else None
        )
        self.store.finish_run(
            run_id,
            status=status,
            final_answer=final_answer,
            final_confidence=final_confidence,
            evidence_score=evidence.score,
            report_path=report_path,
        )
        self._log_evidence_rows(run_id, chunks, used_ids)

        return RlmResult(
            run_id=run_id,
            query=query,
            task_type=task_type,
            status=status,
            final_answer=final_answer,
            final_confidence=final_confidence,
            confidence_level=confidence_level,
            evidence_score=evidence.score,
            retrieval_rounds=rounds_used,
            n_sources=len(chunks),
            supported_claims=[c.claim for c in recorded_supported],
            unsupported_claims=[c.claim for c in unsupported],
            contradictions=[f"{ct.chunk_id_a}↔{ct.chunk_id_b}" for ct in contradictions],
            sources=self._sources_payload(chunks),
            report_path=report_path,
        )

    def _build_envelope(
        self,
        supported: list[Claim],
        used_chunks: list[RetrievedChunk],
        contradictions: list[Contradiction],
        confidence_level: str,
        valid_chunk_ids: set[str],
    ) -> str:
        # Gövdeye giren ham LLM cümlelerindeki uydurma satır-içi atıfları çıkar (kural 7):
        # getirilen sette olmayan [paper:chunk] işaretleri kullanıcıya gösterilmez. Atıf
        # temizliği sonrası içerikten boşalan (yalnız geçersiz atıftan ibaret) iddialar
        # elenir — aksi halde "Kısa cevap" gövdesi tek bir noktaya çökerdi. Çağıran zaten
        # anlamlı iddialarla çağırır; bu filtre fonksiyonu kendi başına da güvenli kılar.
        safe_claims = [
            s
            for s in (_sanitize_citations(c.claim, valid_chunk_ids) for c in supported)
            if _has_content(s)
        ]
        gerekce = "\n".join(f"{i + 1}. {claim}" for i, claim in enumerate(safe_claims))
        kaynaklar = "\n".join(
            f"- {c.citation} | {c.section_name or '—'} | {c.title or '—'}" for c in used_chunks
        )
        limitations = []
        if contradictions:
            limitations.append(
                f"Kaynaklar arasında {len(contradictions)} potansiyel çelişki tespit edildi."
            )
        if confidence_level != "High":
            limitations.append("Kanıt güveni tam değil — bulgular ek doğrulama gerektirir.")
        if not limitations:
            limitations.append("Belirgin bir kısıt tespit edilmedi (token-düzeyi dayanak).")
        limit_block = "\n".join(f"- {x}" for x in limitations)

        parts = [
            "Kısa cevap:",
            safe_claims[0],
            "",
            "Makalelere göre gerekçe:",
            gerekce,
            "",
            "Kaynak dayanakları:",
            kaynaklar,
            "",
            "Sınırlamalar:",
            limit_block,
            "",
            f"Güven seviyesi: {confidence_level}",
        ]
        # Trading uyarısı burada DEĞİL — tek nokta _apply_trading_guard (içerik-tabanlı,
        # görev tipinden bağımsız). Böylece MATH/MULTI/UNCERTAINTY'ye sınıflanan trading
        # soruları da uyarıyı kaçırmaz (kural 1 sızıntısı kapandı).
        return "\n".join(parts)

    def _apply_trading_guard(self, answer: str, query: str) -> str:
        """Kural 1 tek-nokta yaptırımı: canlı sinyal/yatırım tavsiyesi ASLA.

        `rlm_allow_live_trading_signal` (config) burada gerçekten OKUNUR — ölü guard
        değil. MUTLAK olarak False'tur; biri yanlışlıkla True yapsa dahi sistem canlı
        sinyal üretmez, yalnızca bu (kozmetik olmayan) uyarı eklemeyi atlardı. Soru veya
        nihai cevap trading dili taşıyorsa ve uyarı henüz yoksa zorunlu uyarı eklenir.
        """
        return apply_trading_guard(
            answer, query, allow_live_signal=self.settings.rlm_allow_live_trading_signal
        )

    def _write_report(
        self,
        run_id: str,
        query: str,
        task_type: str,
        final_answer: str,
        evidence: EvidenceReport,
        chunks: list[RetrievedChunk],
        supported: list[Claim],
        unsupported: list[Claim],
    ) -> str:
        out_dir = self.settings.reports_dir / "rlm_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}.json"
        payload = {
            "run_id": run_id,
            "query": query,
            "task_type": task_type,
            "evidence": asdict(evidence),
            "final_answer": final_answer,
            "supported_claims": [asdict(c) for c in supported],
            "unsupported_claims": [asdict(c) for c in unsupported],
            "sources": self._sources_payload(chunks),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
