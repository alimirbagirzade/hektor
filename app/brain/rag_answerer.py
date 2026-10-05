"""RAG answerer.

Produces a bilingual (English + Turkish) answer structure:
1. Short Answer / Kısa Cevap
2. Sources Used / Kullanılan Kaynaklar
3. Context Quality / Bağlam Kalitesi
4. Academic Finding / Akademik Bulgu
5. Formula or Argument Analysis / Formül veya Argüman Analizi
6. Trading Hypothesis / Trading Hipotezi
7. Test Plan / Test Planı
8. Risks / Riskler
9. Next Step / Sonraki Adım

The model is explicitly instructed NOT to invent sources. If no chunks are
retrieved, it must say so rather than hallucinate.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.brain.answer_quality import (
    CitationCheck,
    assess_confidence,
    citation_warning,
    is_weak_retrieval,
    reorder_lost_in_middle,
    verify_citations,
)
from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.brain.prompt_loader import load_prompt
from app.config import get_settings
from app.memory.reranking_retriever import RerankingRetriever
from app.memory.retrieval_service import RetrievedChunk, Retriever

# NOT: Cevap formatı ARTIK tek kaynakta — app/prompts/rag_answer.md (sistem
# prompt'u olarak yüklenir). Eskiden burada ayrı bir _BILINGUAL_FORMAT vardı ve
# .md ile çatışıyordu (model iki farklı format alıyordu); birleştirildi (Faz A4).
# Kullanıcı prompt'una format enjekte edilmez; yalnız KAYNAKLAR + SORU verilir.

_FALLBACK_SYSTEM = (
    "Yalnızca verilen KAYNAKLAR'a dayan; kaynak yoksa 'kaynak bulunamadı' de, uydurma. "
    "Her iddiadan sonra [paper_id:chunk_id] satır-içi atıf ver. İki dilli (EN+TR) "
    "yapısal cevap: Kısa Cevap, Kaynaklar, Akademik Bulgu, Trading Hipotezi (test "
    "edilmemiş), Test Planı (OOS+maliyet+look-ahead yok), Riskler. Yatırım tavsiyesi verme."
)


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    llm_used: bool
    # Sohbet kaydı için: modele GERÇEKTEN gönderilen istem + uyarısız ham model metni +
    # atıf kimliği denetimi. LLM çağrılmadıysa boş/None kalır.
    system_prompt: str = ""
    user_prompt: str = ""
    raw_answer: str = ""
    citation_check: CitationCheck | None = None


#: Önceki konuşma bloğunun başlığı — model bu bölümü kaynak sanmasın diye açıkça işaretlenir.
HISTORY_HEADER = (
    "ÖNCEKİ KONUŞMA / PREVIOUS CONVERSATION (yalnız bağlamdır; doğrulanmış kaynak DEĞİLDİR — "
    "buradan atıf verme, iddiaları yalnız KAYNAKLAR'a dayandır):"
)


def format_history(history: list[tuple[int, str, str]]) -> str:
    """(tur_no, soru, cevap) listesini istem bloğuna çevir (boş liste → boş metin)."""
    if not history:
        return ""
    parts = [HISTORY_HEADER]
    for idx, q, a in history:
        parts.append(f"[Tur {idx}] SORU: {q}\n[Tur {idx}] CEVAP: {a}")
    return "\n\n".join(parts)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        head = f"{c.citation}"
        if c.title:
            head += f" — {c.title}"
        blocks.append(f"{head}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    reorder: bool = True,
    history: list[tuple[int, str, str]] | None = None,
) -> tuple[str, str]:
    """Canlı RAG'ın (sistem, kullanıcı) istemi — TEK kaynak.

    `RagAnswerer.answer` ve öz-damıtma verisi (`app.training.self_distill`) aynı fonksiyonu
    kullanır; eğitim örneği canlı istemle bayt-aynı olsun (v13 dersi: eğitimdeki
    `BAĞLAM:/SORU:` biçimi canlı `SOURCES / KAYNAKLAR` biçiminden farklıydı).

    ``history`` (sohbet): (tur_no, soru, cevap) listesi KAYNAKLAR'dan ÖNCE, "doğrulanmış kaynak
    değildir" başlığıyla eklenir. ``None``/boş → istem geçmişsiz sürümle BAYT-AYNI kalır.
    """
    # "Lost in the middle": en alakalı chunk'lar bağlamın başına/sonuna (ekleme/çıkarma yok).
    context_chunks = reorder_lost_in_middle(chunks) if reorder else chunks
    context = _format_context(context_chunks)
    try:
        system = load_prompt("rag_answer")  # tek kaynak format + grounding kuralları
    except FileNotFoundError:
        system = _FALLBACK_SYSTEM
    # Format sistem prompt'undan gelir; kullanıcı prompt'u yalnız bağlam + soru.
    user = f"SOURCES / KAYNAKLAR:\n{context}\n\nQUESTION / SORU: {question}"
    hist = format_history(history or [])
    if hist:
        user = f"{hist}\n\n{user}"
    return system, user


class RagAnswerer:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LocalLLM | None = None,
    ) -> None:
        # Varsayılan: over-fetch + heuristik rerank (robust RAG yolu). Stub/özel
        # retriever enjekte edilirse aynen kullanılır (test izolasyonu korunur).
        self.retriever = retriever or RerankingRetriever()
        self.llm = llm or LocalLLM()
        self.settings = get_settings()

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        *,
        history: list[tuple[int, str, str]] | None = None,
        retrieval_query: str | None = None,
    ) -> RagAnswer:
        """Soruyu yanıtla. ``history``/``retrieval_query`` yalnız sohbet yolunda verilir;
        verilmezse davranış ve istem öncekiyle aynıdır."""
        chunks = self.retriever.retrieve(retrieval_query or question, top_k=top_k)

        if not chunks:
            return RagAnswer(
                question=question,
                answer=(
                    "No sources found. No article chunks in memory to support this query.\n"
                    "Please ingest the relevant PDFs first.\n\n"
                    "---\n\n"
                    "Kaynak bulunamadı. Hafızada bu soruya dayanak oluşturacak chunk yok.\n"
                    "Önce ilgili PDF'leri ingest edin."
                ),
                sources=[],
                llm_used=False,
            )

        # CRAG-lite güven kapısı (opt-in): retrieval ZAYIFSA LLM'i çağırmadan ABSTAIN
        # (Kural 7 — zayıf dayanakla uydurma yok). Deterministik, mesafe-tabanlı.
        if self.settings.rag_abstain:
            conf = assess_confidence(chunks)
            if is_weak_retrieval(
                conf,
                min_similarity=self.settings.rag_abstain_min_similarity,
                min_margin=self.settings.rag_abstain_min_margin,
            ):
                cites = "\n".join(f"- {c.citation} {c.title or ''}".strip() for c in chunks)
                return RagAnswer(
                    question=question,
                    answer=(
                        "Insufficient grounding — retrieved sources are weakly related to the "
                        "question (best similarity "
                        f"{conf.best_similarity:.2f}). Not answering to avoid speculation.\n\n"
                        "---\n\n"
                        "Yetersiz dayanak — getirilen kaynaklar soruyla zayıf ilişkili "
                        f"(en iyi benzerlik {conf.best_similarity:.2f}). Uydurmamak için "
                        "cevap verilmiyor. Soruyu daraltın veya ilgili makaleleri ingest edin.\n\n"
                        "Weak matches / Zayıf eşleşmeler:\n" + cites
                    ),
                    sources=chunks,
                    llm_used=False,
                )

        # "Lost in the middle" (opt, varsayılan açık) — kaynak listesi sıralı kalır; yalnız
        # LLM'e giden bağlam yeniden dizilir.
        system, prompt = build_rag_prompt(
            question, chunks, reorder=self.settings.rag_reorder_context, history=history
        )

        try:
            text = self.llm.generate(prompt, system=system, temperature=0.2, seed=42)
            raw = text
            # Citation-id doğrulama (opt, varsayılan açık): UYDURMA atıfları yakala —
            # citation-forcing prompt yine de getirilmeyen kaynağa atıf verebilir
            # (correctness≠faithfulness). Deterministik son-kontrol, LLM'siz (Kural 7).
            check: CitationCheck | None = None
            if self.settings.rag_verify_citations:
                check = verify_citations(text, chunks)
                text += citation_warning(check)
            return RagAnswer(
                question=question,
                answer=text,
                sources=chunks,
                llm_used=True,
                system_prompt=system,
                user_prompt=prompt,
                raw_answer=raw,
                citation_check=check,
            )
        except LLMUnavailable:
            # Graceful degradation: still return retrieved sources.
            cites = "\n".join(f"- {c.citation} {c.title or ''}".strip() for c in chunks)
            text = (
                "[LLM offline — retrieval results only"
                " / LLM çevrimdışı — yalnızca retrieval sonuçları]\n\n"
                "Available sources / Kullanılabilir kaynaklar:\n" + cites
            )
            return RagAnswer(question=question, answer=text, sources=chunks, llm_used=False)
