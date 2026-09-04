"""Over-fetch + yeniden sıralama (rerank) ile retrieval sarmalayıcı.

`RetrievalService` ile **aynı arayüzü** (`retrieve(query, top_k)`) sunar; böylece
`RagAnswerer` ve eval kodu somut sınıfa değil arayüze bağlanır ve "robust RAG"
yolu tek yerde toplanır.

Çalışma mantığı (LLM-free, çevrimdışı testlerle uyumlu):
1. Dense retrieval'dan `top_k * overfetch` aday çek (geniş havuz).
2. Heuristik `Reranker` ile yeniden sırala (semantik + anahtar kelime + bölüm
   önceliği + formül varlığı).
3. İlk `top_k` adayı döndür.

`rag_rerank` ayarı kapalıysa düz dense retrieval'a indirger (davranış değişmez).
Bu, "yazılı ama bağlanmamış" `Reranker`'ı canlı yola alan Faz A2 adımıdır;
eğitim gerektirmez (bkz. docs/RAG_EGITIM_YENIDEN_TASARIM.md).
"""

from __future__ import annotations

from typing import Protocol

from app.config import get_settings
from app.memory.reranker import Reranker
from app.memory.retrieval_service import RetrievalService, RetrievedChunk


class RerankerLike(Protocol):
    """Heuristik `Reranker` ve `CrossEncoderReranker` bu arayüzü uygular."""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class RerankingRetriever:
    """Dense retrieval + heuristik rerank (over-fetch → rerank → truncate).

    Args:
        base: Temel (dense) retrieval servisi. Verilmezse `RetrievalService()`.
        reranker: Yeniden sıralayıcı. Verilmezse `Reranker()`.
        overfetch: Aday çarpanı; `None` → `settings.rag_overfetch`.
        enabled: Rerank açık mı; `None` → `settings.rag_rerank`. Kapalıysa düz
            dense retrieval döner.
    """

    def __init__(
        self,
        base: RetrievalService | None = None,
        reranker: RerankerLike | None = None,
        overfetch: int | None = None,
        enabled: bool | None = None,
        hybrid: bool | None = None,
        rrf: bool | None = None,
        graph: bool | None = None,
        router: bool | None = None,
    ) -> None:
        self.settings = get_settings()
        self.base = base or RetrievalService()
        self.reranker: RerankerLike = reranker or self._default_reranker()
        self.overfetch = overfetch if overfetch is not None else self.settings.rag_overfetch
        self.enabled = enabled if enabled is not None else self.settings.rag_rerank
        # Hibrit: dense aday havuzunu BM25 (keyword) eşleşmeleriyle genişlet (Faz A3).
        self.hybrid = hybrid if hybrid is not None else self.settings.rag_hybrid
        # RRF füzyon modu (opt-in): dense ve BM25 sıralı listelerini Reciprocal Rank
        # Fusion ile birleştirir (heuristik/cross-encoder rerank yerine). Skor
        # kalibrasyonu gerektirmez → karşılaştırılamaz skorlu kaynaklarda sağlam.
        # Varsayılan kapalı → mevcut over-fetch+rerank davranışı değişmez.
        self.rrf = rrf if rrf is not None else self.settings.rag_rrf
        # Graf modu (SPRIG-lite, opt-in): dense-hit'lerden tohumlanmış PPR + RRF füzyonu
        # (çok-hop recall). Açıksa rrf/rerank yerine graf yolu çalışır. Varsayılan kapalı.
        self.graph = graph if graph is not None else self.settings.rag_graph
        # Sorgu yönlendirici (opt-in, Zincir 2): sorgu-tipine göre yol seç — KISA keyword/
        # entity → konveks-füzyon hibrit (BM25 yardımcı); UZUN semantik → saf dense (ölçülen
        # en iyi). Diğer modlardan önce gelir. Varsayılan kapalı (davranış değişmez).
        self.router = router if router is not None else self.settings.rag_router

    def _default_reranker(self) -> RerankerLike:
        # FlashRank OPT-IN (CPU'da hızlı ONNX cross-encoder); cross_encoder'a göre öncelikli
        # (bge-reranker CPU'da >15s iken FlashRank ~100ms). Yoksa kendi içinde heuristiğe düşer.
        if self.settings.rag_flashrank:
            from app.memory.flashrank_reranker import FlashRankReranker

            return FlashRankReranker()
        # Cross-encoder OPT-IN (Faz A8); açıksa onu kullan (model yoksa kendi içinde
        # heuristiğe düşer). Kapalıysa doğrudan heuristik Reranker.
        if self.settings.rag_cross_encoder:
            from app.memory.cross_encoder_reranker import CrossEncoderReranker

            return CrossEncoderReranker()
        return Reranker()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.settings.rag_top_k

        if not self.enabled:
            return self.base.retrieve(query, top_k=k)

        # Sorgu yönlendirici (opt-in): tip'e göre yol. KISA keyword/entity → konveks-füzyon
        # hibrit; UZUN semantik → saf dense (ölçülen en iyi + en hızlı). Diğer modlardan önce.
        if self.router:
            from app.memory.query_router import classify_query

            if classify_query(query) == "lexical":
                return self._convex_hybrid_retrieve(query, k)
            return self.base.retrieve(query, top_k=k)

        # Graf modu (opt-in): dense tohum → PPR → RRF füzyonu (çok-hop recall).
        if self.graph:
            return self._graph_retrieve(query, k)

        # RRF füzyon modu (opt-in): dense + BM25 sıralı listelerini RRF ile birleştir.
        if self.rrf:
            return self._rrf_retrieve(query, k)

        # Geniş aday havuzu çek (en az k; ideal olarak k * overfetch).
        candidate_k = max(k, k * max(1, self.overfetch))
        candidates = self.base.retrieve(query, top_k=candidate_k)

        # Hibrit: BM25 keyword adaylarını ekle (dense'in kaçırdığı teknik terimler:
        # ATR, Sharpe, RSI…). Korpus boş/erişilemezse sessizce dense-only kalır.
        if self.hybrid:
            candidates = self._add_bm25_candidates(query, candidates, candidate_k)

        if not candidates:
            return []

        ranked = self.reranker.rerank(query, candidates)
        return ranked[:k]

    def _rrf_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Dense ve BM25 sıralı listelerini Reciprocal Rank Fusion ile birleştir.

        Heuristik/cross-encoder rerank yerine saf sıra-füzyonu uygular (deterministik,
        LLM-free). BM25 korpusu boş/erişilemezse veya hibrit kapalıysa dense-only'e
        düşer (davranış güvenli). Metni olmayan (BM25-only, chunk_map'te yok) id'ler
        atlanır — boş kaynak RAG alıntısını bozmasın (Kural 7).
        """
        from app.memory.bm25_corpus import get_corpus_bm25
        from app.memory.rank_fusion import fuse_ranked

        candidate_k = max(k, k * max(1, self.overfetch))
        dense = self.base.retrieve(query, top_k=candidate_k)
        chunk_by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense}
        dense_ids = [c.chunk_id for c in dense]

        bm25_ids: list[str] = []
        if self.hybrid:
            bm25, chunk_map = get_corpus_bm25()
            if bm25 is not None:
                for cid, _score in bm25.search(query, candidate_k):
                    bm25_ids.append(cid)
                    if cid not in chunk_by_id and cid in chunk_map:
                        chunk_by_id[cid] = chunk_map[cid]

        ranked_lists = [lst for lst in (dense_ids, bm25_ids) if lst]
        if not ranked_lists:
            return []

        fused = fuse_ranked(ranked_lists, k=self.settings.rag_rrf_k)
        out = [chunk_by_id[cid] for cid in fused if cid in chunk_by_id]
        return out[:k]

    def _graph_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """SPRIG-lite: dense-hit'lerden tohumlanmış PPR → dense ile RRF füzyonu.

        Korpus term–chunk grafı (lazy) üzerinde, dense aday id'lerinden tohumlanan
        Personalized PageRank çalıştırır; graf-sıralı liste ile dense-sıralı listeyi RRF
        ile birleştirir. Böylece dense'in kaçırdığı ama paylaşılan terimlerle bağlı
        chunk'lar yüzeye çıkabilir (çok-hop recall). Graf yoksa/erişilemezse dense-only'e
        düşer (güvenli). Metni olmayan id'ler atlanır (Kural 7).
        """
        from app.memory.graph_corpus import get_corpus_graph
        from app.memory.graph_retriever import personalized_pagerank, seed_weights_from_ids
        from app.memory.rank_fusion import fuse_ranked

        candidate_k = max(k, k * max(1, self.overfetch))
        dense = self.base.retrieve(query, top_k=candidate_k)
        if not dense:
            return []
        dense_ids = [c.chunk_id for c in dense]
        chunk_by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense}

        graph, corpus_chunks = get_corpus_graph()
        if graph is None:
            return dense[:k]  # graf yok → dense-only

        ppr = personalized_pagerank(
            graph,
            seed_weights_from_ids(dense_ids),
            damping=self.settings.rag_graph_damping,
            iterations=self.settings.rag_graph_iters,
        )
        graph_ids = [
            cid for cid, _s in sorted(ppr.items(), key=lambda kv: (-kv[1], kv[0])) if ppr[cid] > 0.0
        ][:candidate_k]
        for cid in graph_ids:
            if cid not in chunk_by_id and cid in corpus_chunks:
                chunk_by_id[cid] = corpus_chunks[cid]

        ranked_lists = [lst for lst in (dense_ids, graph_ids) if lst]
        fused = fuse_ranked(ranked_lists, k=self.settings.rag_rrf_k)
        out = [chunk_by_id[cid] for cid in fused if cid in chunk_by_id]
        return out[:k]

    def _convex_hybrid_retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """Dense + BM25 skorlarını min-max normalize edip KONVEKS birleştir (router lexical yolu).

        "union + sezgisel rerank" anti-deseni yerine principled füzyon (Bruch et al.):
        skor = α·norm(dense_benzerlik) + (1−α)·norm(bm25); α dense-ağırlıklı. BM25 yoksa/
        boşsa dense-only'e düşer (güvenli, Kural 7 — boş kaynak uydurmaz).
        """
        from app.memory.bm25_corpus import get_corpus_bm25
        from app.memory.query_router import convex_fuse

        candidate_k = max(k, k * max(1, self.overfetch))
        dense = self.base.retrieve(query, top_k=candidate_k)
        chunk_by_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense}
        dense_scores = {
            c.chunk_id: max(0.0, 1.0 - (c.distance if c.distance is not None else 1.0))
            for c in dense
        }
        bm25_scores: dict[str, float] = {}
        bm25, chunk_map = get_corpus_bm25(self.base.chroma)  # sıcak chroma'yı paylaş
        if bm25 is not None:
            for cid, score in bm25.search(query, candidate_k):
                bm25_scores[cid] = float(score)
                if cid not in chunk_by_id and cid in chunk_map:
                    chunk_by_id[cid] = chunk_map[cid]
        if not bm25_scores:
            return dense[:k]  # BM25 yok → dense-only
        fused = convex_fuse(dense_scores, bm25_scores, alpha=self.settings.rag_router_alpha)
        out = [chunk_by_id[cid] for cid in fused if cid in chunk_by_id]
        return out[:k]

    def _add_bm25_candidates(
        self, query: str, candidates: list[RetrievedChunk], candidate_k: int
    ) -> list[RetrievedChunk]:
        from app.memory.bm25_corpus import get_corpus_bm25

        bm25, chunk_map = get_corpus_bm25()
        if bm25 is None:
            return candidates
        have = {c.chunk_id for c in candidates}
        for cid, _score in bm25.search(query, candidate_k):
            if cid not in have and cid in chunk_map:
                candidates.append(chunk_map[cid])
                have.add(cid)
        return candidates
