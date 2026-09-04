"""Korpus geneli term–chunk grafı — Chroma'dan lazy kurulur, chunk sayısına göre cache'lenir.

`graph_retriever` (SPRIG-lite PPR) için korpus grafını sağlar. Chroma tüm chunk metnini
tuttuğundan graf ingestion'a dokunmadan canlı yola alınır; koleksiyon chunk sayısı değişince
(yeni makale) cache yeniden kurulur. Chroma boş/erişilemezse `(None, {})` döner → çağıran
dense-only'e geçer (graceful).

CACHE GEÇERSİZLEŞTİRME — `bm25_corpus`'tan bilinçli farklı:
`bm25_corpus` içerik İMZASI (count + toplam karakter) kullanır çünkü sıcak yolda zaten tüm
satırları yüklüdür → toplam karakteri saymak ~bedava. Graf modu ise sıcak yolda yalnız ucuz
`chroma.count()` çağırır ve pahalı `chroma.get_all()`'ı YENİDEN-KURMA dalına erteler; imzaya
toplam karakter eklemek `get_all()`'ı HER çağrıda zorlardı (perf regresyonu). Bu yüzden burada
imza yalnız chunk SAYISIdır ve aynı-sayıda içerik değişimini (force re-index aynı sayıda chunk
üretir / iyileşen parser farklı chunk üretir / delete+add) tek başına YAKALAYAMAZ. O durumda
`reset_cache()` OTORİTATİF geçersizleştirmedir → chunk mutasyonu yapan TÜM yollar (şu an yalnız
`PaperIndexer.ingest_one`) ingestion sonrası onu çağırmalı, yoksa graf SESSİZCE bayatlar ve eski
chunk'ları sunar (Kural 7 ihlali).
"""

from __future__ import annotations

import threading

from app.memory.chroma_store import ChromaStore
from app.memory.graph_retriever import TermChunkGraph, build_graph
from app.memory.retrieval_service import RetrievedChunk

# Modül düzeyi cache (process ömrü). Anahtar: chunk sayısı.
_cache: dict[str, object] = {"count": -1, "graph": None, "chunks": {}}
# Build kilidi: korpus graf kurulumu (get_all + build_graph) büyük korpusta pahalıdır.
# Eşzamanlı ilk-sorgular (warm-up + kullanıcı) kilitsiz HER BİRİ ayrı build başlatırdı
# (thundering herd → CPU boğulması). Kilit + çift-kontrol → yalnız BİR build, diğerleri
# bekleyip cache'i kullanır (bm25_corpus ile parite).
_build_lock = threading.Lock()


def get_corpus_graph(
    chroma: ChromaStore | None = None,
) -> tuple[TermChunkGraph | None, dict[str, RetrievedChunk]]:
    """Korpus term–chunk grafı + chunk_id→RetrievedChunk haritası döndür (cache'li).

    Returns:
        (graph, chunks). Korpus boş/erişilemezse (None, {}).
    """
    chroma = chroma or ChromaStore()
    try:
        count = chroma.count()
    except Exception:
        return None, {}
    if count == 0:
        return None, {}

    if _cache["count"] != count:
        with _build_lock:
            if _cache["count"] != count:  # çift-kontrol: başka thread bu arada kurmuş olabilir
                texts: dict[str, str] = {}
                chunks: dict[str, RetrievedChunk] = {}
                try:
                    rows = chroma.get_all()
                except Exception:
                    return None, {}
                for row in rows:
                    cid = row["chunk_id"]
                    doc = row.get("document", "") or ""
                    meta = row.get("metadata", {}) or {}
                    if not doc:
                        continue
                    texts[cid] = doc
                    chunks[cid] = RetrievedChunk(
                        chunk_id=cid,
                        paper_id=meta.get("paper_id", "?"),
                        text=doc,
                        page_number=meta.get("page_number"),
                        section_name=meta.get("section_name") or None,
                        title=meta.get("title") or None,
                        distance=None,  # graf kaynaklı; reranker semantiği nötr sayar
                    )
                _cache.update(count=count, graph=build_graph(texts), chunks=chunks)

    return _cache["graph"], _cache["chunks"]  # type: ignore[return-value]


def reset_cache() -> None:
    """Cache'i sıfırla (test izolasyonu / yeniden ingest sonrası).

    Chunk SAYISI anahtarı aynı-sayıda içerik değişimini yakalamaz (force re-index / iyileşen
    parser / delete+add); bu yüzden chunk mutasyonu yapan TÜM yollar (şu an yalnız
    PaperIndexer.ingest_one) bu fonksiyonu çağırmalı — otoritatif geçersizleştirme budur.

    `_build_lock` ALTINDA çalışır: build dalı (get_all + build_graph, saniyeler) kilidi tutarak
    sonunda `_cache.update` yazar. Reset kilitsiz olsaydı, bir build sürerken çağrıldığında
    build'in son update'i reset'i EZER (lost-update) → cache build başında okunan ESKİ chunk
    kümesiyle geri yazılır, ingest sonrası bayat graf sessizce sunulurdu (Kural 7). Kilit, reset'i
    in-flight build TAMAMLANDIKTAN sonra serileştirir. Reset yalnız ingestion thread'inden
    çağrılır (get_corpus_graph aynı çağrı yığınında değil) → reentrancy/deadlock yok.
    """
    with _build_lock:
        _cache.update(count=-1, graph=None, chunks={})
