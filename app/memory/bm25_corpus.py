"""Korpus geneli BM25 indeksi — SQLite'tan lazy kurulur, içerik imzasına göre cache'lenir.

Hibrit retrieval (Faz A3) için: dense (semantik) aday havuzunu keyword (BM25)
eşleşmeleriyle genişletir. `BM25Index` yazılıydı ama hiç doldurulmuyordu; bu modül
onu ingestion'a dokunmadan canlı yola alır (SQLite zaten tüm chunk metnini tutar).

Cache, korpus İÇERİK İMZASI (chunk sayısı + toplam karakter) değişince yeniden kurulur.
Yalnız chunk SAYISI yetmezdi: eşit-sayıda içerik değişiminde (force re-index aynı sayıda
chunk üretir / chunk yerinde yeniden yazılırsa count sabit kalır) indeks SESSİZCE bayatlar
ve eski sonuç dönerdi (Kural 7 ihlali). İmza, sıcak-yolda zaten yüklü satırlar üzerinden
hesaplanır → ek DB turu YOK (ölçüm: ~94k chunk'ta ~6.6ms ≈ list_all_chunks ~234ms yükünün
%3'ü; tam metin-hash'i ~18ms ÖLÇÜLEREK reddedildi). reset_cache() yine otoritatif geçersiz-
leştirmedir (aynı-uzunlukta içerik değişimi imzayı kaçırabilir; mutasyon yolları çağırmalı).
SQLite boş/erişilemezse `(None, {})` döner → çağıran dense-only'e geçer (graceful).
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from app.memory.bm25_index import BM25Index
from app.memory.retrieval_service import RetrievedChunk

if TYPE_CHECKING:
    from app.memory.chroma_store import ChromaStore
    from app.memory.sqlite_store import SqliteStore

# Modül düzeyi cache (process ömrü). Anahtar: korpus içerik imzası (count, toplam_char).
# bm25+chunks TEK anahtarda ("pair") demet olarak tutulur → sıcak-yol dönüşü tek atomik
# subscript okumasıdır. Ayrı iki okuma (`_cache["bm25"]` SONRA `_cache["chunks"]`) arasına
# reset/build girerse okuyucu dolu bm25 + boş chunks yakalar → lexical isabetler sessizce
# düşer, hibrit o sorguda fark ettirmeden dense-only'e geriler (Kural 6/7). Demet bunu önler.
_cache: dict[str, object] = {"sig": None, "pair": (None, {})}
# Build kilidi: 94k chunk tokenizasyonu ~170s sürer. Eşzamanlı ilk-sorgular (warm-up +
# kullanıcı) kilitsiz HER BİRİ ayrı build başlatırdı (thundering herd → CPU boğulması).
# Kilit + çift-kontrol → yalnız BİR build, diğerleri bekleyip cache'i kullanır.
_build_lock = threading.Lock()

# Modül düzeyi paylaşılan SqliteStore (cache; her çağrıda yeni store yaratma).
_shared_store: SqliteStore | None = None


def _corpus_store() -> SqliteStore:
    global _shared_store
    if _shared_store is None:
        from app.memory.sqlite_store import SqliteStore

        _shared_store = SqliteStore()
    return _shared_store


def get_corpus_bm25(
    chroma: ChromaStore | None = None,
    store: SqliteStore | None = None,
) -> tuple[BM25Index | None, dict[str, RetrievedChunk]]:
    """Korpus BM25 indeksi + chunk_id→RetrievedChunk haritası döndür (cache'li).

    KAYNAK = SQLite (Chunk tablosu), Chroma DEĞİL. Önceki Chroma `get_all()` 91k id'yi
    bulk okur, eşzamanlı erişimde (öğrenme döngüsü/MCP/serving) SQLite-kilidine takılır ve
    BM25'i SESSİZCE öldürürdü (chunks=0 → hibrit dense-only'e düşer; Kural 7'ye aykırı sessiz
    çökme). SQLite WAL eşzamanlı-okumayı sorunsuz kaldırır → BM25 buradan kurulur (dayanıklı).
    `chroma` parametresi geriye-uyum için KORUNUR ama YOK SAYILIR. `store` test enjeksiyonu.

    Returns:
        (bm25, chunks). Korpus boş/erişilemezse (None, {}).
    """
    del chroma  # geriye-uyum imzası; kaynak artık SQLite
    st = store or _corpus_store()
    try:
        # embedded=0 chunk'ları DIŞLA: bunlar Chroma'ya yazılamamış (yarım/başarısız ingest);
        # dense yol onları zaten görmez. BM25'e alınsalardı hibrit/RRF/router'da keyword adayı
        # olur, dense-erişilemez metni alıntılardık → BUG-M6 koruması asimetrik kalırdı (Kural 7).
        # embedded kolonu olmayan test stub'ları varsayılan 1 ile korunur (davranış değişmez).
        rows = [ch for ch in st.list_all_chunks() if getattr(ch, "embedded", 1)]
    except Exception:
        return None, {}
    count = len(rows)
    if count == 0:
        return None, {}

    # İçerik imzası: yalnız chunk SAYISI değil toplam karakter de. Eşit-sayıda içerik
    # değişiminde (force re-index aynı sayıda chunk üretir / chunk yerinde yeniden yazılır)
    # count sabit kalır ama toplam karakter değişir → bayat indeks otomatik yeniden kurulur.
    # char_count kolonu chunk'lama anında len(text) ile yazılır; yoksa (test stub) len(text)'e
    # düşer. Satırlar zaten yüklü → yalnız O(n) bellek-içi toplam (~6.6ms/94k chunk, ölçüldü).
    # İmza FİLTRELENMİŞ (embedded) küme üzerinden → sonraki mark_chunks_embedded (0→1) imzayı
    # değiştirir, böylece reset_cache atlanırsa bile doğru rebuild tetiklenir.
    total_chars = sum(getattr(ch, "char_count", None) or len(ch.text or "") for ch in rows)
    sig = (count, total_chars)

    if _cache["sig"] != sig:
        with _build_lock:
            if _cache["sig"] != sig:  # çift-kontrol: başka thread bu arada kurmuş olabilir
                titles = {p.paper_id: p.title for p in st.list_papers()}
                bm25 = BM25Index()
                chunks: dict[str, RetrievedChunk] = {}
                for ch in rows:
                    doc = ch.text or ""
                    if not doc:
                        continue
                    bm25.add_document(ch.chunk_id, doc)
                    chunks[ch.chunk_id] = RetrievedChunk(
                        chunk_id=ch.chunk_id,
                        paper_id=ch.paper_id,
                        text=doc,
                        page_number=ch.page_number,
                        section_name=ch.section_name or None,
                        title=titles.get(ch.paper_id),
                        distance=None,  # BM25 kaynaklı; reranker semantiği nötr (0.5) sayar
                    )
                _cache.update(sig=sig, pair=(bm25, chunks))

    return _cache["pair"]  # type: ignore[return-value]


def reset_cache() -> None:
    """Cache'i sıfırla (test izolasyonu / yeniden ingest sonrası).

    İçerik imzası (count + toplam karakter) aynı-uzunlukta içerik değişimini yakalamaz;
    bu yüzden chunk mutasyonu yapan TÜM yollar (şu an yalnız PaperIndexer.ingest_one)
    bu fonksiyonu çağırmalı — otoritatif geçersizleştirme budur.

    `_build_lock` ALTINDA çalışır: aksi halde sürmekte olan bir build'in (warm-up ~170s)
    son `_cache.update`'i bu reset'i EZER (lost-update) → ingest sonrası eski korpus sessizce
    sunulurdu. Kilit, reset'i in-flight build TAMAMLANDIKTAN sonra serileştirir → sonraki sorgu
    kesin yeniden kurar. Reset yalnız ingestion thread'inden çağrılır (get_corpus_bm25 aynı
    çağrı yığınında değil) → reentrancy/deadlock yok.
    """
    global _shared_store
    with _build_lock:
        _cache.update(sig=None, pair=(None, {}))
        _shared_store = None
