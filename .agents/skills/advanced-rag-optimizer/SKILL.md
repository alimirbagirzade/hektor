# Advanced RAG Optimizer Skill

## Amaç
Retrieval kalitesini artırmak için çoklu sorgu, yeniden sıralama ve
hiyerarşik bağlam genişletme tekniklerini birleştir.

## Tetiklenme Koşulları
- Temel RAG cevabı yetersiz veya alakasız göründüğünde
- Kullanıcı "daha iyi kaynak bul" veya "retrieval'ı iyileştir" dediğinde
- Yüksek hassasiyet gerektiren araştırma soruları için

## Adımlar

### 1. Sorgu Genişletme
```python
from app.brain.query_expander import QueryExpander
from app.brain.multi_query_retriever import MultiQueryRetriever
from app.memory.retrieval_service import RetrievalService

expander = QueryExpander()
expanded = expander.expand(original_query)
print(f"Genişletilmiş sorgular: {expanded}")

retriever = RetrievalService()
multi_retriever = MultiQueryRetriever(retriever=retriever, expander=expander)
chunks = multi_retriever.retrieve(original_query, top_k=8)
```

### 2. Reranking
```python
from app.memory.reranker import Reranker

reranker = Reranker()
reranked = reranker.rerank(original_query, chunks)
```

### 3. Hiyerarşik Genişletme
```python
from app.memory.hierarchical_retriever import HierarchicalRetriever

hier_retriever = HierarchicalRetriever(retriever=retriever)
enriched = hier_retriever.retrieve(original_query, top_k=5)
```

### 4. Hibrit Retrieval (BM25 + Semantik)
```python
from app.memory.bm25_index import BM25Index
from app.memory.hybrid_retriever import HybridRetriever

bm25 = BM25Index()
# Chunk'ları önceden indekse ekle
for chunk in all_chunks:
    bm25.add_document(chunk.chunk_id, chunk.text)

hybrid = HybridRetriever(semantic=retriever, bm25=bm25)
hybrid_results = hybrid.retrieve(original_query, top_k=5, alpha=0.7)
```

### 5. Self-Refining RAG
```python
from app.brain.self_refining_rag import SelfRefiningRAG
from app.memory.contextual_chunker import ContextualChunker

rag = SelfRefiningRAG(
    retriever=multi_retriever,
    reranker=reranker,
    chunker_annotator=ContextualChunker(),
)
final_chunks, quality_report = rag.retrieve_and_refine(original_query, top_k=5, max_rounds=2)
print(quality_report)
```

## Beklenen Çıktı
- Tekilleştirilmiş, kalite puanlı chunk listesi
- Her chunk için semantik + BM25 + bölüm öncelik skoru
- Eksik formül / argüman uyarısı (varsa)

## Performans Notu
- Multi-query: ~3× daha fazla retrieval çağrısı (kabul edilebilir)
- BM25 indeksi önceden hazır olmalı (chunk ekleme pahalı değil)
- Hiyerarşik genişletme yalnızca eksik bağlam tespitinde çalıştır
