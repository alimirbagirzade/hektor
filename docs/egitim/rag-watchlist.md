# RAG İzleme Listesi (Watchlist)

> Ucuz **tarama** turlarının çıktısı: güncel literatürde görülen ama henüz entegre
> edilmemiş RAG teknikleri burada biriktirilir. **Entegrasyon** turu buradan başlar
> (≥1 güçlü aday birikince tam tur koşar). Kod/sürüm/PDF değişmez; bu dosya push edilir.
>
> Durum etiketleri: `aday` (incelenecek) · `entegre` (kodda, dokümanda) · `ertelendi` (gerekçeli) · `red` (uygun değil/hype).
> Aynı teknik tekrar görülürse satırı GÜNCELLE (yeni kaynak/yeni durum), tekrar EKLEME.

| Eklendi | Teknik | Kaynak (URL) | Durum | Not / gerekçe |
|---|---|---|---|---|
| 2026-06-17 | Reciprocal Rank Fusion (RRF) | github.com/Raudaschl/rag-fusion | entegre (v1.2) | `rank_fusion.py` + MultiQuery + opt-in `rag_rrf` |
| 2026-06-17 | bge-reranker-v2-m3 (çok-dilli reranker) | huggingface.co/BAAI/bge-reranker-v2-m3 | entegre (v1.2, kısmi) | model yapılandırılabilir; varsayılan base kaldı (modest CPU) |
| 2026-06-17 | Late chunking (Jina) | arxiv.org/abs/2409.04701 | ertelendi | uzun-bağlam embedding + tutarlı yeniden-embed gerektirir |
| 2026-06-17 | Corrective RAG (CRAG) | arxiv.org/abs/2401.15884 | ertelendi | hafif offline retrieval-evaluator ileride; web-arama kademesi offline değil |
| 2026-06-17 | HyDE | — | ertelendi | LLM + latency/halüsinasyon; opsiyonel graceful HyDE ileride |
| 2026-06-17 | GraphRAG / LightRAG / HippoRAG | arxiv.org/abs/2507.03226 | ertelendi | token-pahalı; LightRAG/HippoRAG hafif varyant olarak ileride |
| 2026-06-17 | SPRIG: CPU-only GraphRAG (PPR) | arxiv.org/abs/2602.23372 | entegre (v1.4) | `graph_retriever.py`+`graph_corpus.py` term–chunk graf + tohumlu PPR + RRF; opt-in `rag_graph` |
| 2026-06-17 | RAGAS offline metrikleri | docs.ragas.io | entegre (v1.3, kısmi) | `app/evals/rag_ragas_offline.py` — faithfulness/context-precision/recall, LLM'siz deterministik |
| 2026-06-17 | RbFT / ALoFTRAG (RAFT varyantları) | arxiv.org/abs/2403.10131 | ertelendi | LoRA reçetesi notu; `discipline_dataset` RbFT ruhunda |
| 2026-06-17 | Matryoshka / nomic-embed v2 (MoE) | nomic.ai | ertelendi | Ollama embedding modeli değişince yükseltme yolu |

## Otomatik tarama adayları (rag-scan)

> `rag-scan` ajanının otomatik eklediği arXiv adayları (heuristik skorlu).
> Entegrasyon turu bunları değerlendirip yukarıdaki ana tabloya taşır.

| Eklendi | arXiv | Skor | Başlık | Sorgu |
|---|---|---|---|---|
| 2026-06-17 | 2602.16974 | 6 | Beyond Chunk-Then-Embed: A Comprehensive Taxonomy and Evaluation of Document Chunking Stra | RAG chunking late chunking contextual retrieval |
| 2026-06-17 | 2601.15518 | 6 | DS@GT at TREC TOT 2025: Bridging Vague Recollection with Fusion Retrieval and Learned Rera | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2504.19754 | 5 | Reconstructing Context: Evaluating Advanced Chunking Strategies for Retrieval-Augmented Ge | RAG chunking late chunking contextual retrieval |
| 2026-06-17 | 2606.01070 | 4 | Test-Time Training for Zero-Resource Dense Retrieval Reranking | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2603.25333 | 4 | Adaptive Chunking: Optimizing Chunking-Method Selection for RAG | RAG chunking late chunking contextual retrieval |
| 2026-06-17 | 2603.14828 | 4 | Toward Robust GraphRAG: Mitigating Retrieval Drift and Hallucination from Imperfect Knowle | GraphRAG knowledge graph retrieval |
| 2026-06-17 | 2511.22858 | 4 | RAG System for Supporting Japanese Litigation Procedures: Faithful Response Generation Com | RAG evaluation faithfulness groundedness hallucination |
| 2026-06-17 | 2511.11017 | 4 | AI Agent-Driven Framework for Automated Product Knowledge Graph Construction in E-Commerce | GraphRAG knowledge graph retrieval |
| 2026-06-17 | 2510.22344 | 4 | FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation | corrective RAG self-RAG adaptive retrieval |
| 2026-06-17 | 2505.21439 | 4 | Towards Better Instruction Following Retrieval Models | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2501.00309 | 4 | Retrieval-Augmented Generation with Graphs (GraphRAG) | GraphRAG knowledge graph retrieval |
| 2026-06-17 | 1811.08772 | 4 | Overcoming low-utility facets for complex answer retrieval | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2601.05264 | 3 | Engineering the RAG Stack: A Comprehensive Review of the Architecture and Trust Frameworks | corrective RAG self-RAG adaptive retrieval |
| 2026-06-17 | 2510.25621 | 3 | FARSIQA: Faithful and Advanced RAG System for Islamic Question Answering | corrective RAG self-RAG adaptive retrieval |
| 2026-06-17 | 2506.09886 | 3 | Probabilistic distances-based hallucination detection in LLMs with RAG | RAG evaluation faithfulness groundedness hallucination |
| 2026-06-17 | 2506.06962 | 3 | AR-RAG: Autoregressive Retrieval Augmentation for Image Generation | retrieval augmented generation reranking |
| 2026-06-17 | 2604.10167 | 6 | Visual Late Chunking: An Empirical Study of Contextual Chunking for Efficient Visual Docum | RAG chunking late chunking contextual retrieval |
| 2026-06-17 | 2404.07220 | 6 | Blended RAG: Improving RAG (Retriever-Augmented Generation) Accuracy with Semantic Search  | corrective RAG self-RAG adaptive retrieval |
| 2026-06-17 | 2108.06279 | 6 | On Single and Multiple Representations in Dense Passage Retrieval | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2605.22834 | 5 | Query-Adaptive Semantic Chunking for Retrieval-Augmented Generation: A Dynamic Strategy wi | RAG chunking late chunking contextual retrieval |
| 2026-06-17 | 2505.21700 | 5 | Rethinking Chunk Size For Long-Document Retrieval: A Multi-Dataset Analysis | RAG chunking late chunking contextual retrieval |
| 2026-06-17 | 2505.04847 | 5 | Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards | RAG evaluation faithfulness groundedness hallucination |
| 2026-06-17 | 2605.12335 | 4 | EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Reco | retrieval augmented generation reranking |
| 2026-06-17 | 2602.23372 | 4 | Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA | GraphRAG knowledge graph retrieval |
| 2026-06-17 | 2602.22225 | 4 | SmartChunk Retrieval: Query-Aware Chunk Compression with Planning for Efficient Document R | RAG chunking late chunking contextual retrieval |
| 2026-06-17 | 2507.23334 | 4 | MUST-RAG: MUSical Text Question Answering with Retrieval Augmented Generation | retrieval augmented fine-tuning RAFT distractor |
| 2026-06-17 | 2506.05690 | 4 | When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generati | GraphRAG knowledge graph retrieval |
| 2026-06-17 | 2504.12920 | 4 | CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval | retrieval augmented fine-tuning RAFT distractor |
| 2026-06-17 | 2408.07303 | 4 | Enhancing Visual Question Answering through Ranking-Based Hybrid Training and Multimodal F | query rewriting HyDE reciprocal rank fusion |
| 2026-06-17 | 2401.15391 | 4 | MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries | RAG evaluation faithfulness groundedness hallucination |
| 2026-06-17 | 2302.04024 | 4 | InMyFace: Inertial and Mechanomyography-Based Sensor Fusion for Wearable Facial Activity R | query rewriting HyDE reciprocal rank fusion |
| 2026-06-17 | 2205.02303 | 4 | Analysing the Robustness of Dual Encoders for Dense Retrieval Against Misspellings | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2510.24652 | 3 | Optimizing Retrieval for RAG via Reinforcement Learning | corrective RAG self-RAG adaptive retrieval |
| 2026-06-17 | 2509.07163 | 3 | Beyond Sequential Reranking: Reranker-Guided Search Improves Reasoning Intensive Retrieval | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2310.11511 | 3 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | corrective RAG self-RAG adaptive retrieval |
| 2026-06-17 | 2604.08082 | 2 | From Binary Groundedness to Support Relations: Towards a Reader-Centred Taxonomy for Compr | RAG evaluation faithfulness groundedness hallucination |
| 2026-06-17 | 2603.25777 | 2 | Challenges and opportunities for AI to help deliver fusion energy | query rewriting HyDE reciprocal rank fusion |
| 2026-06-17 | 2511.06455 | 2 | A Multi-Agent System for Semantic Mapping of Relational Data to Knowledge Graphs | GraphRAG knowledge graph retrieval |
| 2026-06-17 | 2504.13684 | 2 | Intelligent Interaction Strategies for Context-Aware Cognitive Augmentation | retrieval augmented generation reranking |
| 2026-06-17 | 2504.01346 | 2 | RAG over Tables: Hierarchical Memory Index, Multi-Stage Retrieval, and Benchmarking | corrective RAG self-RAG adaptive retrieval |
| 2026-06-17 | 2411.18583 | 2 | Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generat | retrieval augmented generation reranking |
| 2026-06-17 | 2402.12317 | 2 | EVOR: Evolving Retrieval for Code Generation | retrieval augmented generation reranking |
| 2026-06-17 | 2303.14991 | 2 | Empowering Dual-Encoder with Query Generator for Cross-Lingual Dense Retrieval | dense retrieval embedding cross-encoder reranker |
| 2026-06-17 | 2210.02627 | 2 | Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Do | corrective RAG self-RAG adaptive retrieval |
