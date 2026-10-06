# RAG Reliability Engineer Skill

## Amaç
RAG sisteminin güvenilirliğini ölçmek, hataları kaydetmek ve yayın
kapısından geçip geçemeyeceğini değerlendirmek.

## Tetiklenme Koşulları
- "RAG güvenilirliği test et" veya benzeri komutlarda
- Yeni retrieval/cevap modülü eklendikten sonra
- Yayın öncesi kalite değerlendirmesinde

## Adımlar

### 1. Güven Skoru Hesapla
```python
from app.verification.confidence_scorer import ConfidenceScorer
from app.verification.context_sufficiency import ContextSufficiencyClassifier
from app.verification.citation_verifier import CitationVerifier
from app.verification.grounding_verifier import GroundingVerifier
from app.verification.contradiction_detector import ContradictionDetector

# Her bileşeni çalıştır
classifier = ContextSufficiencyClassifier()
sufficiency = classifier.classify(query, chunks)

cit_verifier = CitationVerifier()
citations = cit_verifier.verify(answer_text, chunks)

grnd_verifier = GroundingVerifier()
groundings = grnd_verifier.verify(answer_text, chunks)

detector = ContradictionDetector()
contradictions = detector.detect(chunks)

scorer = ConfidenceScorer()
report = scorer.score(sufficiency, citations, groundings, contradictions)
print(f"Güven skoru: {report.score:.2f} — Karar: {report.decision}")
```

### 2. Çekimser Kalma Kararı
```python
from app.verification.abstention_policy import AbstentionPolicy

policy = AbstentionPolicy()
decision = policy.decide(report, sufficiency)

if decision.should_abstain:
    print(decision.reason)
else:
    # Cevap ver
    pass
```

### 3. Hata Kayıt
```python
from app.reliability.failure_analyzer import FailureAnalyzer, FailureRecord, FailureType

analyzer = FailureAnalyzer()
if report.citation_score < 0.5:
    analyzer.record(FailureRecord(
        question=query,
        answer=answer_text,
        failure_type=FailureType.WRONG_CITATION,
        root_cause="Atıf doğrulaması başarısız",
        wrong_citation=True,
    ))
```

### 4. Yayın Kapısı Kontrolü
```python
from app.reliability.release_gate import ReleaseGate

gate = ReleaseGate()
metrics = {
    "recall_at_10": 0.72,
    "citation_accuracy": 0.88,
    "grounding_score": 0.82,
    "abstention_correct": 0.91,
}
result = gate.check(metrics)

if result.passed:
    print("Yayına hazır!")
else:
    print("Yayın kapısı başarısız:")
    for f in result.failures:
        print(f"  - {f}")
```

### 5. Denetim Kaydı
```python
from app.reliability.retrieval_audit_log import RetrievalAuditLog

audit = RetrievalAuditLog()
run_id = audit.log(query, method="multi_query", top_k=5, chunks=chunks)
print(f"Denetim kaydı: {run_id}")
```

### 6. İnsan İnceleme Kuyruğu
```python
from app.reliability.human_review_queue import HumanReviewQueue

queue = HumanReviewQueue()
if report.score < 0.7:
    review_id = queue.submit(
        question=query,
        chunks=chunks,
        answer=answer_text,
        confidence=report.score,
    )
    print(f"İnceleme kuyruğuna eklendi: {review_id}")
```

## Minimum Eşikler (MVP)
| Metrik | Eşik |
|--------|------|
| Recall@10 | ≥ 0.70 |
| Atıf doğruluğu | ≥ 0.85 |
| Dayanak skoru | ≥ 0.80 |
| Çekimser kalma doğruluğu | ≥ 0.90 |

## Kaynak Güveni
```python
from app.reliability.source_trust import SourceTrustScorer

trust_scorer = SourceTrustScorer()
for chunk in chunks:
    metadata = {"year": "2023", "source": "arxiv", "authors": "[Smith, Jones]"}
    trust = trust_scorer.score(chunk.paper_id, metadata)
    if trust.trust_score < 0.5:
        print(f"Düşük güvenilirlik: {chunk.paper_id} — {trust.reason}")
```
