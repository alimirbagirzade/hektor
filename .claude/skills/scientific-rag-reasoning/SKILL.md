# Scientific RAG Reasoning Skill

## Amaç
Bilimsel makalelerdeki formül, argüman ve metodoloji adımlarını RAG bağlamından
doğru biçimde çıkarmak; eksik formül veya argüman sürekliliği tespit etmek.

## Tetiklenme Koşulları
- Kullanıcı bilimsel/akademik bir soruyu RAG ile sormak istediğinde
- Formül veya matematiksel ifade içeren chunk'lar bağlamda yer aldığında
- Metodoloji adımları veya ispat akışı çıkarılmak istendiğinde

## Adımlar

### 1. Bağlam Kalitesi Kontrolü
```python
from app.memory.contextual_chunker import ContextualChunker
from app.ingestion.chunker import TextChunk

annotator = ContextualChunker()
flags = annotator.annotate(chunks, paper_title="...")

incomplete_formula_chunks = [f for f in flags if f.has_incomplete_formula]
incomplete_arg_chunks = [f for f in flags if f.has_incomplete_argument]
```

### 2. Yeterliliği Değerlendir
```python
from app.verification.context_sufficiency import ContextSufficiencyClassifier

classifier = ContextSufficiencyClassifier()
result = classifier.classify(query, chunks, quality_flags=flags)

if not result.can_answer:
    print("Bağlam yetersiz:", result.missing_items)
```

### 3. Formül Bütünlüğünü Doğrula
```python
from app.verification.formula_verifier import FormulaVerifier

verifier = FormulaVerifier()
for chunk in chunks:
    checks = verifier.verify_chunk(chunk)
    for check in checks:
        if not check.is_complete:
            print(f"Eksik formül: {check.formula_text[:80]}")
```

### 4. Argüman Zincirini İzle
```python
from app.verification.argument_verifier import ArgumentVerifier

arg_verifier = ArgumentVerifier()
for chunk in chunks:
    arg_check = arg_verifier.verify(chunk.text)
    if not arg_check.is_complete:
        print(f"Eksik argüman — öncül: {arg_check.has_premise}, sonuç: {arg_check.has_conclusion}")
```

### 5. Cevap Kalitesi
- Formül eksiği varsa: komşu chunk'ı getir veya kullanıcıya uyarı ver
- Argüman zinciri kırıksa: "Bu argüman devamı için ek kaynak gerekiyor" notunu ekle
- Tüm formüller tam ve argüman zinciri sağlamsa: normal RAG cevabı üret

## Kısıtlar
- Formülleri asla tamamlama — yalnızca mevcut kaynakta olan formülü kullan
- Kaynak olmayan bir matematiksel iddiayı "bu kaynaklarda geçmiyor" olarak işaretle
- LLM çevrimdışıysa retrieval sonuçlarını ham olarak göster
