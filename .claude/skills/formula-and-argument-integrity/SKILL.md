# Formula and Argument Integrity Skill

## Amaç
LaTeX formüllerin sözdizimsel bütünlüğünü ve argüman zincirlerinin
mantıksal tamlığını doğrula; eksiklik tespit edildiğinde komşu chunk'ı öner.

## Tetiklenme Koşulları
- Chunk metninde LaTeX formülü tespit edildiğinde
- "eksik formül", "yarım denklem", "incomplete formula" gibi işaretçiler görüldüğünde
- Argüman zinciri (öncül → sonuç) kesintiye uğradığında

## Formül Bütünlüğü Kontrolü

### LaTeX Delimiter Dengesi
Aşağıdaki eşleştirilmiş çiftlerin dengeli olduğunu doğrula:
- `$...$` (inline)
- `$$...$$` (display)
- `\(...\)` (inline paren)
- `\[...\]` (display bracket)
- `\begin{equation}...\end{equation}`

```python
from app.memory.contextual_chunker import ContextualChunker
from app.rlm.safe_tools import formula_check

flags = ContextualChunker().annotate(chunks, paper_title=title)
for f in flags:
    if f.has_incomplete_formula:
        print(f"EKSIK FORMÜL: {f.chunk_id} — komşu: {f.previous_chunk_id} / {f.next_chunk_id}")

# Tek metin için hızlı denetim (RLM güvenli aracı, deny-by-default allowlist'te):
print(formula_check(chunk.text))
```

### Otomatik Tamamlama Yasağı
Formül eksikse asla tamamlama yapmayın. Bunun yerine:
1. Önceki/sonraki chunk'ta devamı ara
2. Kullanıcıya uyarı ver: "Bu formül kaynak metinde tamamlanmamış"

## Argüman Bütünlüğü Kontrolü

### Öncül-Sonuç Zinciri
Türkçe: dolayısıyla, bu nedenle, zira, çünkü
İngilizce: therefore, thus, since, because, hence

```python
from app.memory.contextual_chunker import ContextualChunker

for f in ContextualChunker().annotate(chunks):
    if f.has_incomplete_argument:
        print(f"Argüman sonuca ulaşmadı — sonraki chunk'a bak: {f.next_chunk_id}")
```

> Not: Ayrı bir `FormulaVerifier`/`ArgumentVerifier` modülü YOKTUR; bütünlük bayrakları
> `ChunkQualityFlags` içinde üretilir, formül denetimi `safe_tools.formula_check` ile yapılır.

## Bağlamsal Chunker ile Entegrasyon
```python
from app.memory.contextual_chunker import ContextualChunker

annotator = ContextualChunker()
flags = annotator.annotate(chunks)

for flag in flags:
    if flag.needs_adjacent_context:
        print(f"{flag.chunk_id} komşu bağlam gerektiriyor")
        print(f"  Önceki: {flag.previous_chunk_id}")
        print(f"  Sonraki: {flag.next_chunk_id}")
```

## Rapor Formatı
```
[FORMÜL BÜTÜNLÜK RAPORU]
Toplam formül: N
  Tam: M
  Eksik: K
    - chunk_id: formül_özeti...

[ARGÜMAN ZİNCİRİ RAPORU]
Toplam argüman içeren chunk: N
  Tam zincir: M
  Eksik öncül: K
  Eksik sonuç: J
```
