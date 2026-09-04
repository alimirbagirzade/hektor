# /hypothesis-evaluator — Trading hipotez denetimi + birleşik eval-runner

Bir trading fikrinin **"kesin sinyal" değil test-edilebilir hipotez** olduğunu denetler (talimat
§9/§12; CLAUDE.md Kural 1-4) ve değerlendirmeleri tek `--type` arayüzünden ReleaseGate'e bağlar.
Tavsiye/kesinlik dili → REJECT. Çıktı asla yatırım tavsiyesi değildir.

## Ne zaman kullan
- Bir araştırma/sentez çıktısı "trading hipotezi" sunuyorsa, üretime/backtest'e geçmeden önce
  test-edilebilirlik denetimi.
- "Bu fikir sinyal gibi mi yazılmış?" (garanti/%100/risksiz dili) taraması.
- RAG retrieval kalitesini eşikten geçirme (`rag-retrieval` tipi).

## Komutlar
```bash
# Trading hipotezlerini değerlendir (JSON listesi veya JSONL; her öğe str ya da dict)
uv run achilles eval-runner --type trading-hypothesis --input hyps.jsonl --json
uv run achilles eval-runner --type trading-hypothesis --input hyps.json --strict   # kapı geçilemezse hata
```

Örnek `hyps.jsonl` (her satır bir hipotez):
```
"Yüksek volatilitede momentum zayıflayabilir; backtest ile test edilmeli, örneklem-dışı doğrulama ve komisyon+slippage maliyetleri dahil, risk stop-loss ile."
{"hypothesis_text": "...", "risk_notes": "drawdown/stop-loss", "assumptions": ["komisyon dahil", "OOS"]}
```

Programatik:
```python
from app.evals.eval_runner import EvalRunner
from app.evals.trading_hypothesis_evaluator import evaluate_hypothesis
r = evaluate_hypothesis("…")                 # r.verdict, r.checklist, r.warnings
res = EvalRunner().run("trading-hypothesis", hypotheses=[...], strict=False)   # res.passed, res.metrics
res = EvalRunner().run("rag-retrieval", questions=golden, retriever=retr)      # ReleaseGate recall@10
```

## Denetim maddeleri (checklist) ve verdict
| Madde | Kural | Anlam |
|-------|-------|-------|
| `testable` | 2 | ölçülebilir koşul / backtest / "test edilmeli" var mı |
| `costs` | 3 | komisyon/slippage/maliyet farkındalığı |
| `out_of_sample` | 2,4 | OOS / look-ahead / walk-forward farkındalığı |
| `no_advice` | 1 | "garanti/%100/risksiz/sure-fire/guarantee/can't lose" YOK |
| `risk_noted` | — | risk/drawdown/stop-loss notu |

| verdict | koşul |
|---------|-------|
| `rejected` | tavsiye dili VAR ya da test-edilemez (HARD — Kural 1,2) |
| `needs_revision` | maliyet veya OOS eksik |
| `candidate` | tüm çekirdek maddeler geçti (yine de "aday; backtest geçmeden strateji değil") |

## Kurallar
- **Tavsiye dili → REJECT** (Kural 1). `candidate` bile olsa strateji değil "aday".
- **--strict:** ReleaseGate eşiği geçilemezse `EvalGateError` (production engellenir).
- **no eval/exec; deterministik** (salt-regex denetçi).
- Bağlanmamış tipler (`rag-answer`/`lora`/`rlm-reward`) → net `NotImplementedError`
  (`rlm-reward` app/rlm bağımlı → eş zamanlı oturum bitene dek ertelendi).
- Rapor `reports/evals/eval_<type>_<ts>.json`'a yazılır.
