---
name: achilles-web
description: Achilles web API'sini MCP tool'ları olarak kullan — RAG sorgusu, durum, eğitim, backtest, sentez makaleleri, formül/kavram. Web arayüzünün yaptığı her şey tool olarak. Web değişince MCP'yi senkronla.
---

# /achilles-web — Achilles Web MCP Köprüsü

Achilles'in yerel web API'si (68 endpoint) MCP tool'ları olarak açılır. Sunucu:
`mcp_server/achilles_mcp.py` (FastMCP → çalışan web'e proxy). Ön koşul: web açık
(`uv run achilles-web` → http://127.0.0.1:8765).

## Kurulum (bir kez)
```bash
bash scripts/sync-mcp.sh            # tool doğrula + 'achilles' MCP kaydı
# sonra Claude Code'u yeniden başlat ki MCP bağlansın
```

## Sık kullanılan tool'lar (MCP adı → ne yapar)
| Tool | İş |
|------|----|
| `api_status` | sistem durumu (model, makale sayısı, ollama) |
| `api_ask` | RAG sorusu sor (kaynaklı cevap; kaynak yoksa uydurmaz) |
| `api_rag_mastery` | "RAG kaç makaleyi anladı %" panosu |
| `api_papers` / `api_ingest` | makale listele / PDF indeksle |
| `api_synthesis_reports*` | sentez makaleleri listele/üret/indir |
| `api_research_formulas` / `api_concept_graph` | formüller / kavram grafiği |
| `api_backtests` | backtest kayıtları |
| `api_learning_training_runs` | eğitim loss eğrileri (grafik verisi) |
| `api_auto_lora_status` | LoRA pipeline durumu |

> ⚠ Ağır/state değiştiren tool'lar (`api_training_run`, `api_research_run`,
> `api_auto_lora_train`) LLM/CPU yoğun ve OOM riski taşır — eğitim döngüsü
> çalışırken tetikleme.

## Senkron protokolü (önemli)
`app/web/server.py` değişip yeni endpoint eklendiğinde → MCP tool listesi
otomatik güncellenmez; **`bash scripts/sync-mcp.sh`** çalıştırıp Claude Code'u
tazele. Spec in-process üretildiği için elle düzenleme gerekmez. Detay:
[docs/PROTOKOL_MCP.md](../../../docs/PROTOKOL_MCP.md).

## Sorun giderme
- Tool'lar boşsa: web sunucusu açık mı? (`curl http://127.0.0.1:8765/api/status`)
- `claude mcp list` → 'achilles' "Failed to connect" ise: `mcp_server/achilles_mcp.py`
  doğrudan çalışıyor mu test et: `uv run python mcp_server/achilles_mcp.py` (stdio bekler).
