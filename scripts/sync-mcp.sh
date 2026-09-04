#!/usr/bin/env bash
# Achilles MCP senkron — web değiştikten sonra MCP tool listesini tazele.
# MCP spec'i in-process üretildiği için "senkron" = MCP'yi yeniden kaydet/başlat.
# Bu script: (1) tool üretimini doğrular, (2) MCP'yi yeniden kaydeder.
set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$(pwd)"

echo "[sync-mcp] tool üretimi doğrulanıyor..."
# Hata çıktısı bir dosyaya alınır; /dev/null'a atmak gerçek nedeni (ör. eksik fastmcp)
# gizliyordu — başarısızlıkta aşağıda gösterilir.
ERRLOG="$(mktemp)"
N=$(uv run python -c "
import asyncio
from mcp_server.achilles_mcp import mcp
print(len(asyncio.run(mcp.list_tools())))
" 2>"$ERRLOG" | tail -1)
echo "[sync-mcp] üretilen tool: ${N:-?}"

if [ -z "${N:-}" ] || [ "${N:-0}" -lt 1 ]; then
  echo "[sync-mcp] HATA: tool üretilemedi. Python çıktısı:"
  sed 's/^/    /' "$ERRLOG" >&2
  if grep -q "No module named 'fastmcp'" "$ERRLOG"; then
    echo "[sync-mcp] ÇÖZÜM: fastmcp opsiyonel 'mcp' extra'sındadır → uv sync --extra mcp" >&2
  fi
  rm -f "$ERRLOG"
  exit 1
fi
rm -f "$ERRLOG"

echo "[sync-mcp] MCP yeniden kaydediliyor (achilles)..."
claude mcp remove achilles 2>/dev/null || true
claude mcp add achilles -- uv run --project "$REPO" python mcp_server/achilles_mcp.py
echo "[sync-mcp] tamam — Claude Code'u yeniden başlat / oturumu tazele ki MCP bağlansın."
