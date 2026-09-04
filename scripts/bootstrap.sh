#!/usr/bin/env bash
# Hektor Trader AI — ilk kurulum + uçtan uca duman testi (çevrimdışı).
# Ollama gerektirmez: fake embedding + sentetik veri ile çalışır.
set -euo pipefail

cd "$(dirname "$0")/.."

export HEKTOR_ALLOW_FAKE_EMBEDDINGS=true

echo "==> Bağımlılıklar kuruluyor"
if command -v uv >/dev/null 2>&1; then
  uv sync --extra dev
  RUN="uv run"
else
  python -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -e ".[dev]"
  RUN=""
fi

echo "==> Sistem başlatılıyor"
$RUN hektor init
$RUN hektor status

echo "==> Sentetik veri üretiliyor"
$RUN hektor gen-data

echo "==> Backtest (örnek strateji) çalıştırılıyor"
$RUN hektor backtest data/market/raw/synthetic.csv

echo "==> Testler"
$RUN pytest -q -m "not ollama and not slow"

echo ""
echo "✅ Duman testi tamam. Sıradaki adımlar:"
echo "   - PDF'leri data/papers/raw_pdf/ içine koy, 'hektor ingest' çalıştır."
echo "   - (Opsiyonel) Ollama kurup 'hektor ask \"...\"' ile RAG dene."
