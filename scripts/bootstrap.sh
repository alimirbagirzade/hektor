#!/usr/bin/env bash
# Achilles Trader AI — ilk kurulum + uçtan uca duman testi (çevrimdışı).
# Ollama gerektirmez: fake embedding + sentetik veri ile çalışır.
set -euo pipefail

cd "$(dirname "$0")/.."

export ACHILLES_ALLOW_FAKE_EMBEDDINGS=true

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
$RUN achilles init
$RUN achilles status

echo "==> Sentetik veri üretiliyor"
$RUN achilles gen-data

echo "==> Backtest (örnek strateji) çalıştırılıyor"
$RUN achilles backtest data/market/raw/synthetic.csv

echo "==> Testler"
$RUN pytest -q -m "not ollama and not slow"

echo ""
echo "✅ Duman testi tamam. Sıradaki adımlar:"
echo "   - PDF'leri data/papers/raw_pdf/ içine koy, 'achilles ingest' çalıştır."
echo "   - (Opsiyonel) Ollama kurup 'achilles ask \"...\"' ile RAG dene."
