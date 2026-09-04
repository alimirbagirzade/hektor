# Achilles Trader AI — geliştirme hedefleri
# uv varsa onu, yoksa pip/python'u kullanır.

UV := $(shell command -v uv 2> /dev/null)

.PHONY: help install test lint format typecheck run clean gen-data ci audit update web web-start web-stop web-log

help:
	@echo "Hedefler:"
	@echo "  install    - bağımlılıkları kur (uv sync veya pip install -e .[dev])"
	@echo "  test       - pytest"
	@echo "  lint       - ruff check"
	@echo "  format     - ruff format"
	@echo "  typecheck  - mypy"
	@echo "  audit      - bağımlılık güvenlik taraması (pip-audit, CVE)"
	@echo "  update     - GitHub'dan son sürüme güncelle (update.sh; macOS/Linux)"
	@echo "  ci         - lint + typecheck + test (CI ile aynı)"
	@echo "  gen-data   - sentetik OHLCV üret"
	@echo "  clean      - cache/artefakt temizle"

install:
ifdef UV
	uv sync --extra dev
	-uv run pre-commit install
else
	python -m pip install -e ".[dev]"
	-pre-commit install
endif

test:
ifdef UV
	uv run pytest -q
else
	pytest -q
endif

lint:
ifdef UV
	uv run ruff check .
else
	ruff check .
endif

format:
ifdef UV
	uv run ruff format .
else
	ruff format .
endif

typecheck:
ifdef UV
	uv run mypy app
else
	mypy app
endif

ci: lint typecheck test

# Tek komut güncelleme (macOS/Linux) — git pull + uv sync --extra dev + web restart.
update:
	@bash update.sh

# Bağımlılık güvenlik taraması — bilinen CVE'ler (PyPI advisory DB; ağ gerekir).
audit:
ifdef UV
	uv run --with pip-audit pip-audit
else
	pip-audit
endif

gen-data:
ifdef UV
	uv run achilles gen-data
else
	achilles gen-data
endif

web:
ifdef UV
	uv run achilles-web
else
	achilles-web
endif

web-start:
	@launchctl load ~/Library/LaunchAgents/com.achilles.web.plist 2>/dev/null; \
	sleep 2 && curl -sf http://localhost:8765/api/status > /dev/null && echo "Achilles Web calisiyor: http://localhost:8765" || echo "Baslatiliyor..."

web-stop:
	@launchctl unload ~/Library/LaunchAgents/com.achilles.web.plist 2>/dev/null; \
	lsof -ti:8765 | xargs kill -9 2>/dev/null; echo "Achilles Web durduruldu."

web-log:
	@tail -f ~/Library/Logs/achilles-web.log

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ build dist *.egg-info
	find . -name "*.pyc" -delete
