"""Shared test fixtures: isolated temp DB / chroma per test session."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest

from app.config.settings import Settings

# --- Geliştiricinin `.env` dosyasını test oturumundan TAMAMEN çıkar ---
#
# `Settings.model_config` `env_file=".env"` taşır ve bu yol CWD'ye görelidir; pytest
# depo kökünden koştuğu için geliştiricinin gerçek `.env`'i ayarlara sızıyordu. Depo
# ağacı izolasyonu (aşağıdaki `_isolate_storage`) yalnız BİRKAÇ anahtarı env ile
# eziyordu; geri kalan her ayar (`*_API_TOKEN`, rate-limit, trusted-hosts, chunk
# boyutları...) makineye göre değişiyor, testleri makineye bağımlı yapıyordu —
# `.env`'inde API token olan geliştiricide 116 test düşüyor, olmayanda hepsi geçiyordu.
#
# Bu satır conftest IMPORT edilirken (test modülleri toplanmadan önce) çalışır; sonraki
# tüm `Settings()` kurulumları — `get_settings()` ve testlerin doğrudan çağrıları dahil —
# `.env`'i görmez. Testler ayarları yalnız açık env değişkeni veya kwarg ile değiştirir.
Settings.model_config["env_file"] = None


def _ollama_running() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _ollama_running():
        return
    skip = pytest.mark.skip(reason="Ollama çalışmıyor — @pytest.mark.ollama testi atlandı")
    for item in items:
        if item.get_closest_marker("ollama"):
            item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _isolate_storage(tmp_path_factory):
    """Tüm veri/durum yollarını tmp'ye al ve GERÇEK ağaca yazımı YASAKLA.

    `HEKTOR_ROOT_PATH` tüm türetilmiş dizinlerin (data/, storage/, models/,
    reports/) tabanıdır → testler artık depo ağacına dosya bırakamaz. Eskiden
    yalnız sqlite+chroma izoleydi; her tam koşu `data/lora_sft/lora_sft.jsonl` ve
    `data/training/jsonl/*.jsonl` üretiyor, bir sonraki koşuda data-gate GO verip
    orkestrasyon testini düşürüyordu (sıra-bağımlı flakiness).

    `HEKTOR_BACKGROUND_LOOPS_ENABLED=false`: TestClient lifespan'i tetiklediğinde
    unattended supervisor GERÇEK bir abonelik motoru (codex/claude) doğurmaya
    çalışıyordu — kota yakımı + CLAUDE.md Kural 8 ihlali.
    """
    base = tmp_path_factory.mktemp("hektor_test")
    os.environ["HEKTOR_ROOT_PATH"] = str(base)
    os.environ["HEKTOR_SQLITE_PATH"] = str(base / "test.db")
    os.environ["HEKTOR_CHROMA_PATH"] = str(base / "chroma")
    os.environ["HEKTOR_ALLOW_FAKE_EMBEDDINGS"] = "true"
    os.environ["HEKTOR_BACKGROUND_LOOPS_ENABLED"] = "false"
    # clear cached settings so env overrides take effect
    from app.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()

    repo_root = Path(settings_mod.PROJECT_ROOT)
    watched = [
        repo_root / "data" / "lora_sft",
        repo_root / "data" / "training" / "jsonl",
        repo_root / "storage",
    ]
    before = {d: _snapshot(d) for d in watched}
    yield
    leaked = sorted(
        str(p.relative_to(repo_root)) for d in watched for p in _snapshot(d) - before[d]
    )
    if leaked:  # pragma: no cover - yalnız izolasyon bozulunca çalışır
        pytest.fail("Testler GERÇEK depo ağacına yazdı (izolasyon bozuk): " + ", ".join(leaked))


def _snapshot(directory: Path) -> set[Path]:
    """Dizindeki dosyaların anlık kümesi (yoksa boş)."""
    if not directory.is_dir():
        return set()
    return {p for p in directory.rglob("*") if p.is_file()}


@pytest.fixture(autouse=True)
def _settings_cache_reset():
    """Test `HEKTOR_*` env'i geçici değiştirdiyse ayar cache'ini SONRA temizle.

    `get_settings` lru_cache'lidir: `monkeypatch.setenv` env'i geri alır ama cache
    eski (ör. tmp kök) nesneyi tutmaya devam ederdi → sonraki testler yanlış kökle
    koşardı. Teardown'da cache'i boşaltmak bu sızıntıyı kapatır.
    """
    yield
    from app.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _reset_web_rate_limiter():
    """Her testten ÖNCE global hız sınırlayıcı pencerelerini temizle.

    TestClient tüm web testlerinde aynı IP'den ("testclient") vurur ve global
    `_rate_limiter` (120/dk, 60s kayan pencere) süreç-boyu paylaşılır. CI suite'i
    ~26s'de bitince TÜM web istekleri tek 60s penceresine paketlenir; toplam 120'yi
    aşınca geç çalışan alakasız testler 429 alır (örn. test_web_training_gate KeyError
    'ok'). Yerelde (yavaş) pencere kendiliğinden sıfırlandığı için gizliydi. Çözüm:
    her teste taze pencere → test-izolasyonu. Yalnız server zaten import edilmişse
    dokunur (import yan etkisi yok). Dedicated rate-limit testi kendi RateLimiter
    örneğini kurar → etkilenmez."""
    import sys

    mod = sys.modules.get("app.web.server")
    if mod is not None:
        for attr in ("_rate_limiter", "_upload_rate_limiter"):
            limiter = getattr(mod, attr, None)
            if limiter is not None:
                limiter._hits.clear()
    yield


@pytest.fixture
def store():
    from app.memory.sqlite_store import SqliteStore

    return SqliteStore()
