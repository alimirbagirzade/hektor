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


# --- Çevrimdışı koruma: işaretsiz testler CANLI Ollama'ya gidemez ---
#
# `LocalLLM` / `EmbeddingService` Ollama ayaktaysa gerçek moda geçer (sahteye düşmez).
# Mock'lanmamış bir test, makinede Ollama açıkken gerçek `/api/embed` / `/api/generate`
# çağırıyordu: meşgul bir Ollama'da (mastery kuyruğu, synth-qa) kapı 30+ dakika takıldı,
# Ollama kapalıyken ise Windows'ta her bağlantı denemesi ~2 sn bekletiyordu (/api/status
# tek istekte 3 deneme → ~6 sn). Testlerin makine durumundan bağımsız olması için
# (CLAUDE.md: testler çevrimdışı) Ollama hedefli GERÇEK ağ istekleri anında ConnectError
# ile düşürülür → kod "Ollama kapalı" yolunu izler. Enjekte edilen httpx.MockTransport
# gerçek taşıyıcı olmadığından etkilenmez; `@pytest.mark.ollama` testleri muaftır.
# Teşhis: HEKTOR_TEST_OLLAMA_GUARD_LOG=<dosya> → engellenen her istek "test url" yazılır.
_OLLAMA_DEFAULT_PORT = 11434
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _is_ollama_target(host: str | None, port: int | None) -> bool:
    from urllib.parse import urlsplit

    from app.config import get_settings

    host = (host or "").lower()
    configured = urlsplit(get_settings().ollama_host)
    if host == (configured.hostname or "").lower() and port == (
        configured.port or _OLLAMA_DEFAULT_PORT
    ):
        return True
    return host in _LOOPBACK and port == _OLLAMA_DEFAULT_PORT


@pytest.fixture(autouse=True)
def _block_live_ollama(request, monkeypatch):
    if request.node.get_closest_marker("ollama"):
        yield
        return
    import requests

    log_path = os.environ.get("HEKTOR_TEST_OLLAMA_GUARD_LOG")

    def _blocked(url: str) -> str:
        if log_path:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"{request.node.nodeid} {url}\n")
        return f"test çevrimdışı: canlı Ollama engellendi ({url})"

    real_httpx = httpx.HTTPTransport.handle_request
    real_requests = requests.adapters.HTTPAdapter.send

    def _httpx_guard(self, req):
        port = req.url.port or (443 if req.url.scheme == "https" else 80)
        if _is_ollama_target(req.url.host, port):
            raise httpx.ConnectError(_blocked(str(req.url)), request=req)
        return real_httpx(self, req)

    def _requests_guard(self, req, *args, **kwargs):
        from urllib.parse import urlsplit

        u = urlsplit(req.url)
        if _is_ollama_target(u.hostname, u.port or (443 if u.scheme == "https" else 80)):
            raise requests.exceptions.ConnectionError(_blocked(req.url))
        return real_requests(self, req, *args, **kwargs)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _httpx_guard)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _requests_guard)
    yield


@pytest.fixture(autouse=True)
def _hermetic_train_load_doctor(request, monkeypatch):
    """`train --run` yolundaki train-load-doctor'ı sahtele (GO).

    Doktor canlı Ollama `/api/ps` + `nvidia-smi` okur; makinede büyük model yüklüyken
    (ör. qwen3:30b, VRAM dolu) NO-GO → exit 4 verip onay/kayıt testlerini makine
    durumuna göre kırıyordu. Doktorun kendi testleri gerçek fonksiyonu sınar.
    """
    if request.module.__name__.endswith("test_train_load_doctor"):
        return
    from app.training import train_load_doctor

    monkeypatch.setattr(
        train_load_doctor,
        "run_train_doctor",
        lambda **kw: train_load_doctor.TrainDoctorReport(verdict="GO"),
    )


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
