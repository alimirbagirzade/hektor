"""Allow-list'teki GET uçları GERÇEKTEN salt-okuma mu — yapısal tarama.

NEDEN VAR (Kademe-2 avı, 2026-07-21): `mcp_server/allowlist.py` "yalnız salt-okuma uçları"
sözleşmesini beyan ediyordu, ama sözleşme HTTP METODUNA göre denetleniyordu. Mevcut test
(`test_yazma_metodlari_tamamen_elenir`) yalnız POST/PUT/DELETE/PATCH'i eliyordu → gövdesinde
kalıcı yazma yapan bir **GET** sessizce geçiyordu. Üç uç tam olarak böyle sızmıştı:

  * GET /api/backtest/{id}/risk   → save_risk_report (sabit anahtar → ÜZERİNE yazma)
  * GET /api/understanding-score  → record=true ile kalıcı snapshot + JSON
  * GET /api/sentinel/overview    → run(persist=True) ile her çağrıda geçmişe yazma

Bu test metoda değil KAYNAK KODUNA bakar: allow-list'teki her ucun handler gövdesini
kalıcılık işaretçilerine karşı tarar. Heuristiktir (çağrı grafiğinin tamamını izlemez),
ama yakaladığı sınıf tam olarak yukarıdaki sınıftır — ve yeni bir yan-etkili GET
eklenirse CI'ı patlatır.
"""

from __future__ import annotations

import inspect
import re

import pytest
from mcp_server.allowlist import ALLOWED

# Handler gövdesinde görülürse "bu uç yazıyor olabilir" diyen işaretçiler.
# Not: yalnız YAZMA fiilleri; `load_`/`list_`/`get_` gibi okuma adları kasten dışarıda.
_YAZMA_ISARETCILERI: tuple[tuple[str, str], ...] = (
    (r"\bpersist\s*=\s*True\b", "persist=True ile kalıcılaştırma"),
    (r"\brecord\b\s*:\s*bool", "record= parametresi (kalıcılaştırma anahtarı)"),
    (r"\.save_\w+\(", "store.save_* çağrısı"),
    (r"\brecord_\w+\(", "record_* çağrısı"),
    (r"\.merge\(", "SQLAlchemy merge (upsert)"),
    (r"\.write_text\(", "dosyaya yazma"),
    (r"\.add_event\(", "olay kaydı yazma"),
    (r"\bcreate_\w+\(", "create_* çağrısı"),
)


def _route_tablosu() -> dict[tuple[str, str], object]:
    """(METOD, path) → endpoint fonksiyonu eşlemesi (FastAPI'nin GERÇEK route tablosu)."""
    from app.web.server import app

    tablo: dict[tuple[str, str], object] = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        endpoint = getattr(route, "endpoint", None)
        for method in getattr(route, "methods", set()) or set():
            if path and endpoint:
                tablo[(method.upper(), path)] = endpoint
    return tablo


@pytest.mark.parametrize(("method", "path"), sorted(ALLOWED))
def test_allowlist_ucu_yazma_isaretcisi_tasimiyor(method: str, path: str) -> None:
    """Allow-list'teki her uç, gövdesinde kalıcı yazma işaretçisi TAŞIMAMALI.

    Bu test başarısız olursa: ucu allow-list'ten ÇIKAR (ya da gerçekten salt-okuma bir
    varyant ekle). "GET olduğu için güvenlidir" gerekçesi bu depoda GEÇERSİZDİR.
    """
    # Denetlenmiş güvenli otomasyon istisnası: tek tur isteği kendi lock/bütçe/training
    # kapılarına sahiptir; enable/config ve eğitim yetkisi vermez.
    if (method, path) == ("POST", "/api/rag-loop/run-once"):
        return

    tablo = _route_tablosu()
    endpoint = tablo.get((method, path))
    if endpoint is None:
        pytest.skip(f"{method} {path} route tablosunda yok (spec drift testi ayrı)")

    try:
        kaynak = inspect.getsource(endpoint)
    except (OSError, TypeError):  # pragma: no cover - kaynak okunamıyorsa atla
        pytest.skip(f"{method} {path} kaynağı okunamadı")

    bulgular = [aciklama for desen, aciklama in _YAZMA_ISARETCILERI if re.search(desen, kaynak)]
    assert not bulgular, (
        f"Allow-list'teki {method} {path} handler'ı kalıcı YAZMA işaretçisi taşıyor: "
        f"{bulgular}. Allow-list yalnız salt-okuma uçları içindir — 'GET olduğu için "
        f"güvenli' varsayımı bu depoda kırıktır (bkz. mcp_server/allowlist.py)."
    )


def test_yan_etkili_getler_allowliste_geri_sizmadi() -> None:
    """REGRESYON: avda çıkarılan üç yan-etkili GET tekrar EKLENMEMELİ."""
    yasak = {
        ("GET", "/api/backtest/{backtest_id}/risk"),
        ("GET", "/api/understanding-score"),
        ("GET", "/api/sentinel/overview"),
    }
    geri_gelen = yasak & set(ALLOWED)
    assert not geri_gelen, (
        f"Yan etkili GET uçları allow-list'e geri sızdı: {geri_gelen}. "
        "Bunlar GET olmalarına rağmen kalıcı yazma yapar."
    )
