"""MCP tool yüzeyi allow-list'i — TEK KAYNAK (single source of truth).

Tasarım ilkesi: **varsayılan kapalı, açıkça izin ver.**

``FastMCP.from_openapi()`` kendisine verilen spec'teki HER operasyonu bir tool'a
çevirir. Hektor web API'si 110+ operasyon sunar; bunların çoğu dış bir ajanın
görmemesi gereken yazma/tetikleme uçlarıdır (eğitim, onay, kill-switch, autodrive).
Bu yüzden spec, FastMCP'ye verilmeden ÖNCE burada budanır.

Neden spec budama (FastMCP ``route_maps`` yerine)?
- ``route_maps`` desen-sıralı ve "varsayılan açık" çalışır: yeni bir route eklendiğinde
  hiçbir desene uymazsa **tool olur**. Budama ise tersini garanti eder — allow-list'te
  olmayan hiçbir şey FastMCP'ye ulaşmaz.
- FastMCP sürüm/API değişimlerinden bağımsızdır (bkz. 2026-07-28 spec geçişi);
  saf sözlük işlemi olduğu için test edilmesi de kolaydır.

Bakım: web'e yeni bir SALT-OKUMA ucu eklenip MCP'den görünmesi isteniyorsa buraya
açıkça eklenir. Yazma/tetikleme uçları buraya EKLENMEZ.

⚠️ YAN-ETKİLİ GET TUZAĞI (Kademe-2 avında YAKALANDI — 2026-07-21):
"GET = salt-okuma" varsayımı bu depoda YANLIŞTIR. Allow-list'e "okuma" başlığı altında
eklenmiş ÜÇ GET ucu aslında kalıcı yazıyordu:

  * ``GET /api/backtest/{id}/risk``    → ``save_risk_report`` ile ``rr_<id>`` satırını
                                          EZİYORDU; içerik motorun verdiği sorgu
                                          parametrelerinden türüyordu (uyarıları
                                          susturup pozisyon boyutunu şişirmek mümkündü).
  * ``GET /api/understanding-score``   → ``record=true`` ile kalıcı snapshot + JSON yazar.
  * ``GET /api/sentinel/overview``     → ``run(persist=True)`` ile her çağrıda geçmişe yazar.

Bir ucu buraya eklerken HTTP metoduna DEĞİL, handler'ın gövdesine bak. Bu sınıfı
``tests/test_mcp_allowlist_side_effects.py`` yapısal olarak tarar (metod değil, KAYNAK
kodu inceler) — yeni bir yan-etkili GET eklenirse CI patlar.
"""

from __future__ import annotations

from typing import Any

# --- İZİN VERİLEN uçlar: (HTTP metodu, OpenAPI path'i) ---------------------------
#
# Yalnız salt-okuma / durum sorgulama uçları. Sıralama önemsizdir (küme semantiği).
# Path'ler OpenAPI spec'indeki HAM string'lerdir ({param} şablonları dahil) —
# `verify_allowlist()` her girdinin spec'te gerçekten var olduğunu doğrular, böylece
# bir route yeniden adlandırılırsa allow-list sessizce ölü kalmaz.
ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        # --- Soru-cevap (RAG) ---
        ("POST", "/api/ask"),
        # --- RAG öğrenme hattı (güvenli otomasyon) ---
        # Motor kalıcı enable/config yetkisi ALMAZ; yalnız durumu okuyup idempotent tek tur
        # isteyebilir. Tur kendi eşzamanlılık, bütçe ve training-pause kapılarını uygular.
        ("GET", "/api/rag-loop/status"),
        ("POST", "/api/rag-loop/run-once"),
        # --- Bilgi kartları (okuma) ---
        ("GET", "/api/cards/pending"),
        ("GET", "/api/cards/approved"),
        ("GET", "/api/card/{paper_id}"),
        # --- Backtest (okuma) ---
        ("GET", "/api/backtests"),
        ("GET", "/api/backtest/{backtest_id}/pine"),
        # ⛔ `/api/backtest/{id}/risk` BİLİNÇLİ OLARAK YOK: GET olmasına rağmen risk
        #    raporunu `rr_<backtest_id>` sabit anahtarıyla SQLite'a upsert eder
        #    (server.py `save_risk_report`) → her çağrı insanın gördüğü raporu EZER,
        #    üstelik içerik motorun seçtiği sorgu parametrelerinden türer. Bkz. YAN-ETKİLİ
        #    GET notu (aşağıda).
        # --- Sistem durumu ---
        ("GET", "/api/status"),
        ("GET", "/api/healthz"),
        ("GET", "/api/version"),
        ("GET", "/api/profile"),
        # --- Nöbetçi (Sentinel) — YALNIZ geçmiş okuma ---
        # ⛔ `/api/sentinel/overview` BİLİNÇLİ OLARAK YOK: GET olmasına rağmen
        #    `run(persist=True)` çağırıp her istekte geçmişe YAZAR (kendi docstring'i
        #    "Koşuyu geçmişe yazar" der). Tekrarlı çağrı, budama ile eski nöbetçi
        #    kayıtlarını dışarı iter. Okuma ihtiyacını `/history` zaten karşılar.
        ("GET", "/api/sentinel/history"),
        # --- Ajan haritası / koşu geçmişi (okuma) ---
        ("GET", "/api/agents"),
        ("GET", "/api/agents/graph"),
        ("GET", "/api/agents/runs"),
        ("GET", "/api/agents/runs/{run_id}"),
        # --- Makale havuzu (okuma) ---
        ("GET", "/api/papers"),
        # --- Öğrenme metrikleri (okuma) ---
        ("GET", "/api/learning/summary"),
        ("GET", "/api/rag-mastery"),
        # ⛔ `/api/understanding-score` BİLİNÇLİ OLARAK YOK: `record=true` sorgu
        #    parametresiyle çağrıldığında kalıcı snapshot + JSON rapor YAZAR
        #    (`record_understanding`). Parametre OpenAPI'den tool argümanı olarak
        #    türediği için motor onu kendisi `true` yapabilirdi. Salt-okuma ihtiyacını
        #    aşağıdaki `/history` ucu karşılar.
        ("GET", "/api/understanding-score/history"),
    }
)

# --- ASLA SUNULMAYAN uçlar (savunma katmanı) -------------------------------------
#
# Varsayılan-kapalı budama bunları zaten dışarıda bırakır; bu liste allow-list'e
# yanlışlıkla tehlikeli bir uç eklenirse **kurulumu düşürmek** için vardır (fail-closed).
# Path substring'i olarak eşleşir.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "/api/approvals",  # onay verme/reddetme — yalnız insan (Kural 8)
    "/api/supervisor/stop-all",  # kill-switch
    "/api/supervisor/clear-stop-all",  # kill-switch temizleme
    "/api/training/run",  # eğitim başlatma
    "/api/orchestration/autodrive",  # otonom sürüş
    "/api/orchestration/start",  # eğitim orkestrasyonu başlatma
    "/api/orchestration/resume",
    "/api/auto-lora/promote",  # adaptör terfisi
    "/api/auto-lora/train",
    "/api/auto-lora/enable",
    "/api/rag-loop/enable",  # kalıcı arka-plan yetkisi yalnız insan/panel
    "/api/rag-loop/config",  # motor kendi bütçe ve ağ kadansını değiştiremez
)

# NOT: bu liste ELLE bakım edilir ve tek başına yeterli DEĞİLDİR — asıl garanti
# `ALLOWED`'ın kapalı bir küme olmasıdır. Ayrıca `tests/test_mcp_allowlist.py`
# içindeki `test_hicbir_human_only_route_allowlist_te_degil`, FastAPI'nin GERÇEK
# bağımlılık grafiğini tarayarak hiçbir `require_human` ucunun allow-list'e
# sızmadığını doğrular (bu string listesine güvenmez).

_HTTP_METHODS = ("get", "post", "put", "delete", "patch", "head", "options", "trace")


class AllowlistError(RuntimeError):
    """Allow-list tutarsız — MCP sunucusu kurulmamalı (fail-closed)."""


def verify_allowlist(spec: dict[str, Any]) -> None:
    """Allow-list'in tutarlılığını doğrula; bozuksa ``AllowlistError`` fırlat.

    İki kontrol:
    1. Allow-list'teki hiçbir girdi ``FORBIDDEN_SUBSTRINGS`` ile çakışmamalı.
    2. Allow-list'teki her girdi spec'te GERÇEKTEN var olmalı (drift tespiti) —
       aksi halde route yeniden adlandırılınca allow-list sessizce ölü kalır.
    """
    for method, path in sorted(ALLOWED):
        for bad in FORBIDDEN_SUBSTRINGS:
            if bad in path:
                raise AllowlistError(
                    f"Allow-list yasaklı bir ucu içeriyor: {method} {path} "
                    f"(yasak desen: {bad}). MCP yüzeyi açılmadı."
                )

    paths = spec.get("paths", {})
    missing = [
        f"{method} {path}"
        for method, path in sorted(ALLOWED)
        if method.lower() not in paths.get(path, {})
    ]
    if missing:
        raise AllowlistError(
            "Allow-list'teki şu uçlar OpenAPI spec'inde bulunamadı (route yeniden "
            f"adlandırıldı mı?): {', '.join(missing)}"
        )


def filter_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Spec'i allow-list'e indirge — allow-list'te olmayan HER operasyon düşer.

    Girdi spec'i değiştirilmez (kopya döner). ``paths`` dışındaki üst-seviye alanlar
    (``components``, ``info``, ...) korunur; FastMCP şema çözümlemesi için gerekir.
    """
    verify_allowlist(spec)

    out = dict(spec)
    filtered: dict[str, Any] = {}

    for path, item in spec.get("paths", {}).items():
        kept = {
            method: op
            for method, op in item.items()
            if method.lower() not in _HTTP_METHODS or (method.upper(), path) in ALLOWED
        }
        # Yalnız metod-dışı anahtar (ör. "parameters") kaldıysa path'i tamamen at.
        if any(m.lower() in _HTTP_METHODS for m in kept):
            filtered[path] = kept

    out["paths"] = filtered
    return out
