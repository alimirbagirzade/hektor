"""Motor (yerel abonelikli CLI ajanı) uçları — SALT-OKUMA motor seçici beslemesi.

`server.py`'ye TEK satır (`include_router`) ile bağlanır → sıcak dosya minimal dokunulur.

SÖZLEŞME (üçü de test'le sabitlenmiştir — bkz. tests/test_engines_api.py):
1. **Salt-okuma** — hiçbir uç süreç doğurmaz, eğitim başlatmaz, dosya yazmaz. PATH yoklaması
   (`shutil.which`) tek yan etkidir ve o da cache'lidir.
2. **Kimlik bilgisi DÖNMEZ** — token / e-posta / API anahtarı / oturum çerezi ASLA. Motorlar
   kendi CLI oturumlarıyla girişlidir; Achilles bu bilgiye ne bakar ne de sahiptir
   (CLAUDE.md + [[no-api-local-subscription-only]]).
3. **Giriş durumu uydurulmaz** — `logged_in` daima `null`. Bir CLI'nin abonelik oturumu
   ancak çalıştırılınca anlaşılır; yoklamak için spawn gerekirdi (kota yakar + salt-okuma
   sözleşmesini bozar). Kural 7: bilinmiyorsa "bilinmiyor" denir.

Kimlik doğrulama: `require_auth` (token boşsa lokal-açık, diğer router'larla aynı davranış).
Sürücü scope'u bu ucu görebilir — motor listesi bir yetki değil, salt bilgidir.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.web.security import require_auth

router = APIRouter(prefix="/api/engines", tags=["engines"], dependencies=[Depends(require_auth)])


@router.get("")
def list_engines() -> dict[str, Any]:
    """Kayıtlı motorları listele — ad, etiket, kurulu mu, girişli mi, kota uyarısı.

    Hiçbir şey tetiklemez. `selectable=false` olan motor ⚡ RUN için seçilemez ve
    `/api/orchestration/autodrive` ucu onu zaten reddeder (fail-closed, çift kapı).
    """
    from app.orchestration import engines

    return {
        "engines": engines.describe_all(),
        "default": engines.DEFAULT_ENGINE,
        "login_note": engines.LOGIN_UNKNOWN_NOTE,
    }


@router.post("/rescan")
def rescan_engines() -> dict[str, Any]:
    """PATH yoklama cache'ini temizleyip yeniden tara (kullanıcı motoru yeni kurduysa).

    Yan etkisi YALNIZ süreç-içi cache'in boşaltılmasıdır — süreç doğurmaz, dosya yazmaz.
    """
    from app.orchestration import engines

    engines.reset_probe_cache()
    return {
        "engines": engines.describe_all(),
        "default": engines.DEFAULT_ENGINE,
        "login_note": engines.LOGIN_UNKNOWN_NOTE,
    }
