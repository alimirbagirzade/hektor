"""Bulut sağlayıcı kullanım politikası (B · öğretmen çıktısı) — kod tarafındaki kapı.

Bulut modelinin çıktısını yerel modelin EĞİTİM verisine koymak, sağlayıcıların kullanım
koşullarında "rakip model eğitimi" yasağına girebilir (docs/TASARIM_FAZ3_BULUT.md §1). Bu yüzden:

- Eğitim için izin VARSAYILMAZ. Abonelik ve API sağlayıcıları için politika ``izin yok``.
- Açık ağırlıklı öğretmen için bile politika ancak lisans kaydı + insan onayıyla açılır; bu
  dosyada açık bir girdi YOKTUR (açmak kod değişikliği + Kademe 2 + kullanıcı kararı ister).
- ``cloud_origin_lines`` eğitim kapıları tarafından çağrılır: bulut kökenli satır = NO-GO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Eğitim satırının ``metadata`` alanında bulut kökenini gösteren anahtar/değerler.
CLOUD_ORIGIN_KEYS = ("origin", "source", "teacher")
CLOUD_ORIGIN_PREFIXES = ("cloud", "bulut", "teacher_api", "claude", "openai", "gpt", "codex")


@dataclass(frozen=True)
class ProviderPolicy:
    provider: str
    second_opinion: str  # izinli_insan_tiklamasi | api_anahtari_gerekir
    training_use: bool
    reason: str
    sources: tuple[str, ...]


POLICIES: dict[str, ProviderPolicy] = {
    "claude_code_cli": ProviderPolicy(
        provider="claude_code_cli",
        second_opinion="izinli_insan_tiklamasi",
        training_use=False,
        reason="Tüketici Şartları (8 Eki 2025): rakip ürün geliştirmek ve AI/ML modeli eğitmek "
        "yasak; API anahtarı dışında otomatik erişim yasak. Claude Code OAuth 'ordinary use' "
        "içindir; ürün entegrasyonu için API anahtarı önerilir. Bu yüzden yalnız tur başına "
        "insan tıklaması, resmi ve değiştirilmemiş `claude` ikilisi; toplu/otomatik döngü YOK.",
        sources=(
            "https://www.anthropic.com/legal/consumer-terms",
            "https://code.claude.com/docs/en/legal-and-compliance",
        ),
    ),
    "anthropic_api": ProviderPolicy(
        provider="anthropic_api",
        second_opinion="api_anahtari_gerekir",
        training_use=False,
        reason="Ticari Şartlar D.4 (17 Haz 2025): 'train competing AI models' Anthropic'in açık "
        "onayı olmadan yasak. İkinci görüş uygun; eğitim için yazılı onay gerekir. İstemci bu "
        "dalda UYGULANMADI (proje kararı: API anahtarı yok).",
        sources=("https://www.anthropic.com/legal/commercial-terms",),
    ),
    "openai": ProviderPolicy(
        provider="openai",
        second_opinion="api_anahtari_gerekir",
        training_use=False,
        reason="OpenAI Kullanım Şartları: 'Use Output to develop models that compete with "
        "OpenAI' ve programatik çıktı çıkarma yasak; Codex otomasyon için API anahtarı öneriyor. "
        "Resmi sayfa bu ortamdan okunamadı (403) — insan okumalı.",
        sources=("https://openai.com/policies/row-terms-of-use/",),
    ),
    "fake": ProviderPolicy(
        provider="fake",
        second_opinion="izinli_insan_tiklamasi",
        training_use=False,
        reason="Test sağlayıcısı; ağ çağrısı yapmaz. Çıktısı da eğitime girmez.",
        sources=(),
    ),
}


def policy_for(provider: str) -> ProviderPolicy | None:
    return POLICIES.get(provider)


def training_use_allowed(provider: str) -> tuple[bool, str]:
    """Bu sağlayıcının çıktısı eğitim verisine girebilir mi? (bu dalda HİÇBİRİ için evet değil)"""
    pol = POLICIES.get(provider)
    if pol is None:
        return False, f"'{provider}' için kayıtlı politika yok — eğitim için izin varsayılmaz."
    return pol.training_use, pol.reason


def _is_cloud_meta(meta: Any) -> bool:
    if not isinstance(meta, dict):
        return False
    for key in CLOUD_ORIGIN_KEYS:
        val = str(meta.get(key) or "").strip().lower()
        if val and val.startswith(CLOUD_ORIGIN_PREFIXES):
            return True
    return False


def cloud_origin_lines(lines: list[str]) -> list[int]:
    """Eğitim JSONL satırlarından bulut kökenli olanların 0-tabanlı indeksleri (NO-GO nedeni)."""
    bad: list[int] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and _is_cloud_meta(row.get("metadata")):
            bad.append(i)
    return bad
