"""driver.py — AutoDriver: deep-hunt aşamasını headless `claude -p` ile otonom sürer.

Kullanıcının "tek tuş → Claude aboneliğiyle devreye soksun" isteğinin çekirdeği. Orchestrator'ın
salt-okuma aşamalarını ilerletir, ZORUNLU `deep-hunt` (Kademe-2 derin av) aşamasını headless
`claude -p` ile otonom çalıştırır, PASS ise onay kapısına kadar ilerletir.

GÜVENLİK (Kural 8): onay kapısında DURUR — gerçek eğitim yine TAZE insan onayı bekler; train
aşaması varsayılan HANDOFF. AutoDriver eğitim BAŞLATMAZ.

LOCAL-FIRST: `claude -p` ABONELİK CLI'sidir (API key DEĞİL; weekly-bug-scan.ps1 ile aynı yol).
Spawn yalnız `execute=True` + `claude` PATH'te iken yapılır; varsayılan DRY-RUN komutu döner
(Kural 8 deseni). `runner` enjekte edilebilir → testler gerçek spawn olmadan çalışır.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.orchestration import engine_procs, engines, verdict_audit
from app.orchestration.engine_procs import DEFAULT_GRACE_S
from app.orchestration.engines import DEFAULT_ENGINE
from app.orchestration.orchestrator import TrainingOrchestrator
from app.web import driver_scope

log = logging.getLogger(__name__)

# Depo kökü: app/orchestration/driver.py → parents[2]. MCP config'te mutlak yol gerekir
# (motor başka bir cwd'de doğabilir).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# claude -p çıktısının SON satırındaki verdict işaretçisi.
_VERDICT_RE = re.compile(r"HEKTOR_HUNT_VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)
HUNT_TIMEOUT_S = 1800  # 30 dk — CPU'da derin av uzun sürebilir.

# "Sür" modu verdict işaretçisi — av modunun DESENİ birebir aynı (son satır, PASS|FAIL,
# bulunamazsa güvenli tarafta FAIL). İşaretçi ADI kasıtlı FARKLIDIR: aynı olsaydı bir sür
# koşusunun "PASS"i `parse_hunt_verdict` tarafından "derin av geçti" sanılıp `hunt_ack=true`
# yazdırabilirdi — yani ajan, av yapmadan eğitim kapısını açardı (Kural 8).
_DRIVE_VERDICT_RE = re.compile(r"HEKTOR_DRIVE_VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)

# Sür modu av modundan UZUNDUR: veri hattı adımları (carding → RLM → curate → assemble)
# CPU'da saatler alabilir; av tek geçişlik salt-okuma taramadır. Ayrı sabit → biri
# değişince diğeri sessizce kaymaz.
DRIVE_TIMEOUT_S = 3600  # 60 dk

# Sürücü token TTL'i sür modu için AYRI hesaplanır. driver_scope.DEFAULT_TTL_S (2100s) av
# moduna (1800s) göre ayarlıdır; sür modunda kullanılsaydı token koşunun ORTASINDA
# (~35. dk) ölür, MCP çağrıları 401 almaya başlar ve ajan sebebini anlamadan tıkanırdı.
# ⚠️ SORUN 2 (Kademe-2 avında yakalandı, P7'de düzeltildi): sür modu bağlandığında
# `_drive_pipeline` bu TTL'i AÇIKÇA `mint(run_id, ttl_s=DRIVE_TOKEN_TTL_S)` ile geçirir
# (aşağıda). Aksi halde varsayılan TTL (2100s) DRIVE_TIMEOUT_S'den (3600s) kısa olduğundan
# token koşunun ~35. dakikasında ölür ve MCP çağrıları sebepsiz 401 almaya başlar.
# Test yalnız sabitin büyüklüğünü değil, mint çağrısının GERÇEKTEN bu TTL ile yapıldığını
# doğrular (bkz. tests/test_drive_mode.py::test_drive_mint_ttl_gecer).
DRIVE_TOKEN_TTL_S = DRIVE_TIMEOUT_S + 300

# Araç deny-list'i motor kayıt tablosunda durur (motora ÖZGÜ bayraklar).
# Geriye dönük uyum + tek kaynak: buradan yeniden dışa aktarılır.
DISALLOWED_TOOLS = engines.DISALLOWED_TOOLS

# Çocuğa GEÇİRİLMEYECEK insan sırları. NOT: anahtarı silmek YETMEZ — Settings
# `env_file=".env"` kullandığı için silinen anahtar dotenv'den geri okunur
# (deneyle doğrulandı). Bu yüzden anahtar açıkça BOŞ STRING'e ezilir; env kaynağı
# pydantic-settings'te dotenv'den önceliklidir.
_HUMAN_SECRET_ENV = ("HEKTOR_API_TOKEN",)

# Çocuğa geçirilmeyecek AYAR-EZME değişkenleri. `--safe-mode` "admin-managed (policy)
# settings still apply" der; bu değişkenler harici bir ayar dosyası işaret ederek
# safe-mode'a rağmen hook/ayar geri getirebilir. Ebeveynin ortamı kirlenmişse
# (kalıcı shell profili, sistem-genel env) sürücü kısıtı sessizce delinirdi.
# Derinlemesine savunma: spawn öncesi açıkça temizlenir.
_SETTINGS_OVERRIDE_ENV = (
    "CLAUDE_CODE_MANAGED_SETTINGS_PATH",
    "CLAUDE_CODE_REMOTE_SETTINGS_PATH",
    "CLAUDE_CODE_MOCK_REMOTE_SETTINGS",
)

# runner sözleşmesi: (command, timeout_s, env) -> (returncode, combined_output)
Runner = Callable[..., tuple[int, str]]


def build_hunt_prompt(run: dict[str, Any]) -> str:
    """Headless claude -p için Kademe-2 derin av promptu (SALT RAPOR; kod/eğitim YOK).

    ⚠️ BAĞIMSIZ VERDICT (P8): motor artık serbest bir "PASS" değil, YAPILANDIRILMIŞ KANIT
    (taranan dosyalar + alt-sistemler + bulgular) üretmek zorundadır. Kanıt, motordan
    bağımsız olarak dosya sistemiyle doğrulanır (bkz. ``verdict_audit``); uydurulmuş, boş
    ya da iç-tutarsız kanıt PASS'i düşürür → motorun öz-beyanı TEK kanıt değildir (Kural 8).
    """
    adapter = run.get("adapter_name", "hektor_lora")
    return (
        "Sen Hektor deposunda KADEME-2 derin adversarial bug-avı çalıştıran bir ajansın. "
        "İLK İŞ: depo kökündeki CLAUDE.md'yi Read ile OKU (safe-mode'da oto-keşif kapalıdır; "
        "kurallar oradadır). Bu, eğitim ÖNCESİ ZORUNLU denetimdir. KESİNLİKLE: yalnız RAPOR üret; "
        "KOD DEĞİŞTİRME, git commit/push YAPMA, EĞİTİM BAŞLATMA, hiçbir dosyaya yazma. "
        "Alt-sistem başına paralel bul → şüpheci adversarial doğrula (varsayılan çürütülmüş) "
        "→ yalnız onaylanan CİDDİ (HIGH/BLOCKER) bulguları say.\n"
        "\n"
        "SON İKİ ÇIKTI (sırayla). ÖNCE tek satırda yapılandırılmış KANIT bloğu — çıktının "
        "verdict'ten hemen önceki bölümüne şu işaretçiyle bir JSON nesnesi yaz:\n"
        f"{verdict_audit.EVIDENCE_MARKER}\n"
        '{"scanned_files": [{"path": "<depo köküne göreli yol>", "line": <1-tabanlı satır no>, '
        '"quote": "<o satırın BİREBİR metni>"}, ...], '
        '"subsystems": ["<taradığın alt-sistem adı>", ...], '
        '"findings": [{"severity": "HIGH|BLOCKER|MEDIUM|LOW", "file": "<yol>", '
        '"summary": "<kısa>"}]}\n'
        f"KANIT KURALI: en az {verdict_audit.MIN_SCANNED_FILES} FARKLI ve depoda GERÇEKTEN "
        f"var olan dosya, en az {verdict_audit.MIN_SUBSYSTEMS} farklı alt-sistem listele. "
        "OKUMA-KANITI ZORUNLU: her dosya için Read çıktısındaki AYIRT EDİCİ bir satırın "
        f"numarasını ve o satırın BİREBİR metnini (kırpılınca ≥{verdict_audit.MIN_QUOTE_LEN} "
        "karakter) ver; her dosyada FARKLI bir satır seç (aynı jenerik satırı, ör. "
        "'from __future__ import annotations', birden çok dosyaya kanıt olarak KULLANMA). "
        "Denetim bu satırı dosyayı bağımsızca okuyup doğrular → dosyayı gerçekten okumadan "
        "kanıt üretemezsin. Uydurma/eksik yol, yanlış satır alıntısı ya da PASS derken "
        "HIGH/BLOCKER bulgu listelemek verdict'i DÜŞÜRÜR. PASS ise 'findings' HIGH/BLOCKER "
        "içermemeli (boş ya da yalnız MEDIUM/LOW olabilir).\n"
        "\n"
        "SONRA çıktının SON SATIRI tam olarak şu biçimde olmalı:\n"
        "HEKTOR_HUNT_VERDICT: PASS    (ciddi bulgu yok — eğitim güvenli)\n"
        "veya\n"
        "HEKTOR_HUNT_VERDICT: FAIL    (ciddi bulgu var — önce düzeltilmeli)\n"
        f"Bağlam: orkestrasyon koşusu, adapter={adapter}."
    )


def build_drive_prompt(run: dict[str, Any]) -> str:
    """Headless motor için "sür" modu promptu (MCP araçlarıyla veri hattını ilerletir).

    Av modundan farkı: burada ajanın İŞ YAPMASI beklenir — ama yalnız Hektor MCP
    araçlarıyla, dosyaya doğrudan dokunmadan ve EĞİTİME ASLA başlamadan.
    """
    adapter = run.get("adapter_name", "hektor_lora")
    return (
        "Sen Hektor deposunda VERİ HATTINI ilerleten bir sürücü ajansın. "
        "İLK İŞ: depo kökündeki CLAUDE.md'yi Read ile OKU — bağlayıcı kurallar oradadır.\n"
        "\n"
        "ARAÇ KULLANIMI: işleri YALNIZCA Hektor MCP araçlarıyla (`mcp__*`) yap. "
        "Dosyaları DOĞRUDAN DÜZENLEME (Edit/Write yok), kabuk komutu çalıştırma, "
        "git commit/push yapma. Read/Grep/Glob yalnız DURUM ANLAMAK içindir.\n"
        "\n"
        "HEDEF: veri hattı adımlarını sırayla ilerlet — RAG durumu → gerekiyorsa tek RAG "
        "öğrenme turu → carding → RLM → curate → assemble. Önce RAG durumunu oku; döngü "
        "çalışmıyorsa yalnız `run-once` ile bir tur iste. Kalıcı enable/config değiştirme. "
        "Her adımdan önce durumu MCP ile oku; adım zaten tamamsa ATLA, baştan çalıştırma.\n"
        "\n"
        "⛔ EĞİTİM BAŞLATMA. Gerçek LoRA eğitimini ASLA tetikleme; onay uçlarını "
        "(`/api/approvals/{id}/approve`) ve eğitim uçlarını (`/api/training/run`) ÇAĞIRMA. "
        "Bunlar insan yetkisidir (CLAUDE.md Kural 8) ve sürücü kimliğinle 403 alırsın — "
        "403'ü aşmaya ÇALIŞMA, beklenen davranıştır. TAZE İNSAN ONAYI gerektiren her "
        "adımda DUR ve neyin beklediğini raporla.\n"
        "\n"
        "Çıktının SON SATIRI tam olarak şu biçimde olmalı:\n"
        "HEKTOR_DRIVE_VERDICT: PASS    (ilerletilebilen adımlar ilerletildi; "
        "onay kapısında ya da yapılacak iş kalmadığı için temiz durdu)\n"
        "veya\n"
        "HEKTOR_DRIVE_VERDICT: FAIL    (ilerlenemedi — sebebi raporda açıkla)\n"
        f"Bağlam: orkestrasyon koşusu, adapter={adapter}."
    )


def build_mcp_config(root: str | None = None) -> dict[str, Any]:
    """Doğurulan motora verilecek MCP config'i üret (kullanıcı kaydına BAĞIMLI DEĞİL).

    `--strict-mcp-config` ile birlikte kullanılır → kullanıcı düzeyindeki `claude mcp add`
    kayıtları YOK SAYILIR, yalnız burada tanımlanan sunucu yüklenir. Böylece spawn kendi
    kendine yeter (makine değişse de çalışır).

    ⚠️ SIR YAZILMAZ: sürücü token'ı bu dosyaya KONULMAZ. MCP sunucusu motorun ÇOCUĞU olarak
    doğar ve ortamı ondan miras alır; token zaten `build_child_env` ile motorun ortamındadır
    (bkz. mcp_server/hektor_mcp.py:driver_headers). Token'ı config'e yazmak, kısa ömürlü
    bir sırrı diske düşürürdü.
    """
    repo_root = root or str(_REPO_ROOT)
    return {
        "mcpServers": {
            "hektor": {
                "command": "uv",
                "args": [
                    "run",
                    "--project",
                    repo_root,
                    "--extra",
                    "mcp",  # fastmcp opsiyonel `mcp` extra'sındadır
                    "python",
                    str(Path(repo_root) / "mcp_server" / "hektor_mcp.py"),
                ],
            }
        }
    }


def write_mcp_config(path: str | Path, root: str | None = None) -> str:
    """MCP config'i diske yaz ve yolunu döndür (sır içermez — bkz. build_mcp_config)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(build_mcp_config(root), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(target)


# Sür modu MCP config dosyalarının konduğu dizin. Depo içinde SABİT (motor başka bir cwd'de
# doğsa da mutlak yol geçilir); .gitignore `storage/` çıktısını zaten yok sayar.
_DRIVE_MCP_DIR = _REPO_ROOT / "storage" / "mcp"
# run_id dosya adına gömüleceği için yol-geçişi/enjeksiyon karşısında güvenli tabana indir.
_SAFE_RUN_RE = re.compile(r"[^A-Za-z0-9_-]+")


def drive_mcp_config_path(run_id: str) -> str:
    """Bir koşu için MCP config dosyasının mutlak yolu (koşu başına ayrı dosya).

    run_id serbest metin olabileceğinden dosya adına gömülmeden önce güvenli tabana
    indirilir (yol-geçişi savunması); config'in kendisi sır taşımaz (bkz. build_mcp_config).
    """
    safe = _SAFE_RUN_RE.sub("_", run_id).strip("_")[:64] or "run"
    return str(_DRIVE_MCP_DIR / f"drive-{safe}.json")


def build_drive_command(
    run: dict[str, Any], mcp_config_path: str, engine: str = DEFAULT_ENGINE
) -> list[str]:
    """ "Sür" modu argv'si — MCP erişimli, sertleştirilmiş şablon.

    GÜVENLİK: sür modu `--safe-mode` KULLANAMAZ (o bayrak MCP'yi de kapatır). Bunun yerine
    safe-mode'un kapattığı kanallar tek tek kapatılır; gerekçe `engines._CLAUDE_DRIVE_ARGV`
    ve `docs/SCOPE_ISOLATION.md`.
    """
    return engines.build_drive_command(engine, build_drive_prompt(run), mcp_config_path)


def build_hunt_command(run: dict[str, Any], engine: str = DEFAULT_ENGINE) -> list[str]:
    """Seçili motorun argv'si (shell yok → enjeksiyon yok; prompt argv ÖĞESİ olarak geçer).

    Motor adı kayıt tablosundan gelir; bilinmeyen ad ValueError ile REDDEDİLİR.

    GÜVENLİK: sertleştirme bayrakları motorun `argv_template`'inde durur
    (`app/orchestration/engines.py`). `claude` motoru `--safe-mode` +
    `--strict-mcp-config` + `--disallowedTools` ile doğurulur; gerekçe için
    `engines.Engine.hardened` ve `docs/SCOPE_ISOLATION.md`.
    """
    return engines.build_command(engine, build_hunt_prompt(run))


def build_child_env(token: str, run_id: str) -> dict[str, str]:
    """Doğurulan motorun ortamı: insan sırları BOŞA ezilir, sürücü kimliği eklenir.

    Anahtarı ``del`` etmek yerine boş string'e ezmek ZORUNLU — bkz. ``_HUMAN_SECRET_ENV``.
    """
    env = dict(os.environ)
    for key in _HUMAN_SECRET_ENV:
        env[key] = ""
    # Ayar-ezme kanalları: burada SİLMEK doğru (boş string de geçerli bir yol sayılıp
    # hataya yol açabilir; yokluk = "ayar yok" demektir).
    for key in _SETTINGS_OVERRIDE_ENV:
        env.pop(key, None)
    env[driver_scope.DRIVER_TOKEN_ENV] = token
    env[driver_scope.DRIVER_RUN_ID_ENV] = run_id
    return env


def _parse_verdict(output: str, pattern: re.Pattern[str], marker: str) -> dict[str, Any]:
    """Ortak verdict ayıklayıcı: SONDAN ilk eşleşme; bulunamazsa FAIL (fail-closed).

    Av ve sür modları AYNI deseni paylaşır — yalnız işaretçi adı farklıdır.
    """
    match = None
    for line in reversed((output or "").splitlines()):
        match = pattern.search(line)
        if match:
            break
    if match is None:
        return {
            "verdict": "unknown",
            "passed": False,
            "summary": f"Verdict satırı ({marker}) bulunamadı — güvenli tarafta FAIL.",
        }
    verdict = match.group(1).upper()
    return {"verdict": verdict, "passed": verdict == "PASS", "summary": (output or "")[-600:]}


def parse_hunt_verdict(output: str) -> dict[str, Any]:
    """claude çıktısından verdict satırını (sondan) ayıkla → {verdict, passed, summary}."""
    return _parse_verdict(output, _VERDICT_RE, "HEKTOR_HUNT_VERDICT")


def parse_drive_verdict(output: str) -> dict[str, Any]:
    """Sür modu çıktısından verdict satırını ayıkla (av moduyla AYNI sözleşme)."""
    return _parse_verdict(output, _DRIVE_VERDICT_RE, "HEKTOR_DRIVE_VERDICT")


def engine_available(engine: str = DEFAULT_ENGINE) -> bool:
    """Motor CLI'si PATH'te KURULU mu (giriş durumu DEĞİL — o ancak çalıştırınca anlaşılır)."""
    return engines.available(engine)


def claude_available() -> bool:
    """Geriye dönük uyumluluk: varsayılan (claude) motorunun kurulu olup olmadığı."""
    return engine_available(DEFAULT_ENGINE)


# ⛔ DURDUR ile kesilen sürecin dönüş kodu (POSIX SIGTERM sözleşmesiyle uyumlu negatif
# değil, kendi işaretçimiz): drive() bunu görünce "kullanıcı durdurdu" der, verdict
# yokluğunu "av başarısız" gibi raporlamaz.
STOPPED_RC = -99

# STOP_ALL yoklama aralığı (saniye). Küçük tutulur: DURDUR'a basan kullanıcı sürecin
# saniyeler içinde ölmesini bekler. Yoklama yalnız bir dosya-var-mı kontrolüdür (ucuz).
STOP_POLL_S = 1.0


def _resolve_executable(command: list[str]) -> list[str]:
    """argv[0]'ı MUTLAK yola çöz — çalışma dizinindeki taklitçi binary'yi engeller.

    `shutil.which` çağrılırken cwd BİLİNÇLİ olarak arama yolundan çıkarılır: Windows'ta
    `which`/`CreateProcess` varsayılan olarak önce çalışma dizinine bakar, bu da depo
    köküne bırakılmış sahte bir `claude.exe`'nin gerçek CLI yerine koşmasına yol açardı.

    Çözülemezse komut OLDUĞU GİBİ bırakılır → `Popen` `FileNotFoundError` verir ve
    çağıran bunu "motor kurulu değil" olarak temiz biçimde raporlar (sessiz düşüş yok).
    """
    if not command:
        return command
    # PATH'ten cwd ve boş girdileri at (boş girdi POSIX'te "geçerli dizin" demektir).
    yol = os.pathsep.join(
        p
        for p in os.environ.get("PATH", "").split(os.pathsep)
        if p and Path(p).resolve() != Path.cwd().resolve()
    )
    tam = shutil.which(command[0], path=yol)
    return [tam, *command[1:]] if tam else command


def _stop_all_active() -> bool:
    """STOP_ALL kill-switch'i etkin mi (savunmacı — yoklama patlarsa koşuyu kesme)."""
    try:
        from app.agents.runtime import supervisor

        return bool(supervisor.is_stop_all_active())
    except Exception:
        log.debug("STOP_ALL yoklanamadı", exc_info=True)
        return False


def _default_runner(
    command: list[str],
    timeout: int,
    env: dict[str, str] | None = None,
    *,
    run_id: str = "",
) -> tuple[int, str]:
    """Motoru doğur ve KESİLEBİLİR biçimde bekle.

    ⛔ NEDEN Popen + yoklama (eski `subprocess.run` DEĞİL): `subprocess.run` süreç tutamacını
    kimseye vermeden bloklar. Tutamaç olmadan ⛔ DURDUR koşan motoru KESEMEZ — STOP_ALL
    yalnız bir bayrak dosyası yazar, süreci öldürmez. Eskiden motor 30 dk zaman aşımına
    kadar koşup abonelik kotasını yakmaya devam ediyordu (bkz. app/orchestration/
    engine_procs.py). Burada süreç kaydedilir, STOP_ALL periyodik yoklanır ve etkinleşirse
    süreç gerçekten sonlandırılır.

    shell=False + argv listesi → kabuk enjeksiyonu yok. Çıktı (stdout+stderr) birleşik.
    env: insan sırları temizlenmiş + sürücü kimliği eklenmiş ortam (build_child_env).

    ⛔ MUTLAK YOL + SABİT CWD (Kademe-2 avında yakalandı, 3/3 onay):
    Motor eskiden argv[0]='claude' ile, mutlak yol olmadan ve `cwd` pinlenmeden
    doğuruluyordu. Windows'ta hem `shutil.which` hem `CreateProcess` arama yolunun
    BAŞINDA süreç çalışma dizinine bakar → çalışma dizinine bırakılan sahte bir
    `claude.exe` gerçek CLI'nin YERİNE koşardı. Sertleştirme bayrakları sahte binary'ye
    hiçbir şey yaptıramaz: taklitçi yalnız son satıra `HEKTOR_HUNT_VERDICT: PASS`
    yazarak ZORUNLU derin av kapısını düşürürdü (Kural 8).
    Ayrıca sabitlenmemiş cwd, avın YANLIŞ AĞACI tarayıp "temiz" demesine yol açardı.
    """
    command = _resolve_executable(command)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        # Av HER ZAMAN bu deponun kökünü tarar — sunucuyu kim nereden başlattıysa değil.
        cwd=str(_REPO_ROOT),
    )
    engine_procs.register(run_id, proc)
    stopped = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Zaman aşımı: süreci öldür, sözleşmeyi koru (çağıran TimeoutExpired bekler).
                with contextlib.suppress(Exception):
                    proc.kill()
                with contextlib.suppress(Exception):
                    proc.communicate(timeout=DEFAULT_GRACE_S)
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                # communicate(timeout=...) zaman aşımında ÇIKTIYI KAYBETMEZ (belgelenmiş
                # davranış): yakalayıp yeniden çağırmak okumaya kaldığı yerden devam eder.
                out, err = proc.communicate(timeout=min(STOP_POLL_S, remaining))
                break
            except subprocess.TimeoutExpired:
                if _stop_all_active():
                    stopped = True
                    if run_id:
                        engine_procs.terminate_run(run_id)
                    else:
                        _terminate_single(proc)
                    out, err = "", ""
                    with contextlib.suppress(Exception):
                        out, err = proc.communicate(timeout=DEFAULT_GRACE_S)
                    break
    finally:
        engine_procs.unregister(run_id, proc)
    if stopped:
        return STOPPED_RC, (out or "") + (err or "")
    return proc.returncode, (out or "") + (err or "")


def _terminate_single(proc: subprocess.Popen) -> None:
    """run_id verilmediğinde (kayıt yokken) tek süreci kes — kayıt yolunun yedeği."""
    with contextlib.suppress(Exception):
        proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=DEFAULT_GRACE_S)
    with contextlib.suppress(Exception):
        if proc.poll() is None:
            proc.kill()


class AutoDriver:
    """deep-hunt'ı headless claude -p ile süren otonom orkestrasyon sürücüsü."""

    def __init__(self, orchestrator: TrainingOrchestrator | None = None) -> None:
        self.orch = orchestrator or TrainingOrchestrator()

    def drive(
        self,
        run_id: str,
        *,
        execute: bool = False,
        engine: str = DEFAULT_ENGINE,
        runner: Runner | None = None,
        timeout: int = HUNT_TIMEOUT_S,
        mode: str = "hunt",
    ) -> dict[str, Any]:
        """Otonom sürüş — `mode` iki AYRI işi ayırır:

        * ``mode="hunt"`` (varsayılan, geriye dönük): salt-okuma aşamaları → ZORUNLU
          Kademe-2 derin avı headless motorla sür → PASS ise ``hunt_ack``+onaya ilerlet.
          Av modu ``--safe-mode`` ile doğar (MCP KAPALI) — salt rapor üretir (Kural 8 kapısı).
        * ``mode="drive"`` (⚡ RUN): motora Hektor MCP araçlarına erişim ver ve VERİ HATTINI
          ilerlet. MCP açık, ``--safe-mode`` YOK; verdict AYRI işaretçi kullanır
          (``HEKTOR_DRIVE_VERDICT``) → sür PASS'i av ``hunt_ack``'ini AÇMAZ. Bkz.
          ``_drive_pipeline`` ve docs/ROADMAP_MOTOR_BAGLAMA.md P7.

        Varsayılan `execute=False`: gerçek spawn YOK, çalıştırılacak komutu döner (DRY-RUN).
        `runner` enjekte edilirse gerçek spawn yerine o kullanılır (test). `engine` kayıt
        tablosundan gelir (bilinmeyen ad reddedilir); süreç başlatmayan motorlar (ör. `local`)
        otonom sürüşte kullanılamaz."""
        if mode == "drive":
            return self._drive_pipeline(run_id, execute=execute, engine=engine, runner=runner)
        if mode != "hunt":
            return {
                "ok": False,
                "reason": f"Bilinmeyen sürüş modu: {mode!r} (beklenen: hunt|drive).",
            }
        try:
            eng = engines.get_engine(engine)
        except ValueError as exc:
            return {"ok": False, "reason": str(exc)}
        if not eng.spawns:
            return {
                "ok": False,
                "reason": (
                    f"Motor '{eng.name}' süreç başlatmaz (doğrudan yerel hat) — "
                    "otonom derin av için spawn eden bir motor seç."
                ),
            }

        run = self.orch.store.get_run(run_id)
        if run is None:
            return {"ok": False, "reason": f"Koşu bulunamadı: {run_id}"}

        # 1) Salt-okuma aşamalarını deep-hunt bloğuna kadar ilerlet.
        snap = self.orch.run_until_blocked(run_id)
        run_now = snap.get("run") or {}
        cur, status = run_now.get("current_stage"), run_now.get("status")

        if cur != "deep-hunt" or status != "blocked":
            # Zaten geçmiş ya da başka yerde bloklanmış/bitmiş → otonom sürüş gereksiz.
            return {
                "ok": True,
                "drove": False,
                "reason": f"deep-hunt'ta blocked değil (current={cur}, status={status})",
                **snap,
            }

        command = build_hunt_command(run, eng.name)

        if not execute:
            self.orch.store.add_event(
                run_id,
                "deep-hunt",
                "info",
                f"Otonom sürüş DRY-RUN: {eng.label} komutu hazır (execute=False; spawn yok).",
            )
            return {
                "ok": True,
                "drove": False,
                "dry_run": True,
                "engine": eng.name,
                "quota_warning": eng.quota_warning,
                "command": command,
                "needs": f"execute=True + kurulu `{eng.binary}` CLI (giriş kendi CLI'sinde)",
                **self.orch.status(run_id),
            }

        # Gerçek spawn yolunda `run_id`'yi bağla → süreç kaydedilir ve ⛔ DURDUR onu
        # gerçekten kesebilir. Enjekte edilen runner'ın imzası KORUNUR (test yolu
        # (command, timeout, env) bekler; ona run_id geçirmek imzayı kırardı).
        run_fn = runner or functools.partial(_default_runner, run_id=run_id)
        # FAIL-CLOSED: sertleştirilmemiş motor DOĞURULMAZ. Araç kısıtı olmayan bir motor
        # auth'suz yerel CLI'yi (`hektor approval-approve`) veya 127.0.0.1:8765'i
        # çağırıp kendi eğitimini onaylayabilir → scope katmanı tamamen delinir (Kural 8).
        # `runner` enjekte edilmişse gerçek spawn yoktur (test yolu) → kısıt aranmaz.
        if runner is None and not eng.hardened:
            self.orch.store.add_event(
                run_id,
                "deep-hunt",
                "error",
                f"Otonom sürüş REDDEDİLDİ: {eng.label} araç-seviyesinde kısıtlanamıyor.",
            )
            return {
                "ok": False,
                "reason": (
                    f"{eng.label} sertleştirilmiş değil — AutoDriver yalnız araç-kısıtlı "
                    "motor doğurur. Kısıtsız motor kendi eğitimini onaylayabilir (Kural 8). "
                    "Bkz. docs/SCOPE_ISOLATION.md."
                ),
                "engine": eng.name,
                "hardened": False,
                "command": command,
            }
        if runner is None and not engine_available(eng.name):
            return {
                "ok": False,
                "reason": f"`{eng.binary}` CLI PATH'te yok — {eng.label} kurulu olmalı.",
                "engine": eng.name,
                "command": command,
            }

        self.orch.store.add_event(
            run_id,
            "deep-hunt",
            "info",
            f"Otonom sürüş: {eng.label} ile Kademe-2 derin av başladı. {eng.quota_warning}",
        )
        # Sürücü kimliği: bu koşuya bağlı, kısa ömürlü token. Motor bununla insan-yalnız
        # uçlarda 403 alır (Kural 8). Koşu bitince `finally` ile MUTLAKA iptal edilir.
        token = driver_scope.mint(run_id)
        try:
            rc, output = run_fn(command, timeout, build_child_env(token, run_id))
        except Exception as exc:  # spawn/timeout — koşuyu çökertme, deep-hunt bloklu kalır
            log.exception("AutoDriver: %s çalıştırılamadı", eng.name)
            self.orch.store.add_event(run_id, "deep-hunt", "error", f"Otonom sürüş hatası: {exc}")
            return {
                "ok": False,
                "reason": f"{eng.label} çalıştırılamadı: {exc}",
                "engine": eng.name,
                "command": command,
            }
        finally:
            driver_scope.revoke_run(run_id)

        # ⛔ Kullanıcı DURDUR'a bastı → motor kesildi. Bunu "av başarısız" gibi raporlamak
        # YANILTICI olurdu (av koşmadı, kesildi); ayrı bir durum olarak döner.
        if rc == STOPPED_RC:
            self.orch.store.add_event(
                run_id,
                "deep-hunt",
                "warning",
                "⛔ DURDUR (STOP_ALL): motor süreci kesildi — deep-hunt bloklu kalır.",
            )
            return {
                "ok": True,
                "drove": True,
                "stopped": True,
                "hunt_passed": False,
                "reason": "Koşu ⛔ DURDUR ile kesildi (STOP_ALL etkin).",
                "engine": eng.name,
                **self.orch.status(run_id),
            }

        verdict = parse_hunt_verdict(output)
        # ⚠️ BAĞIMSIZ VERDICT (P8, Kural 8): motorun "PASS" beyanı TEK kanıt değildir. Yapısal
        # kanıt (taranan dosyalar/alt-sistemler) motordan BAĞIMSIZ olarak dosya sistemiyle
        # doğrulanır — kanıtsız/uydurma/iç-tutarsız bir PASS zorunlu av kapısını AÇAMAZ.
        # Denetim ASLA koşuyu çökertmemeli (fail-closed): motordan gelen çıktı düşman-kontrollü
        # sayılır, beklenmedik bir hata güvenli tarafta "geçmedi" olarak sonuçlanır.
        try:
            audit = verdict_audit.audit_hunt_evidence(output, root=_REPO_ROOT)
        except Exception:  # pragma: no cover — verdict_audit zaten iç-fail-closed; bu son kale
            log.exception("Bağımsız verdict denetimi beklenmedik hata verdi — reddedildi")
            audit = {
                "ok": False,
                "reason": "Bağımsız denetim beklenmedik hata verdi (fail-closed → reddedildi).",
                "scanned_count": 0,
                "subsystem_count": 0,
            }
        if not (verdict["passed"] and audit["ok"]):
            if verdict["passed"] and not audit["ok"]:
                # Motor PASS dedi ama bağımsız denetim geçmedi → en kritik durum: öz-beyanın
                # tek başına kapıyı açmasını engelleyen tam olarak budur.
                msg = (
                    "Derin av PASS beyanı BAĞIMSIZ denetimden geçemedi → deep-hunt bloklu "
                    f"kalır (motorun öz-beyanı tek kanıt değildir — Kural 8): {audit['reason']}"
                )
            else:
                msg = f"Derin av {verdict['verdict']} → deep-hunt bloklu kalır (eğitim ilerlemez)."
            self.orch.store.add_event(run_id, "deep-hunt", "warning", msg)
            return {
                "ok": True,
                "drove": True,
                "hunt_passed": False,
                "verdict": verdict,
                "audit": audit,
                **self.orch.status(run_id),
            }

        # 2) PASS + bağımsız denetim GEÇTİ → hunt_ack=true işaretle + onay kapısına kadar
        #    ilerlet (orada DURUR — Kural 8).
        params = dict(run.get("params") or {})
        params["hunt_ack"] = True
        params["hunt_driven_by"] = f"{eng.name}-autodrive"
        self.orch.store.update_run(
            run_id, params_json=json.dumps(params, ensure_ascii=False, default=str)
        )
        self.orch.store.add_event(
            run_id,
            "deep-hunt",
            "info",
            "Derin av PASS + bağımsız kanıt denetimi GEÇTİ "
            f"({audit['scanned_count']} dosya / {audit['subsystem_count']} alt-sistem teyit "
            "edildi) → hunt_ack=true; onay kapısına ilerletiliyor (eğitim onay bekler).",
        )
        final = self.orch.run_until_blocked(run_id)
        return {
            "ok": True,
            "drove": True,
            "hunt_passed": True,
            "verdict": verdict,
            "audit": audit,
            **final,
        }

    def _drive_pipeline(
        self,
        run_id: str,
        *,
        execute: bool = False,
        engine: str = DEFAULT_ENGINE,
        runner: Runner | None = None,
    ) -> dict[str, Any]:
        """⚡ RUN "sür" modu: motoru MCP araçlarıyla veri hattını ilerletmeye SÜRER.

        Av modundan (``drive`` mode="hunt") kritik farkları:
        - MCP açık (``build_drive_command`` → ``--mcp-config`` + ``--strict-mcp-config``,
          ``--safe-mode`` YOK). Motor Hektor MCP araçlarını GÖRÜR; ama araç yüzeyi P4
          allow-list'iyle (21 uç: salt-okuma GET'ler + POST ask/run-once) + sürücü
          token'ıyla sınırlıdır.
        - Verdict AYRI işaretçi (``HEKTOR_DRIVE_VERDICT``): sür PASS'i ``hunt_ack`` YAZMAZ
          — zorunlu Kademe-2 av kapısı bağımsız kalır (Kural 8; av ayrı tetiklenir).
        - Token DRIVE_TOKEN_TTL_S ile mint edilir (SORUN 2): koşu 60 dk sürebilir, varsayılan
          2100s TTL koşu ortasında ölüp MCP çağrılarına 401 attırırdı.
        - Sür promptu gerçek eğitimi ASLA başlatmaz; onay/eğitim uçları sürücü kimliğiyle
          403 alır (Kural 8) — bu zaten P1 scope katmanında kurulu, burada KORUNUR.

        `execute=False`: DRY-RUN (spawn yok, komut + MCP config yolu döner). `runner` enjekte
        edilirse gerçek spawn/kısıt aranmaz (test yolu)."""
        try:
            eng = engines.get_engine(engine)
        except ValueError as exc:
            return {"ok": False, "reason": str(exc)}
        if not eng.spawns:
            return {
                "ok": False,
                "reason": (
                    f"Motor '{eng.name}' süreç başlatmaz (doğrudan yerel hat) — "
                    "sür modu için spawn eden bir motor seç."
                ),
            }
        if not eng.drive_argv_template:
            # Sessizce av moduna DÜŞMEZ: av MCP'siz olduğundan sessiz düşüş, araçsız bir
            # ajan doğururdu (veri hattını ilerletemez). Açık ret doğru davranıştır.
            return {
                "ok": False,
                "reason": (
                    f"{eng.label} sür (drive) modunu desteklemiyor — MCP erişimli "
                    "sertleştirilmiş şablonu yok."
                ),
                "engine": eng.name,
            }

        run = self.orch.store.get_run(run_id)
        if run is None:
            return {"ok": False, "reason": f"Koşu bulunamadı: {run_id}"}

        # Koşu başına ayrı MCP config YOLU (eşzamanlı koşular çakışmaz). ⚠️ Dosya YALNIZ
        # gerçek spawn'da (execute=True) diske yazılır ve koşu bitince `finally` içinde
        # SİLİNİR — dry-run yalnız komutu gösterir, depoya artefakt bırakmaz (repo hijyeni;
        # rlm-security-reviewer bulgusu). Config sır içermez (token ortamdan miras alınır).
        mcp_path = drive_mcp_config_path(run_id)
        command = build_drive_command(run, mcp_path, eng.name)

        if not execute:
            self.orch.store.add_event(
                run_id,
                "drive",
                "info",
                f"Sür DRY-RUN: {eng.label} komutu hazır (execute=False; spawn yok).",
            )
            return {
                "ok": True,
                "drove": False,
                "dry_run": True,
                "mode": "drive",
                "engine": eng.name,
                "quota_warning": eng.quota_warning,
                "command": command,
                "mcp_config": mcp_path,
                "needs": f"execute=True + kurulu `{eng.binary}` CLI (giriş kendi CLI'sinde)",
                **self.orch.status(run_id),
            }

        run_fn = runner or functools.partial(_default_runner, run_id=run_id)
        # FAIL-CLOSED: sür profili DOĞRULANMAMIŞ motor doğurulmaz. `drive_hardened`, av
        # modundaki `hardened`'DAN AYRIDIR (sür `--safe-mode` kullanamaz → farklı bayrak seti).
        if runner is None and not eng.drive_hardened:
            self.orch.store.add_event(
                run_id,
                "drive",
                "error",
                f"Sür REDDEDİLDİ: {eng.label} sür modunda araç-seviyesinde kısıtlanamıyor.",
            )
            return {
                "ok": False,
                "reason": (
                    f"{eng.label} sür modunda sertleştirilmiş değil — AutoDriver yalnız "
                    "araç-kısıtlı motor doğurur (Kural 8). Bkz. docs/SCOPE_ISOLATION.md."
                ),
                "engine": eng.name,
                "drive_hardened": False,
                "command": command,
            }
        if runner is None and not engine_available(eng.name):
            return {
                "ok": False,
                "reason": f"`{eng.binary}` CLI PATH'te yok — {eng.label} kurulu olmalı.",
                "engine": eng.name,
                "command": command,
            }

        self.orch.store.add_event(
            run_id,
            "drive",
            "info",
            f"Sür modu: {eng.label} ile veri hattı MCP üzerinden sürülüyor. {eng.quota_warning}",
        )
        # MCP config'i spawn'dan HEMEN ÖNCE yaz (motor `--mcp-config` ile okuyacak).
        write_mcp_config(mcp_path)
        # SORUN 2 FIX: token AÇIKÇA DRIVE_TOKEN_TTL_S ile mint edilir (varsayılan 2100s
        # sür koşusunun ortasında ölürdü). Koşu bitince `finally` ile MUTLAKA iptal edilir.
        token = driver_scope.mint(run_id, ttl_s=DRIVE_TOKEN_TTL_S)
        try:
            rc, output = run_fn(command, DRIVE_TIMEOUT_S, build_child_env(token, run_id))
        except Exception as exc:  # spawn/timeout — koşuyu çökertme
            log.exception("AutoDriver sür: %s çalıştırılamadı", eng.name)
            self.orch.store.add_event(run_id, "drive", "error", f"Sür hatası: {exc}")
            return {
                "ok": False,
                "reason": f"{eng.label} çalıştırılamadı: {exc}",
                "engine": eng.name,
                "command": command,
                "mode": "drive",
            }
        finally:
            driver_scope.revoke_run(run_id)
            # Koşu bitti (MCP sunucusu motorun çocuğuydu, o da öldü) → config artefaktını sil.
            with contextlib.suppress(OSError):
                Path(mcp_path).unlink(missing_ok=True)

        if rc == STOPPED_RC:
            self.orch.store.add_event(
                run_id, "drive", "warning", "⛔ DURDUR (STOP_ALL): sür motoru kesildi."
            )
            return {
                "ok": True,
                "drove": True,
                "stopped": True,
                "mode": "drive",
                "drive_passed": False,
                "reason": "Sür koşusu ⛔ DURDUR ile kesildi (STOP_ALL etkin).",
                "engine": eng.name,
                **self.orch.status(run_id),
            }

        # ⚠️ KRİTİK (Kural 8): sür verdict'i AV işaretçisinden AYRI okunur ve hunt_ack YAZMAZ.
        # Zorunlu Kademe-2 av kapısı yalnız gerçek av (mode="hunt") ile açılır.
        verdict = parse_drive_verdict(output)
        self.orch.store.add_event(
            run_id,
            "drive",
            "info" if verdict["passed"] else "warning",
            f"Sür {verdict['verdict']} → veri hattı sürüşü bitti (av kapısı ETKİLENMEZ).",
        )
        return {
            "ok": True,
            "drove": True,
            "mode": "drive",
            "drive_passed": verdict["passed"],
            "verdict": verdict,
            "engine": eng.name,
            **self.orch.status(run_id),
        }
