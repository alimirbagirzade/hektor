"""verdict_audit.py — av (hunt) verdict'inin BAĞIMSIZ doğrulaması (Kural 8 sertleştirme).

SORUN (P8, doğrulanmış): `driver.parse_hunt_verdict` verdict'i TAMAMEN motorun kendi
stdout'unun son satırından okur (``ACHILLES_HUNT_VERDICT: PASS``). Bu satır, ZORUNLU
Kademe-2 derin av kapısının (her eğitimden önce, CLAUDE.md Kural 8) TEK dayanağıdır. Motor
derin avı HİÇ yapmadan son satıra ``PASS`` yazsa hat açılırdı → **motor kendi başarısının
tek kaynağı**. Fail-closed korumalar (satır yoksa FAIL, sahte-binary savunması, Write/Edit
yasağı) vardı ama BAĞIMSIZ kanıt yoktu.

ÇÖZÜM: motor artık serbest bir "PASS" değil, YAPILANDIRILMIŞ KANIT üretmek zorunda
(taranan dosyalar + alt-sistemler + bulgular). Bu modül o kanıtı **motordan bağımsız bir
oracle** ile — dosya sisteminin kendisiyle — doğrular:

* Kanıt bloğu yok/bozuk  → PASS reddedilir (öz-beyan tek kanıt olamaz).
* İddia edilen "taranan" dosyalar depoda GERÇEKTEN yoksa sayılmaz; asgari kapsama
  (dosya + alt-sistem sayısı) tutmuyorsa reddedilir → uydurulmuş ya da boş bir taramanın
  "PASS"ı geçemez.
* Motor "PASS" derken kanıtında HIGH/BLOCKER bir bulgu listelemişse (kendi kendiyle
  çelişiyorsa) reddedilir.

Böylece motorun verdict'i tek kanıt olmaktan çıkar: dosya sistemi zorunlu bir teyit
katmanıdır.

## P9 — OKUMA-KANITI katmanı (var-olma ≠ okundu)

P8'in dar boşluğu: yukarıdaki kapı dosyanın VAR OLDUĞUnu teyit eder, OKUNDUĞUNU değil.
Motor 5 GERÇEK var-olan dosya adını (driver.py, engines.py…) listeleyip **hiç okumadan**
PASS yazsa P8 kapısı geçerdi. P9 bunu kapatır: motor artık her "taranan dosya" için o
dosyanın İÇERİĞİNDEN türeyen bir **okuma-kanıtı** vermek zorunda — belirli bir 1-tabanlı
satır numarası + o satırın BİREBİR metni. Denetim dosyayı bağımsızca okuyup
``lines[line-1].strip() == quote.strip()`` kontrol eder.

Bu kanıt **uydurulamaz çünkü içeriğe bağlıdır**: motor, belirli bir dosyanın GÜNCEL
durumundaki belirli bir satırın birebir içeriğini ancak o dosyayı Read ile okuyarak
bilebilir (eğitim verisindeki eski sürüm satır numarası/içeriği kayar; worktree'deki
kaydedilmemiş değişiklikler de). Aynı jenerik satırın (``from __future__ import
annotations``) birden çok dosyaya kanıt diye tekrar kullanılması, kanıt-satırı ve dosya
tekilleştirilerek engellenir. İçerik-hash'i seçilmedi: ``--safe-mode`` avında motorun
yalnız Read/Grep/Glob'u vardır, sha256 hesaplayamaz → meşru avlar YANLIŞ reddedilirdi.

**Not (dürüstlük sınırı):** motor bir dosyayı okuyup satırı doğru alıntılar ama o dosyayı
gerçekten *anlamadan/denetlemeden* geçebilir — okuma-kanıtı "hiç açmadan PASS" sınıfını
kapatır, "açtı ama düşünmedi" sınıfını değil. Onu tam kapatmak ikinci bir bağımsız LLM
doğrulayıcı gerektirir (çevrimdışı test edilemez + kota yakar → bilinçli ertelendi).

DETERMİNİZM: saf fonksiyonlar; ağ yok, LLM yok, rastgelelik yok. `root` enjekte edilebilir
→ testler gerçek depo yerleşimine bağlı değildir. Yalnız salt-okuma dosya erişimi (içerik
okuma dahil), yan etki YOK.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Depo kökü: app/orchestration/verdict_audit.py → parents[2]. Motorun bildirdiği yollar
# depo köküne GÖRELİdir; iddiaları burada dosya sistemiyle doğrularız.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Motorun yapılandırılmış kanıt bloğunu başlatan işaretçi. Verdict satırından AYRIDIR:
# verdict son satırdadır (parse_hunt_verdict onu okur), kanıt ondan ÖNCE gelir.
EVIDENCE_MARKER = "ACHILLES_HUNT_EVIDENCE"

# Asgari kapsama eşikleri — "hiç bakmadan PASS yaz" sınıfını yakalamak için taban.
# Derin av alt-sistem başına paralel tarar → gerçek bir av bunların çok üstündedir; bunlar
# yalnız GROSS tembelliği (sıfıra yakın tarama) eler, meşru odaklı bir avı reddetmeyecek
# kadar düşük tutulur. Ayarlanabilir sabitler → tek yerde, testte de kullanılır.
MIN_SCANNED_FILES = 5
MIN_SUBSYSTEMS = 2

# P9 okuma-kanıtı eşikleri.
# En az kaç FARKLI dosya için geçerli okuma-kanıtı istenir. Kapsama tabanıyla (breadth)
# hizalı tutulur → "5 dosya listeledim ama hiçbirini okumadım" sınıfı elenir.
MIN_READ_PROVEN = MIN_SCANNED_FILES
# Alıntılanan satır (kenar boşlukları kırpıldıktan sonra) en az bu kadar uzun olmalı.
# Boş/tek-karakterlik ("`)`") ya da ayırt-edici olmayan satırların kanıt sayılmasını önler;
# meşru bir satırı (ör. "MIN_SUBSYSTEMS = 2") reddetmeyecek kadar düşük.
MIN_QUOTE_LEN = 12

# Okuma-kanıtı yalnız KAYNAK dosyalarında anlamlıdır. Motor kanıt hedefi olarak diskte var
# olan büyük bir artefaktı (data/ SQLite DB, models/ ağırlıkları, vector_db/ indeksi) işaret
# edip denetim sürecinin belleğini/CPU'sunu tüketmeye çalışamasın diye üst sınır (DoS savunması,
# rlm-security-reviewer önerisi). Kaynak dosyalar bunun çok altındadır → meşru avı etkilemez.
MAX_PROOF_FILE_BYTES = 5_000_000

# "PASS" beyanıyla ÇELİŞEN bulgu ciddiyetleri (büyük harfe normalize edilerek kıyaslanır).
# Motor bunlardan birini kanıtına yazıp yine de PASS derse iç-tutarsızdır → reddedilir.
_BLOCKING_SEVERITIES = frozenset({"HIGH", "BLOCKER", "CRITICAL", "SEVERE"})


def extract_evidence(output: str | None) -> dict[str, Any] | None:
    """Motor çıktısındaki yapılandırılmış kanıt bloğunu ayıkla (yoksa/bozuksa ``None``).

    ``EVIDENCE_MARKER``dan sonraki İLK JSON nesnesi ``raw_decode`` ile okunur; markdown
    kod-çiti (```json ... ```) gibi arkasındaki metin YOK SAYILIR. Herhangi bir hata
    fail-closed ``None`` döner → çağıran bunu "kanıt yok" olarak reddeder.
    """
    if not output:
        return None
    idx = output.find(EVIDENCE_MARKER)
    if idx == -1:
        return None
    rest = output[idx + len(EVIDENCE_MARKER) :]
    brace = rest.find("{")
    if brace == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(rest[brace:])
    except Exception:
        # FAIL-CLOSED: HERHANGİ bir ayrıştırma hatası → None. `ValueError`/`JSONDecodeError`
        # yetmez: derinden iç-içe JSON (`[[[...`) `RecursionError` fırlatır ve o `ValueError`
        # DEĞİL `RuntimeError` alt sınıfıdır → motorun kontrol ettiği tek bir çıktı sürücüyü
        # çökertirdi (rlm-security-reviewer bulgusu). Geniş yakalama, docstring'in "her hata
        # fail-closed None döner" sözleşmesiyle bilinçli olarak hizalıdır.
        log.debug("Kanıt bloğu ayrıştırılamadı — güvenli tarafta None", exc_info=True)
        return None
    return obj if isinstance(obj, dict) else None


def _resolve_in_root(rel: Any, root: Path) -> Path | None:
    """``rel`` yolunu ``root`` altında güvenle çöz; gerçek bir dosya değilse ``None``.

    Yol-geçişi savunması: çözülen yol kökün DIŞINA çıkıyorsa (``../../etc``) reddedilir.
    Böylece motor, depo dışındaki bir dosyayı "taradım" diye sayamaz.
    """
    if not isinstance(rel, str) or not rel.strip():
        return None
    try:
        candidate = (root / rel).resolve()
        root_resolved = root.resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None  # kökün dışına taştı → güvenli tarafta reddet
    return candidate if candidate.is_file() else None


def _entry_path(item: Any) -> str | None:
    """Bir ``scanned_files`` öğesinin yolunu çıkar.

    Geriye dönük uyumluluk: öğe düz bir ``str`` olabilir (P8, yalnız var-olma) ya da
    ``{"path": ..., "line": ..., "quote": ...}`` nesnesi (P9, okuma-kanıtlı). İkisi de
    var-olma kapısında sayılır; yalnız nesne biçimi okuma-kanıtı taşıyabilir.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        path = item.get("path")
        return path if isinstance(path, str) else None
    return None


def _distinct_existing_files(scanned: Any, root: Path) -> int:
    """Depoda GERÇEKTEN var olan farklı dosyaların sayısı (çözülmüş yola göre tekilleştirir)."""
    if not isinstance(scanned, (list, tuple)):
        return 0
    seen: set[Path] = set()
    for item in scanned:
        resolved = _resolve_in_root(_entry_path(item), root)
        if resolved is not None:
            seen.add(resolved)
    return len(seen)


def _read_proof_of(item: Any) -> tuple[int, str] | None:
    """Öğedeki okuma-kanıtını (1-tabanlı satır no + o satırın birebir metni) çıkar.

    Kanıt yoksa/biçimsizse ``None``. ``line`` gerçek bir tamsayı olmalı (``bool`` DEĞİL —
    ``True`` bir ``int`` alt sınıfıdır ve 1'e eşittir, kazara kanıt sayılmamalı); ``quote``
    kırpılmış hâliyle en az ``MIN_QUOTE_LEN`` uzunlukta bir metin olmalı.
    """
    if not isinstance(item, dict):
        return None
    line = item.get("line")
    quote = item.get("quote")
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        return None
    if not isinstance(quote, str) or len(quote.strip()) < MIN_QUOTE_LEN:
        return None
    return line, quote


def _line_at_matches(resolved: Path, line: int, quote: str) -> bool:
    """Dosyayı bağımsızca oku ve ``line`` numaralı satırın ``quote`` ile eşleştiğini doğrula.

    Karşılaştırma iki taraf da ``strip()`` ile yapılır → girinti/satır-sonu farklarına
    dayanıklı, ama satırın öz içeriği birebir eşleşmeli (uydurulamazlık buradan gelir).
    Herhangi bir okuma/çözme hatası fail-closed ``False`` döner (ikili dosya, silinmiş vb.).

    BELLEK SINIRI (DoS savunması): tüm dosya belleğe alınmaz; ``MAX_PROOF_FILE_BYTES`` üstü
    dosyalar (kaynak değil, artefakt) reddedilir ve dosya yalnız hedef satıra kadar
    satır-satır okunur → dev bir dosya işaret edilse bile bellek bir satırla sınırlı kalır.
    """
    try:
        if resolved.stat().st_size > MAX_PROOF_FILE_BYTES:
            return False  # kaynak dosya değil (DB/model/indeks) → kanıt hedefi olamaz
        with resolved.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                if lineno == line:
                    return raw.strip() == quote.strip()
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return False  # satır no dosya sonundan büyük → eşleşme yok


def _read_proven_files(scanned: Any, root: Path) -> int:
    """Okuma-kanıtı dosya sistemiyle DOĞRULANAN farklı dosya sayısı.

    Bir öğe ancak şu durumda sayılır: (a) depoda var olan bir dosyaya çözülür, (b) geçerli
    bir okuma-kanıtı taşır, (c) alıntılanan satır o dosyada gerçekten o konumdadır. Aynı
    dosya ve aynı alıntı-satırı yalnız BİR kez sayılır → tek jenerik satırı birçok dosyaya
    kanıt diye tekrar kullanarak kapsamayı şişirmek engellenir.
    """
    if not isinstance(scanned, (list, tuple)):
        return 0
    seen_files: set[Path] = set()
    seen_quotes: set[str] = set()
    for item in scanned:
        proof = _read_proof_of(item)
        if proof is None:
            continue
        line, quote = proof
        resolved = _resolve_in_root(_entry_path(item), root)
        if resolved is None or resolved in seen_files:
            continue
        norm_quote = quote.strip()
        if norm_quote in seen_quotes:
            continue  # aynı satır birden çok dosyaya kanıt olamaz (jenerik-satır forgery savunması)
        if _line_at_matches(resolved, line, quote):
            seen_files.add(resolved)
            seen_quotes.add(norm_quote)
    return len(seen_files)


def _distinct_subsystems(subsystems: Any) -> int:
    """Farklı, boş-olmayan alt-sistem adlarının sayısı (büyük/küçük harf duyarsız)."""
    if not isinstance(subsystems, (list, tuple)):
        return 0
    return len({s.strip().casefold() for s in subsystems if isinstance(s, str) and s.strip()})


def _has_blocking_finding(findings: Any) -> bool:
    """Kanıttaki bulgular arasında PASS ile çelişen (HIGH/BLOCKER…) biri var mı?"""
    if not isinstance(findings, (list, tuple)):
        return False
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        sev = finding.get("severity")
        if isinstance(sev, str) and sev.strip().upper() in _BLOCKING_SEVERITIES:
            return True
    return False


def audit_hunt_evidence(output: str | None, *, root: Path | None = None) -> dict[str, Any]:
    """Motorun av "PASS" beyanını dosya sistemiyle BAĞIMSIZ doğrula.

    Yalnız verdict PASS iken çağrılması ANLAMLIdır (çağıran öyle yapar); FAIL zaten
    deep-hunt'ı bloklu tutar. Döndürülen sözlük:

    * ``ok``            — kanıt bağımsız denetimi geçti mi (PASS'in gerçekten ilerlemesi bu).
    * ``reason``        — geçmediyse Türkçe sebep (geçtiyse "").
    * ``scanned_count`` — depoda var olan farklı dosya sayısı (bağımsızca sayıldı).
    * ``subsystem_count`` — farklı alt-sistem sayısı.
    * ``claimed_files`` — motorun iddia ettiği ham dosya sayısı (var olmayanlar dahil).
    * ``read_proven_count`` — okuma-kanıtı dosya sistemiyle doğrulanan farklı dosya sayısı (P9).

    Hiçbir yan etki YOK, ağ YOK: yalnız salt-okuma dosya erişimi (var-mı + içerik okuma).
    """
    base = root or _REPO_ROOT
    result: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "scanned_count": 0,
        "subsystem_count": 0,
        "claimed_files": 0,
        "read_proven_count": 0,
    }

    evidence = extract_evidence(output)
    if evidence is None:
        result["reason"] = (
            "Yapılandırılmış kanıt bloğu (ACHILLES_HUNT_EVIDENCE) yok ya da bozuk — "
            "motorun öz-beyanı tek kanıt olamaz (Kural 8)."
        )
        return result

    scanned = evidence.get("scanned_files")
    result["claimed_files"] = len(scanned) if isinstance(scanned, (list, tuple)) else 0
    scanned_count = _distinct_existing_files(scanned, base)
    subsystem_count = _distinct_subsystems(evidence.get("subsystems"))
    result["scanned_count"] = scanned_count
    result["subsystem_count"] = subsystem_count

    if scanned_count < MIN_SCANNED_FILES:
        result["reason"] = (
            f"Bağımsız denetim: depoda var olan taranmış dosya sayısı yetersiz "
            f"({scanned_count} < {MIN_SCANNED_FILES}) — iddia edilen tarama teyit edilemedi."
        )
        return result

    if subsystem_count < MIN_SUBSYSTEMS:
        result["reason"] = (
            f"Bağımsız denetim: taranan alt-sistem çeşitliliği yetersiz "
            f"({subsystem_count} < {MIN_SUBSYSTEMS}) — derin av kapsaması teyit edilemedi."
        )
        return result

    if _has_blocking_finding(evidence.get("findings")):
        result["reason"] = (
            "Bağımsız denetim: 'PASS' beyanı kanıttaki HIGH/BLOCKER bulguyla ÇELİŞİYOR — "
            "iç-tutarsız verdict reddedildi."
        )
        return result

    # P9 OKUMA-KANITI KAPISI (var-olma kapılarından SONRA, en sıkı kapı). Var-olma ≠ okundu:
    # motor gerçek dosya adlarını hiç okumadan listeleyebilir. Her dosya için içerik-bağımlı
    # bir satır alıntısı, dosya bağımsızca okunarak doğrulanır → uydurulamaz.
    read_proven = _read_proven_files(scanned, base)
    result["read_proven_count"] = read_proven
    if read_proven < MIN_READ_PROVEN:
        result["reason"] = (
            f"Bağımsız denetim: okuma-kanıtı doğrulanan dosya sayısı yetersiz "
            f"({read_proven} < {MIN_READ_PROVEN}) — dosyaların GERÇEKTEN okunduğu teyit "
            f"edilemedi (var-olma ≠ okundu; Kural 8)."
        )
        return result

    result["ok"] = True
    return result
