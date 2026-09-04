"""verdict_audit — av verdict'inin bağımsız doğrulaması (P8, Kural 8).

Çevrimdışı + deterministik: `root` enjekte edilir → gerçek depo yerleşimine bağlı değil.
Kritik iddia: motorun 'PASS' beyanı TEK kanıt olamaz; yapısal kanıt dosya sistemiyle teyit
edilmeli. Sahte/eksik/iç-tutarsız kanıt reddedilir; gerçek kanıt geçer.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.orchestration import verdict_audit


# Her dosyanın 2. satırı AYIRT EDİCİ ve dosyaya özgüdür → geçerli bir okuma-kanıtı olarak
# alıntılanabilir (P9). 1. satır yalnız dolgu.
def _quote_for(subdir: str, i: int) -> str:
    return f"DISTINCT_TOKEN_{subdir}_{i} = {i}"


def _make_files(root: Path, n: int, subdir: str = "sub") -> list[str]:
    """`root/subdir` altında `n` gerçek dosya yarat, depo-göreli yollarını döndür (düz str)."""
    d = root / subdir
    d.mkdir(parents=True, exist_ok=True)
    rels = []
    for i in range(n):
        f = d / f"f{i}.py"
        f.write_text(
            f"# saf içerik dosya {subdir} {i}\n{_quote_for(subdir, i)}\n", encoding="utf-8"
        )
        rels.append(f"{subdir}/f{i}.py")
    return rels


def _proofs(root: Path, n: int, subdir: str = "sub") -> list[dict]:
    """`_make_files` gibi ama her öğe GEÇERLİ okuma-kanıtı taşır (P9): {path, line, quote}."""
    rels = _make_files(root, n, subdir)
    return [{"path": rel, "line": 2, "quote": _quote_for(subdir, i)} for i, rel in enumerate(rels)]


def _evidence(scanned: list, subsystems: list[str], findings: list[dict] | None = None) -> str:
    payload = {"scanned_files": scanned, "subsystems": subsystems, "findings": findings or []}
    return (
        "denetim raporu...\n"
        f"{verdict_audit.EVIDENCE_MARKER}\n```json\n{json.dumps(payload)}\n```\n"
        "ACHILLES_HUNT_VERDICT: PASS\n"
    )


# ── extract_evidence ──────────────────────────────────────────────────────────


def test_extract_evidence_fenced_and_plain() -> None:
    payload = {"scanned_files": ["a.py"], "subsystems": ["x"], "findings": []}
    fenced = f"{verdict_audit.EVIDENCE_MARKER}\n```json\n{json.dumps(payload)}\n```\nson"
    plain = f"{verdict_audit.EVIDENCE_MARKER} {json.dumps(payload)} kuyruk metni"
    assert verdict_audit.extract_evidence(fenced) == payload
    assert verdict_audit.extract_evidence(plain) == payload


def test_extract_evidence_missing_or_broken() -> None:
    assert verdict_audit.extract_evidence(None) is None
    assert verdict_audit.extract_evidence("hiç kanıt yok") is None
    assert verdict_audit.extract_evidence(f"{verdict_audit.EVIDENCE_MARKER} {{bozuk json") is None
    # JSON ama nesne değil (liste) → reddedilir
    assert verdict_audit.extract_evidence(f"{verdict_audit.EVIDENCE_MARKER} [1,2,3]") is None


def test_extract_evidence_recursion_error_fail_closed(monkeypatch) -> None:
    """`raw_decode` `RecursionError` fırlatırsa fail-closed None (sürücüyü çökertmez).

    rlm-security-reviewer bulgusu: derinden iç-içe JSON `RecursionError` fırlatır ve o
    `ValueError` DEĞİL `RuntimeError` alt sınıfıdır → dar `except (ValueError, ...)` kaçırırdı.
    Tetikleyen ÖZYİNELEME DERİNLİĞİ platforma göre değişir (CI'da 5000 sorunsuz ayrışabilir),
    bu yüzden istisnayı DOĞRUDAN zorlarız → handler'ı deterministik test ederiz.
    """

    def boom(self, s, idx=0):
        raise RecursionError("derin iç-içe")

    monkeypatch.setattr(json.JSONDecoder, "raw_decode", boom)
    payload = f'{verdict_audit.EVIDENCE_MARKER} {{"scanned_files": []}}'
    assert verdict_audit.extract_evidence(payload) is None
    # Uçtan uca: audit da çökmeden reddetmeli.
    res = verdict_audit.audit_hunt_evidence(
        f"{payload}\nACHILLES_HUNT_VERDICT: PASS", root=Path(".")
    )
    assert res["ok"] is False


# ── audit_hunt_evidence ───────────────────────────────────────────────────────


def test_missing_evidence_rejected() -> None:
    res = verdict_audit.audit_hunt_evidence("ACHILLES_HUNT_VERDICT: PASS", root=Path("."))
    assert res["ok"] is False
    assert "kanıt" in res["reason"].lower()


def test_valid_evidence_passes(tmp_path: Path) -> None:
    scanned = _proofs(tmp_path, 3, "orchestration") + _proofs(tmp_path, 2, "web")
    out = _evidence(scanned, ["orchestration", "web"])
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["ok"] is True
    assert res["scanned_count"] == 5 and res["subsystem_count"] == 2
    assert res["read_proven_count"] == 5


def test_too_few_files_rejected(tmp_path: Path) -> None:
    scanned = _make_files(tmp_path, 2, "orchestration")
    out = _evidence(scanned, ["orchestration", "web"])
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["ok"] is False and "dosya" in res["reason"]
    assert res["scanned_count"] == 2


def test_too_few_subsystems_rejected(tmp_path: Path) -> None:
    scanned = _make_files(tmp_path, 6, "orchestration")
    out = _evidence(scanned, ["orchestration"])  # tek alt-sistem
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["ok"] is False and "alt-sistem" in res["reason"]


def test_nonexistent_files_do_not_count(tmp_path: Path) -> None:
    """Uydurulmuş yollar depoda yok → sayılmaz → eşik altında reddedilir."""
    fake = [f"uydurma/yok{i}.py" for i in range(9)]
    out = _evidence(fake, ["orchestration", "web"])
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["ok"] is False
    assert res["claimed_files"] == 9 and res["scanned_count"] == 0


def test_path_traversal_rejected(tmp_path: Path) -> None:
    """Kök dışına (../) çıkan yollar sayılmaz (yol-geçişi savunması)."""
    outside = tmp_path.parent / "disarida.py"
    outside.write_text("x\n", encoding="utf-8")
    scanned = [*_make_files(tmp_path, 4, "orchestration"), "../disarida.py"]
    out = _evidence(scanned, ["orchestration", "web"])
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    # 4 gerçek + kök dışı sayılmaz = 4 < 5 → reddedilir
    assert res["scanned_count"] == 4 and res["ok"] is False


def test_pass_with_high_finding_is_inconsistent(tmp_path: Path) -> None:
    """PASS derken kanıtta HIGH bulgu listelemek iç-tutarsız → reddedilir."""
    scanned = _make_files(tmp_path, 3, "orchestration") + _make_files(tmp_path, 2, "web")
    out = _evidence(
        scanned,
        ["orchestration", "web"],
        findings=[{"severity": "HIGH", "file": "x.py", "summary": "sızıntı"}],
    )
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["ok"] is False and "HIGH/BLOCKER" in res["reason"]


def test_medium_low_findings_allowed(tmp_path: Path) -> None:
    """MEDIUM/LOW bulgular PASS ile çelişmez (yalnız HIGH/BLOCKER kapatır)."""
    scanned = _proofs(tmp_path, 3, "orchestration") + _proofs(tmp_path, 2, "web")
    out = _evidence(
        scanned,
        ["orchestration", "web"],
        findings=[{"severity": "MEDIUM", "file": "x.py", "summary": "ufak"}],
    )
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["ok"] is True


def test_duplicate_files_counted_once(tmp_path: Path) -> None:
    """Aynı dosyayı tekrar tekrar listelemek kapsamayı ŞİŞİREMEZ."""
    one = _make_files(tmp_path, 1, "orchestration")
    out = _evidence(one * 9, ["orchestration", "web"])
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["scanned_count"] == 1 and res["ok"] is False


# ── P9: okuma-kanıtı (var-olma ≠ okundu) ──────────────────────────────────────


def test_listed_but_not_read_rejected(tmp_path: Path) -> None:
    """P9 ÇEKİRDEK: GERÇEK var-olan dosyalar listelenip HİÇ OKUNMADAN PASS → REDDEDİLİR.

    Bu tam olarak P8'in kaçırdığı sınıf: düz str yollar (okuma-kanıtı yok) var-olma
    kapısını (5 dosya / 2 alt-sistem) geçer ama okuma-kanıtı kapısına takılır.
    """
    scanned = _make_files(tmp_path, 3, "orchestration") + _make_files(tmp_path, 2, "web")
    out = _evidence(scanned, ["orchestration", "web"])  # düz str → kanıt yok
    res = verdict_audit.audit_hunt_evidence(out, root=tmp_path)
    assert res["ok"] is False
    assert res["scanned_count"] == 5  # var-olma kapısını GEÇTİ (P8 korunur)
    assert res["read_proven_count"] == 0  # ama okuma-kanıtı yok
    assert "okundu" in res["reason"] or "okun" in res["reason"]


def test_valid_read_proof_passes(tmp_path: Path) -> None:
    """Gerçekten okunup doğru içerik-alıntısı verilen kanıt → GEÇER."""
    scanned = _proofs(tmp_path, 3, "orchestration") + _proofs(tmp_path, 2, "web")
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["ok"] is True and res["read_proven_count"] == 5


def test_wrong_quote_rejected(tmp_path: Path) -> None:
    """Dosya adı gerçek ama alıntı YANLIŞ (o satırda değil) → kanıt sayılmaz → reddedilir."""
    scanned = _proofs(tmp_path, 3, "orchestration") + _proofs(tmp_path, 2, "web")
    for e in scanned:
        e["quote"] = "BU_SATIR_DOSYADA_YOK = 999"  # uydurma içerik
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["ok"] is False and res["read_proven_count"] == 0


def test_wrong_line_number_rejected(tmp_path: Path) -> None:
    """Doğru içerik ama YANLIŞ satır numarası → konum eşleşmez → sayılmaz."""
    scanned = _proofs(tmp_path, 5, "orchestration")
    for e in scanned:
        e["line"] = 99  # alıntı 2. satırda; 99 yanlış
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["ok"] is False and res["read_proven_count"] == 0


def test_reused_generic_quote_counts_once(tmp_path: Path) -> None:
    """Aynı satırı (jenerik) birçok dosyaya kanıt diye tekrar kullanmak kapsamayı şişiremez.

    Jenerik-satır forgery savunması: dosyalar gerçekten var ve her biri o satırı İÇERSE
    bile, aynı alıntı-metni yalnız BİR dosya için sayılır → 5 farklı kanıta ulaşılamaz.
    """
    d = tmp_path / "orchestration"
    d.mkdir(parents=True)
    generic = "from __future__ import annotations"
    scanned = []
    for i in range(6):
        rel = f"orchestration/g{i}.py"
        (tmp_path / rel).write_text(f"{generic}\n", encoding="utf-8")
        scanned.append({"path": rel, "line": 1, "quote": generic})
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["read_proven_count"] == 1 and res["ok"] is False


def test_short_quote_not_counted(tmp_path: Path) -> None:
    """Kırpılınca çok kısa (< MIN_QUOTE_LEN) alıntı ayırt edici değil → kanıt sayılmaz."""
    d = tmp_path / "orchestration"
    d.mkdir(parents=True)
    scanned = []
    for i in range(5):
        rel = f"orchestration/s{i}.py"
        (tmp_path / rel).write_text(")\n", encoding="utf-8")  # 1 karakterlik satır
        scanned.append({"path": rel, "line": 1, "quote": ")"})
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["read_proven_count"] == 0 and res["ok"] is False


def test_read_proof_indentation_tolerant(tmp_path: Path) -> None:
    """Alıntı girinti/satır-sonu farkına dayanıklı (strip); öz içerik eşleşmesi yeterli."""
    scanned = _proofs(tmp_path, 5, "orchestration")
    for e in scanned:  # motor girintiyle/boşlukla alıntılamış olabilir
        e["quote"] = f"   {e['quote']}  "
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["ok"] is True and res["read_proven_count"] == 5


def test_line_bool_not_treated_as_proof(tmp_path: Path) -> None:
    """`line: true` (bool) kazara satır-1 kanıtı sayılmamalı (bool, int alt sınıfıdır)."""
    scanned = _proofs(tmp_path, 5, "orchestration")
    for e in scanned:
        e["line"] = True
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["read_proven_count"] == 0 and res["ok"] is False


def test_oversize_file_not_proof_target(tmp_path: Path, monkeypatch) -> None:
    """MAX_PROOF_FILE_BYTES üstü dosya (artefakt) kanıt hedefi olamaz → sayılmaz (DoS savunması).

    Motor diskte var olan büyük bir artefaktı (DB/model/indeks) 'taradım' diye işaret edip
    denetim sürecinin belleğini tüketmeye çalışamaz. Eşiği düşürerek küçük dosyalarla test
    ederiz → gerçekte 5 MB'lık dosya yazmaya gerek yok (deterministik + hızlı).
    """
    monkeypatch.setattr(verdict_audit, "MAX_PROOF_FILE_BYTES", 5)
    scanned = _proofs(tmp_path, 5, "orchestration")  # her dosya birkaç on bayt → 5 baytı aşar
    res = verdict_audit.audit_hunt_evidence(
        _evidence(scanned, ["orchestration", "web"]), root=tmp_path
    )
    assert res["read_proven_count"] == 0 and res["ok"] is False
