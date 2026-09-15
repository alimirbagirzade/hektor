"""Eğitim-öncesi dataset kalite denetimi (#3 offline kapı) — eğitMEDEN kalite kanıtı.

v5 LoRA REJECT'in dersi (memory/v5-adapter-regression.md): zehirli veri GPU'ya GİTMEDEN
yakalanmalı. Bu modül birleşik SFT setini LLM'siz, deterministik tarar ve GO / NO-GO kararı
verir — pahalı eğitimden ÖNCE dürüst bir kapı (CLAUDE.md Kural 2: test edilmeden "hazır" deme).

Yakalanan v5 başarısızlık modları:
- **Garanti/kesinlik vaadi** (Kural 1 ihlali) → assistant cevabı `guaranteed_profit` red-flag'i
  tetikliyorsa ZEHİR → NO-GO.
- **Sabit açılış ezberi** → tek bir açılış (v5'te "pasaja göre") cevapların çoğunu açıyorsa
  model onu koşulsuz ezberler → NO-GO.
- **Sabit kapanış ezberi** (v8 dersi) → tek bir SON cümle cevapların anlamlı payını
  bitiriyorsa model onu koşulsuz tekrarlar → NO-GO.
- **Boş / okunamayan veri** ve **sır / kişisel veri** → NO-GO (eğitilen dosyanın kendisi
  taranır; lora-audit yalnız kartları görür).
Uyarı (bloklamaz ama raporlanır): sızıntı öneki payı, maliyet-token eksiği, disiplin kapsamı,
toplam < 1000 (overfit riski).

Hiçbir eğitim başlatmaz (kural 8). Çıktı determinist (kural 6).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.training.evaluate_model import RED_FLAGS

# --- Eşikler (v5 dersleri) ---------------------------------------------------
# Tek bir açılış-bigramı bu payı geçerse → ezber riski (NO-GO).
_OPENING_SHARE_BLOCK = 0.40
# Tek bir SON cümle cevapların bu payından fazlasını bitiriyorsa → kapanış ezberi (NO-GO).
# v8 verisinde en sık kapanış %5.9, ikincisi %3.8 idi (iki sabit disiplin kuyruğu); açılış
# kuralı bunu görmedi. Şablon havuzu tek başına ~%3 taban taşır (şablon × 16 strateji).
_CLOSING_SHARE_BLOCK = 0.04
# Kapanış payı küçük sette anlamsız (1 satır = %100) → bu sayının altında uygulanmaz.
_CLOSING_MIN_ANSWERS = 100
_SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# Sızıntı öneki ("pasaja göre") bu payı geçerse → uyarı (Fix A sonrası azalmalı).
_LEAKAGE_SHARE_WARN = 0.02
# Sağlıklı LoRA için pratik alt sınır.
_MIN_EXAMPLES = 1000
# Disiplin kapsamı hedefin bu oranının altındaysa uyar.
_DISCIPLINE_COVERAGE_WARN = 0.9

_COST_RE = re.compile(r"komisyon|slippage|spread|commission|slip", re.I)
_STRAT_RE = re.compile(r"strateji|strategy", re.I)
# "Olası strateji hipotezleri" / "strategy hypothesis" → araştırma özeti başlığı; gerçek
# al/sat tavsiyesi değil. Bu desenle başlayan cevaplar maliyet-körü sayılmaz.
_HYPOTHESIS_STRAT_RE = re.compile(
    r"olası strateji hipotez|strategy hypothesis|strateji hipotez|hypothes", re.I
)
_WORD_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]+")
_LEAKAGE_PREFIXES = ("pasaja göre", "pasaja gore", "pasaj a göre")


@dataclass
class DatasetQualityReport:
    """Eğitim-öncesi kalite raporu — GO/NO-GO + gerekçe + metrikler."""

    total: int
    verdict: str  # "GO" | "NO-GO"
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    guaranteed_profit_hits: int = 0
    top_opening: str = ""
    top_opening_share: float = 0.0
    unreadable_lines: int = 0
    top_closing: str = ""
    top_closing_share: float = 0.0
    secret_hits: int = 0
    pii_hits: int = 0
    leakage_prefix_hits: int = 0
    leakage_prefix_share: float = 0.0
    ignores_costs_hits: int = 0
    discipline_present: int = 0
    discipline_target: int = 0
    recommended_epochs: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assistant_answer(line: str) -> str | None:
    """JSONL satırından son assistant cevabını çıkar (parse edilemezse None).

    `messages` (chat) biçiminin yanında `prompt`/`completion` biçimi de okunur — aksi halde
    o biçimdeki satırlar sessizce atlanıp garanti vaadi dahil hiçbir denetime girmiyordu.
    """
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    msgs = obj.get("messages")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content", ""))
        return None
    completion = obj.get("completion")
    return str(completion) if completion is not None else None


def _closing_sentence(answer: str) -> str:
    """Cevabın son anlamlı cümlesi (küçük harf, boşluk normalize) — kapanış-ezberi için."""
    sents = [s for s in _SENT_BOUNDARY_RE.split(answer.strip()) if len(s.strip()) > 20]
    return " ".join(sents[-1].lower().split()) if sents else ""


def _opening_bigram(answer: str) -> str:
    """Cevabın ilk iki kelimesi (küçük harf) — açılış-ezberi tespiti için."""
    words = _WORD_RE.findall(answer.lower())
    return " ".join(words[:2])


def recommend_epochs(n: int) -> int:
    """Dataset boyutuna göre epoch öner (Fix C — overfit azalt).

    v5 az/tekrarlı veride aşırı overfit etti. Küçük sette daha az tur, büyük sette
    biraz daha fazla; T4 + ~%25 disiplin karışımı için makul aralık 1-3.
    """
    if n < 500:
        return 1
    if n <= 2500:
        return 2
    return 3


def audit_dataset(
    lines: list[str],
    *,
    discipline_lines: list[str] | None = None,
) -> DatasetQualityReport:
    """Birleşik SFT satırlarını LLM'siz denetle → GO/NO-GO raporu.

    Args:
        lines: Birleşik (eğitilecek) JSONL satırları.
        discipline_lines: Disiplin havuzu (verilirse kapsam = havuz ∩ set raporlanır).
    """
    answers = [a for ln in lines if (a := _assistant_answer(ln)) is not None]
    total = len(lines)

    blockers: list[str] = []
    warnings: list[str] = []

    # 0) Boş / okunamayan veri — sessiz boş ya da denetimsiz eğitim yok (HARD NO-GO).
    unreadable = total - len(answers)
    if total == 0:
        blockers.append("veri boş: denetlenecek satır yok (boş veriyle eğitim başlatılmaz)")
    elif unreadable:
        blockers.append(
            f"{unreadable} satırda assistant cevabı okunamadı (bozuk/bilinmeyen biçim) — "
            "denetlenemeyen satır eğitime giremez"
        )

    # 1) Garanti/kesinlik vaadi — Kural 1 zehiri (HARD NO-GO).
    gp_re = RED_FLAGS["guaranteed_profit"]
    gp_hits = sum(1 for a in answers if gp_re.search(a))
    if gp_hits:
        blockers.append(f"{gp_hits} cevap garanti/kesinlik vaadi içeriyor (Kural 1 zehiri)")

    # 2) Açılış-ezberi — tek bigram cevapların çoğunu açıyorsa (HARD NO-GO; v5 mekanizması).
    openings: dict[str, int] = {}
    for a in answers:
        bg = _opening_bigram(a)
        if bg:
            openings[bg] = openings.get(bg, 0) + 1
    top_opening, top_count = ("", 0)
    if openings:
        top_opening, top_count = max(openings.items(), key=lambda kv: kv[1])
    top_share = (top_count / len(answers)) if answers else 0.0
    if top_share > _OPENING_SHARE_BLOCK:
        blockers.append(
            f"açılış ezberi riski: '{top_opening}' cevapların %{top_share * 100:.0f}'ini açıyor "
            f"(eşik %{_OPENING_SHARE_BLOCK * 100:.0f})"
        )

    # 2b) Kapanış-ezberi — tek bir son cümle cevapların anlamlı payını bitiriyorsa (HARD NO-GO).
    closings: dict[str, int] = {}
    for a in answers:
        cl = _closing_sentence(a)
        if cl:
            closings[cl] = closings.get(cl, 0) + 1
    top_closing, top_closing_count = ("", 0)
    if closings:
        top_closing, top_closing_count = max(closings.items(), key=lambda kv: kv[1])
    closing_share = (top_closing_count / len(answers)) if answers else 0.0
    if len(answers) >= _CLOSING_MIN_ANSWERS and closing_share > _CLOSING_SHARE_BLOCK:
        blockers.append(
            f"kapanış ezberi riski: '{top_closing[:60]}' cevapların %{closing_share * 100:.1f}'ini "
            f"bitiriyor (eşik %{_CLOSING_SHARE_BLOCK * 100:.0f}) — model bu cümleyi tekrarlar"
        )

    # 3) Sızıntı öneki ("pasaja göre") — Fix A sonrası azalmalı (WARN).
    leak_hits = sum(
        1 for a in answers if any(a.lower().lstrip().startswith(p) for p in _LEAKAGE_PREFIXES)
    )
    leak_share = (leak_hits / len(answers)) if answers else 0.0
    if leak_share > _LEAKAGE_SHARE_WARN:
        warnings.append(
            f"sızıntı öneki: {leak_hits} cevap 'pasaja göre' ile başlıyor "
            f"(%{leak_share * 100:.1f}) — synth-qa'yı yeniden üret (Fix A)"
        )

    # 4) Maliyet-körü strateji cevabı (WARN; ignores_costs flag'inin veri-tarafı).
    # Araştırma özetlerindeki "Olası strateji hipotezleri" başlıkları hariç:
    # bunlar gerçek al/sat tavsiyesi değil, kart-özeti formatının bir parçası.
    cost_hits = sum(
        1
        for a in answers
        if _STRAT_RE.search(a) and not _COST_RE.search(a) and not _HYPOTHESIS_STRAT_RE.search(a)
    )
    if cost_hits:
        warnings.append(
            f"{cost_hits} 'strateji' cevabı maliyet token'ı (komisyon/slippage) içermiyor"
        )

    # 5) Disiplin kapsamı (WARN — havuz verildiyse).
    disc_present = 0
    disc_target = 0
    if discipline_lines:
        present = set(lines) & set(discipline_lines)
        disc_present = len(present)
        disc_target = len(discipline_lines)
        if disc_present < _DISCIPLINE_COVERAGE_WARN * disc_target:
            warnings.append(f"disiplin kapsamı düşük: {disc_present}/{disc_target} örnek karışımda")

    # 6) Boyut (WARN — overfit riski).
    if total < _MIN_EXAMPLES:
        warnings.append(
            f"toplam {total} < {_MIN_EXAMPLES}: az veride overfit eder (synth-qa ile büyüt)"
        )

    # 7) Sır / kişisel veri — eğitilen DOSYANIN kendisi taranır. lora-audit Gate 7 yalnız
    # kartları görür; sentetik QA + disiplin satırları başka hiçbir kapıdan geçmiyordu.
    from app.registry.promotion_gates import scan_secret_pii

    scan = scan_secret_pii(lines)
    if scan.secret_findings:
        blockers.append(
            f"{len(scan.secret_findings)} sır bulgusu (anahtar/parola deseni) — "
            "veri eğitime giremez"
        )
    if scan.pii_findings:
        blockers.append(
            f"{len(scan.pii_findings)} kişisel veri bulgusu (e-posta/telefon vb.) — "
            "veri eğitime giremez"
        )

    verdict = "NO-GO" if blockers else "GO"
    return DatasetQualityReport(
        total=total,
        verdict=verdict,
        blockers=blockers,
        warnings=warnings,
        guaranteed_profit_hits=gp_hits,
        top_opening=top_opening,
        top_opening_share=round(top_share, 4),
        unreadable_lines=unreadable,
        top_closing=top_closing,
        top_closing_share=round(closing_share, 4),
        secret_hits=len(scan.secret_findings),
        pii_hits=len(scan.pii_findings),
        leakage_prefix_hits=leak_hits,
        leakage_prefix_share=round(leak_share, 4),
        ignores_costs_hits=cost_hits,
        discipline_present=disc_present,
        discipline_target=disc_target,
        recommended_epochs=recommend_epochs(total),
    )
