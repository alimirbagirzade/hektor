"""Kart kürasyonu — orphan karantina + per-paper version-collapse.

İki sınıf 'zehirli' kartı LoRA eğitiminden çıkarır (DB'yi yıkıcı silmeden, yalnız
``lora_eligible`` bayrağını düşürerek — GERİ ALINABİLİR):

1. ORPHAN: ``paper_id``'si ``papers`` tablosunda OLMAYAN onaylı kart. Dayandığı
   makale yok → iddiası ('%92 doğruluk' vb.) hiçbir kaynağa bağlanamaz
   (CLAUDE.md kural 7 — kaynak uydurma yasak).
2. ÇOK-VERSİYON (version-collapse): aynı ``paper_id``'den üretilmiş BİRDEN ÇOK kart.
   LLM aynı metni farklı turlarda işleyince farklı (çoğu kez ÇELİŞKİLİ) ``main_claim``
   üretir; model aynı konuya birden çok 'doğru' cevap görür → disiplinli/kararlı cevap
   veremez (v5 disiplin-gerilemesinin veri-kökü). Paper başına EN İYİ tek kart tutulur,
   gerisi ``lora_eligible=0`` yapılır.

Saf planlama (``plan_curation`` — DB'siz birim-testlenebilir) + güvenli uygula
(``apply_curation`` — ``dry_run`` VARSAYILAN). EĞİTİM BAŞLATMAZ.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.lora.gates import _card_text

if TYPE_CHECKING:
    from app.memory.sqlite_store import SqliteStore


def card_richness(card: dict) -> int:
    """İçerik zenginliği ölçüsü: denetlenebilir serbest metnin karakter sayısı.

    ``gates._card_text`` ile AYNI metni kullanır (title+summary+main_claim+methods+
    formüller+hipotezler…) → version-collapse'te 'en iyi' = en zengin kart.
    """
    return len(_card_text(card))


def pick_canonical_card_id(cards: list[dict]) -> str:
    """Bir paper'ın kartları arasından TUTULACAK (kanonik) kartın ``card_id``'sini seç.

    Deterministik sıralama (azalan öncelik): içerik-zenginliği → ``created_at`` (en yeni)
    → ``difficulty`` → ``card_id`` (sabit tie-break). Aynı girdi → aynı seçim (kural 6).
    """

    def key(c: dict) -> tuple[int, str, float, str]:
        return (
            card_richness(c),
            str(c.get("created_at") or ""),
            float(c.get("difficulty") or 0.0),
            str(c.get("card_id") or ""),
        )

    return str(max(cards, key=key).get("card_id") or "")


@dataclass
class CurationPlan:
    """Saf plan: hangi kartlar tutulacak / orphan / fazla-versiyon (DB yazmaz)."""

    keep_card_ids: set[str] = field(default_factory=set)
    orphan_card_ids: set[str] = field(default_factory=set)
    redundant_card_ids: set[str] = field(default_factory=set)
    orphan_paper_ids: set[str] = field(default_factory=set)
    collapsed_paper_ids: set[str] = field(default_factory=set)
    # paper_id -> tutulan kanonik card_id (version-collapse şeffaflığı)
    canonical_by_paper: dict[str, str] = field(default_factory=dict)

    @property
    def demote_card_ids(self) -> set[str]:
        """``lora_eligible=0`` yapılacak kartlar (orphan ∪ fazla-versiyon)."""
        return self.orphan_card_ids | self.redundant_card_ids


def plan_curation(cards: list[dict], valid_paper_ids: set[str]) -> CurationPlan:
    """Saf planlama: yalnız approved+eligible kartlar üzerinde orphan + collapse hesapla.

    Args:
        cards: ``list_approved_cards`` çıktısı (review_status/lora_eligible alanları dahil).
        valid_paper_ids: ``papers`` tablosundaki gerçek paper_id seti.

    DB'ye DOKUNMAZ; yalnız hangi kartların düşürüleceğini deterministik hesaplar.
    """
    plan = CurationPlan()
    eligible = [c for c in cards if c.get("review_status") == "approved" and c.get("lora_eligible")]

    # 1) Orphan: paper_id papers tablosunda yok → hepsini düşür.
    survivors: list[dict] = []
    for c in eligible:
        pid = str(c.get("paper_id") or "")
        if pid not in valid_paper_ids:
            plan.orphan_card_ids.add(str(c.get("card_id") or ""))
            plan.orphan_paper_ids.add(pid)
        else:
            survivors.append(c)

    # 2) Version-collapse: paper başına tek kanonik kart tut, gerisini düşür.
    by_paper: dict[str, list[dict]] = defaultdict(list)
    for c in survivors:
        by_paper[str(c.get("paper_id") or "")].append(c)
    for pid, group in by_paper.items():
        canonical = pick_canonical_card_id(group)
        plan.canonical_by_paper[pid] = canonical
        plan.keep_card_ids.add(canonical)
        if len(group) > 1:
            plan.collapsed_paper_ids.add(pid)
        for c in group:
            cid = str(c.get("card_id") or "")
            if cid != canonical:
                plan.redundant_card_ids.add(cid)
    return plan


@dataclass
class CurationReport:
    """``apply_curation`` özeti — şeffaf sayım + uygulanan plan."""

    total_eligible: int
    orphan_demoted: int
    redundant_demoted: int
    kept: int
    distinct_orphan_papers: int
    distinct_collapsed_papers: int
    dry_run: bool
    plan: CurationPlan

    @property
    def total_demoted(self) -> int:
        return self.orphan_demoted + self.redundant_demoted


def apply_curation(store: SqliteStore, *, dry_run: bool = True) -> CurationReport:
    """Onaylı kartları yükle, planla ve (``dry_run=False`` ise) ``lora_eligible=0`` uygula.

    Mutasyon YALNIZ ``lora_eligible`` bayrağını düşürür (``review_status`` korunur) →
    geri alınabilir; kart silinmez. EĞİTİM BAŞLATMAZ (CLAUDE.md kural 8). ``dry_run``
    VARSAYILAN True: yalnız rapor üretir, DB'ye yazmaz.
    """
    cards = store.list_approved_cards()
    valid_paper_ids = store.list_paper_ids()
    plan = plan_curation(cards, valid_paper_ids)

    total_eligible = sum(
        1 for c in cards if c.get("review_status") == "approved" and c.get("lora_eligible")
    )
    if not dry_run:
        # Deterministik sıra (sabit log/audit izi); her kart bağımsız flag düşürme.
        for cid in sorted(plan.demote_card_ids):
            store.set_card_lora_eligible(cid, False)

    return CurationReport(
        total_eligible=total_eligible,
        orphan_demoted=len(plan.orphan_card_ids),
        redundant_demoted=len(plan.redundant_card_ids),
        kept=len(plan.keep_card_ids),
        distinct_orphan_papers=len(plan.orphan_paper_ids),
        distinct_collapsed_papers=len(plan.collapsed_paper_ids),
        dry_run=dry_run,
        plan=plan,
    )


def curation_markdown(report: CurationReport) -> str:
    """Kürasyon raporunu Markdown'a çevir (düşürülen card_id'ler = geri-alma izi)."""
    p = report.plan
    mode = "DRY-RUN (yazılmadı)" if report.dry_run else "UYGULANDI (lora_eligible=0)"
    lines = [
        "# LoRA Kart Kürasyon Raporu",
        "",
        f"- Mod: **{mode}**",
        f"- Uygun (approved+eligible) kart: {report.total_eligible}",
        f"- Orphan düşürülen: {report.orphan_demoted} "
        f"({report.distinct_orphan_papers} farklı paper_id)",
        f"- Fazla-versiyon düşürülen: {report.redundant_demoted} "
        f"({report.distinct_collapsed_papers} paper collapse edildi)",
        f"- Tutulan (kanonik): {report.kept}",
        f"- Toplam düşürülen: {report.total_demoted}",
        "",
        "## Orphan card_id'ler (paper_id papers tablosunda yok)",
        "",
    ]
    lines += [f"- {cid}" for cid in sorted(p.orphan_card_ids)] or ["- (yok)"]
    lines += ["", "## Version-collapse: düşürülen card_id'ler", ""]
    lines += [f"- {cid}" for cid in sorted(p.redundant_card_ids)] or ["- (yok)"]
    return "\n".join(lines) + "\n"
