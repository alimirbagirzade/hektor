"""Korpus bütünlük denetimi — katmanlar arası tutarlılık, SALT-OKUMA (Kural 7 bekçisi).

Korpus dört katmanda yaşar: disk (`raw_pdf/`) → SQLite `papers` → SQLite `chunks`
(`embedded` bayrağı) → Chroma vektörleri → `knowledge_cards`. Her katman kendi içinde
tutarlı olabilir ama ARALARINDAKİ sapmalar sessizdir: hiçbir yol hata vermez, en fazla
INFO log düşer. Bu modül tam o sapmaları sayar ve suçlu kimlikleri listeler.

Ölçülmüş gerçek vakalar (2026-09-06, 155 makalelik korpus) — hepsi elle bulundu:
- 155 PDF → 153 makale: başlık-dedup'ı iki farklı makaleyi birleştirdi (yalnız INFO).
- 43.005 chunk SQLite'ta, 750'si Chroma'da: Ollama zaman aşımı → yarım ingest.
- 20 karttan 18'i boş: kart üretici zaman aşımında `{}` yazıp `pending`'e koydu.
- "EN ACİL" işaretli PDF sıfır metin: aylardır bozuk, kimse fark etmedi.

Tasarım sözleşmesi:
- **Salt-okuma.** Silmez, yeniden-ingest etmez, kart onaylamaz, eğitim başlatmaz.
  Her bulguya bir `oneri` (komut) eklenir ama ÇALIŞTIRILMAZ — karar insanın.
- **Saf çekirdek.** `denetle()` düz Python verisi alır → çevrimdışı test edilir. Canlı
  veriyi yalnız `topla()` okur.
- **Deterministik.** Sıralama sabit; aynı girdi aynı rapor.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.memory.paper_indexer import is_boilerplate_title

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_SEVIYE_SIRA = {PASS: 0, WARN: 1, FAIL: 2}

# Sayfa başına bu kadar karakterden azı büyük olasılıkla taranmış/OCR'sız PDF'tir.
TARANMIS_ESIK_KRK_PER_SAYFA = 200


@dataclass
class Bulgu:
    """Tek bir bütünlük kuralının sonucu."""

    kural: str
    seviye: str
    sayi: int
    mesaj: str
    kimlikler: list[str] = field(default_factory=list)
    oneri: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KorpusDenetimi:
    bulgular: list[Bulgu]
    ozet: dict[str, int]

    @property
    def durum(self) -> str:
        return max((b.seviye for b in self.bulgular), key=lambda s: _SEVIYE_SIRA[s], default=PASS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "durum": self.durum,
            "ozet": self.ozet,
            "bulgular": [b.to_dict() for b in self.bulgular],
        }


# ---------------------------------------------------------------------------
# Girdi (düz veri — canlı sistemden `topla()` doldurur, testler elle kurar)
# ---------------------------------------------------------------------------
@dataclass
class KorpusGirdisi:
    disk_pdf_adlari: set[str]
    # paper_id -> (dosya adı, başlık, n_chars, n_pages)
    makaleler: dict[str, tuple[str, str | None, int | None, int | None]]
    # paper_id -> SQLite'ta embedded=1 chunk sayısı (0 dahil)
    sqlite_gomulu: dict[str, int]
    # paper_id -> SQLite'taki toplam chunk sayısı
    sqlite_chunk: dict[str, int]
    # paper_id -> Chroma'daki vektör sayısı
    chroma: dict[str, int]
    # (card_id, paper_id, review_status, card_json)
    kartlar: list[tuple[str, str, str, str]]


# ---------------------------------------------------------------------------
# Kurallar — her biri saf fonksiyon
# ---------------------------------------------------------------------------
def _kirp(ids: Iterable[str], limit: int) -> list[str]:
    s = sorted(ids)
    return s[:limit] if limit > 0 else s


def kural_disk_vs_db(g: KorpusGirdisi, limit: int) -> Bulgu:
    """Diskteki her PDF `papers`'ta olmalı. Eksik = sessiz kayıp (dedup/parse düşürdü)."""
    db_adlari = {ad for ad, _t, _c, _p in g.makaleler.values()}
    eksik = g.disk_pdf_adlari - db_adlari
    fazla = db_adlari - g.disk_pdf_adlari
    if eksik:
        return Bulgu(
            "disk_vs_db",
            FAIL,
            len(eksik),
            f"Diskte olup DB'ye girmemiş PDF: {len(eksik)} (sessiz kayıp).",
            _kirp(eksik, limit),
            "uv run hektor ingest  # idempotent; nedeni için ingest logunda "
            "'Ayni baslikli' / 'Yarım kalmış' satırlarına bak",
        )
    if fazla:
        return Bulgu(
            "disk_vs_db",
            WARN,
            len(fazla),
            f"DB'de olup diskte olmayan makale: {len(fazla)} (dosya taşınmış/silinmiş).",
            _kirp(fazla, limit),
            "Dosyayı raw_pdf/'e geri koy ya da makaleyi bilinçli olarak DB'den çıkar.",
        )
    return Bulgu("disk_vs_db", PASS, 0, f"Disk ↔ DB birebir: {len(db_adlari)} makale.")


def kural_gomme(g: KorpusGirdisi, limit: int) -> Bulgu:
    """Chunk'ı olan her makalenin vektörü Chroma'da olmalı (yarım ingest tespiti)."""
    yarim = [p for p, n in g.sqlite_chunk.items() if n > 0 and g.chroma.get(p, 0) == 0]
    sapma = [
        p
        for p, n in g.sqlite_gomulu.items()
        if p not in yarim and g.chroma.get(p, 0) != n and (n > 0 or g.chroma.get(p, 0) > 0)
    ]
    if yarim:
        return Bulgu(
            "gomme_tutarliligi",
            FAIL,
            len(yarim),
            f"Chunk'ı var ama Chroma'da HİÇ vektörü yok (yarım ingest): {len(yarim)} makale. "
            "Retrieval bu makaleleri GÖREMEZ.",
            _kirp(yarim, limit),
            "uv run hektor ingest  # 'Yarım kalmış ingest' yolu bunları onarır",
        )
    if sapma:
        return Bulgu(
            "gomme_tutarliligi",
            WARN,
            len(sapma),
            f"SQLite embedded sayısı ≠ Chroma vektör sayısı: {len(sapma)} makale.",
            _kirp(sapma, limit),
            "uv run hektor ingest --force  # yalnız sapan makaleler için (yeniden gömer)",
        )
    toplam = sum(g.chroma.values())
    return Bulgu("gomme_tutarliligi", PASS, 0, f"Gömme tutarlı: {toplam} vektör.")


def kart_bos_mu(card_json: str) -> bool:
    """Kart, `paper_id` dışında hiçbir alan taşımıyorsa BOŞTUR (LLM zaman aşımı artığı)."""
    try:
        d = json.loads(card_json)
    except (json.JSONDecodeError, TypeError):
        return True
    if not isinstance(d, dict):
        return True
    return all(v in (None, "", [], {}) for k, v in d.items() if k != "paper_id")


def kural_bos_kartlar(g: KorpusGirdisi, limit: int) -> Bulgu:
    """Boş kart onay kuyruğunu kirletir; onaylanırsa EĞİTİM VERİSİNE girer."""
    bos = [(cid, st) for cid, _p, st, cj in g.kartlar if kart_bos_mu(cj)]
    if not bos:
        return Bulgu("bos_kartlar", PASS, 0, f"Boş kart yok ({len(g.kartlar)} kart).")
    onayli = [cid for cid, st in bos if st == "approved"]
    seviye = FAIL if onayli else WARN
    mesaj = f"Boş kart: {len(bos)} / {len(g.kartlar)}"
    if onayli:
        mesaj += f" — {len(onayli)} tanesi ONAYLI (eğitim verisini zehirler)"
    return Bulgu(
        "bos_kartlar",
        seviye,
        len(bos),
        mesaj + ".",
        _kirp((cid for cid, _ in bos), limit),
        "uv run hektor cards reject <card_id>  # boşları reddet; sonra Ollama boşken "
        "`hektor card <paper_id>` ile yeniden üret",
    )


def kural_boilerplate_baslik(g: KorpusGirdisi, limit: int) -> Bulgu:
    """Şablon/banner başlık = metadata bozuk + başlık-dedup'ı için çakışma riski."""
    kotu = [p for p, (_ad, t, _c, _pg) in g.makaleler.items() if is_boilerplate_title(t or "")]
    if not kotu:
        return Bulgu("boilerplate_baslik", PASS, 0, "Tüm başlıklar gerçek başlık.")
    return Bulgu(
        "boilerplate_baslik",
        WARN,
        len(kotu),
        f"Şablon/banner başlıklı makale: {len(kotu)} (ör. 'Published as a conference paper at').",
        _kirp(kotu, limit),
        "Başlığı elle düzelt ya da `uv run hektor ingest --force` (yeni kural dosya "
        "adından türetir)",
    )


def kural_sifir_metin(g: KorpusGirdisi, limit: int) -> Bulgu:
    """Sıfır karakter = bozuk PDF; sayfa başına çok az karakter = taranmış/OCR'sız."""
    sifir = [p for p, (_ad, _t, c, _pg) in g.makaleler.items() if c is not None and c == 0]
    az = [
        p
        for p, (_ad, _t, c, pg) in g.makaleler.items()
        if c is not None and pg and c > 0 and c / pg < TARANMIS_ESIK_KRK_PER_SAYFA
    ]
    if sifir:
        return Bulgu(
            "sifir_metin",
            FAIL,
            len(sifir),
            f"Metni ÇIKMAYAN PDF: {len(sifir)} (bozuk/şifreli/boş dosya).",
            _kirp(sifir, limit),
            "PDF'i yeniden indir (arXiv) ve `uv run hektor ingest --force`",
        )
    if az:
        return Bulgu(
            "sifir_metin",
            WARN,
            len(az),
            f"Sayfa başına <{TARANMIS_ESIK_KRK_PER_SAYFA} karakter: {len(az)} makale "
            "(muhtemelen taranmış, OCR gerekir).",
            _kirp(az, limit),
            "uv run hektor ingestion-quality --paper-id <id>  # ocr bileşenine bak",
        )
    return Bulgu("sifir_metin", PASS, 0, "Tüm makalelerin metni var.")


def kural_oksuz_kartlar(g: KorpusGirdisi, limit: int) -> Bulgu:
    """paper_id'si `papers`'ta olmayan kart = kaynağı yok (Kural 7)."""
    oksuz = [cid for cid, p, _st, _cj in g.kartlar if p not in g.makaleler]
    if not oksuz:
        return Bulgu("oksuz_kartlar", PASS, 0, "Öksüz kart yok.")
    return Bulgu(
        "oksuz_kartlar",
        WARN,
        len(oksuz),
        f"Kaynağı DB'de olmayan kart: {len(oksuz)}.",
        _kirp(oksuz, limit),
        "uv run hektor lora-curate --run  # orphan kartları eğitimden çıkarır",
    )


def kural_cok_versiyon(g: KorpusGirdisi, limit: int) -> Bulgu:
    """Aynı makaleye birden çok kart = çelişki riski (v5 disiplin kökü)."""
    sayac = Counter(p for _cid, p, _st, _cj in g.kartlar)
    cok = [p for p, n in sayac.items() if n > 1]
    if not cok:
        return Bulgu("cok_versiyon_kart", PASS, 0, "Makale başına en fazla bir kart.")
    return Bulgu(
        "cok_versiyon_kart",
        WARN,
        len(cok),
        f"Birden çok kartı olan makale: {len(cok)}.",
        _kirp(cok, limit),
        "uv run hektor lora-curate --run  # makale başına en zengin tek kartı tutar",
    )


KURALLAR = (
    kural_disk_vs_db,
    kural_gomme,
    kural_bos_kartlar,
    kural_boilerplate_baslik,
    kural_sifir_metin,
    kural_oksuz_kartlar,
    kural_cok_versiyon,
)


def denetle(g: KorpusGirdisi, *, limit: int = 10) -> KorpusDenetimi:
    """Tüm kuralları koş. Saf: yan etkisiz, deterministik."""
    bulgular = [k(g, limit) for k in KURALLAR]
    ozet = {
        "disk_pdf": len(g.disk_pdf_adlari),
        "makale": len(g.makaleler),
        "sqlite_chunk": sum(g.sqlite_chunk.values()),
        "sqlite_gomulu": sum(g.sqlite_gomulu.values()),
        "chroma_vektor": sum(g.chroma.values()),
        "kart": len(g.kartlar),
    }
    return KorpusDenetimi(bulgular=bulgular, ozet=ozet)


def cikis_kodu(d: KorpusDenetimi, *, strict: bool = False) -> int:
    """FAIL → 2 (zincir kapısı), WARN → strict ise 1 yoksa 0, PASS → 0."""
    if d.durum == FAIL:
        return 2
    if d.durum == WARN and strict:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Canlı veri toplama — tek yan-etkisiz okuma noktası
# ---------------------------------------------------------------------------
def topla(store: Any = None, chroma: Any = None, raw_pdf_dir: Path | None = None) -> KorpusGirdisi:
    """SQLite + Chroma + diski oku. Hiçbir şey yazmaz."""
    from sqlalchemy import text

    from app.config import get_settings
    from app.memory.chroma_store import ChromaStore
    from app.memory.sqlite_store import SqliteStore

    s = get_settings()
    store = store or SqliteStore()
    chroma = chroma or ChromaStore()
    raw = raw_pdf_dir or s.raw_pdf_dir

    disk = {p.name for p in Path(raw).glob("*.pdf")} if Path(raw).exists() else set()

    with store.engine.connect() as con:
        makaleler = {
            r[0]: (Path(r[1]).name, r[2], r[3], r[4])
            for r in con.execute(
                text("select paper_id, source_path, title, n_chars, n_pages from papers")
            )
        }
        sqlite_chunk = {
            r[0]: int(r[1])
            for r in con.execute(text("select paper_id, count(*) from chunks group by paper_id"))
        }
        sqlite_gomulu = {
            r[0]: int(r[1])
            for r in con.execute(
                text("select paper_id, sum(embedded) from chunks group by paper_id")
            )
        }
        kartlar = [
            (r[0], r[1], r[2], r[3])
            for r in con.execute(
                text("select card_id, paper_id, review_status, card_json from knowledge_cards")
            )
        ]

    chroma_sayac: Counter[str] = Counter()
    for item in chroma.get_all():
        pid = (item.get("metadata") or {}).get("paper_id")
        if pid:
            chroma_sayac[str(pid)] += 1

    return KorpusGirdisi(
        disk_pdf_adlari=disk,
        makaleler=makaleler,
        sqlite_gomulu={p: sqlite_gomulu.get(p, 0) for p in makaleler} | sqlite_gomulu,
        sqlite_chunk={p: sqlite_chunk.get(p, 0) for p in makaleler} | sqlite_chunk,
        chroma=dict(chroma_sayac),
        kartlar=kartlar,
    )
