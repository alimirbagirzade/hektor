"""Sentez makalesi üretici — araştırma oturumlarından yapılandırılmış Markdown.

Amaç: var olan bilgilerin (makaleler + bilgi kartları) üzerine üretilen yeni
hipotez/indikatör önerilerini, insan tarafından incelenebilir tek bir
"sentez makalesi" formatında dışa vermek. Web arayüzünden indirilebilir.

İlkeler (CLAUDE.md):
- Çıktı her zaman HİPOTEZ + TEST NOKTASI'dır; yatırım tavsiyesi değildir.
- verdict != pass ise öneri "aday"dır, "hazır" değildir.
- Kaynaklar gerçek oturum verisinden gelir; retrieval boşsa açıkça yazılır.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.md$")


def synthesis_reports_dir() -> Path:
    """Sentez makalelerinin yazıldığı dizin (reports/synthesis)."""
    d = get_settings().reports_dir / "synthesis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def synthesis_mirror_dir() -> Path | None:
    """Sentezlerin aynalandığı (kopyalandığı) ek dizin — yapılandırılmamışsa None.

    Ayar `ACHILLES_SYNTHESIS_MIRROR_DIR` (config: ``synthesis_mirror_dir``) ile
    verilir. Boşsa aynalama kapalıdır.
    """
    raw = (get_settings().synthesis_mirror_dir or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def mirror_synthesis_report(path: Path) -> Path | None:
    """Üretilen sentez makalesini yapılandırılmış ayna dizinine kopyala.

    Best-effort: ayna dizini yoksa oluşturur; herhangi bir hata sentez üretimini
    KIRMAZ, yalnızca uyarı loglar. Aynalama kapalıysa (dizin yok) None döner.
    """
    mirror = synthesis_mirror_dir()
    if mirror is None:
        return None
    try:
        mirror.mkdir(parents=True, exist_ok=True)
        dest = mirror / path.name
        shutil.copy2(path, dest)
        logger.info("Sentez aynalandı: %s", dest)
        return dest
    except OSError as exc:
        logger.warning("Sentez aynalama başarısız (%s): %s", mirror, exc)
        return None


def is_safe_report_name(name: str) -> bool:
    """Yol kaçışlarına izin vermeyen dosya adı kontrolü."""
    return bool(_SAFE_NAME.match(name)) and ".." not in name


def list_synthesis_reports() -> list[dict[str, Any]]:
    """Mevcut sentez makalelerini (yeniden eskiye) listele."""
    out: list[dict[str, Any]] = []
    for p in sorted(synthesis_reports_dir().glob("*.md"), reverse=True):
        stat = p.stat()
        out.append(
            {
                "name": p.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="minutes"),
            }
        )
    return out


def _fmt_metrics(bt: dict | None) -> list[str]:
    """Backtest metriklerini madde listesine çevir (yoksa açıkça söyle)."""
    if not bt:
        return ["- Backtest sonucu kaydedilmemiş."]
    metrics = bt.get("metrics", bt)
    lines = []
    for key in (
        "total_return",
        "sharpe",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "n_trades",
        "profit_factor",
    ):
        if isinstance(metrics, dict) and key in metrics and metrics[key] is not None:
            lines.append(f"- **{key}**: {metrics[key]}")
    return lines or [f"- Ham sonuç: `{json.dumps(metrics, ensure_ascii=False)[:300]}`"]


def _paper_titles(store: SqliteStore, paper_ids: list[str]) -> list[str]:
    """Kaynak makale başlıklarını getir; bulunamayanı paper_id ile belirt."""
    titles = []
    for pid in paper_ids:
        title = None
        try:
            for p in store.list_papers():
                if p.paper_id == pid:
                    title = p.title
                    break
        except Exception:
            title = None
        titles.append(f"- {title or '(başlık bulunamadı)'} — `{pid}`")
    return titles or ["- (kaynak oturumda kayıtlı değil — retrieval boş olabilir)"]


def generate_synthesis_paper(
    store: SqliteStore | None = None,
    max_sessions: int = 5,
    question_filter: str | None = None,
) -> Path | None:
    """Son araştırma oturumlarından bir sentez makalesi (Markdown) üret.

    Oturum yoksa None döner (kaynak uydurmayız).
    """
    store = store or SqliteStore()
    sessions = store.list_research_sessions(limit=50)
    if question_filter:
        sessions = [s for s in sessions if question_filter.lower() in (s["question"] or "").lower()]
    sessions = sessions[:max_sessions]
    if not sessions:
        return None

    now = dt.datetime.now()
    question = sessions[0]["question"] or "—"
    any_pass = any(s.get("verdict") == "pass" for s in sessions)
    status = "ADAY (backtest PASS)" if any_pass else "HİPOTEZ (backtest geçmedi / doğrulanmadı)"

    lines: list[str] = [
        f"# Sentez Makalesi — {question}",
        "",
        f"_Üretim: {now.strftime('%Y-%m-%d %H:%M')} · Achilles Trader AI · Durum: **{status}**_",
        "",
        "> ⚠ **Bu bir araştırma çıktısıdır — yatırım tavsiyesi DEĞİLDİR.** Tüm öneriler",
        "> test edilmesi gereken hipotezlerdir; out-of-sample doğrulama olmadan hiçbir",
        "> strateji 'hazır' sayılmaz (maliyetler: komisyon + slippage dahil edilmelidir).",
        "",
        "## Özet",
        "",
        f"Bu makale, '{question}' sorusu için {len(sessions)} araştırma iterasyonunun",
        "sentezidir. Var olan literatür kartlarındaki bilgiler birleştirilerek yeni",
        "indikatör/strateji hipotezleri üretilmiş, her biri aynı veri ve maliyet",
        "varsayımlarıyla backtest edilmiş ve yansıma (reflection) ile yorumlanmıştır.",
        "",
    ]

    for i, s in enumerate(sessions, start=1):
        ind = s.get("proposed_indicator") or {}
        ir = s.get("strategy_ir") or {}
        verdict = s.get("verdict") or "—"
        v_emoji = {"pass": "✅", "fail": "❌"}.get(verdict, "🟡")
        lines += [
            f"## Hipotez H{i} — iterasyon {s.get('iteration', '?')} {v_emoji} `{verdict}`",
            "",
            f"_Oturum: `{s['session_id']}` · {str(s.get('created_at') or '')[:16]}_",
            "",
            "### Kaynak makaleler",
            *_paper_titles(store, s.get("source_paper_ids") or []),
            "",
            "### Önerilen indikatör / fikir",
            "",
            f"- **Ad:** {ind.get('name', '—')}",
            f"- **Formül/Tanım:** `{ind.get('formula', ind.get('description', '—'))}`",
            f"- **Gerekçe (sentez):** {s.get('synthesis_reasoning') or ind.get('rationale', '—')}",
            "",
            "### Strateji kuralları (StrategyIR)",
            "",
            "```json",
            json.dumps(ir, ensure_ascii=False, indent=2)[:1500] if ir else "{}",
            "```",
            "",
            "### Backtest sonucu",
            "",
            *_fmt_metrics(s.get("backtest_result")),
            "",
            "### Yansıma / yorum",
            "",
            (s.get("reflection") or "_Yansıma kaydedilmemiş._"),
            "",
            "### Test noktaları (sonraki adımlar)",
            "",
            "- [ ] Out-of-sample dönemde tekrar backtest (farklı tarih aralığı).",
            "- [ ] Komisyon + slippage duyarlılık taraması (2×, 5× maliyet).",
            "- [ ] Parametre sağlamlığı: ±%20 parametre oynatma ile Sharpe değişimi.",
            "- [ ] Look-ahead kontrolü: pozisyonlar `shift(1)` ile mi? (denetle).",
            "",
        ]

    lines += [
        "---",
        "",
        "## Genel değerlendirme",
        "",
        f"- Toplam iterasyon: {len(sessions)} · PASS: "
        f"{sum(1 for s in sessions if s.get('verdict') == 'pass')} · "
        f"FAIL: {sum(1 for s in sessions if s.get('verdict') == 'fail')}",
        "- PASS olmayan hiçbir hipotez 'çalışıyor' kabul edilmez (CLAUDE.md kural 2).",
        "- Bu makale RAG'a geri beslenebilir (kavram makalesi olarak ingest) —",
        "  böylece sonraki sentezler bu bulguların üzerine inşa edilir.",
        "",
    ]

    out = synthesis_reports_dir() / f"sentez_{now.strftime('%Y%m%d_%H%M')}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    mirror_synthesis_report(out)
    return out
