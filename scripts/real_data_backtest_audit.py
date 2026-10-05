"""real_data_backtest_audit.py — GERÇEK OHLCV ile backtest hesap/veri akışı denetimi.

Başarı ölçütü KÂR DEĞİL: ürün yolu (kaydet → nihai onay → veri denetimi → dönem testi) ile
üretilen işlemlerin, üretim motorunu kullanmayan bağımsız hesapla (``tests/independent_backtest``)
aynı çıkması ve veri akışının (satır, boşluk, saat dilimi, dönem sınırları) tutarlı olması.

- İZOLE kökte koşar (``--root``; ``HEKTOR_ROOT_PATH`` + ayrı SQLite) → ana kurulumun veritabanı,
  dönem erişim kayıtları ve raporları DEĞİŞMEZ.
- Yalnız geliştirme ve doğrulama dönemleri koşulur; final dönemi KOŞULMAZ (bu denetim bir
  geliştirme kullanımıdır, sonuç "dokunulmamış final" değildir).
- Nihai strateji onayı bu betikte otomatik verilir ve öyle etiketlenir: kullanıcının strateji
  onayı DEĞİLDİR, yalnız teknik doğrulama içindir.

Kullanım::

    uv run python scripts/real_data_backtest_audit.py \
        --csv C:/HP/hektor/data/market/raw/BTCUSDT_1h_2023-01_2024-12.csv \
        --root tmp/real_data_audit --report docs/evidence/gercek_veri_backtest_denetimi
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COSTS = {
    "commission_bps_per_side": 10.0,
    "slippage_bps_per_side": 2.0,
    "spread_bps": 2.0,
    "funding_bps_per_day": 0.0,
    "zero_reasons": {"funding_bps_per_day": "spot piyasa: kaldıraçsız, fonlama/taşıma yok"},
}
ASSUMPTION_NOTES = {
    "commission_bps_per_side": "Binance spot standart taker ücreti %0.10 = 10 bp/dolum "
    "(indirimsiz; hesap seviyesine göre değişir — varsayım)",
    "slippage_bps_per_side": "2 bp/dolum aleyhe (1h BTCUSDT için varsayım; ölçülmedi)",
    "spread_bps": "2 bp tam makas (kline verisinde kotasyon yok → ölçülmedi, varsayım)",
    "funding_bps_per_day": "0 — spot, kaldıraçsız (gerekçe kayıtlı)",
}


def strategies() -> list[dict]:
    base = {
        "market": "BTCUSDT spot (teknik doğrulama piyasası)",
        "timeframe": "1h",
        "indicators": [{"name": "EMA", "period": 20}, {"name": "EMA", "period": 50}],
        "stop": {"type": "atr_initial", "value": 2.0, "atr_period": 14},
        "take_profit": {"type": "r_multiple", "value": 3.0},
        "sizing": {"type": "risk_per_trade", "risk_pct": 0.01, "max_leverage": 1.0},
        "costs": COSTS,
    }
    return [
        {
            **base,
            "name": "denetim_ema20_50_long",
            "direction": "long",
            "entry_rules": ["ema_20 > ema_50"],
            "exit_rules": ["ema_20 < ema_50"],
        },
        {
            **base,
            "name": "denetim_ema20_50_short",
            "direction": "short",
            "entry_rules": ["ema_20 < ema_50"],
            "exit_rules": ["ema_20 > ema_50"],
        },
    ]


def independent_data_check(path: Path) -> dict:
    from tests.independent_backtest import read_csv

    b = read_csv(path)
    ts = [dt.datetime.fromisoformat(t) for t in b.time]
    steps = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
    gaps = [(b.time[i - 1], b.time[i]) for i, s in enumerate(steps, start=1) if s != 3600]
    bad = [
        i
        for i in range(len(b.close))
        if not (b.low[i] <= min(b.open[i], b.close[i]) <= max(b.open[i], b.close[i]) <= b.high[i])
    ]
    return {
        "rows": len(b.close),
        "first": b.time[0],
        "last": b.time[-1],
        "monotonic": all(s > 0 for s in steps),
        "gaps": gaps,
        "missing_bars": int(sum(s / 3600 - 1 for s in steps if s > 3600)),
        "invalid_ohlc_rows": bad[:10],
        "tz_offsets": sorted({t[-6:] for t in b.time}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--root", default=str(ROOT / "tmp" / "real_data_audit"))
    ap.add_argument("--report", default=str(ROOT / "docs/evidence/gercek_veri_backtest_denetimi"))
    a = ap.parse_args(argv)
    src = Path(a.csv)
    prov_path = src.with_name(src.stem + ".provenance.json")
    if not prov_path.is_file():
        raise SystemExit(f"Köken kaydı yok: {prov_path} — kaynağı belirsiz veri kullanılmaz.")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    csv_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    if csv_sha != prov["csv_sha256"]:
        raise SystemExit("CSV özeti köken kaydıyla tutmuyor — veri değişmiş.")

    root = Path(a.root).resolve()
    if root.exists():
        shutil.rmtree(root)
    (root / "data" / "market" / "raw").mkdir(parents=True)
    shutil.copy2(src, root / "data" / "market" / "raw" / src.name)
    os.environ["HEKTOR_ROOT_PATH"] = str(root)
    os.environ["HEKTOR_SQLITE_PATH"] = str(root / "audit.db")

    from app.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    from tests import independent_backtest as ib

    from app.trading import event_engine as ee
    from app.trading.chat_strategy import approve, review, save_spec
    from app.trading.strategy_testing import check_data, run_stage

    data = check_data(src.name, timeframe="1h", tz=None)
    ind_data = independent_data_check(src)
    bars = ib.read_csv(src)
    out: dict = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "engine_version": ee.ENGINE_VERSION,
        "metrics_version": ee.METRICS_VERSION,
        "data": {
            "provenance": {k: v for k, v in prov.items() if k != "files"},
            "n_source_files": len(prov.get("files", [])),
            "csv_sha256_verified": csv_sha,
            "production_report": data["report"],
            "protocol_preview": data["protocol"],
            "independent_check": ind_data,
        },
        "costs": COSTS,
        "assumption_notes": ASSUMPTION_NOTES,
        "runs": [],
        "scope_note": "Yalnız geliştirme + doğrulama dönemleri; final dönemi koşulmadı. "
        "Bağımsız hesap kapsamı: EMA/ATR, atr_initial stop, R hedef, risk tabanlı boyut "
        "(takip eden stop ve ATR-katı hedef kapsam dışı).",
        "approval_note": "Nihai strateji onayı denetim betiği tarafından otomatik verildi — "
        "kullanıcının strateji onayı DEĞİL; yalnız teknik doğrulama.",
    }
    rep = data["report"]
    out["data"]["consistency"] = {
        "rows_equal": rep["rows_clean"] == ind_data["rows"] == prov["n_rows"],
        "start_equal": rep["start"][:19] == ind_data["first"][:19],
        "end_equal": rep["end"][:19] == ind_data["last"][:19],
        "gap_count_equal": int((rep.get("gaps") or {}).get("n_gaps", -1)) == len(ind_data["gaps"]),
    }
    for spec in strategies():
        rec = save_spec(spec, source={"origin": "real_data_backtest_audit"})
        rv = review(rec["strategy_id"])
        approve(
            rec["strategy_id"],
            review_sha=rv["review_sha"],
            acknowledged=[i["key"] for i in rv["items"]],
            note="otomatik denetim betiği — teknik doğrulama; kullanıcı strateji onayı değil",
        )
        for stage in ("gelistirme", "dogrulama"):
            run = run_stage(rec["strategy_id"], data_file=src.name, tz=None, stage=stage)
            res = run["result"]
            ind = ib.run(bars, spec, run["period_start"], run["period_end"], bar_days=1 / 24)
            cmp = ib.compare_trades(res["trades"], ind.trades, n=10)
            prod_ret = res["metrics"]["total_return_pct"]
            ind_ret = round((ind.final_equity - 1.0) * 100, 4)
            out["runs"].append(
                {
                    "strategy": spec["name"],
                    "strategy_id": rec["strategy_id"],
                    "direction": spec["direction"],
                    "stage": stage,
                    "oos_status": run["oos_status"],
                    "period": [run["period_start"], run["period_end"]],
                    "fingerprint": run["fingerprint"],
                    "leak_check_ok": res["leak_check"]["ok"],
                    "metrics": res["metrics"],
                    "counters": res["counters"],
                    "independent_total_return_pct": ind_ret,
                    "total_return_equal": abs(prod_ret - ind_ret) <= 1e-4,
                    "comparison": cmp,
                }
            )
    ok = all(out["data"]["consistency"].values()) and all(
        r["comparison"]["same_structure"]
        and r["comparison"]["max_rel_diff"] < 1e-9
        and r["total_return_equal"]
        and r["leak_check_ok"]
        for r in out["runs"]
    )
    out["verdict"] = "HESAP TUTARLI" if ok else "FARK VAR — incele"
    rp = Path(a.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.with_suffix(".json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    rp.with_suffix(".md").write_text(render_md(out), encoding="utf-8")
    print(out["verdict"])
    return 0 if ok else 1


def _f(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}f}"


def render_md(o: dict) -> str:
    p = o["data"]["provenance"]
    d = o["data"]
    lines = [
        "# Gerçek piyasa verisiyle backtest hesap denetimi",
        "",
        f"_Üretim: {o['generated_at']} · motor `{o['engine_version']}` · metrik "
        f"`{o['metrics_version']}` · sonuç: **{o['verdict']}**_",
        "",
        "> Başarı ölçütü kâr değil, hesapların ve veri akışının doğruluğudur. Sayılar "
        "**kayıtlı hesap**tır; strateji önerisi ya da yatırım tavsiyesi değildir. Final dönemi "
        "koşulmadı; doğrulama dönemi bu denetimde **geliştirme amaçlı** kullanıldı.",
        "",
        "## Veri",
        "",
        f"- {p['label']}",
        f"- Kaynak: {p['source']} — `{p['source_base_url']}` ({d['n_source_files']} aylık dosya, "
        "her biri `.CHECKSUM` SHA256 ile doğrulandı)",
        f"- Sembol / tür / aralık: **{p['symbol']}** · {p['market_type']} · {p['interval']}",
        f"- Saat dilimi: {p['timezone']}",
        f"- Dönem: {p['period_actual']['first_open']} → {p['period_actual']['last_open']} "
        f"({p['n_rows']} bar)",
        f"- İndirme zamanı: {p['downloaded_at']}",
        f"- CSV SHA256: `{p['csv_sha256']}` (denetimde yeniden hesaplandı, eşleşti)",
        f"- Sınır: {p['limits']}",
        "",
        "### Veri akışı tutarlılığı (ürün denetimi ↔ bağımsız okuma)",
        "",
        "| Kontrol | Sonuç |",
        "|---|---|",
    ]
    for k, v in d["consistency"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")
    ic = d["independent_check"]
    lines += [
        "",
        f"Bağımsız okuma: {ic['rows']} satır, artan sıralı={ic['monotonic']}, boşluk "
        f"{len(ic['gaps'])} ({ic['missing_bars']} eksik bar; doldurulmadı): "
        + ", ".join(f"{a} → {b}" for a, b in ic["gaps"][:5])
        + f"; geçersiz OHLC satırı {len(ic['invalid_ohlc_rows'])}; ofsetler {ic['tz_offsets']}.",
        f"Dönem sınırları (sabit, temiz veri özeti başına): geliştirme → "
        f"{d['protocol_preview'].get('dev_end')} · doğrulama → "
        f"{d['protocol_preview'].get('val_end')}.",
        "",
        "## Varsayımlar",
        "",
        "- Giriş: kapanışta `ema_20 > ema_50` (long) / `ema_20 < ema_50` (short); dolum sonraki "
        "açılışta.",
        "- Çıkış: kapanışta ters kesişim → sonraki açılış; stop 2×ATR(14) girişte sabit (sinyal "
        "barının ATR'si); hedef 3R; aynı barda stop+hedef → stop önce; gap → açılıştan.",
        "- Pozisyon büyüklüğü: işlem başına özsermayenin %1 riski (stop mesafesine göre), "
        "kaldıraç ≤ 1.",
    ]
    for k, v in o["assumption_notes"].items():
        lines.append(f"- `{k}` = {o['costs'][k]}: {v}")
    lines += ["", f"Kapsam: {o['scope_note']}", "", f"Onay: {o['approval_note']}", ""]
    lines += ["## Koşular", ""]
    lines += [
        "| Strateji | Dönem | OOS etiketi | İşlem (ürün/bağımsız) | Getiri ürün / bağımsız (%) "
        "| Azami göreli fark | Yapı aynı |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in o["runs"]:
        c = r["comparison"]
        lines.append(
            f"| {r['strategy']} | {r['stage']} ({r['period'][0][:10]} → {r['period'][1][:10]}) "
            f"| {r['oos_status']} | {c['n_prod']}/{c['n_independent']} | "
            f"{r['metrics']['total_return_pct']} / {r['independent_total_return_pct']} | "
            f"{c['max_rel_diff']:.2e} | {'✅' if c['same_structure'] else '❌'} |"
        )
    for r in o["runs"]:
        c = r["comparison"]
        lines += [
            "",
            f"### {r['strategy']} · {r['stage']} — {c['compared']} işlem karşılaştırıldı "
            f"(toplam {c['n_prod']}; >10 ise ilk, son ve aradan eşit aralıklı)",
            "",
            "| # | Giriş | Çıkış | Neden | Giriş fiyatı | Çıkış fiyatı | Miktar | Maliyet "
            "| Net PnL | Δfiyat (g/ç) | Δmiktar | Δmaliyet | ΔPnL |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for row in c["rows"]:
            df = row["diff"]
            lines.append(
                f"| {row['index']} | {row['entry_time'][:16]} | {row['exit_time'][:16]} | "
                f"{row['exit_reason']} | {row['entry_fill']:.2f} | {row['exit_fill']:.2f} | "
                f"{_f(row['qty'], 8)} | {_f(row['cost'], 8)} | {_f(row['net_pnl'], 8)} | "
                f"{df['entry_fill']:.1e}/{df['exit_fill']:.1e} | {df['qty']:.1e} | "
                f"{df['cost']:.1e} | {df['net_pnl']:.1e} |"
            )
        cn = r["counters"]
        lines.append(
            f"\nÇıkış sayaçları (ürün): stop {cn['stop_exits']} · gap-stop {cn['gap_stop_exits']} "
            f"· hedef {cn['tp_exits']} · kural {cn['rule_exits']} · dönem sonu {cn['end_exits']} "
            f"· aynı bar stop önce {cn['same_bar_stop_first']}. Sızıntı (önek) kontrolü: "
            f"{'geçti' if r['leak_check_ok'] else 'KALDI'}."
        )
    lines += [
        "",
        "Not: Tabloda 10 işlem gösterilir; yapı (giriş/çıkış zamanı, neden) ve fiyat/PnL göreli "
        "farkı TÜM işlemlerde ayrıca taranır (`all_trades_checked`, `max_rel_diff`). "
        "Fiyat/miktar/PnL birimi özsermaye = 1.0 ölçeğindedir (miktar = nominal / fiyat). "
        "Gözlem: EMA'lar ilk bardan başlar (ısınma eşiği yok) — verinin ilk ~50 barındaki "
        "sinyaller olgunlaşmamış göstergeyle üretilir; bu bir tasarım sınırıdır, hata değil. "
        "Gerçek veride gap ve aynı-bar stop/hedef olayları az ya da hiç oluşmayabilir; bu uç "
        "durumlar `tests/test_independent_backtest_crosscheck.py` kontrollü bar testleriyle "
        "ayrıca doğrulanır.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
