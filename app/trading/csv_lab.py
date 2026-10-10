"""csv_lab.py — kullanıcının bıraktığı CSV'den gösterge/strateji ADAYI üret ve disiplinli test et.

Tasarım: ``docs/TASARIM_GECE_DONGUSU.md`` §CSV. Gece döngüsünün bir adımıdır; elle de koşar
(``hektor csv-lab``). Hiçbir sonuç "hazır" değildir — çıktı her zaman *aday* + test noktasıdır.

Akış (dosya başına, temiz veri özeti başına BİR kez):

1. **Meta**: piyasa maliyet profili + saat dilimi + zaman dilimi. Öncelik: yan dosya
   ``<ad>.meta.json`` > dosya adı > ayar. Zaman dilimi verinin adımından ÇIKARILIR (medyan adım
   bilinen bir dilime tam eşit olmalı). Profil bulunamazsa ``ihtiyatli`` (yüksek maliyet)
   kullanılır ve raporda VARSAYIM olarak yazılır — maliyet asla sıfırlanmaz (Kural 3).
2. **Veri denetimi**: ``data_quality.load_checked_csv`` (sıra, yinelenen, OHLC, saat dilimi).
3. **Dönemler**: ``strategy_testing.protocol_for`` — sohbet stratejileriyle AYNI sabit bölme
   (%60 geliştirme · %20 doğrulama · %20 final). Final dönemine bu modül HİÇ dokunmaz.
4. **Arama**: sabit şablon ızgarası (kayıtlı göstergeler, güvenli kural dili — eval yok, Kural 5)
   yalnız GELİŞTİRME döneminde koşar. Deneme sayısı N kaydedilir.
5. **Seçim**: en az ``MIN_TRADES`` işlemli adaylar geliştirme Sharpe'ına göre sıralanır; çoklu
   deneme düzeltmesi olarak Deflated Sharpe (Bailey & López de Prado 2014) hesaplanır.
6. **Doğrulama**: ilk K aday için önek-değişmezliği (sızıntı) kontrolü + doğrulama dönemi (OOS)
   BİR kez; erişim ``csvlab`` ailesine kaydedilir (aynı veride ikinci bakış etiketlenir).
7. **Çıktı**: rapor (JSON + Markdown) + seçilen adaylar için TradingView Pine v5 *gösterge*
   taslağı (yalnız Pine'da karşılığı olan göstergeler).

Pozisyon motoru ``event_engine.run``: sinyal bar kapanışında, dolum sonraki barın açılışında
(look-ahead yok, Kural 4); komisyon + kayma + makas + fonlama her dolumda/barda düşülür.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from app.config import get_settings
from app.feedback.chat_store import new_id, utcnow
from app.trading import event_engine as ee
from app.trading.data_quality import _TF_SECONDS, DataQualityError, load_checked_csv
from app.trading.strategy_spec import TestableStrategy
from app.trading.strategy_store import StrategyStore
from app.trading.strategy_testing import (
    CHECK_SEED,
    DISCLAIMER,
    MIN_TRADES,
    StrategyTestError,
    protocol_for,
    result_warnings,
    stage_window,
)

FAMILY = "csvlab"
LAB_VERSION = "csvlab-1"
DSR_PASS = 0.95

# ── maliyet profilleri (bp) ──────────────────────────────────────────────────
# Kaba, TEMKİNLİ perakende varsayımları; gerçek hesabınızın maliyeti farklıysa yan dosyada
# ``costs`` ile ezin. Sıfır değer yalnız gerekçeyle (CostModel kuralı).
PROFILES: dict[str, dict[str, Any]] = {
    "kripto_vadeli": {
        "label": "Kripto vadeli (perp)",
        "costs": {
            "commission_bps_per_side": 5.0,
            "slippage_bps_per_side": 3.0,
            "spread_bps": 2.0,
            "funding_bps_per_day": 3.0,
        },
        "short": True,
    },
    "kripto_spot": {
        "label": "Kripto spot",
        "costs": {
            "commission_bps_per_side": 10.0,
            "slippage_bps_per_side": 3.0,
            "spread_bps": 2.0,
            "funding_bps_per_day": 0.0,
            "zero_reasons": {"funding_bps_per_day": "spot, kaldıraçsız; fonlama yok"},
        },
        "short": False,
    },
    "forex_cfd": {
        "label": "Forex / CFD (altın dahil)",
        "costs": {
            "commission_bps_per_side": 0.5,
            "slippage_bps_per_side": 1.0,
            "spread_bps": 3.0,
            "funding_bps_per_day": 1.0,
        },
        "short": True,
    },
    "bist": {
        "label": "Hisse / BIST",
        "costs": {
            "commission_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "spread_bps": 5.0,
            "funding_bps_per_day": 0.0,
            "zero_reasons": {"funding_bps_per_day": "spot hisse, kaldıraçsız; fonlama yok"},
        },
        "short": False,
    },
    "ihtiyatli": {
        "label": "Bilinmiyor → ihtiyatlı (yüksek maliyet VARSAYIMI)",
        "costs": {
            "commission_bps_per_side": 10.0,
            "slippage_bps_per_side": 5.0,
            "spread_bps": 5.0,
            "funding_bps_per_day": 3.0,
        },
        "short": True,
    },
}

_FX = {"usd", "eur", "gbp", "jpy", "chf", "aud", "nzd", "cad", "try", "xau", "xag"}


def detect_profile(stem: str) -> tuple[str, str]:
    """Dosya adından (profil, nasıl bulundu). Bulunamazsa ``ihtiyatli``."""
    low = stem.lower()
    tokens = set(re.split(r"[^a-z0-9]+", low))
    if tokens & {"perp", "futures", "future", "vadeli", "usdtm", "swap", "perpetual"}:
        return "kripto_vadeli", "dosya adı (vadeli)"
    if "spot" in tokens:
        return "kripto_spot", "dosya adı (spot)"
    if tokens & {"bist", "xu100", "xu030", "is"} or low.endswith(".is"):
        return "bist", "dosya adı (BIST)"
    if "fx" in tokens or "forex" in tokens or "cfd" in tokens:
        return "forex_cfd", "dosya adı (forex/cfd)"
    for t in tokens:
        if len(t) == 6 and t[:3] in _FX and t[3:] in _FX:
            return "forex_cfd", f"dosya adı ({t.upper()})"
    return "ihtiyatli", "bulunamadı → ihtiyatlı maliyet varsayımı"


# ── zaman dilimi tespiti ─────────────────────────────────────────────────────

_TIME_COLS = ("time", "date", "datetime", "timestamp", "open time")


def detect_timeframe(raw: bytes) -> tuple[str, float, bool]:
    """(zaman dilimi, medyan adım sn, damgada saat dilimi var mı). Tam eşleşme yoksa ``""``."""
    df = pd.read_csv(io.BytesIO(raw), nrows=5000)
    cols = {str(c).strip().lower(): c for c in df.columns}
    tcol = next((cols[c] for c in _TIME_COLS if c in cols), None)
    if tcol is None:
        raise DataQualityError("Zaman kolonu yok (time/date/datetime/timestamp).")
    s = df[tcol]
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all() and len(s):
        unit = "ms" if numeric.abs().median() > 1e11 else "s"
        idx = pd.to_datetime(numeric, unit=unit, utc=True)
        aware = True
    else:
        sample = s.dropna().astype(str).head(50)
        aware = bool(sample.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})\s*$", regex=True).any())
        idx = pd.to_datetime(s, errors="coerce", utc=aware, format="mixed")
    steps = pd.Series(pd.DatetimeIndex(idx).sort_values()).diff().dt.total_seconds()
    steps = steps[steps > 0].dropna()
    if steps.empty:
        raise DataQualityError("Zaman adımı hesaplanamadı (tek damga ya da okunamadı).")
    med = float(steps.median())
    for tf, sec in _TF_SECONDS.items():
        if med == sec:
            return tf, med, aware
    # Günlük hisse verisi: hafta sonu/tatil boşlukları medyanı 1 günde tutar; yine de tam değil
    # ise dilim belirsizdir.
    return "", med, aware


# ── şablon ızgarası ──────────────────────────────────────────────────────────


@dataclass
class Variant:
    template: str
    params: dict[str, Any]
    spec: TestableStrategy
    pine_ok: bool = True


def _spec(
    name: str,
    market: str,
    tf: str,
    direction: str,
    entry: list[str],
    exit_: list[str],
    stop: dict[str, Any],
    costs: dict[str, Any],
) -> TestableStrategy:
    return TestableStrategy.model_validate(
        {
            "name": name,
            "market": market,
            "timeframe": tf,
            "direction": direction,
            "entry_rules": entry,
            "exit_rules": exit_,
            "stop": stop,
            "sizing": {"type": "fixed_fraction", "fraction": 1.0, "max_leverage": 1.0},
            "costs": costs,
        }
    )


def build_variants(market: str, tf: str, costs: dict[str, Any], short: bool) -> list[Variant]:
    """Sabit, küçük ızgara — deneme sayısı raporda N olarak görünür (aşırı arama yok)."""
    out: list[Variant] = []
    trail = {"type": "atr_trailing", "value": 3.0, "atr_period": 14}
    fixed = {"type": "atr_initial", "value": 2.0, "atr_period": 14}
    dirs = ["long", "short"] if short else ["long"]
    for d in dirs:
        up, dn = (">", "<") if d == "long" else ("<", ">")
        for f, sl in ((10, 50), (20, 50), (20, 100), (50, 200)):
            out.append(
                Variant(
                    "ema_trend",
                    {"yon": d, "hizli": f, "yavas": sl},
                    _spec(
                        f"EMA{f}/{sl} trend {d}",
                        market,
                        tf,
                        d,
                        [f"ema_{f} {up} ema_{sl}"],
                        [f"ema_{f} {dn} ema_{sl}"],
                        trail,
                        costs,
                    ),
                )
            )
        for p, th in ((7, 25), (14, 30)):
            lo = th if d == "long" else 100 - th
            ent = f"rsi_{p} < {lo}" if d == "long" else f"rsi_{p} > {lo}"
            ext = f"rsi_{p} > 50" if d == "long" else f"rsi_{p} < 50"
            out.append(
                Variant(
                    "rsi_donus",
                    {"yon": d, "periyot": p, "esik": lo},
                    _spec(f"RSI{p} dönüş {d}", market, tf, d, [ent], [ext], fixed, costs),
                )
            )
        for p, pe in ((20, 0.9), (50, 0.95)):
            out.append(
                Variant(
                    "sma_kirilim_entropi",
                    {"yon": d, "sma": p, "permentropi_esik": pe},
                    _spec(
                        f"SMA{p} kırılım + PE<{pe} {d}",
                        market,
                        tf,
                        d,
                        [f"close {up} sma_{p}", f"permentropy_20 < {pe}"],
                        [f"close {dn} sma_{p}"],
                        trail,
                        costs,
                    ),
                    pine_ok=False,
                )
            )
    return out


# ── istatistik ───────────────────────────────────────────────────────────────

_N = NormalDist()
_EULER = 0.5772156649015329


def _sr_stats(equity: pd.Series) -> tuple[float, int, float, float]:
    """(bar başına Sharpe, gözlem sayısı, çarpıklık, basıklık [normal=3])."""
    r = equity.pct_change().dropna().to_numpy(dtype=float)
    if len(r) < 3 or float(np.std(r)) == 0.0:
        return 0.0, len(r), 0.0, 3.0
    mu, sd = float(np.mean(r)), float(np.std(r))
    z = (r - mu) / sd
    return mu / sd, len(r), float(np.mean(z**3)), float(np.mean(z**4))


def deflated_sharpe(
    sr: float, n_obs: int, skew: float, kurt: float, trial_srs: list[float]
) -> float:
    """Deflated Sharpe Ratio (bar başına SR). N deneme içinden en iyiyi seçmenin şans payı düşülür.

    SR0 = sqrt(Var[SR_i]) · ((1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))); DSR = Φ((SR-SR0)·√(T-1) /
    √(1 - γ3·SR + (γ4-1)/4·SR²)).
    """
    n = len(trial_srs)
    if n_obs < 3:
        return 0.0
    if n >= 2:
        var = float(np.var(trial_srs, ddof=1))
        sr0 = math.sqrt(max(var, 0.0)) * (
            (1 - _EULER) * _N.inv_cdf(1 - 1 / n) + _EULER * _N.inv_cdf(1 - 1 / (n * math.e))
        )
    else:
        sr0 = 0.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0:
        return 0.0
    return float(_N.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)))


# ── Pine v5 gösterge taslağı ─────────────────────────────────────────────────

_PINE = {
    "ema": "ta.ema(close, {p})",
    "sma": "ta.sma(close, {p})",
    "rsi": "ta.rsi(close, {p})",
    "atr": "ta.atr({p})",
}


def to_pine(spec: TestableStrategy) -> str:
    """Kuralları Pine v5 GÖSTERGESİ olarak (al/çık işaretleri). Yalnız desteklenen göstergeler."""

    def col(c: str) -> str:
        if c in ("open", "high", "low", "close", "volume"):
            return c
        name, p = c.rsplit("_", 1)
        if name not in _PINE:
            raise ValueError(f"Pine karşılığı yok: {c}")
        return _PINE[name].format(p=p)

    def cond(rules: list[str]) -> str:
        parts = []
        for r in rules:
            lhs, op, rhs = r.split()
            rr = rhs if re.fullmatch(r"-?\d+(?:\.\d+)?", rhs) else col(rhs)
            parts.append(f"({col(lhs)} {op} {rr})")
        return " and ".join(parts) or "false"

    return "\n".join(
        [
            "//@version=5",
            f'indicator("Hektor aday: {spec.name}", overlay=true)',
            "// ADAY — test noktası, yatırım tavsiyesi DEĞİL. Hektor csv_lab çıktısı.",
            f"// Maliyet varsayımı (bp): {json.dumps(spec.costs.model_dump(mode='json'))}",
            f"giris = {cond(spec.entry_rules)}",
            f"cikis = {cond(spec.exit_rules)}",
            "// Sinyal bar KAPANIŞINDA; işlem bir sonraki barın açılışında varsayılır.",
            'plotshape(giris and barstate.isconfirmed, "Giriş", '
            "shape.triangleup, location.belowbar, color.green)",
            'plotshape(cikis and barstate.isconfirmed, "Çıkış", '
            "shape.triangledown, location.abovebar, color.red)",
        ]
    )


# ── ana akış ─────────────────────────────────────────────────────────────────


@dataclass
class FileResult:
    file: str
    status: str  # done | skipped | failed | seen
    reason: str = ""
    report: dict[str, Any] = field(default_factory=dict)


def _ledger_path() -> Path:
    p = get_settings().state_dir / "nightly" / "csv_lab_ledger.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _ledger() -> dict[str, Any]:
    try:
        return json.loads(_ledger_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_ledger(led: dict[str, Any]) -> None:
    _ledger_path().write_text(json.dumps(led, ensure_ascii=False, indent=2), encoding="utf-8")


def _meta(path: Path) -> dict[str, Any]:
    side = path.with_name(path.stem + ".meta.json")
    if not side.is_file():
        return {}
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataQualityError(f"Yan dosya okunamadı ({side.name}): {exc}") from exc
    return data if isinstance(data, dict) else {}


def _run_window(df: pd.DataFrame, spec: TestableStrategy, start: Any, end: Any) -> ee.RunResult:
    return ee.run(df, spec, start=start, end=end)


def analyze_file(
    path: Path, *, top_k: int | None = None, store: StrategyStore | None = None
) -> dict:
    """Tek CSV → rapor sözlüğü. Hata → ``DataQualityError``/``StrategyTestError``."""
    s = get_settings()
    top_k = s.csv_lab_top_k if top_k is None else top_k
    raw = path.read_bytes()
    meta = _meta(path)
    tf_detected, med_step, aware = detect_timeframe(raw)
    tf = str(meta.get("timeframe") or tf_detected)
    if not tf:
        raise DataQualityError(
            f"Zaman dilimi belirlenemedi (medyan adım {med_step:g} sn bilinen bir dilime eşit "
            f'değil). Yan dosyaya "timeframe" yazın ({path.stem}.meta.json).'
        )
    assumptions: list[str] = []
    tz = meta.get("tz") or s.csv_lab_default_tz or None
    if not aware and not tz:
        if _TF_SECONDS.get(tf, 0) >= 86400:
            tz = "UTC"
            assumptions.append("Günlük/haftalık veride saat dilimi yok → UTC varsayıldı.")
        else:
            raise DataQualityError(
                "Damgalarda saat dilimi yok ve belirtilmedi — sessizce UTC varsayılmaz. Yan "
                'dosyaya "tz" yazın (ör. "UTC", "Europe/Istanbul") ya da '
                "HEKTOR_CSV_LAB_DEFAULT_TZ."
            )
    if meta.get("profile"):
        profile_key, how = str(meta["profile"]), "yan dosya"
        if profile_key not in PROFILES:
            raise DataQualityError(f"Bilinmeyen profil '{profile_key}' ({', '.join(PROFILES)}).")
    else:
        profile_key, how = detect_profile(path.stem)
    profile = PROFILES[profile_key]
    costs = dict(profile["costs"]) | dict(meta.get("costs") or {})
    if profile_key == "ihtiyatli" and not meta.get("costs"):
        assumptions.append(
            "Piyasa bilinmiyor → ihtiyatlı (yüksek) maliyet varsayıldı; gerçek maliyet için "
            'yan dosyaya "profile" ya da "costs" yazın.'
        )
    market = str(meta.get("market") or path.stem)[:40]

    df, report = load_checked_csv(path, timeframe=tf, tz=tz)
    store = store or StrategyStore()
    protocol = protocol_for(store, report.clean_sha256, df.index)
    dev_s, dev_e = stage_window(protocol, "gelistirme", df.index)
    val_s, val_e = stage_window(protocol, "dogrulama", df.index)

    variants = build_variants(market, tf, costs, bool(profile["short"]))
    trials: list[dict[str, Any]] = []
    srs: list[float] = []
    for var in variants:
        try:
            res = _run_window(df, var.spec, dev_s, dev_e)
        except ValueError as exc:
            trials.append({"name": var.spec.name, "error": str(exc)[:200]})
            continue
        sr, nobs, sk, ku = _sr_stats(res.equity)
        srs.append(sr)
        trials.append(
            {
                "name": var.spec.name,
                "template": var.template,
                "params": var.params,
                "strategy_id": var.spec.strategy_id(),
                "dev": res.metrics,
                "_sr": (sr, nobs, sk, ku),
                "_v": var,
            }
        )
    ok = [t for t in trials if "dev" in t and int(t["dev"]["n_trades"]) >= MIN_TRADES]
    # Doğrulama dönemine yalnız geliştirmede maliyet sonrası POZİTİF olanlar bakar: kötü adayla
    # OOS bakışı harcamak, sonraki bakışları "kullanılmış" yapar ve bilgi vermez.
    positive = [t for t in ok if float(t["dev"]["sharpe"]) > 0]
    positive.sort(key=lambda t: (-float(t["dev"]["sharpe"]), t["name"]))
    selected: list[dict[str, Any]] = []
    prior_val = {a["strategy_id"] for a in store.accesses(FAMILY, report.clean_sha256, "dogrulama")}
    out_dir = get_settings().reports_dir / "csv_lab" / datetime.now(UTC).strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in positive[: max(0, top_k)]:
        v: Variant = t["_v"]
        sr, nobs, sk, ku = t["_sr"]
        dsr = deflated_sharpe(sr, nobs, sk, ku, srs)
        leak = ee.prefix_invariance_check(df.loc[:val_e], v.spec, seed=CHECK_SEED)
        entry: dict[str, Any] = {
            "name": t["name"],
            "template": t["template"],
            "params": t["params"],
            "strategy_id": t["strategy_id"],
            "rules": {"giris": v.spec.entry_rules, "cikis": v.spec.exit_rules},
            "readable": v.spec.readable(),
            "dev": t["dev"],
            "dsr_dev": round(dsr, 4),
            "leak_check": leak,
        }
        if not leak["ok"]:
            entry["verdict"] = "sizinti_supheli"
            entry["note"] = "Önek değişmezliği başarısız — doğrulama dönemine bakılmadı."
            selected.append(entry)
            continue
        run_id = new_id("csvlab_")
        val = _run_window(df, v.spec, val_s, val_e)
        store.add_access(FAMILY, report.clean_sha256, "dogrulama", t["strategy_id"], run_id)
        entry["run_id"] = run_id
        entry["val"] = val.metrics
        entry["val_warnings"] = result_warnings(val.metrics)
        entry["val_reused"] = bool(prior_val)
        good = (
            dsr >= DSR_PASS
            and float(val.metrics["sharpe"]) > 0
            and int(val.metrics["n_trades"]) >= MIN_TRADES
            and float(val.metrics["total_return_pct"]) > 0
        )
        entry["verdict"] = "aday_oos_tutarli" if good else "aday_zayif"
        # Final dönemine giden TEK yol: strateji deposuna ``csvlab`` ailesiyle kaydedilir; final
        # aile+veri başına bir kez çalışır (``run_final`` → ``strategy_testing.run_stage``).
        saved, _created = store.save_strategy(
            t["strategy_id"],
            family_id=FAMILY,
            parent_id="",
            name=v.spec.name,
            spec=v.spec.model_dump(mode="json"),
            source={
                "csvlab": {
                    "file": path.name,
                    "clean_sha256": report.clean_sha256,
                    "tz": tz or "",
                    "verdict": entry["verdict"],
                    "dsr_dev": entry["dsr_dev"],
                    "lab_version": LAB_VERSION,
                }
            },
            origin="csvlab",
        )
        entry["store_family"] = saved["family_id"]
        if v.pine_ok:
            pine = out_dir / f"{path.stem}_{t['strategy_id']}.pine"
            pine.write_text(to_pine(v.spec), encoding="utf-8")
            entry["pine"] = str(pine)
        else:
            entry["pine"] = ""
            entry["pine_note"] = "Permütasyon entropisinin Pine karşılığı yok; taslak üretilmedi."
        selected.append(entry)

    return {
        "lab_version": LAB_VERSION,
        "file": path.name,
        "file_sha256": report.file_sha256,
        "clean_sha256": report.clean_sha256,
        "data": report.to_dict(),
        "timeframe": tf,
        "tz": tz or "",
        "timeframe_source": "yan dosya" if meta.get("timeframe") else f"veri adımı {med_step:g} sn",
        "profile": profile_key,
        "profile_label": profile["label"],
        "profile_source": how,
        "costs": costs,
        "assumptions": assumptions,
        "periods": {
            "gelistirme": [str(dev_s), str(dev_e)],
            "dogrulama": [str(val_s), str(val_e)],
            "final": "DOKUNULMADI (insan kararıyla, aile başına bir kez)",
        },
        "n_trials": len(variants),
        "n_trials_ok": len(ok),
        "n_trials_positive": len(positive),
        "min_trades": MIN_TRADES,
        "trials": [{k: val for k, val in t.items() if not k.startswith("_")} for t in trials],
        "selected": selected,
        "disclaimer": DISCLAIMER,
        "note": (
            "Seçim yalnız geliştirme döneminde yapıldı; doğrulama dönemine yalnız seçilen "
            f"{len(selected)} aday BİR kez baktı. 'aday_oos_tutarli' bile hazır strateji DEĞİLDİR: "
            "final dönem, gerçek maliyet ve insan denetimi (/backtest-auditor) gerekir."
        ),
        "created_at": utcnow(),
    }


def _write_report(rep: dict[str, Any], stem: str) -> dict[str, str]:
    out_dir = get_settings().reports_dir / "csv_lab" / datetime.now(UTC).strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{stem}_{rep['clean_sha256'][:8]}"
    base.with_suffix(".json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    base.with_suffix(".md").write_text(render_markdown(rep), encoding="utf-8")
    return {"json": str(base.with_suffix(".json")), "md": str(base.with_suffix(".md"))}


def render_markdown(rep: dict[str, Any]) -> str:
    lines = [
        f"# CSV laboratuvarı — {rep['file']}",
        "",
        f"- Zaman dilimi: **{rep['timeframe']}** ({rep['timeframe_source']})",
        f"- Maliyet profili: **{rep['profile_label']}** ({rep['profile_source']}) · "
        f"{json.dumps(rep['costs'], ensure_ascii=False)}",
        f"- Veri: {rep['data']['rows_clean']} bar · {rep['data']['start']} → {rep['data']['end']}",
        f"- Dönemler: geliştirme {rep['periods']['gelistirme'][0]} → "
        f"{rep['periods']['gelistirme'][1]} · doğrulama {rep['periods']['dogrulama'][0]} → "
        f"{rep['periods']['dogrulama'][1]} · final {rep['periods']['final']}",
        f"- Deneme sayısı N = {rep['n_trials']} (≥{rep['min_trades']} işlemli: "
        f"{rep['n_trials_ok']}; maliyet sonrası pozitif: {rep['n_trials_positive']})",
    ]
    for a in rep["assumptions"]:
        lines.append(f"- ⚠ VARSAYIM: {a}")
    lines += ["", "## Seçilen adaylar", ""]
    if not rep["selected"]:
        lines.append(
            f"Hiçbir deneme geliştirme döneminde hem ≥{rep['min_trades']} işleme hem maliyet "
            "sonrası pozitif Sharpe'a ulaşmadı — aday yok. Bu da bir bulgudur: bu şablonlar bu "
            "veride maliyetleri karşılamıyor."
        )
    for e in rep["selected"]:
        d, v = e["dev"], e.get("val") or {}
        lines += [
            f"### {e['name']} — `{e['verdict']}` · `{e['strategy_id']}`",
            f"- Kurallar: giriş `{' VE '.join(e['rules']['giris'])}` · çıkış "
            f"`{' VE '.join(e['rules']['cikis']) or '-'}`",
            f"- Geliştirme: Sharpe {d['sharpe']} · getiri %{d['total_return_pct']} · DD "
            f"%{d['max_drawdown_pct']} · işlem {d['n_trades']} · maliyet %{d['costs_pct']} · "
            f"DSR {e['dsr_dev']}",
        ]
        if v:
            lines.append(
                f"- Doğrulama (OOS): Sharpe {v['sharpe']} · getiri %{v['total_return_pct']} · DD "
                f"%{v['max_drawdown_pct']} · işlem {v['n_trades']} · maliyet %{v['costs_pct']}"
            )
            for w in e.get("val_warnings") or []:
                lines.append(f"  - ⚠ {w}")
            if e.get("val_reused"):
                lines.append(
                    "  - ⚠ Bu veride doğrulama dönemine daha önce bakıldı (csvlab ailesi)."
                )
        if e.get("note"):
            lines.append(f"- {e['note']}")
        if e.get("pine"):
            lines.append(f"- Pine v5 gösterge taslağı: `{Path(e['pine']).name}`")
        elif e.get("pine_note"):
            lines.append(f"- {e['pine_note']}")
        lines.append("")
    lines += [
        "## Test noktası",
        "",
        "Bu bir hipotez listesidir. Sıradaki adım: /backtest-auditor ile denetim, gerçek maliyet "
        "ve final dönem testi: `hektor csv-lab-final <strategy_id> --dosya <ad>.csv --gerekce "
        '"..."` — bu veride YALNIZ BİR aday için, bir kez. Yatırım tavsiyesi değildir.',
        "",
        f"_{rep['disclaimer']}_",
    ]
    return "\n".join(lines) + "\n"


def run_final(
    strategy_id: str, *, data_file: str, reason: str, tz: str | None = None
) -> dict[str, Any]:
    """(İnsan) CSV adayını dokunulmamış FİNAL döneminde BİR KEZ test et.

    Önkoşullar: aday bu veride csv_lab tarafından doğrulama dönemine bakılmış olmalı (seçim
    disiplini — geliştirmede seçilmemiş bir varyant finale gidemez) · gerekçe ≥10 karakter.
    Final ``csvlab`` ailesi + veri başına tek kullanımdır (``run_stage`` zorlar): aynı veride
    ikinci bir CSV adayı finale giremez. Sonuç "kayıtlı hesap"tır, tavsiye değildir.
    """
    from app.trading.chat_strategy import approve, review
    from app.trading.strategy_testing import resolve_data_file, run_stage

    reason = (reason or "").strip()
    if len(reason) < 10:
        raise StrategyTestError("Final testi gerekçesi en az 10 karakter olmalı.")
    store = StrategyStore()
    rec = store.get_strategy(strategy_id)
    if rec is None:
        raise StrategyTestError(f"Strateji yok: {strategy_id}")
    lab = (rec.get("source") or {}).get("csvlab") or {}
    if rec.get("origin") != "csvlab" or not lab:
        raise StrategyTestError("Bu komut yalnız CSV laboratuvarı adayları içindir.")
    spec = TestableStrategy.model_validate(rec["spec"])
    _df, report = load_checked_csv(
        resolve_data_file(data_file), timeframe=spec.timeframe, tz=tz or lab.get("tz") or None
    )
    looked = {
        a["strategy_id"] for a in store.accesses(rec["family_id"], report.clean_sha256, "dogrulama")
    }
    if strategy_id not in looked:
        raise StrategyTestError(
            "Bu aday bu veride doğrulama dönemine bakılarak seçilmedi — finale gidemez "
            "(önce `hektor csv-lab` bu veriyi işlemeli)."
        )
    rv = review(strategy_id, store=store)
    if not rv["approved"]:
        approve(
            strategy_id,
            review_sha=rv["review_sha"],
            acknowledged=[i["key"] for i in rv["items"]],
            note=f"csv_lab final (insan): {reason}"[:1000],
            store=store,
        )
    return run_stage(
        strategy_id,
        data_file=data_file,
        tz=tz or lab.get("tz") or None,
        stage="final",
        store=store,
    )


def run_inbox(*, max_files: int | None = None, top_k: int | None = None) -> dict[str, Any]:
    """``market_raw_dir`` altındaki YENİ CSV'leri işle (dosya özeti başına bir kez)."""
    s = get_settings()
    d = s.market_raw_dir
    files = sorted(p for p in d.glob("*.csv") if p.is_file()) if d.is_dir() else []
    led = _ledger()
    limit = s.csv_lab_max_files if max_files is None else max_files
    results: list[FileResult] = []
    done = 0
    for p in files:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        meta_p = p.with_name(p.stem + ".meta.json")
        meta_sha = hashlib.sha256(meta_p.read_bytes()).hexdigest()[:16] if meta_p.is_file() else ""
        key = f"{sha}:{meta_sha}:{LAB_VERSION}"
        if key in led:
            results.append(FileResult(p.name, "seen", led[key].get("status", "")))
            continue
        if done >= max(0, limit):
            results.append(FileResult(p.name, "skipped", "gece sınırı — sonraki gece"))
            continue
        done += 1
        try:
            rep = analyze_file(p, top_k=top_k)
            paths = _write_report(rep, p.stem)
            led[key] = {"status": "done", "at": utcnow(), **paths}
            results.append(FileResult(p.name, "done", "", {**_brief(rep), **paths}))
        except (DataQualityError, StrategyTestError) as exc:
            # Kalıcı kullanıcı-düzeltmeli hata: dosya/yan dosya değişene kadar tekrar denenmez.
            led[key] = {"status": "skipped", "reason": str(exc)[:500], "at": utcnow()}
            results.append(FileResult(p.name, "skipped", str(exc)[:500]))
        except Exception as exc:  # beklenmeyen: kaydetme → ertesi gece yeniden denenir
            results.append(FileResult(p.name, "failed", f"{type(exc).__name__}: {exc}"[:500]))
        _save_ledger(led)
    return {
        "dir": str(d),
        "files": [r.__dict__ for r in results],
        "n_done": sum(1 for r in results if r.status == "done"),
        "n_new_candidates": sum(len(r.report.get("selected", [])) for r in results),
    }


def _brief(rep: dict[str, Any]) -> dict[str, Any]:
    return {
        "timeframe": rep["timeframe"],
        "profile": rep["profile"],
        "n_trials": rep["n_trials"],
        "assumptions": rep["assumptions"],
        "selected": [
            {
                "name": e["name"],
                "verdict": e["verdict"],
                "dev_sharpe": e["dev"]["sharpe"],
                "dsr": e["dsr_dev"],
                "val_sharpe": (e.get("val") or {}).get("sharpe"),
                "val_trades": (e.get("val") or {}).get("n_trades"),
            }
            for e in rep["selected"]
        ],
    }
