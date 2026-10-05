"""strategy_testing.py — sohbet stratejisini gerçek veride, dönem protokolüyle test et (Faz 2B).

Dönemler (temiz veri özeti başına SABİT, ilk kullanımda kaydedilir):
- ``gelistirme`` : ilk %60 bar. Serbestçe tekrarlanabilir.
- ``dogrulama``  : sonraki %20 (örneklem dışı geliştirme). Her bakış kaydedilir. Aynı AİLEDEN
  başka bir varyant bu döneme daha önce baktıysa sonuç "geliştirmede kullanılmış (N varyant)"
  olarak etiketlenir — bağımsız örneklem dışı kanıt SAYILMAZ.
- ``final``      : son %20, DOKUNULMAMIŞ. Aile başına (aynı veri için) YALNIZ BİR KEZ; ikinci
  istek reddedilir (yeni veri gerekir).

Isınma: her dönemde göstergeler önceki TÜM veriyle hesaplanır, ama işlem/getiri yalnız dönem
içindedir (``event_engine.run`` start/end).

Koşu önkoşulları: stratejide ÖNEMLİ desteklenmeyen kural yok (yoksa test DURUR; kullanıcı açıkça
"basitleştirilmiş strateji" oluşturabilir), nihai strateji orijinal öneri/taslak farkları
görülerek ONAYLANDI (``chat_strategy.approve``; fark listesi değişince onay geçersiz), veri
denetimi geçti, önek değişmezliği (sızıntı) kontrolü geçti.

Sonuç dili: "kayıtlı hesap". Bu sonuç stratejinin gelecekte başarılı olacağını ya da modelin daha
iyi trader olduğunu GÖSTERMEZ (Kural 1/2).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import get_settings
from app.feedback.chat_store import new_id, utcnow
from app.trading import event_engine as ee
from app.trading.data_quality import load_checked_csv
from app.trading.strategy_spec import TestableStrategy
from app.trading.strategy_store import StrategyStore

STAGES = ("gelistirme", "dogrulama", "final")
STAGE_TR = {"gelistirme": "Geliştirme", "dogrulama": "Doğrulama (OOS)", "final": "Final (tek)"}
DEV_FRAC = 0.6
VAL_FRAC = 0.2
CHECK_SEED = 42
DISCLAIMER = (
    "Kayıtlı hesap: bu sayılar yalnız bu strateji tanımının, bu veride, bu dönemde, bu maliyet "
    "varsayımlarıyla ve bu motor sürümüyle hesaplanmış sonucudur. Gelecekte başarı ya da modelin "
    "daha iyi trader olduğu anlamına GELMEZ; yatırım tavsiyesi değildir."
)


MIN_TRADES = 30


def result_warnings(metrics: dict[str, Any]) -> list[str]:
    """Sonucun yorumunu sınırlayan uyarılar (sonucu DEĞİŞTİRMEZ, görünür kılar)."""
    out = []
    n = int(metrics.get("n_trades") or 0)
    if n < MIN_TRADES:
        out.append(f"Az işlem ({n} < {MIN_TRADES}): istatistiksel anlam zayıf.")
    if float(metrics.get("sharpe") or 0) > 4:
        out.append("Gerçek dışı yüksek Sharpe (>4): az işlem, aşırı uyum ya da sızıntı olabilir.")
    pf = metrics.get("profit_factor")
    if pf is not None and float(pf) > 5 and n < 100:
        out.append(f"Şüpheli yüksek profit factor ({pf}) ve az işlem.")
    if n == 0:
        out.append("Hiç işlem yok — strateji bu dönemde test edilmiş sayılmaz.")
    return out


class StrategyTestError(ValueError):
    """Test başlatılamadı (kullanıcıya gösterilir)."""


def market_dir() -> Path:
    return get_settings().market_raw_dir


def resolve_data_file(name: str) -> Path:
    """Yalnız ``market_raw_dir`` altındaki dosya adları (yol geçişi yok)."""
    name = Path(name or "").name
    if not name or not name.lower().endswith(".csv"):
        raise StrategyTestError("Geçerli bir CSV dosya adı seçin.")
    p = market_dir() / name
    if not p.is_file():
        raise StrategyTestError(f"Veri dosyası yok: {name}")
    return p


def list_data_files() -> list[dict[str, Any]]:
    d = market_dir()
    if not d.is_dir():
        return []
    return [
        {"name": p.name, "bytes": p.stat().st_size} for p in sorted(d.glob("*.csv")) if p.is_file()
    ]


def protocol_for(store: StrategyStore, clean_sha: str, index: pd.DatetimeIndex) -> dict[str, Any]:
    n = len(index)
    dev_i = int(n * DEV_FRAC)
    val_i = int(n * (DEV_FRAC + VAL_FRAC))
    if dev_i < 30 or val_i - dev_i < 10 or n - val_i < 10:
        raise StrategyTestError(f"Dönemlere bölmek için veri az ({n} bar).")
    return store.ensure_protocol(clean_sha, index[dev_i].isoformat(), index[val_i].isoformat(), n)


def stage_window(protocol: dict[str, Any], stage: str, index: pd.DatetimeIndex) -> tuple:
    dev_end = pd.Timestamp(protocol["dev_end"])
    val_end = pd.Timestamp(protocol["val_end"])
    if stage == "gelistirme":
        return index[0], index[index < dev_end][-1]
    if stage == "dogrulama":
        return dev_end, index[index < val_end][-1]
    return val_end, index[-1]


def fingerprint(spec: TestableStrategy, clean_sha: str, start: str, end: str) -> str:
    blob = json.dumps(
        {
            "strategy": spec.content_hash(),
            "data": clean_sha,
            "period": [start, end],
            "costs": spec.costs.model_dump(mode="json"),
            "metrics_version": ee.METRICS_VERSION,
            "engine_version": ee.ENGINE_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_stage(
    strategy_id: str,
    *,
    data_file: str,
    tz: str | None,
    stage: str,
    store: StrategyStore | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise StrategyTestError(f"Geçersiz dönem: {stage}")
    store = store or StrategyStore()
    rec = store.get_strategy(strategy_id)
    if rec is None:
        raise StrategyTestError(f"Strateji yok: {strategy_id}")
    spec = TestableStrategy.model_validate(rec["spec"])
    blocking = spec.important_unsupported()
    if blocking:
        raise StrategyTestError(
            "Stratejide desteklenmeyen ÖNEMLİ kural var — asıl strateji test edilemez: "
            + "; ".join(u.text for u in blocking[:3])
            + ". İsterseniz bu kuralları açıkça çıkaran ayrı bir 'basitleştirilmiş strateji' "
            "oluşturup onu test edin."
        )
    from app.trading.chat_strategy import review as final_review

    rv = final_review(strategy_id, store=store)
    if not rv["approved"]:
        why = (
            "fark listesi onaydan sonra değişti"
            if rv.get("approval")
            else f"{rv['n_changes']} fark (orijinal öneri → taslak → nihai) henüz onaylanmadı"
        )
        raise StrategyTestError(
            f"Nihai strateji onaylanmadı ({why}). Test başlatmadan önce farkları görüp onaylayın."
        )
    approval = rv["approval"] or {}
    path = resolve_data_file(data_file)
    df, report = load_checked_csv(path, timeframe=spec.timeframe, tz=tz)
    protocol = protocol_for(store, report.clean_sha256, df.index)
    start, end = stage_window(protocol, stage, df.index)
    leak = ee.prefix_invariance_check(df.loc[:end], spec, seed=CHECK_SEED)
    if not leak["ok"]:
        raise StrategyTestError(
            f"Sızıntı kontrolü başarısız: veri kesildiğinde '{leak['what']}' değişti (kesim "
            f"{leak['mismatch_cut']}). Hesap gelecekten okuyor olabilir; test yapılmadı."
        )

    family = rec["family_id"]
    oos_status = "gelistirme"
    oos_note = "Geliştirme dönemi — örneklem dışı kanıt değildir."
    if stage == "dogrulama":
        prior = store.accesses(family, report.clean_sha256, "dogrulama")
        variants = {a["strategy_id"] for a in prior} - {strategy_id}
        if variants:
            oos_status = "gelistirmede_kullanildi"
            oos_note = (
                f"Bu aile doğrulama dönemine daha önce {len(variants)} başka varyantla baktı; "
                "dönem artık geliştirmede kullanılmış sayılır — bağımsız örneklem dışı kanıt "
                "DEĞİLDİR. Son değerlendirme için dokunulmamış final dönemi kullanılır."
            )
        else:
            oos_status = "ilk_bakis"
            oos_note = "Bu aile doğrulama dönemine ilk kez bakıyor."
    elif stage == "final":
        if store.accesses(family, report.clean_sha256, "final"):
            raise StrategyTestError(
                "Bu strateji ailesi bu verinin final dönemini ZATEN kullandı. Final dönemi "
                "tekrar kullanılırsa geliştirme setine dönüşür — yeni (görülmemiş) veri gerekir."
            )
        oos_status = "final_tek_kullanim"
        oos_note = "Dokunulmamış final dönemi — bu aile için tek kullanım."

    result = ee.run(df, spec, start=start, end=end)
    trials = len(
        {a["strategy_id"] for a in store.accesses(family, report.clean_sha256, "dogrulama")}
        | ({strategy_id} if stage == "dogrulama" else set())
    )
    run_id = new_id("run_")
    run_at = utcnow()
    fp = fingerprint(spec, report.clean_sha256, result.window["start"], result.window["end"])
    payload = {
        **result.summary(),
        "warnings": result_warnings(result.metrics),
        "stage": stage,
        "stage_label": STAGE_TR[stage],
        "oos_status": oos_status,
        "oos_note": oos_note,
        "family_validation_trials": trials,
        "protocol": protocol,
        "leak_check": leak,
        "assumptions": ee.ASSUMPTIONS,
        "metric_definitions": ee.METRIC_DEFINITIONS,
        "engine_version": ee.ENGINE_VERSION,
        "metrics_version": ee.METRICS_VERSION,
        "costs": spec.costs.model_dump(mode="json"),
        "strategy_readable": spec.readable(),
        # Test edilen = ONAYLI NİHAİ strateji. Orijinal öneriden farklıysa sonuç orijinale ait
        # DEĞİLDİR (öğrenme adayı bu alanla bağlanır).
        "tested_strategy": {
            "strategy_id": strategy_id,
            "approval_id": approval.get("approval_id", ""),
            "review_sha": rv["review_sha"],
            "translation_id": rv.get("translation_id", ""),
            "original_sha": rv.get("original_sha", ""),
            "n_changes_from_original": len(rv.get("original_vs_final") or []),
            "n_changes_total": rv["n_changes"],
            "is_original_proposal": bool(rv.get("translation_id"))
            and not rv.get("original_vs_final"),
        },
        "disclaimer": DISCLAIMER,
        "times": {
            "data_start": report.start,
            "data_end": report.end,
            "period_start": result.window["start"],
            "period_end": result.window["end"],
            "strategy_created_at": rec["created_at"],
            "backtest_run_at": run_at,
            # Sonuç yalnız koşu bittiğinde bilinir: bilgi zamanı = koşu zamanı (veri bitişi DEĞİL).
            "knowledge_available_at": run_at,
        },
    }
    if stage in ("dogrulama", "final"):
        store.add_access(family, report.clean_sha256, stage, strategy_id, run_id)
    reports = get_settings().reports_dir / "strategy_tests"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"{run_id}.json"
    report_path.write_text(
        json.dumps(
            {"run_id": run_id, "strategy_id": strategy_id, "data": report.to_dict(), **payload},
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return store.save_run(
        run_id=run_id,
        strategy_id=strategy_id,
        family_id=family,
        stage=stage,
        clean_sha256=report.clean_sha256,
        data_report=report.to_dict(),
        period_start=result.window["start"],
        period_end=result.window["end"],
        oos_status=oos_status,
        result_json=payload,
        engine_version=ee.ENGINE_VERSION,
        metrics_version=ee.METRICS_VERSION,
        fingerprint=fp,
        report_path=str(report_path),
        run_at=run_at,
    )


def check_data(data_file: str, *, timeframe: str, tz: str | None) -> dict[str, Any]:
    """Testten önce veri denetimi (yazma yok; dönem protokolü önizlemesi)."""
    df, report = load_checked_csv(resolve_data_file(data_file), timeframe=timeframe, tz=tz)
    store = StrategyStore()
    existing = store.get_protocol(report.clean_sha256)
    n = len(df)
    preview = existing or {
        "dev_end": df.index[int(n * DEV_FRAC)].isoformat(),
        "val_end": df.index[int(n * (DEV_FRAC + VAL_FRAC))].isoformat(),
        "n_bars": n,
        "note": "Önizleme — ilk testte kalıcı olarak sabitlenir.",
    }
    return {"report": report.to_dict(), "protocol": preview}
