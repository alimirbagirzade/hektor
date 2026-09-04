"""Zincir Veri Üreticisi — araştırma oturumlarını LoRA eğitim verisine çevirir.

Mevcut eğitim formatı: soru → cevap (basit)
Bu modülün ürettiği format: tam reasoning zinciri

Örnek eğitim kaydı:
  prompt: "Araştırma: momentum + volatilite kombinasyonu\n
           Mevcut formüller: RSI (momentum), ATR (volatilite), EMA (trend)\n
           Kavram ilişkileri: RSI --[measures]--> momentum\n
           Görev: Yeni indikatör öner"

  completion: "Düşünce: RSI yüksek volatilitede çok sinyal üretiyor...
               Öneri: VolumeFilteredRSI = RSI(14) sadece ATR < ortalama_ATR iken
               Giriş: rsi_14 > 55 ve atr_14 < avg_atr_20
               Beklenen avantaj: Volatilite filtrelemesi gürültüyü azaltır
               Sınırlama: Yavaş piyasada fırsat kaçırabilir
               [Backtest: Sharpe 1.2, OOS pozitif — PASS]
               Yansıma: ATR eşiği iyi çalıştı, ancak..."

Bu format modele "nasıl düşünmesi" gerektiğini öğretir.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import get_settings
from app.memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


def _format_reasoning_prompt(session: dict[str, Any], formulas_text: str, graph_text: str) -> str:
    return (
        f"Araştırma sorusu: {session['question']}\n\n"
        f"Mevcut formüller:\n{formulas_text}\n\n"
        f"Kavram grafiği:\n{graph_text}\n\n"
        "Görev: Bu bilgileri kullanarak yeni bir trading indikatörü öner. "
        "Önce düşün, sonra yapısal olarak öner, sonra beklentilerini ve sınırlamalarını açıkla."
    )


def _format_reasoning_completion(session: dict[str, Any]) -> str:
    parts: list[str] = []

    ind = session.get("proposed_indicator") or {}
    reasoning = session.get("synthesis_reasoning") or ind.get("combination_reasoning", "")
    if reasoning:
        parts.append(f"Düşünce: {reasoning}")

    ind_name = ind.get("indicator_name", "")
    if ind_name:
        parts.append(f"Önerilen indikatör: {ind_name}")

    components = ind.get("formula_components", [])
    if components:
        comp_list = ", ".join(f"{c.get('name', '?')} ({c.get('role', '')})" for c in components[:4])
        parts.append(f"Bileşenler: {comp_list}")

    ir = session.get("strategy_ir") or {}
    entry = ir.get("entry_rules", [])
    if entry:
        parts.append(f"Giriş kuralları: {entry}")

    expected = ind.get("expected_edge", "")
    if expected:
        parts.append(f"Beklenen avantaj: {expected}")

    failures = ind.get("failure_conditions", [])
    if failures:
        parts.append(f"Başarısızlık koşulları: {'; '.join(failures[:3])}")

    bt = session.get("backtest_result") or {}
    verdict = session.get("verdict", "")
    if verdict and bt:
        m = bt.get("metrics", {})
        sharpe = m.get("sharpe")
        ret = m.get("total_return_pct")
        sharpe_str = f"Sharpe {sharpe:.3f}, " if sharpe else ""
        ret_str = f"getiri {ret:.3f}%" if ret else ""
        parts.append(f"Backtest: {sharpe_str}{ret_str} → {verdict.upper()}")

    reflection = session.get("reflection", "")
    if reflection:
        parts.append(f"Yansıma: {reflection}")

    improvement = session.get("improvement_notes", "")
    if improvement:
        parts.append(f"İyileştirme: {improvement}")

    return "\n".join(parts) if parts else "(boş zincir)"


class ChainDataBuilder:
    """Araştırma oturumlarından MLX-LM uyumlu JSONL üretir."""

    def __init__(self, store: SqliteStore | None = None) -> None:
        self.store = store or SqliteStore()
        self.settings = get_settings()

    def build(
        self,
        only_successful: bool = False,
        min_iterations: int = 1,
    ) -> dict[str, Any]:
        """
        Tüm research_session kayıtlarından reasoning chain eğitim örnekleri üret.
        Returns: {n_records, output_path, content_hash}
        """
        import hashlib

        sessions = self.store.list_research_sessions(limit=1000)
        formulas = self.store.list_formulas()
        formulas_text = self._format_formulas(formulas)
        graph_links = self.store.list_concept_links()
        graph_text = self._format_graph(graph_links)

        records: list[dict[str, str]] = []
        seen: set[str] = set()

        for s in sessions:
            if only_successful and s.get("verdict") != "pass":
                continue
            if not s.get("proposed_indicator"):
                continue

            prompt = _format_reasoning_prompt(s, formulas_text, graph_text)
            completion = _format_reasoning_completion(s)

            key = f"{s['question']}||{s.get('iteration', 1)}"
            if key in seen:
                continue
            seen.add(key)

            records.append({"prompt": prompt, "completion": completion})

        out_dir = self.settings.jsonl_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "research_chains.jsonl"

        hasher = hashlib.sha256()
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                line = json.dumps(rec, ensure_ascii=False)
                f.write(line + "\n")
                hasher.update(line.encode())

        logger.info("Zincir veri seti: %d kayıt → %s", len(records), out_path)
        return {
            "n_records": len(records),
            "output_path": str(out_path),
            "content_hash": hasher.hexdigest()[:16],
        }

    def _format_formulas(self, formulas: list[dict]) -> str:
        if not formulas:
            return "(henüz formül yok)"
        lines = [
            f"• {f['name']} [{f.get('category', '?')}]: {f.get('description', '')[:80]}"
            for f in formulas[:20]
        ]
        return "\n".join(lines)

    def _format_graph(self, links: list[dict]) -> str:
        if not links:
            return "(kavram grafiği boş)"
        lines = [
            f"  {lk['from_concept']} --[{lk['relation']}]--> {lk['to_concept']}"
            for lk in links[:20]
        ]
        return "\n".join(lines)
