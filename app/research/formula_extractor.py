"""Formül Çıkarıcı — PDF chunk'larından matematiksel formülleri yapısal olarak çıkarır.

Her chunk için LLM'e sorar:
  "Bu metindeki trading/finans formüllerini JSON olarak ver."

Çıktı doğrudan `formulas` tablosuna yazılır.
Ollama çevrimdışıysa chunk başına kural tabanlı yedek çalışır.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)

# Basit kural tabanlı yedek: bilinen indikatör adlarını tespit et
_KNOWN_INDICATORS = {
    "RSI": ("momentum", "Relative Strength Index — aşırı alım/satım tespiti"),
    "MACD": ("trend", "Moving Average Convergence Divergence — momentum değişimi"),
    "EMA": ("trend", "Exponential Moving Average — ağırlıklı hareketli ortalama"),
    "SMA": ("trend", "Simple Moving Average — basit hareketli ortalama"),
    "ATR": ("volatility", "Average True Range — volatilite ölçümü"),
    "Bollinger": ("volatility", "Bollinger Bands — fiyat kanalı"),
    "Sharpe": ("risk", "Sharpe Ratio — risk-getiri oranı"),
    "drawdown": ("risk", "Maximum Drawdown — maksimum kayıp"),
    "OBV": ("volume", "On-Balance Volume — hacim akışı"),
    "VWAP": ("volume", "Volume Weighted Average Price — hacim ağırlıklı fiyat"),
    "Stochastic": ("momentum", "Stochastic Oscillator — momentum indikatörü"),
}

_EXTRACT_PROMPT = """\
Aşağıdaki akademik metin parçasındaki trading ve finans formüllerini, göstergelerini \
ve matematiksel ifadelerini çıkar.

METİN:
{text}

Her formül için şu JSON dizisini döndür (formül yoksa boş dizi []):
[
  {{
    "name": "RSI",
    "latex": "100 - \\\\frac{{100}}{{1 + RS}}",
    "plain": "RSI = 100 - 100/(1+RS), RS = ortalama_kazanç/ortalama_kayıp",
    "description": "Momentum indikatörü, aşırı alım/satım tespiti",
    "variables": {{"period": "geri bakış penceresi", "RS": "kazanç/kayıp oranı"}},
    "category": "momentum"
  }}
]

Kategoriler: momentum, trend, volatility, volume, risk, statistical
Eğer formül yoksa sadece [] döndür. Başka açıklama yapma.
"""


class FormulaExtractor:
    def __init__(
        self,
        store: SqliteStore | None = None,
        llm: LocalLLM | None = None,
    ) -> None:
        self.store = store or SqliteStore()
        self.llm = llm or LocalLLM()

    def extract_from_paper(self, paper_id: str, *, force: bool = False) -> list[dict[str, Any]]:
        """Bir makalenin tüm chunk'larından formül çıkar."""
        chunks = self.store.list_chunks(paper_id)
        all_formulas: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for chunk in chunks:
            try:
                formulas = self._extract_from_chunk(chunk.text, paper_id, force=force)
            except Exception as exc:
                logger.warning("Chunk formül çıkarımı başarısız: %s", exc)
                formulas = []

            for f in formulas:
                name = f.get("name", "").strip()
                if not name or name in seen_names:
                    continue
                seen_names.add(name)
                if not force and self.store.formula_exists(paper_id, name):
                    continue

                fid = f"fml_{uuid.uuid4().hex[:12]}"
                self.store.save_formula(
                    formula_id=fid,
                    paper_id=paper_id,
                    name=name,
                    latex=f.get("latex"),
                    plain=f.get("plain"),
                    description=f.get("description"),
                    variables_json=json.dumps(f.get("variables", {}), ensure_ascii=False),
                    category=f.get("category"),
                )
                all_formulas.append({"formula_id": fid, **f})

        logger.info(
            "paper=%s → %d yeni formül çıkarıldı (toplam chunk=%d)",
            paper_id,
            len(all_formulas),
            len(chunks),
        )
        return all_formulas

    def extract_from_all_papers(self) -> dict[str, list[dict[str, Any]]]:
        """Tüm makalelerden formül çıkar."""
        results: dict[str, list[dict[str, Any]]] = {}
        for paper in self.store.list_papers():
            results[paper.paper_id] = self.extract_from_paper(paper.paper_id)
        return results

    def _extract_from_chunk(
        self, text: str, paper_id: str, *, force: bool = False
    ) -> list[dict[str, Any]]:
        if not self.llm.available():
            return self._rule_based_extract(text, paper_id)
        try:
            raw = self.llm.generate(
                _EXTRACT_PROMPT.format(text=text[:3000]),
                fmt="json",
                timeout=60,
                max_tokens=1024,
            )
            parsed = self._parse_json_list(raw)
            if parsed is not None:
                return parsed
        except LLMUnavailable:
            logger.debug("LLM çevrimdışı — kural tabanlı yedek kullanılıyor")
        except Exception as exc:
            logger.debug("LLM formül çıkarımı başarısız: %s", exc)

        # Yedek: kural tabanlı tespit
        return self._rule_based_extract(text, paper_id)

    def _rule_based_extract(self, text: str, paper_id: str) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for name, (category, description) in _KNOWN_INDICATORS.items():
            # Kelime-sınırı eşleşmesi: kısa akronimler (EMA/SMA/ATR/OBV) başka kelimelerin
            # içinde ("scheme", "plasma", "theatre", "obvious") YANLIŞ eşleşmesin → kaynak
            # uydurma (latex=None sahte formül) önlenir (CLAUDE.md Kural 7).
            if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                found.append(
                    {
                        "name": name,
                        "latex": None,
                        "plain": None,
                        "description": description,
                        "variables": {},
                        "category": category,
                    }
                )
        return found

    @staticmethod
    def _parse_json_list(raw: str) -> list[dict] | None:
        raw = raw.strip()
        # JSON array bul
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [f for f in parsed if isinstance(f, dict) and f.get("name")]
        except (json.JSONDecodeError, ValueError):
            pass
        return None
