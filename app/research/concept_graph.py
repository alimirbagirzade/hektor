"""Kavram Grafiği — formüller ve fikirler arası ilişkileri yönetir.

Her kenar: (from_concept) --[relation]--> (to_concept)
Relation tipleri: extends, measures, limits, combines, opposite_of, requires

Örnekler:
  RSI --[measures]--> momentum
  MACD --[combines]--> EMA_fast + EMA_slow
  volatility --[limits]--> RSI_effectiveness
  Bollinger --[extends]--> SMA
"""

from __future__ import annotations

import logging
from typing import Any

from app.brain.local_llm import LLMUnavailable, LocalLLM
from app.memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)

_VALID_RELATIONS = {"extends", "measures", "limits", "combines", "opposite_of", "requires"}

# Determinizm (CLAUDE.md kural 6): bağ-çıkarımı aynı makale için aynı grafı vermeli;
# seedsiz LLM çağrısı turdan tura farklı kenarlar üretip kavram-grafını kararsız
# kılıyordu. dataset_splitter/detached_launch ile aynı sabit seed konvansiyonu.
_LINK_EXTRACTION_SEED = 42

_LINK_PROMPT = """\
Aşağıdaki formüller ve kavramlar arasındaki ilişkileri belirle.

BİLİNEN FORMÜLLER:
{formula_list}

METİN BAĞLAMI:
{context}

Her ilişki için şu JSON dizisini döndür:
[
  {{"from": "RSI", "relation": "measures", "to": "momentum"}},
  {{"from": "Bollinger", "relation": "extends", "to": "SMA"}},
  {{"from": "high_volatility", "relation": "limits", "to": "RSI"}}
]

Geçerli ilişkiler: extends, measures, limits, combines, opposite_of, requires
Yalnız gerçek ilişkileri ekle. Yanlış kesin, az ama doğru olsun.
"""


class ConceptGraph:
    def __init__(
        self,
        store: SqliteStore | None = None,
        llm: LocalLLM | None = None,
    ) -> None:
        self.store = store or SqliteStore()
        self.llm = llm or LocalLLM()

    def build_from_papers(self) -> int:
        """Tüm makalelerden kavram bağlantılarını çıkar. Kaydedilen link sayısını döner."""
        formulas = self.store.list_formulas()
        formula_names = [f["name"] for f in formulas]
        if not formula_names:
            return 0

        total = 0
        for paper in self.store.list_papers():
            chunks = self.store.list_chunks(paper.paper_id)
            paper_formulas = [f for f in formulas if f["paper_id"] == paper.paper_id]
            if not paper_formulas or not chunks:
                continue

            # Makale için bağlam: ilk 5 chunk
            context = "\n\n".join(c.text[:500] for c in chunks[:5])
            formula_list = "\n".join(f["name"] for f in paper_formulas)

            # İdempotency: bu makalenin eski bağlantılarını sil, sonra yeniden ekle.
            # (Aksi halde her ingest'te aynı kenarlar tekrar eklenip tabloyu şişirir.)
            self.store.delete_concept_links_for_paper(paper.paper_id)
            links = self._extract_links(formula_list, context, paper.paper_id)
            for link in links:
                self.store.save_concept_link(
                    from_concept=link["from"],
                    relation=link["relation"],
                    to_concept=link["to"],
                    source_paper_id=paper.paper_id,
                )
                total += 1

        logger.info("Kavram grafiği: %d bağlantı eklendi", total)
        return total

    def add_link(
        self,
        from_concept: str,
        relation: str,
        to_concept: str,
        source_paper_id: str | None = None,
        weight: float = 1.0,
    ) -> None:
        if relation not in _VALID_RELATIONS:
            raise ValueError(f"Geçersiz ilişki: {relation}. Geçerliler: {_VALID_RELATIONS}")
        self.store.save_concept_link(
            from_concept=from_concept,
            relation=relation,
            to_concept=to_concept,
            source_paper_id=source_paper_id,
            weight=weight,
        )

    def neighbors(self, concept: str) -> list[dict[str, Any]]:
        """Bir kavramın tüm bağlantılarını döner."""
        return self.store.list_concept_links(concept)

    def as_text(self) -> str:
        """Grafiği LLM prompt'u için okunabilir metin olarak döner."""
        links = self.store.list_concept_links()
        if not links:
            return "(kavram grafiği boş)"
        lines = [
            f"  {lk['from_concept']} --[{lk['relation']}]--> {lk['to_concept']}" for lk in links
        ]
        return "\n".join(lines)

    def _extract_links(
        self, formula_list: str, context: str, paper_id: str
    ) -> list[dict[str, str]]:
        try:
            raw = self.llm.generate(
                _LINK_PROMPT.format(formula_list=formula_list, context=context[:2000]),
                fmt="json",
                timeout=60,
                max_tokens=512,
                seed=_LINK_EXTRACTION_SEED,
            )
            import json
            import re

            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return [
                        lk
                        for lk in parsed
                        if isinstance(lk, dict)
                        and lk.get("from")
                        and lk.get("relation") in _VALID_RELATIONS
                        and lk.get("to")
                    ]
        except LLMUnavailable:
            pass
        except Exception as exc:
            logger.debug("Kavram bağlantı çıkarımı başarısız: %s", exc)
        return []
