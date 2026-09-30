"""RAG yapılandırmasının LoRA'dan BAĞIMSIZ sürümlenmesi.

``RagSnapshot`` retrieval davranışını belirleyen tüm ayarları (embedding, reranker, top_k,
füzyon bayrakları, indeks hash'i) dondurur; ``rag_version`` bunların hash'idir.
Bir LoRA/profil eval'i boyunca RAG yapılandırması DEĞİŞMEMELİDİR —
``assert_same_rag`` bunu denetler (``RagConfigDrift``).

``RetrievalTrace`` her inference koşusunda getirilen doküman/chunk kimliklerini ve
skorlarını kaydeder (hangi bağlamla cevap üretildiği tekrar-üretilebilir olsun).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.lora.mix_common import hash_obj
from app.memory.doc_purpose import parse_purposes

_LATE_DEFAULTS: dict[str, str] = {
    "exclude_purposes": "",
    "query_translate": "off",
    "translate_model": "",
}


class RagConfigDrift(RuntimeError):
    """Aynı eval koşusu içinde RAG yapılandırması değişti."""


@dataclass(frozen=True)
class RagSnapshot:
    embedding_model: str
    embedding_model_version: str
    reranker: str
    reranker_version: str
    top_k: int
    overfetch: int
    hybrid: bool
    rrf: bool
    graph: bool
    router: bool
    contextual_embed: bool
    chunk_size: int
    chunk_overlap: int
    index_hash: str = "unknown"
    # Sonradan eklenen alanlar: VARSAYILAN değerdeyken hash'e girmez → mevcut rag_version'lar
    # (ve onlara bağlı regression gate kıyasları) değişmez; değer verilince sürüm ayrışır.
    exclude_purposes: str = ""
    query_translate: str = "off"
    translate_model: str = ""

    @property
    def rag_version(self) -> str:
        d = asdict(self)
        for key, default in _LATE_DEFAULTS.items():
            if d.get(key) == default:
                d.pop(key)
        return "rag-" + hash_obj(d)[:12]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rag_version"] = self.rag_version
        return d


def _reranker_name(s: Any) -> tuple[str, str]:
    if not s.rag_rerank:
        return "none", "-"
    if s.rag_flashrank:
        return "flashrank", str(s.rag_flashrank_model)
    if s.rag_cross_encoder:
        return "cross_encoder", str(s.rag_cross_encoder_model)
    if s.rag_rrf:
        return "rrf", f"k={s.rag_rrf_k}"
    return "heuristic", "builtin"


def index_fingerprint(chroma_path: Path) -> str:
    """Chroma dizininin (dosya adı, boyut) parmak izi — içerik değişince değişir.

    Tam içerik hash'i büyük indekste pahalıdır; boyut+ad listesi pratik bir parmak izidir.
    Dizin yoksa 'missing'.
    """
    if not chroma_path.exists():
        return "missing"
    entries = sorted(
        (str(p.relative_to(chroma_path)).replace("\\", "/"), p.stat().st_size)
        for p in chroma_path.rglob("*")
        if p.is_file()
    )
    return hash_obj(entries)[:16]


def current_rag_snapshot(
    *,
    embedding_model_version: str = "unknown",
    index_hash: str | None = None,
    top_k: int | None = None,
) -> RagSnapshot:
    """Geçerli ayarlardan RAG anlık görüntüsü üret (salt-okuma)."""
    s = get_settings()
    reranker, reranker_version = _reranker_name(s)
    if index_hash is None:
        chroma = Path(s.chroma_path)
        if not chroma.is_absolute():
            chroma = s.root / chroma
        index_hash = index_fingerprint(chroma)
    return RagSnapshot(
        embedding_model=str(s.embed_model),
        embedding_model_version=embedding_model_version,
        reranker=reranker,
        reranker_version=reranker_version,
        top_k=int(top_k if top_k is not None else s.rag_top_k),
        overfetch=int(s.rag_overfetch),
        hybrid=bool(s.rag_hybrid),
        rrf=bool(s.rag_rrf),
        graph=bool(s.rag_graph),
        router=bool(s.rag_router),
        contextual_embed=bool(s.rag_contextual_embed),
        chunk_size=int(s.chunk_size),
        chunk_overlap=int(s.chunk_overlap),
        index_hash=index_hash,
        exclude_purposes=",".join(sorted(parse_purposes(s.rag_exclude_purposes))),
        query_translate=str(s.rag_query_translate).lower(),
        translate_model=(
            str(s.rag_translate_model) if str(s.rag_query_translate).lower() != "off" else ""
        ),
    )


def assert_same_rag(expected: RagSnapshot, actual: RagSnapshot) -> None:
    if expected != actual:
        diff = {
            k: (v, getattr(actual, k))
            for k, v in asdict(expected).items()
            if getattr(actual, k) != v
        }
        raise RagConfigDrift(f"RAG yapılandırması eval sırasında değişti: {diff}")


@dataclass
class RetrievalTrace:
    """Tek inference için retrieval kaydı."""

    rag_version: str
    embedding_model: str
    embedding_model_version: str
    reranker: str
    reranker_version: str
    top_k: int
    retrieved_document_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float | None] = field(default_factory=list)

    @classmethod
    def from_chunks(cls, snapshot: RagSnapshot, chunks: list[Any]) -> RetrievalTrace:
        return cls(
            rag_version=snapshot.rag_version,
            embedding_model=snapshot.embedding_model,
            embedding_model_version=snapshot.embedding_model_version,
            reranker=snapshot.reranker,
            reranker_version=snapshot.reranker_version,
            top_k=snapshot.top_k,
            retrieved_document_ids=[str(getattr(c, "paper_id", "")) for c in chunks],
            retrieved_chunk_ids=[str(getattr(c, "chunk_id", "")) for c in chunks],
            retrieval_scores=[getattr(c, "distance", None) for c in chunks],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
