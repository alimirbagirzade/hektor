"""LoRA karışım (adapter mixing) altyapısının ortak yardımcıları.

- Domain listesi, profil YAML okuma, ağırlık ayrıştırma/doğrulama.
- Deterministik hash'ler (kanonik JSON, dosya, model/tokenizer parmak izi).
- Registry dizini (``<root>/registry/lora``).

Profil ağırlıkları SEMANTİK YÜZDE DEĞİLDİR; PEFT adapter ölçek katsayısıdır.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings

DOMAINS: tuple[str, ...] = ("math", "statistics", "reasoning", "trading", "coding")
MERGE_METHODS: tuple[str, ...] = ("svd", "ties", "dare_ties", "linear")
# PEFT: svd dışındaki yöntemler tüm adapter'larda aynı rank ister.
SAME_RANK_METHODS: frozenset[str] = frozenset({"ties", "dare_ties", "linear"})
MAX_WEIGHT = 2.0

_WEIGHT_PAIR = re.compile(r"^\s*([a-z_]+)\s*[=:]\s*([0-9]*\.?[0-9]+)\s*$")


class MixConfigError(ValueError):
    """Profil/ağırlık yapılandırması geçersiz."""


def repo_root() -> Path:
    """Depo kökü (configs/ ve evals/ buradan çözülür)."""
    return Path(__file__).resolve().parents[2]


def default_mix_config_path() -> Path:
    return repo_root() / "configs" / "lora" / "mix_profiles.yaml"


def registry_dir() -> Path:
    """Karışım registry'lerinin dizini (veri kökü altında, git'e girmez)."""
    return get_settings().root / "registry" / "lora"


# ---------------------------------------------------------------- hashing


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_obj(obj: Any) -> str:
    """Kanonik JSON'un sha256'sı (anahtar sırası bağımsız)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_files(paths: Iterable[Path]) -> str:
    """Birden çok dosyanın (ad + içerik) birleşik hash'i; sıra bağımsız."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode("utf-8"))
        h.update(hash_file(p).encode("ascii"))
    return h.hexdigest()


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)


def fingerprint_model_dir(model_dir: Path) -> tuple[str, str]:
    """Yerel HF model dizininden (base_model_hash, tokenizer_hash) üret.

    base_model_hash, 30B ağırlıklarını baştan sona okumamak için bilinçli olarak
    ``config.json`` + ağırlık indeks dosyası + ağırlık dosyalarının (ad, boyut)
    listesinden türetilir — bir "parmak izi"dir, kriptografik tam içerik hash'i değil.
    tokenizer_hash tokenizer dosyalarının tam içerik hash'idir.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model dizini yok: {model_dir}")
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = model_dir / name
        if p.exists():
            h.update(name.encode())
            h.update(hash_file(p).encode())
    weights = sorted(
        [*model_dir.glob("*.safetensors"), *model_dir.glob("*.bin")], key=lambda p: p.name
    )
    for w in weights:
        h.update(f"{w.name}:{w.stat().st_size}".encode())
    tok = [model_dir / n for n in _TOKENIZER_FILES if (model_dir / n).exists()]
    if not tok:
        raise FileNotFoundError(f"tokenizer dosyası bulunamadı: {model_dir}")
    return h.hexdigest(), hash_files(tok)


def git_commit(cwd: Path | None = None) -> str:
    """Geçerli git commit'i (bulunamazsa 'unknown'); salt-okuma."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or repo_root()),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


# ---------------------------------------------------------------- jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL oku; bozuk satırları atla (kayıt defteri tek satır yüzünden çökmesin)."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Tüm satırları yazıcıya özgü geçici dosyaya yaz, sonra atomik değiştir.

    Geçici dosya adı her yazımda benzersizdir (pid + rastgele ek): eşzamanlı iki yazıcı aynı
    ``.tmp`` dosyasını ezmez. Windows'ta hedef o an bir okuyucu tarafından açıksa
    ``replace`` kısa süre ``PermissionError`` verebilir → birkaç kez yeniden denenir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    text = "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in rows)
    try:
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(10):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
    finally:
        tmp.unlink(missing_ok=True)


LOCK_STALE_SECONDS = 60.0


@contextmanager
def file_lock(path: Path, *, timeout: float = 10.0, poll: float = 0.05) -> Iterator[None]:
    """``<path>.lock`` ile basit süreçler-arası kilit (O_CREAT|O_EXCL; Windows uyumlu).

    Oku-değiştir-yaz döngüsünü (ör. karar kaydı/tüketimi) kayıp güncellemeye karşı korur.
    Çöken bir süreçten kalan ``LOCK_STALE_SECONDS``'tan eski kilit kırılır. Süre dolarsa
    ``TimeoutError``.
    """
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_SECONDS:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"kilit alınamadı: {lock} (başka bir süreç yazıyor olabilir)"
                ) from None
            time.sleep(poll)
            continue
        except PermissionError:
            # Windows: silinmekte olan kilit dosyası kısa süre erişilemez olabilir.
            if time.monotonic() >= deadline:
                raise TimeoutError(f"kilit alınamadı: {lock}") from None
            time.sleep(poll)
            continue
        break
    try:
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- weights


def validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Ağırlık sözlüğünü doğrula ve tüm domain'leri içeren kopyasını döndür.

    Toplamın 1 olması ŞART DEĞİL (semantik yüzde değil, ölçek katsayısı).
    """
    unknown = set(weights) - set(DOMAINS)
    if unknown:
        raise MixConfigError(f"bilinmeyen domain: {sorted(unknown)} (geçerli: {list(DOMAINS)})")
    out: dict[str, float] = {}
    for d in DOMAINS:
        v = float(weights.get(d, 0.0))
        if not 0.0 <= v <= MAX_WEIGHT:
            raise MixConfigError(f"{d} ağırlığı [0, {MAX_WEIGHT}] dışında: {v}")
        out[d] = round(v, 6)
    if not any(v > 0 for v in out.values()):
        raise MixConfigError("en az bir domain ağırlığı > 0 olmalı")
    return out


def parse_weights(text: str) -> dict[str, float]:
    """'math=0.3,statistics=0.2,...' biçimini ayrıştır (eval/exec YOK, regex)."""
    pairs: dict[str, float] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        m = _WEIGHT_PAIR.match(part)
        if not m:
            raise MixConfigError(f"ağırlık biçimi geçersiz: {part!r} (ör. math=0.3)")
        pairs[m.group(1)] = float(m.group(2))
    return validate_weights(pairs)


def format_weights(weights: Mapping[str, float]) -> str:
    return ", ".join(f"{d}={weights.get(d, 0.0):.2f}" for d in DOMAINS)


def weights_to_arg(weights: Mapping[str, float]) -> str:
    """Ağırlıkları ``--mix-weights`` bayrağına KAYIPSIZ geri verilebilir biçime çevir.

    ``format_weights`` 2 haneye yuvarlar (görüntü içindir); bu ise ``parse_weights`` ile
    birebir aynı sözlüğü geri üretir (``validate_weights`` 6 haneye yuvarlar) ve boşluk
    içermez → PowerShell/komut satırı argümanı olarak güvenle taşınır.
    """

    def _num(v: float) -> str:
        return f"{float(v):.6f}".rstrip("0").rstrip(".") or "0"

    return ",".join(f"{d}={_num(weights.get(d, 0.0))}" for d in DOMAINS)


# ---------------------------------------------------------------- config


def load_mix_config(path: Path | None = None) -> dict[str, Any]:
    """mix_profiles.yaml'ı oku ve doğrula."""
    p = path or default_mix_config_path()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict) or not profiles:
        raise MixConfigError(f"{p}: 'profiles' boş")
    data["profiles"] = {name: validate_weights(w or {}) for name, w in profiles.items()}
    methods = data.get("merge_methods") or {m: {} for m in MERGE_METHODS}
    bad = set(methods) - set(MERGE_METHODS)
    if bad:
        raise MixConfigError(f"desteklenmeyen merge yöntemi: {sorted(bad)}")
    data["merge_methods"] = {k: dict(v or {}) for k, v in methods.items()}
    router = data.get("router") or {}
    for dom, prof in (router.get("domain_profiles") or {}).items():
        if prof not in data["profiles"]:
            raise MixConfigError(f"router.{dom} bilinmeyen profile işaret ediyor: {prof}")
    fb = router.get("fallback_profile")
    if fb and fb not in data["profiles"]:
        raise MixConfigError(f"router.fallback_profile bilinmiyor: {fb}")
    data["router"] = router
    return data


def profile_weights(name: str, config: Mapping[str, Any] | None = None) -> dict[str, float]:
    cfg = config or load_mix_config()
    try:
        return dict(cfg["profiles"][name])
    except KeyError as exc:
        raise MixConfigError(
            f"bilinmeyen profil: {name} (mevcut: {sorted(cfg['profiles'])})"
        ) from exc
