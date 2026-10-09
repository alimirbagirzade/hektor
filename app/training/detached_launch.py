"""Detached LoRA eğitim başlatıcı + canlı durum/hazırlık tespiti.

Neden ayrı modül: web sunucusu (veya Claude Code) kapansa da eğitim sürmeli.
`training_manager` eğitimi web süreci içinde subprocess olarak çalıştırır →
web yeniden başlayınca/çökünce eğitim de ölür (storage/auto_lora_state.json'daki
"Eğitim COMPLETED olmadı" hataları bundandı). Bu modül eğitimi **detached** süreç
olarak başlatır: çıktılar log dosyalarına yazılır, ilerleme `training_status()`
tarafından log'dan okunur. start-train.ps1 ile aynı mekanik.

Veri kaynağı tek: `data/lora_sft/lora_sft.jsonl` (sentetik + kart birleşik, ~1266).
Başlatıcı her seferinde bunu train/valid'e böler → `DatasetBuilder` kaynaklı
clobber (train.jsonl'in 0'a düşmesi) başlatmada otomatik onarılır. Bölme
KAYNAK-GRUPLUdur (aynı `source_id`/`paper_id` tek tarafta) → valid metrikleri
sızıntısız (Gate 8 sözleşmesiyle aynı).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from app.config import DEFAULT_TRAIN_PROFILE, get_settings
from app.training import resource_lock

log = logging.getLogger(__name__)

# "HAZIR" rozetinin görünme eşiği (insan yine de tek tıkla onaylar — Kural 8).
MIN_READY_EXAMPLES = 200
# Determinist split (Kural 6) — lora-split ile aynı.
_SPLIT_SEED = 42
_VALID_RATIO = 0.05
# `hektor train` PeftTrainConfig'i batch_size GEÇMEDEN kurar (app/main.py) → dataclass
# varsayılanı 1 kullanılır. Adım sayısı hesabı bununla hizalı: 1 örnek = 1 örnek-adımı
# (mikro-batch). Profilde gradient_accumulation_steps>1 ise trainer bunu optimizer adımına
# kendisi çevirir (peft_lora_train.optimizer_steps); plan/durum/nöbetçi mikro-adımda kalır.
_TRAIN_BATCH_SIZE = 1
# Log'a son yazımdan bu kadar dakika geçmediyse eğitim "canlı" sayılır.
# Yavaş CPU eğitiminde adım ~dakikalar sürer + log seyrek yazılabilir → 45 dk
# (tamamlanma zaten step>=total ile anında algılanır; bu yalnız ara boşluklar için).
# Kademe 2 F2-4 (v14): uzun dizi örneklerinde eval (sessiz) + bir optimizer adımı 45 dk'yı
# aşabiliyor → canlı koşu "durmuş" görünüp ikinci başlatmaya kapı açıyordu → 150 dk.
_LIVE_LOG_AGE_MIN = 150.0
# Geçerli adapter adı (path traversal/CLI argüman güvenliği): yalnız harf/rakam/_/-.
_ADAPTER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# Eski adın uyumu: başlatma penceresinin TTL'i artık ortak kilitte (resource_lock).
_LAUNCH_LOCK_TTL = resource_lock.LAUNCH_TTL_S


def _launch_lock_path(root: Path) -> Path:
    """Ortak ağır iş kilidinin yolu (eski ``.training_launching`` yerine)."""
    return resource_lock.lock_path(root)


# Bu süreçte alınmış başlatma kilitlerinin token'ları (kök → token).
_LAUNCH_TOKENS: dict[str, str] = {}


def _acquire_launch_lock(root: Path) -> bool:
    """Ortak ağır iş kilidini ``launching`` durumunda al (atomik, O_EXCL).

    Eğitim, dönüşüm ve karşılaştırma aynı kilidi paylaşır (``resource_lock``). Çöken
    sahibin kilidi pid ölçümüyle, yarım kalan başlatma ``LAUNCH_TTL_S`` ile bayat sayılır.
    """
    info, _why = resource_lock.acquire(
        "training", "launch", root=root, pid=os.getpid(), state="launching"
    )
    if info is None:
        return False
    _LAUNCH_TOKENS[str(root)] = str(info["token"])
    return True


def _release_launch_lock(root: Path) -> None:
    resource_lock.release(_LAUNCH_TOKENS.pop(str(root), None), root=root)


def _count_lines(path: Path) -> int:
    """Dosyadaki boş-olmayan satır sayısı (yoksa 0)."""
    if not path.exists():
        return 0
    try:
        return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())
    except Exception:
        return 0


def _combined_source(settings) -> Path:
    return settings.root / "data" / "lora_sft" / "lora_sft.jsonl"


def _source_key(line: str) -> str:
    """Bir JSONL satırının KAYNAK GRUBU anahtarı (Gate 8'in `source_id` sözleşmesi).

    Öncelik: `metadata.source_id` → `metadata.paper_id` → satır kökündeki aynı alanlar.
    (Gerçek `lora_sft.jsonl`'de bu iki alan aynı `paper_...` kimliğini taşır; sentetik QA
    ve kart satırları makaleye bağlıdır, disiplin/adversarial satırlarının kaynağı yoktur.)

    GERİ-DÜŞÜŞ: kaynak kimliği olmayan satır için anahtar satır İÇERİĞİNİN sha256'sıdır →
    (a) belirlenimci (Kural 6: aynı içerik = aynı anahtar, makineden bağımsız),
    (b) her satır kendi grubudur — makale-sızıntısı kavramı yoktur, dolayısıyla oran
    bozulmaz; birebir aynı (kopya) satırlar ise yine aynı tarafta kalır.
    """
    row: object = None
    with contextlib.suppress(Exception):
        row = json.loads(line)
    if isinstance(row, dict):
        meta = row.get("metadata")
        if isinstance(meta, dict):
            sid = meta.get("source_id") or meta.get("paper_id")
            if sid:
                return f"src:{sid}"
            # Disiplin satırlarının makale kaynağı yoktur ama aynı cevap İSKELETİNİN strateji
            # kopyaları neredeyse ikizdir; satır-hash'i her birini ayrı gruba koyup ikizleri
            # train ve valid'e dağıtıyordu → valid loss iyimser (Kademe 2, 2026-09-15).
            skeleton = meta.get("skeleton_id")
            if skeleton:
                return f"skel:{skeleton}"
        sid = row.get("source_id") or row.get("paper_id")
        if sid:
            return f"src:{sid}"
    return "line:" + hashlib.sha256(line.encode("utf-8")).hexdigest()[:16]


def _chunk_refs(line: str) -> list[str]:
    """Satırın içerdiği korpus parçalarının kimlikleri (`metadata.chunk_id` + `context_chunk_ids`).

    Damıtma satırlarının BAĞLAM'ı başka makalelerden parça çeker; yalnız `source_id` ile
    gruplamak aynı parça metnini train ve valid'e düşürüyordu (Kademe 2 2026-10-06 F3-4:
    gerçek veride 80 valid satırının 5'i). Makale düzeyinde birleştirme denendi: 1569
    satırın 1068'i tek bileşene çöküyor → valid temsil gücünü yitirir; parça düzeyi en
    büyük bileşeni 83 satırda tutuyor. Kalan zayıf ilişki (aynı makalenin FARKLI parçası
    iki tarafta) bilinçli kabul edilir.
    """
    row: object = None
    with contextlib.suppress(Exception):
        row = json.loads(line)
    if not isinstance(row, dict):
        return []
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        return []
    refs: list[str] = []
    if meta.get("chunk_id"):
        refs.append(str(meta["chunk_id"]))
    ctx = meta.get("context_chunk_ids")
    if isinstance(ctx, list):
        refs.extend(str(c) for c in ctx if c)
    return refs


def _merge_groups_by_chunk(keys: list[str], refs: list[list[str]]) -> dict[str, str]:
    """Ortak parça paylaşan kaynak gruplarını birleştir (union-find).

    Dönüş: her kaynak anahtarı → bileşen anahtarı. Bileşen anahtarı bileşendeki EN KÜÇÜK
    kaynak anahtarıdır → satır sırasından bağımsız, belirlenimci (Kural 6).
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Küçük kök kazanır → kök her zaman bileşenin en küçük düğümüdür.
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    for key, row_refs in zip(keys, refs, strict=True):
        find(key)
        for ref in row_refs:
            union(key, "chunk:" + ref)
    # "chunk:" düğümleri "src:"/"skel:"/"line:" düğümlerinden sıralamada önce gelebilir;
    # bileşen adını yalnız KAYNAK anahtarlarından seç.
    comp_name: dict[str, str] = {}
    for key in sorted(set(keys)):
        comp_name.setdefault(find(key), key)
    return {key: comp_name[find(key)] for key in set(keys)}


def split_lines_by_source(lines: list[str], seed: int = _SPLIT_SEED) -> tuple[list[str], list[str]]:
    """Satırları KAYNAK-GRUPLU böl: (train, valid). Saf fonksiyon → test edilebilir.

    Neden gruplu: satır-düzeyinde karıştırma aynı makalenin (hatta aynı chunk'ın) sentetik
    QA'larını train ve valid'e dağıtıyordu → valid metrikleri sızıntı yüzünden iyimser
    (Gate 8'in `app/lora/dataset_splitter.py` içindeki kaynak-gruplu bölmesi eğitim
    dosyalarına hiç ulaşmıyordu). Artık aynı `source_id` TEK tarafta kalır; ortak bir
    korpus parçası (`chunk_id`/`context_chunk_ids`) paylaşan kaynaklar da tek gruptur.

    Nasıl: gruplar `sorted()` ile kanonik sıraya konur, `random.Random(seed)` ile karıştırılır
    (Kural 6: aynı seed → aynı bölme), valid hedef orana ULAŞANA kadar grup grup doldurulur.
    Grup bütünlüğü bölünemediği için valid hedefi biraz AŞABİLİR (en çok bir grup kadar).
    En az bir grup daima train'de bırakılır (tek-gruplu veri train'i boşaltmasın).
    """
    import random

    if not lines:
        return [], []

    src_keys = [_source_key(ln) for ln in lines]
    comp = _merge_groups_by_chunk(src_keys, [_chunk_refs(ln) for ln in lines])
    groups: dict[str, list[str]] = {}
    for ln, key in zip(lines, src_keys, strict=True):
        groups.setdefault(comp[key], []).append(ln)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    target_valid = max(1, int(len(lines) * _VALID_RATIO)) if len(lines) > 1 else 0
    valid: list[str] = []
    valid_keys: set[str] = set()
    for key in keys:
        if len(valid) >= target_valid or len(valid_keys) >= len(keys) - 1:
            break
        valid.extend(groups[key])
        valid_keys.add(key)

    train = [ln for key in keys if key not in valid_keys for ln in groups[key]]
    return train, valid


def _pretrain_gate_blockers(settings) -> list[str]:
    """Kanonik eğitim verisini pretrain-gate'ten geçir; NO-GO gerekçelerini döndür.

    Kapı ÇALIŞTIRILAMAZSA da boş olmayan liste döner → eğitim başlamaz (Kural 2).
    """
    try:
        from app.training.dataset_quality import audit_dataset
        from app.training.discipline_dataset import discipline_jsonl_lines

        src = _combined_source(settings)
        lines = (
            [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if src.exists()
            else []
        )
        report = audit_dataset(lines, discipline_lines=discipline_jsonl_lines())
        # Kademe 2 (2026-10-06) F3-1: dosya ŞU ANKİ DB + sentetik QA'dan kanonik birleştirmeyle
        # aynı mı? Eskiden yalnız CLI pretrain-gate'te; web/Auto-LoRA/kolay akış atlıyordu.
        canonical = settings.root / "data" / "lora_sft" / "lora_sft.jsonl"
        if src.exists() and src.resolve() == canonical.resolve():
            from app.training.sft_assembly import check_assembly_freshness

            report.blockers.extend(check_assembly_freshness(lines, settings).blockers)
    except Exception as exc:
        return [f"kalite kapısı çalıştırılamadı: {exc}"]
    # Seçili sohbet veri sürümünde sonradan reddedilen/hariç tutulan/düzenlenen kayıt → dur.
    from app.cloud.policy import cloud_origin_lines
    from app.feedback.chat_dataset import chat_selection_blockers

    blockers = list(report.blockers) + chat_selection_blockers()
    # Faz 3 (B): bulut kökenli satır eğitime GİRMEZ — sağlayıcı şartları izin varsaymaz.
    cloud = cloud_origin_lines(lines)
    if cloud:
        blockers.append(
            f"Eğitim verisinde bulut kökenli {len(cloud)} satır var (ilk satır {cloud[0] + 1}); "
            "bulut çıktısıyla eğitim yasak (docs/TASARIM_FAZ3_BULUT.md §3)."
        )
    return blockers


def ensure_train_split(settings=None) -> tuple[int, int]:
    """`lora_sft.jsonl` → `jsonl_dir/{train,valid}.jsonl` (determinist, KAYNAK-GRUPLU).

    Birleşik kaynak doluysa HER ZAMAN yeniden böler (clobber onarımı). Kaynak
    boş/yoksa mevcut train.jsonl'e dokunmaz ama (0, 0) döndürür: eldeki bayat split'i
    saymak, kanonik kaynak kaybolduğunda eğitimin denetlenmemiş eski veriyle sessizce
    sürmesine yol açıyordu (Kademe-2 av bulgusu). (n_train, n_valid) döndürür.
    """
    s = settings or get_settings()
    src = _combined_source(s)
    lines = (
        [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if src.exists()
        else []
    )
    if not lines:
        # Kaynak yok → dosyalara dokunma, ama eğitilebilir veri YOK say.
        return 0, 0

    train, valid = split_lines_by_source(lines)

    jd = s.jsonl_dir
    jd.mkdir(parents=True, exist_ok=True)
    _write_if_changed(jd / "train.jsonl", "\n".join(train) + ("\n" if train else ""))
    _write_if_changed(jd / "valid.jsonl", "\n".join(valid) + ("\n" if valid else ""))
    log.info("Eğitim verisi bölündü: train=%d valid=%d → %s", len(train), len(valid), jd)
    return len(train), len(valid)


def _write_if_changed(path: Path, text: str) -> None:
    """İçerik AYNIYSA dosyaya dokunma (mtime korunur).

    Her başlatma/kurtarma yeniden böler; aynı içerik yeniden yazılınca `train.jsonl`
    mtime'ı ilerler ve nöbetçinin mtime tabanlı veri-kayması kontrolü yanlış alarm verirdi
    (Kademe-2 A5). Belirlenimci bölme → aynı kaynak = aynı bayt.
    """
    with contextlib.suppress(OSError, UnicodeDecodeError):
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return
    path.write_text(text, encoding="utf-8")


@dataclass
class TrainingSplit:
    """Kanonik eğitim verisi bölme sonucu (eski `DatasetResult` ile alan-uyumlu)."""

    train_path: Path
    valid_path: Path
    n_train: int
    n_valid: int
    content_hash: str


def _hash_jsonl_lines(lines: list[str]) -> str:
    """Train satırlarından kısa içerik hash'i (adapter provenance; 16 hex)."""
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8"))
    return h.hexdigest()[:16]


def _auto_register_dataset(settings, src: Path) -> None:
    """Kanonik veri setini (lora_sft.jsonl) ``DatasetVersion`` olarak sürümle (Modül 8).

    Eğitim verisi hazırlandığında otomatik kayıt → ``model-data-registry`` Kural-8 kapısı
    gerçek veriyle dolar. İçerik-hash ile İDEMPOTENT (aynı veri → tek sürüm). BEST-EFFORT:
    registry hatası eğitimi/önizlemeyi BOZMAZ. Onay/terfi yine İNSAN ELİYLE (CLI
    ``registry-promote-dataset``; Kural 8 — burada eğitim BAŞLATILMAZ, terfi YAPILMAZ).
    """
    if _count_lines(src) <= 0:
        return
    with contextlib.suppress(Exception):
        from app.memory.sqlite_store import SqliteStore
        from app.registry import RegistryStore

        RegistryStore(SqliteStore(settings.sqlite_file)).register_dataset_from_file(
            src, name="lora_sft", source_type="sft"
        )


def build_training_split(settings=None) -> TrainingSplit:
    """TEK KANONİK eğitim verisi: `lora_sft.jsonl` → `jsonl_dir/{train,valid}.jsonl`.

    Web uçları (`/api/training/dataset|dry-run|colab-notebook`) artık `DatasetBuilder`
    (SQLite tabanlı, cılız `{prompt,completion}`) yerine BUNU çağırır → üretilen train.jsonl
    daima kanonik `lora_sft.jsonl` ile AYNI format (`{messages}`) ve sayıda olur; iki-hat
    drifti (CLI zengin / web cılız, aynı dosyayı karşılıklı ezme) kökten kalkar. CLI
    `train`/`launch()` ile AYNI `ensure_train_split` yolu kullanılır (tek doğruluk kaynağı).

    Kanonik kaynak yok/boşsa `assemble_sft_lines` (synth + onaylı kart + ~%25 disiplin;
    determinist seed=0) ile BİR KEZ üretilir; üretilen de boşsa dosyaya DOKUNULMAZ
    (clobber guard). Kaynak doluysa olduğu gibi bölünür — CLI'nin kurduğu zengin seti
    web'den ezme riski yok. Eğitim BAŞLATMAZ (CLAUDE.md kural 8).
    """
    s = settings or get_settings()
    src = _combined_source(s)
    if _count_lines(src) == 0:
        # Kanonik kaynak yok → bir kez birleştir (scripts/assemble_sft.py ile aynı yol).
        with contextlib.suppress(Exception):
            from app.training.sft_assembly import assemble_sft_lines

            res = assemble_sft_lines(s, seed=0)
            if res.lines:  # boşsa yazma → mevcut kaynağı/eğitimi bozma
                src.parent.mkdir(parents=True, exist_ok=True)
                src.write_text("\n".join(res.lines) + "\n", encoding="utf-8")
                from app.feedback.chat_dataset import note_assembly

                note_assembly(src, res.chat)

    n_train, n_valid = ensure_train_split(s)
    train_path = s.jsonl_dir / "train.jsonl"
    valid_path = s.jsonl_dir / "valid.jsonl"
    train_lines = (
        [ln for ln in train_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if train_path.exists()
        else []
    )
    _auto_register_dataset(s, src)  # Modül 8: kanonik veri setini sürümle (best-effort, idempotent)
    return TrainingSplit(
        train_path=train_path,
        valid_path=valid_path,
        n_train=n_train,
        n_valid=n_valid,
        content_hash=_hash_jsonl_lines(train_lines) if train_lines else "",
    )


def readiness(settings=None) -> dict:
    """Eğitime hazırlık: gerçek birleşik veri (lora_sft.jsonl) örnek sayısı."""
    s = settings or get_settings()
    n = _count_lines(_combined_source(s))
    if n == 0:  # kaynak yoksa eldeki bölünmüş train'i baz al
        n = _count_lines(s.jsonl_dir / "train.jsonl")
    ready = n >= MIN_READY_EXAMPLES
    if not n:
        label = "veri yok"
    elif ready:
        label = f"{n} örnek hazır"
    else:
        label = f"{n}/{MIN_READY_EXAMPLES} örnek"
    return {"ready": ready, "ready_examples": n, "ready_label": label}


def _detached_status(settings) -> dict | None:
    """Detached/CLI eğitimi log tazeliği + tqdm satırından algıla (yoksa None)."""
    s = settings
    logf = s.root / "logs" / "train-full-err.log"
    if not logf.exists():
        return None
    if (time.time() - logf.stat().st_mtime) / 60.0 >= _LIVE_LOG_AGE_MIN:
        return None
    try:
        txt = logf.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    # tqdm örn: "30/1203 [1:28:58<48:43:23, 149.53s/it]"
    m = re.findall(r"(\d+)/(\d+)\s*\[[^<\]]*<([^,\]]*)", txt)
    if not m:
        return None
    step, total, eta = int(m[-1][0]), int(m[-1][1]), m[-1][2].strip()
    if total <= 0 or step >= total:
        return None
    adapter = "LoRA"
    st = s.root / "storage" / "train_status.json"
    if st.exists():
        with contextlib.suppress(Exception):
            adapter = json.loads(st.read_text(encoding="utf-8")).get("adapter", "LoRA")
    return {
        "running": True,
        "source": "detached",
        "adapter": adapter,
        "step": step,
        "total": total,
        "pct": round(step * 100 / total, 1),
        "eta": eta,
    }


def training_status() -> dict:
    """Birleşik eğitim durumu (üst-bar rozeti + sekme için).

    Sıra: web (training_manager) → detached log → çalışmıyorsa hazırlık bilgisi.
    """
    s = get_settings()
    # 1) Web'den başlatılan (legacy in-process yol)
    try:
        from app.web.training_manager import get_training_manager

        prog = get_training_manager().progress
        state = getattr(getattr(prog, "state", None), "value", "")
        if state == "running":
            return {
                "running": True,
                "source": "web",
                "adapter": getattr(prog, "adapter_name", "LoRA"),
                "step": getattr(prog, "current_iter", 0),
                "total": getattr(prog, "total_iters", 0),
                "pct": round(getattr(prog, "pct", 0.0), 1),
                "eta": "",
            }
    except Exception:
        pass
    # 2) Detached/CLI eğitim
    det = _detached_status(s)
    if det:
        return det
    # 3) Çalışmıyor → hazırlık
    return {"running": False, **readiness(s)}


def is_running() -> bool:
    """Herhangi bir eğitim (web veya detached) canlı mı?"""
    return bool(training_status().get("running"))


def _find_hektor(root: Path) -> list[str] | None:
    """`hektor` konsol betiğini bul → komut öneki (yoksa uv run yedeği)."""
    exe = "hektor.exe" if os.name == "nt" else "hektor"
    # 1) Aktif venv (web sunucusunun python'ı) yanındaki Scripts/bin
    cand = Path(sys.executable).parent / exe
    if cand.exists():
        return [str(cand)]
    # 2) Proje .venv
    sub = "Scripts" if os.name == "nt" else "bin"
    cand2 = root / ".venv" / sub / exe
    if cand2.exists():
        return [str(cand2)]
    # 3) PATH
    found = shutil.which("hektor")
    if found:
        return [found]
    # 4) uv run yedeği
    uv = shutil.which("uv") or str(
        Path(os.environ.get("USERPROFILE", "")) / ".local" / "bin" / "uv.exe"
    )
    if uv and Path(uv).exists():
        return [uv, "run", "--project", str(root), "hektor"]
    return None


def _build_train_cmd(
    base: list[str],
    adapter_name: str,
    iters: int,
    base_model: str | None,
    profile: str | None,
    max_examples: int,
) -> list[str]:
    """Detached eğitim için `hektor train --run …` komut listesini kur (saf, test edilebilir).

    profile TRUTHY ise `--profile` EKLENİR; boş/None ise EKLENMEZ. Bu ayrım kritik:
    profile geçmezse spawn edilen `train --run` PeftTrainConfig varsayılanlarıyla (vanilya:
    assistant_only_loss=False, NEFTune=0, lr=2e-4) koşar → prompt maskeleme FİİLEN kapalı
    kalır (v5 disiplin-regresyon reçetesi). Bu yüzden `launch()` yerel-güvenli bir profili
    varsayılan yapar; bu helper yalnız listeyi kurar.
    """
    cmd = [
        *base,
        "train",
        "--run",
        "--backend",
        "peft",
        "--adapter-name",
        adapter_name,
        "--iterations",
        str(iters),
    ]
    if base_model:
        cmd += ["--base-model", base_model]
    if profile:
        cmd += ["--profile", profile]
    if max_examples > 0:
        cmd += ["--max-examples", str(max_examples)]
    return cmd


def _status_payload(
    adapter_name: str,
    dtype: str,
    iters: int,
    base_model: str | None,
    profile: str | None,
    max_examples: int,
    pid: int,
    approval_id: str = "",
    *,
    mix_weights: str = "",
    mix_profile: str = "",
    mix_decision_id: str = "",
    data_sha256: str = "",
) -> dict:
    """``storage/train_status.json`` içeriği — reçetenin TAMAMI (saf, test edilebilir).

    Neden tamamı: ``scripts/training-watchdog.ps1`` çöken eğitimi YALNIZ bu dosyadan
    okuduğu reçeteyle diriltir. Dosya (adapter, dtype, iterations) ile sınırlı kaldığı
    sürece yeniden başlatma ``max_examples`` ve ``profile``ı UNUTUR: adım sayısı korunur
    ama örnek tavanı profil varsayılanına düşer → aynı küçük alt-küme üzerinde sessizce
    çok-epoch koşulur, profilin ``epochs: 1`` vaadi ihlal edilir (bkz. plan_iterations;
    v5 disiplin-regresyonunun sınıfı). 2026-09-08'de v8 koşusu tam bu boşluk yüzünden
    600 örnek yerine 300 örnek × 2 epoch eğitti.

    ``approval_id``: bu koşuyu yetkilendiren TAZE onayın kimliği (``require_fresh_approval``
    tarafından TÜKETİLMİŞ). Kurtarma/nöbetçi yolu bu alanı bir ZAMAN PENCERESİ ("son N
    saatte başladı" gibi) yerine gerçek onay kaydını doğrulamak için kullanır — dosyanın
    varlığı ya da tazeliği tek başına kalıcı yetki SAYILMAZ (bkz. HANDOFF §4: nöbetçi
    2026-09-15'te onaysız bir koşuyu tam bu yüzden diriltti).

    ``mix_weights``/``mix_profile``/``mix_decision_id``: bu koşunun tükettiği karışım
    ağırlık kararı (``weights_to_arg`` biçimi). Nöbetçi kurtarmada bunları `--mix-weights`
    ile geri verir; yoksa etkileşimsiz alt süreç bir SONRAKİ eğitimin bekleyen kararını
    tüketir ya da 5 ile çıkardı (Kademe-2 A2).

    ``data_sha256``: başlangıçtaki kanonik kaynak (`lora_sft.jsonl`) hash'i — veri kayması
    mtime yerine içerikle ölçülür (Kademe-2 A5; bkz. ``train_guard.recovery_allowed``).
    """
    return {
        "adapter": adapter_name,
        "dtype": dtype,
        "iterations": iters,
        "base_model": base_model or "",
        "profile": profile or "",
        "max_examples": max(0, int(max_examples)),
        "pid": pid,
        # Kademe 2 J-2: pid yeniden kullanımı → "durdur" ilgisiz süreci öldürmesin.
        "pid_create_time": resource_lock.process_create_time(pid) if pid else None,
        "approval_id": approval_id or "",
        "started_at": _utcnow_iso(),
        "mix_weights": mix_weights or "",
        "mix_profile": mix_profile or "",
        "mix_decision_id": mix_decision_id or "",
        "data_sha256": data_sha256 or "",
    }


def _profile_limits(profile: str | None) -> tuple[int, int]:
    """Profilden (max_examples, epochs) oku. Profil yok/okunamazsa (0, 1) — kırpmasız, 1 epoch."""
    if not profile:
        return 0, 1
    try:
        from app.training.peft_lora_train import load_lora_profile

        prof = load_lora_profile(profile)
    except Exception:
        log.warning(
            "LoRA profili okunamadı (%s) — adım sayısı kırpma/epoch bilgisi olmadan hesaplanıyor.",
            profile,
        )
        return 0, 1

    def _as_int(value: object, fallback: int) -> int:
        """YAML alanını güvenle int'e çevir (yok/None/çöp değer → fallback)."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return fallback
        return fallback

    return max(0, _as_int(prof.get("max_examples"), 0)), max(1, _as_int(prof.get("epochs"), 1))


def plan_iterations(n_train: int, max_examples: int, profile: str | None) -> tuple[int, int, int]:
    """Adım sayısını FİİLEN eğitilecek örnek sayısından hesapla. (iters, n_effective, epochs).

    Neden: `hektor train` PeftTrainConfig'i batch_size'sız kurar → batch_size=1, yani
    1 epoch = eğitilen örnek sayısı kadar adım. Ama trainer `max_steps=cfg.iterations`'ı
    KIRPILMIŞ alt-kümeden bağımsız uygular. Adım sayısı tam `train.jsonl` satır sayısından
    hesaplanırsa (eski davranış), profilin `max_examples` kırpması yüzünden aynı küçük
    alt-küme üzerinde birden çok epoch koşulur ve profilin "epochs: 1" vaadi SESSİZCE
    ihlal edilir (ör. 1099 adım / 300 örnek ≈ 3.7 epoch → ezber/aşırı-uyum; v5 disiplin
    regresyonunun sınıfı).

    Hesap:
      1. Etkin kırpma tavanı = CLI `max_examples` (>0 ise; profili EZER, app/main.py ile
         aynı öncelik), yoksa profilin `max_examples`'ı, o da yoksa kırpma yok.
      2. n_effective = min(n_train, tavan)  → trainer'ın `sample_rows` ile alacağı sayı.
      3. steps_per_epoch = n_effective // batch_size (batch_size=1) — en az 1.
      4. iters = steps_per_epoch * profildeki `epochs` → profilin epoch vaadi KARŞILANIR.
    """
    prof_max, epochs = _profile_limits(profile)
    cap = max_examples if max_examples > 0 else prof_max
    n_effective = min(n_train, cap) if cap > 0 else n_train
    steps_per_epoch = max(1, n_effective // _TRAIN_BATCH_SIZE)
    return steps_per_epoch * epochs, n_effective, epochs


# Spawn sonrası alt sürecin HEMEN çıkıp çıkmadığını izleme süresi (sn). `train --run`
# ağırlık/sızıntı/yük kapılarında birkaç saniyede 3/4/5/6 ile çıkar; bunu görmeden "başladı"
# demek ölü bir durum kaydı bırakır. Testler 0 geçer (tek yoklama, beklemesiz).
# Kademe 2 (2026-10-06) L-3: sabit 8 sn, alt sürecin kapılara ulaşma süresinden (içe aktarma +
# sızıntı taraması ~10 sn) kısaydı → onaydan sonra kapıda düşen koşu "başlatıldı" görünüyordu.
# Artık alt süreç TÜM kapıları geçince işaret dosyası yazar; ebeveyn işareti ya da çıkışı bekler.
# Üst sınır yalnız takılan alt süreç için (dolarsa eski davranış: canlıysa başlatıldı sayılır).
_EARLY_EXIT_WAIT_S = 180.0
GATES_MARKER_ENV = "HEKTOR_TRAIN_GATES_MARKER"
# Kademe 2 L-2: üst sürecin tükettiği onayın kimliği → alt süreç `train --run` doğrular.
APPROVAL_ENV = "HEKTOR_TRAIN_APPROVAL_ID"
_EARLY_EXIT_HINTS = {
    3: "taze insan onayı gerekiyor (Kural 8)",
    4: "train-load-doctor NO-GO (rakip GPU/LLM yükü) — ayrıntı logs/train-full.log",
    5: "karışım ağırlığı kararı yok — `uv run hektor mix weights`",
    6: "eğitim verisinde eval sızıntısı — `uv run hektor mix leakage`",
    8: "ortak ağır iş kilidi tutuluyor ya da sohbet cevabı üretiliyor",
    10: "Kademe 2 kaydı yok/geçersiz — derin av sonrası `uv run hektor kademe2-kayit`",
    11: "koşu onaylanan reçeteyle aynı değil — yeni anlık görüntü gerekir",
    1: "kalite kapısı / veri kapısı NO-GO — ayrıntı logs/train-full.log",
}


def _fail(message: str) -> dict:
    return {"ok": False, "message": message, "adapter": ""}


def _adapter_dir_blocker(adapter_dir: Path) -> str | None:
    """Sıfırdan koşu için adapter klasörü temiz mi? (Kademe-2 B4)

    Aynı adla kalan ``checkpoint-*`` klasörleri kurtarmayı bozar: HF döndürmesi adım
    numarasına göre sildiğinden eski koşunun büyük numaralı checkpoint'i kalır → çöken yeni
    koşu "tamamlanmış" sayılır ya da ESKİ ağırlıklardan devam edilir.
    """
    if not adapter_dir.is_dir():
        return None
    stale = sorted(p.name for p in adapter_dir.glob("checkpoint-*") if p.is_dir())
    if stale or (adapter_dir / "run_complete.json").is_file():
        return (
            f"Adapter klasörü dolu ({adapter_dir.name}: {', '.join(stale[:3]) or 'bitmiş koşu'})"
            " — sıfırdan eğitim için YENİ bir adapter adı seç (eski checkpoint'ler kurtarmayı "
            "bozar)."
        )
    return None


def _recipe_blockers(base_model: str | None, profile: str | None) -> list[str]:
    """Reçete ↔ makine uyumu: profil hedefleri base'te var mı + CPU RAM yeter mi.

    Onay TÜKETİLMEDEN önce (Kademe-2 B3/B6): eskiden ikisi de alt süreçte, 8 sn erken-çıkış
    penceresinden SONRA düşüyordu → onay yanıyor, ölü durum kaydı kalıyordu.
    """
    from app.training.peft_lora_train import (
        TARGET_MODULES,
        check_cpu_ram,
        load_lora_profile,
        precheck_target_modules,
    )

    base = base_model or get_settings().peft_base_model
    targets: tuple[str, ...] = TARGET_MODULES
    if profile:
        try:
            targets = tuple(load_lora_profile(profile).get("target_modules") or TARGET_MODULES)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            return [f"Profil hatası: {exc}"]
    out: list[str] = []
    target_err = precheck_target_modules(base, targets)
    if target_err:
        out.append(target_err)
    try:
        import torch

        on_cuda = bool(torch.cuda.is_available())
    except Exception:
        on_cuda = False
    if not on_cuda:
        # launch() eğitimi bf16 başlatır (dtype varsayılanı) → 2 bayt.
        ram_ok, ram_msg = check_cpu_ram(base, 2)
        if not ram_ok:
            out.append(ram_msg)
    return out


def preflight_launch(
    adapter_name: str = "hektor_lora",
    *,
    run_load_doctor: bool = True,
    base_model: str | None = None,
    profile: str | None = None,
    check_recipe: bool = False,
    recipe_sha: str = "",
) -> dict:
    """Başlatma öncesi UCUZ, deterministik kontroller — onay TÜKETİLMEDEN önce çağrılır.

    Kademe-2 A6: web/Auto-LoRA yolu tek kullanımlık onayı `launch()`'tan ÖNCE tüketiyordu;
    ardından "zaten çalışıyor", "ağırlık kararı yok", boş bölme ya da kalite kapısı NO-GO
    çıkınca onay boşa yanıyordu. Bu fonksiyon HİÇBİR ŞEY tüketmez/başlatmaz; yalnız
    ``train.jsonl``'i (içerik aynıysa dokunmadan) yeniden böler.

    Sıra: adapter adı → koşan eğitim → bekleyen ağırlık kararı → bölme (boş değil) →
    pretrain-gate → eval sızıntısı → train-load-doctor. ``{ok, message, n_train}`` döner.
    """
    if not _ADAPTER_RE.match(adapter_name or ""):
        return _fail("Geçersiz adapter adı — yalnız harf, rakam, _ ve - (en çok 64).")

    if is_running():
        return _fail("Zaten eğitim çalışıyor.")

    # Sohbet cevabı üretiliyorsa (bellek yarışı) onay tüketilmeden önce dur.
    from app.feedback.resource_guard import chat_lease_blocker

    lease_blocker = chat_lease_blocker(get_settings().root)
    if lease_blocker:
        return _fail(lease_blocker)

    dir_blocker = _adapter_dir_blocker(get_settings().adapters_dir / adapter_name)
    if dir_blocker:
        return _fail(dir_blocker)

    # Her eğitimden önce karışım ağırlığı sorulur (kullanıcı kuralı). Alt süreç etkileşimsiz
    # olduğundan kararın ÖNCEDEN kaydedilmiş olması gerekir; yoksa alt süreç exit 5 ile
    # log'a gömülü düşerdi — burada açık mesajla erken dön.
    from app.lora.weight_decision import WeightDecisionStore

    if WeightDecisionStore().pending() is None:
        return _fail(
            "Eğitim öncesi karışım ağırlığı kararı yok. Önce: `uv run hektor mix weights` "
            "(profil seç ya da math/statistics/reasoning/trading/coding ağırlıklarını gir)."
        )

    s = get_settings()
    n_train, _n_valid = ensure_train_split(s)
    if n_train <= 0:
        return _fail(
            "Eğitim verisi yok (lora_sft.jsonl boş). "
            "Önce sentetik veri üret (synth-qa / lora-cloud-prep)."
        )

    # Kalite kapısı TÜM başlatma yollarında — eskiden yalnız scripts/start-train.ps1
    # içindeydi; web butonu ve auto_pipeline denetlenmemiş veriyi eğitebiliyordu.
    gate_blockers = _pretrain_gate_blockers(s)
    if gate_blockers:
        return _fail("Kalite kapısı NO-GO — eğitim başlatılmadı: " + " | ".join(gate_blockers[:3]))

    # Kademe 2 derin av kaydı HER eğitimden önce zorunlu (CLAUDE.md) — onay tüketilmeden önce.
    from app.training import easy_train

    # Kademe 2 (2026-10-06) F1-1: kolay akış reçete kapsamlı kayıtla gelir; argümansız
    # çağrı yalnız veri kapsamlı kayıt arıyordu → onay yandıktan sonra "kayıt yok".
    k2 = easy_train.kademe2_check(recipe_sha=recipe_sha)
    if k2:
        return _fail(k2)

    # Sızıntı kapısı (alt süreç de uygular, 6 ile çıkar) — onay yanmadan önce burada.
    try:
        from app.lora.mix_cli import run_leakage_check

        leak = run_leakage_check(s.jsonl_dir / "train.jsonl")
    except Exception as exc:
        return _fail(f"Sızıntı kapısı çalıştırılamadı — eğitim başlatılmadı (Kural 2): {exc}")
    if not leak.get("clean"):
        return _fail(
            f"Eğitim verisinde eval sızıntısı: {leak.get('counts')} — "
            "ayrıntı: `uv run hektor mix leakage`."
        )

    if check_recipe:
        blockers = _recipe_blockers(base_model, profile)
        if blockers:
            return _fail("Reçete ön-kontrolü başarısız — " + " | ".join(blockers))

    if run_load_doctor:
        from app.training.train_load_doctor import run_train_doctor

        doctor = run_train_doctor()
        if doctor.verdict == "NO-GO":
            return _fail(
                "train-load-doctor NO-GO — "
                + (" ".join(doctor.reasons) or "rakip GPU/LLM yükü, boş VRAM eşiğin altında.")
            )

    return {"ok": True, "message": "Ön-kontroller geçti.", "n_train": n_train}


def _early_exit_code(
    proc: subprocess.Popen, wait_s: float, marker: Path | None = None
) -> int | None:
    """Alt süreç kapılarda çıktıysa çıkış kodunu, kapıları geçtiyse (işaret dosyası) ya da
    ``wait_s`` doldu ve hâlâ yaşıyorsa ``None`` döndür."""
    deadline = time.monotonic() + max(0.0, wait_s)
    while True:
        rc = proc.poll()
        if rc is not None:
            return int(rc)
        if marker is not None and marker.exists():
            return None
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def launch(
    adapter_name: str = "hektor_lora",
    iterations: int = 0,
    dtype: str = "bf16",
    base_model: str | None = None,
    profile: str | None = DEFAULT_TRAIN_PROFILE,
    max_examples: int = 0,
    approval_id: str = "",
    early_exit_wait_s: float | None = None,
    skip_register: bool = False,
    recipe_sha: str = "",
) -> dict:
    """Eğitimi DETACHED başlat (web/terminal kapansa da sürer).

    - recipe_sha (kolay akış): yeniden bölünen veri + ayarlar onaylanan reçeteyle AYNI olmalı
      (alt süreç doğmadan denetlenir); alt sürece reçete ve beklenen train/valid özetleri
      geçirilir → ``train --run`` aynı bağlamayı onay/kilit öncesi yeniden denetler, eğitici
      okuduğu baytların özetini doğrular.

    - Veriyi `lora_sft.jsonl`'den yeniden böler (clobber-proof, KAYNAK-GRUPLU).
    - iterations<=0 → adım sayısı `plan_iterations` ile: FİİLEN eğitilecek örnek sayısı
      (profil/CLI `max_examples` kırpması UYGULANDIKTAN sonra) × profildeki `epochs`.
      Tam `train.jsonl` satır sayısını kullanmak, kırpılmış alt-küme üzerinde sessizce
      birkaç epoch koşulmasına (aşırı-uyum) yol açıyordu.
    - profile: LoRA reçete profili. VARSAYILAN `DEFAULT_TRAIN_PROFILE` (maskeli + NEFTune)
      — web butonu / auto_pipeline / CLI-detached çağıranlarının HİÇBİRİ profil geçmediği
      için varsayılan vanilya olsaydı bu yollarla başlatılan eğitim SESSİZCE maskesiz koşar
      ve tam v5 disiplin-regresyonunu üretirdi (Kademe-2 av bulgusu). Güvenli-varsayılan
      olarak yerel maskeli profil seçilir; farklı reçete isteyen çağıran açıkça geçebilir,
      profili tümüyle atlamak isteyen `profile=""`/`None` verebilir.
    - max_examples>0: yalnız N örnekle eğit (CPU'da makul süre). profile içinde de olabilir.
    - approval_id: çağıranın (web endpoint / auto_pipeline) bu koşu için TÜKETTİĞİ taze
      onayın kimliği. `train_status.json`'a yazılır ki kurtarma/nöbetçi yolu bir ZAMAN
      PENCERESİ yerine gerçek onay kaydını doğrulayabilsin (bkz. `_status_payload`).
    - early_exit_wait_s: spawn sonrası alt sürecin erken çıkışını izleme süresi (None →
      ``_EARLY_EXIT_WAIT_S``). Alt süreç bu sürede çıkarsa durum dosyası SİLİNİR ve
      ``ok=False`` + çıkış kodu döner (ölü koşu kaydı bırakılmaz; Kademe-2 A6).
    - {ok, message, adapter} döndürür. Eğitimi GERÇEKTEN başlatır (Kural 8: bu
      çağrı yalnızca açık kullanıcı eylemiyle — buton/onay — tetiklenir).
    """
    s = get_settings()
    root = s.root

    # Ucuz ön-kontroller (adapter adı, koşan eğitim, ağırlık kararı, bölme, kapı, sızıntı,
    # yük doktoru). Çağıran (web/Auto-LoRA) bunları onayı tüketmeden ÖNCE de çağırır;
    # burada durum değişmiş olabileceği için TEKRAR denetlenir.
    # F1-2: kolay akışta (reçete bağlı) RAM/hedef modül ön-kontrolü de burada — alt süreçte
    # 8 sn erken-çıkış penceresinden sonra düşüp "başlatıldı" görünmesin.
    pre = preflight_launch(
        adapter_name,
        base_model=base_model,
        profile=profile,
        check_recipe=bool(recipe_sha),
        recipe_sha=recipe_sha,
    )
    if not pre.get("ok"):
        return _fail(str(pre.get("message", "Ön-kontrol başarısız.")))
    n_train = int(pre.get("n_train", 0))
    if recipe_sha:
        from app.training import easy_train

        binding = easy_train.recipe_binding_problems(
            recipe_sha,
            adapter_name=adapter_name,
            base_model=base_model,
            profile=profile,
            max_examples=max_examples,
        )
        if binding:
            return _fail(
                "Onaylanan reçeteyle başlatılacak koşu aynı değil — eğitim BAŞLATILMADI: "
                + " | ".join(binding[:3])
            )

    # Kademe 2 P-3: alt süreç üst sürecin TÜKETTİĞİ onayı doğrular (L-2). Onaysız başlatma
    # (ör. gözetimsiz politika) alt süreçte sessizce çıkış 3 verirdi → burada açıkça reddet.
    if not (approval_id or "").strip():
        return _fail(
            "Bu başlatmanın tüketilmiş insan onayı yok — gerçek eğitim her seferinde taze onay "
            "ister (Kural 8). Gözetimsiz politika (HEKTOR_UNATTENDED_TRAINING_ENABLED) eğitim "
            "alt sürecini yetkilendiremez."
        )
    # Atomik ortak kilit: iki eş-zamanlı istek (çift-tık/retry) ya da terminalden başlatılmış
    # eğitim/dönüşüm/karşılaştırma varken çift süreç başlamasın.
    if not _acquire_launch_lock(root):
        return _fail(
            (resource_lock.blocker(root) or "Eğitim şu an başlatılıyor") + " — lütfen bekle."
        )
    lock_token = _LAUNCH_TOKENS.get(str(root), "")
    # Kademe 2 P-1: tüketilen onay BU kilide bağlanır — alt süreç kilit kaydındaki kimliği
    # ister; elle alınmış kilit (`resource_lock acquire`) onay taşımaz → yetki vermez.
    resource_lock.annotate(lock_token, root=root, approval_id=approval_id or "")

    spawned = False
    try:
        # Yarış kapanışı: kilit ALINDIKTAN SONRA sohbet kiralarına tekrar bak. Sohbet tarafı
        # önce kirasını yazıp sonra bu kilide baktığından ikisi birden ilerleyemez.
        from app.feedback.resource_guard import chat_lease_blocker

        lease_blocker = chat_lease_blocker(root)
        if lease_blocker:
            return _fail(lease_blocker)
        # Adım sayısı FİİLEN eğitilecek örnek sayısından (profil kırpması UYGULANDIKTAN
        # sonra) hesaplanır; profildeki `epochs` gerçekten karşılanır (bkz. plan_iterations).
        planned, n_effective, epochs = plan_iterations(n_train, max_examples, profile)
        # Kademe-2 B5: plandan fazla adım = aynı alt-küme üzerinde sessizce çok-epoch (ezber).
        # start-train.ps1 bunu uyarıyordu, web/launch sessizce kabul ediyordu.
        if iterations > planned > 0:
            return _fail(
                f"İstenen {iterations} adım planı ({planned} = {n_effective} örnek × {epochs} "
                "epoch) aşıyor → aynı örnekler üzerinde fazladan epoch (ezber riski). "
                "İterasyonu 0 bırak (plandan) ya da daha çok VERİ için örnek tavanını büyüt."
            )
        iters = iterations if iterations > 0 else planned
        base = _find_hektor(root)
        if not base:
            return {
                "ok": False,
                "message": "hektor çalıştırıcısı bulunamadı (venv/uv yok).",
                "adapter": "",
            }

        cmd = _build_train_cmd(base, adapter_name, iters, base_model, profile, max_examples)

        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["HEKTOR_TRAIN_DTYPE"] = dtype
        # Bu yol (auto_pipeline/web buton) onayı ÜST katmanda alır; spawn edilen
        # `hektor train --run` iç onay kapısını atlasın (çift onay olmasın). STOP_ALL
        # iç komutta yine de geçerlidir. Manuel `hektor train --run` bu env'i ALMAZ.
        env["HEKTOR_TRAIN_SUPERVISED"] = "1"
        # Bu yol hiçbir zaman nöbetçi kurtarması değildir (o start-train.ps1 -Supervised).
        env.pop("HEKTOR_TRAIN_RECOVERY", None)
        # Ortak kilit alt sürece devredilir: `train --run` bu token'la kilidi DEVRALIR,
        # kendi başına ikinci kilit almaya çalışmaz.
        env[resource_lock.TOKEN_ENV] = lock_token
        # Kilit belirteci onay kanıtı DEĞİL (L-2): alt süreç bu onay kaydını yeniden doğrular.
        env[APPROVAL_ENV] = approval_id or ""
        gates_marker = root / "storage" / f"train_gates_{lock_token[:12]}.ok"
        with contextlib.suppress(OSError):
            gates_marker.unlink()
        env[GATES_MARKER_ENV] = str(gates_marker)
        # Taze başlatma ASLA checkpoint'ten devam etmez (Kademe-2 B12): sunucu, önceden
        # `start-train.ps1 -Resume` koşmuş bir kabuktan açıldıysa RESUME=1 miras kalırdı.
        env["HEKTOR_TRAIN_RESUME"] = "0"
        # Kayıt defteri (Kademe-2 C3): alt süreç biten adapter'ı CANDIDATE kaydeder; yalnız
        # kendi kaydını açan auto_pipeline çift kaydı önlemek için bunu kapatır.
        if skip_register:
            env["HEKTOR_TRAIN_SKIP_REGISTER"] = "1"
        else:
            env.pop("HEKTOR_TRAIN_SKIP_REGISTER", None)
        # Reçete bağlaması alt sürece taşınır (yoksa miras kalan değer temizlenir).
        from app.training.easy_train import RECIPE_ENV

        if recipe_sha:
            env[RECIPE_ENV] = recipe_sha
        else:
            env.pop(RECIPE_ENV, None)

        # Alt süreç (etkileşimsiz) BU bekleyen kararı tüketecek; ağırlıklar durum dosyasına
        # yazılır ki kurtarma aynı ağırlıkları bayrakla geri verebilsin (Kademe-2 A2).
        from app.lora.mix_common import weights_to_arg
        from app.lora.weight_decision import WeightDecisionStore
        from app.training.train_guard import sha256_file

        pending = WeightDecisionStore().pending()
        data_sha256 = sha256_file(_combined_source(s)) or ""

        popen_kwargs: dict = {"cwd": str(root), "env": env, "close_fds": True}
        if os.name == "nt":
            # Gizli konsol + yeni süreç grubu (DETACHED_PROCESS DEĞİL): eğitimin açtığı
            # git/nvidia-smi gibi torunlar pencere açamaz — bkz. app/procutil.py.
            from app.procutil import DETACHED_HIDDEN

            popen_kwargs["creationflags"] = DETACHED_HIDDEN
        else:
            popen_kwargs["start_new_session"] = True

        # Log dosyaları child sürece devredilir (with-bloğu kullanılamaz: Popen
        # detached çalışacağı için handle'lar spawn'dan sonra kapatılır).
        out_f = open(logs / "train-full.log", "ab")  # noqa: SIM115
        err_f = open(logs / "train-full-err.log", "ab")  # noqa: SIM115
        try:
            proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, **popen_kwargs)
        except (OSError, ValueError) as exc:
            log.exception("Detached eğitim başlatılamadı")
            return {
                "ok": False,
                "message": f"Eğitim başlatılamadı (süreç oluşturulamadı): {exc}",
                "adapter": "",
            }
        finally:
            out_f.close()
            err_f.close()

        # Kilit artık alt sürecin ömrüne bağlı (sunucu kapansa da eğitim sürdükçe tutulur;
        # alt süreç çökerse pid ölçümüyle bayat sayılır).
        resource_lock.transfer(lock_token, proc.pid, root=root)
        (root / "storage").mkdir(parents=True, exist_ok=True)
        # pid kaydı (Phase 2): /api/training/stop detached koşuyu pid ile durdurabilsin.
        # Reçetenin tamamı yazılır — nöbetçi yeniden başlatırken profil/örnek tavanını
        # unutmasın (bkz. _status_payload).
        status_file = root / "storage" / "train_status.json"
        status_file.write_text(
            json.dumps(
                _status_payload(
                    adapter_name,
                    dtype,
                    iters,
                    base_model,
                    profile,
                    max_examples,
                    proc.pid,
                    approval_id,
                    mix_weights=weights_to_arg(pending.weights) if pending else "",
                    mix_profile=pending.profile_name if pending else "",
                    mix_decision_id=pending.decision_id if pending else "",
                    data_sha256=data_sha256,
                )
            ),
            encoding="utf-8",
        )

        # Erken çıkış: alt süreç kapılarda (3/4/5/6) hemen düşerse "başlatıldı" deme ve
        # ölü durum kaydı bırakma (nöbetçi onu diriltmeye çalışırdı).
        wait_s = _EARLY_EXIT_WAIT_S if early_exit_wait_s is None else early_exit_wait_s
        rc = _early_exit_code(proc, wait_s, gates_marker)
        with contextlib.suppress(OSError):
            gates_marker.unlink()
        if rc is not None:
            with contextlib.suppress(OSError):
                status_file.unlink()
            resource_lock.release(lock_token, root=root)
            _LAUNCH_TOKENS.pop(str(root), None)
            hint = _EARLY_EXIT_HINTS.get(rc, "ayrıntı: logs/train-full-err.log")
            log.warning("Detached eğitim hemen çıktı (çıkış kodu %d): %s", rc, hint)
            return _fail(f"Eğitim BAŞLAMADI — alt süreç hemen çıktı (çıkış kodu {rc}): {hint}.")
        spawned = True
        log.info(
            "Detached eğitim başlatıldı: %s (%d adım = %d örnek × %d epoch, dtype=%s)",
            adapter_name,
            iters,
            n_effective,
            epochs,
            dtype,
        )
        # Süreç spawn edildi; canlı kalıp kalmadığı üst-bar rozetinden/log'dan izlenir
        # (Kural 2: "başladı" değil "başlatıldı" + nereden doğrulanacağı belirtilir).
        return {
            "ok": True,
            "message": (
                f"Eğitim başlatıldı (detached): {adapter_name} — "
                f"{n_effective}/{n_train} örnek (profil kırpması sonrası), "
                f"{iters} adım ≈ {epochs} epoch, {dtype}. "
                "İlerlemeyi üst-bar rozetinden izle."
            ),
            "adapter": adapter_name,
        }
    finally:
        # Spawn başarısızsa kilidi hemen bırak; başarılıysa kilit alt sürecindir (devredildi;
        # alt süreç bitince bırakır, çökerse pid ölçümüyle bayat sayılır).
        if not spawned:
            _release_launch_lock(root)
        else:
            _LAUNCH_TOKENS.pop(str(root), None)


# --------------------------------------------------------------------------
# Detached eğitim DURDURMA (Phase 2) — /api/training/stop gerçek durdurma
# --------------------------------------------------------------------------
def _utcnow_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def read_detached_training_status(root: Path | None = None) -> dict:
    """``storage/train_status.json`` içeriğini döndür (yoksa/bozuksa {})."""
    r = root or get_settings().root
    st = r / "storage" / "train_status.json"
    if not st.exists():
        return {}
    try:
        data = json.loads(st.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(pid))
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
        except Exception:
            return False


def is_detached_training_running(root: Path | None = None) -> bool:
    """Detached eğitim süreci canlı mı? Önce pid (varsa), sonra log tazeliği.

    ``root`` verilirse (test/izolasyon) log-tazeliği yedeği de YALNIZ o root'tan
    okunur; yoksa global ayar kökü kullanılır. Eskiden yedek her zaman
    ``get_settings()`` (gerçek makine kökü) okuyup verilen ``root``'u yok sayıyordu
    → ölü-pid testi gerçek makinedeki taze ``logs/train-full-err.log``'u görüp
    yanlış ``True`` dönüyordu (Linux CI'de log olmadığından gizliydi).
    """
    info = read_detached_training_status(root)
    pid = info.get("pid")
    if isinstance(pid, int) and _pid_alive(pid):
        return True
    try:
        settings = SimpleNamespace(root=Path(root)) if root is not None else get_settings()
        return bool(_detached_status(settings))
    except Exception:
        return False


def _terminate_tree(pid: int) -> tuple[bool, str]:
    """Süreci ve (varsa) çocuklarını nazikçe sonlandır; gerekirse kill. (psutil)."""
    try:
        import psutil
    except Exception:
        # psutil yoksa tek-süreç kaba terminate (yine de çalışır).
        try:
            if _pid_alive(pid):
                os.kill(pid, signal.SIGTERM)
                return True, f"SIGTERM pid {pid} (psutil yok)"
            return False, f"pid {pid} zaten ölü — stop_requested"
        except Exception as exc:
            return False, f"stop_requested (terminate hatası: {exc})"
    try:
        if not psutil.pid_exists(pid):
            return False, f"pid {pid} zaten ölü — stop_requested"
        from app.training.resource_lock import process_tree

        procs = process_tree(pid)  # yalnız gerçek alt süreçler (bayat ppid'li yetim değil)
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate()
        _gone, alive = psutil.wait_procs(procs, timeout=8)
        for p in alive:
            with contextlib.suppress(Exception):
                p.kill()
        return True, f"terminated pid {pid} (+{len(procs) - 1} çocuk)"
    except Exception as exc:
        return False, f"stop_requested (terminate hatası: {exc})"


def _stop_event(detail: str, terminated: bool) -> None:
    try:
        from app.agents.runtime.tracker import log_system_event

        log_system_event(
            f"Detached eğitim durdurma: {detail}",
            agent_id="lora-trainer",
            level="warning",
            action="train_stop",
            payload={"terminated": terminated},
        )
    except Exception:
        log.debug("stop event yazılamadı", exc_info=True)


def mark_detached_status(adapter_name: str, root: Path | None = None, **fields: object) -> bool:
    """Durum dosyasına sonuç alanları işle (``finished_at`` / ``failed_at`` + ``error``).

    Yalnız dosya AYNI adapter'a aitse yazar (başka koşunun kaydı ezilmez). Alt süreç
    (`train --run`) bitişte çağırır: başarılı ya da deterministik hatayla biten koşu artık
    "çökmüş" sayılıp diriltilmez (Kademe-2 A1/A2/B2/B3). True → yazıldı.
    """
    r = root or get_settings().root
    st = r / "storage" / "train_status.json"
    info = read_detached_training_status(r)
    if not info or info.get("adapter") != adapter_name:
        return False
    info.update(fields)
    try:
        st.write_text(json.dumps(info), encoding="utf-8")
    except OSError:
        log.warning("Durum dosyasına sonuç işlenemedi: %s", st)
        return False
    return True


def _stale_stop_target(info: dict) -> str:
    """Durum dosyasındaki pid hâlâ BU eğitimin süreci mi? Değilse gerekçe ("" = öldürülebilir).

    Kademe 2 J-2: durum dosyası koşu bittikten sonra diskte kalır; Windows pid'i başka bir
    sürece (dönüşüm PowerShell'i, ollama.exe, yeni eğitim) verebilir. Bitmiş koşuda ya da
    başlangıç zamanı tutmayan süreçte hiçbir şey öldürülmez.
    """
    if info.get("finished_at") or info.get("failed_at"):
        return "koşu zaten bitmiş"
    pid = info.get("pid")
    want = info.get("pid_create_time")
    if isinstance(pid, int) and pid > 0 and isinstance(want, int | float):
        have = resource_lock.process_create_time(pid)
        if have is not None and abs(have - float(want)) > 2.0:
            return f"pid {pid} artık başka bir süreç (başlangıç zamanı tutmuyor)"
    return ""


def request_stop_detached_training(root: Path | None = None) -> dict:
    """Detached eğitimi durdurmayı iste (Windows/Linux/macOS uyumlu).

    1) ``storage/STOP_TRAINING`` bırak (loop-script'ler + güvenlik sinyali).
    2) ``train_status.json``'dan pid oku; süreç (ve çocukları) canlıysa terminate
       (gerekirse kill) et. pid yok/ölüyse HATA VERME → 'stop_requested' döndür.
    Olay genel akışa yazılır.
    """
    r = root or get_settings().root
    storage = r / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "STOP_TRAINING").write_text(_utcnow_iso(), encoding="utf-8")

    info = read_detached_training_status(r)
    pid = info.get("pid")
    terminated = False
    stale = _stale_stop_target(info)
    if stale:
        detail = f"stop_requested ({stale} — süreç öldürülmedi)"
    elif isinstance(pid, int) and pid > 0:
        terminated, detail = _terminate_tree(pid)
    else:
        detail = "stop_requested (pid kaydı yok — STOP_TRAINING bırakıldı)"

    info["stop_requested_at"] = _utcnow_iso()
    info["stop_detail"] = detail
    with contextlib.suppress(Exception):
        (storage / "train_status.json").write_text(json.dumps(info), encoding="utf-8")

    _stop_event(detail, terminated)
    return {"ok": True, "stopped": terminated, "detail": detail, "pid": pid}
