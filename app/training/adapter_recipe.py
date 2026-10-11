"""Adapter'ın EĞİTİM reçetesini (profil / max_seq_length / istem biçimi) adapter klasöründen bul.

Neden: web LoRA sohbeti (`app/web/lora_chat_service.py`) tek bir istem biçimi ve bağlam bütçesi
varsayıyordu — v11–v13'ün reçetesi (1024 token, sentetik QA "BAĞLAM: … SORU: …"). v14 ise
`moe30b_attn_long` (max_seq_length 6144) ile canlı RAG istemiyle (`build_rag_prompt`,
`rag_answer.md` + "SOURCES / KAYNAKLAR … QUESTION / SORU:") eğitildi → v14'e eski biçim ve
2400 karakterlik bağlam göndermek modeli eğitim dağılımının dışında sorgular.

Kaynak önceliği (ilk bulunan kazanır; hepsi SALT-OKUMA):
  1. ``run_plan.json`` — trainer'ın koşu işareti. Bu modülden sonraki koşular ``profile``,
     ``max_seq_length`` ve ``prompt_formats`` (FİİLİ eğitim satırlarındaki biçim sayımı) yazar;
     daha eski koşularınkinde yalnız adım alanları vardır → sonraki kaynağa düşülür.
  2. ``train_status.final.json`` — bitmiş koşunun arşivlenmiş durum kaydı (``profile``).
  3. Kayıt defteri (``registry/adapters/registry.jsonl``) — ``notes`` içindeki ``profile=…``.
  Profil adı bulunursa ``max_seq_length`` ve istem biçimi ipucu
  ``configs/lora/lora_profiles.yaml``'dan okunur (``chat_prompt_format``; yoksa sentetik QA).
  Hiçbiri yoksa reçete BİLİNMEZ: çağıran eski davranışa düşer ve görünür uyarı verir.

Eğitim BAŞLATMAZ; model yüklemez (çevrimdışı test edilebilir).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

PromptFormat = Literal["synthetic_qa", "live_rag"]

#: Sentetik QA / disiplin verisinin biçimi (`synthetic_qa_builder`, `discipline_dataset`).
SYNTHETIC_QA: PromptFormat = "synthetic_qa"
#: Canlı RAG istemi (`app.brain.rag_answerer.build_rag_prompt`; v14 öz-damıtma verisi).
LIVE_RAG: PromptFormat = "live_rag"

RUN_PLAN_FILE = "run_plan.json"  # peft_lora_train.RUN_PLAN_FILE ile aynı
FINAL_STATUS_FILE = "train_status.final.json"  # detached_launch.FINAL_STATUS_NAME ile aynı

_LIVE_RAG_PREFIX = "SOURCES / KAYNAKLAR:"
_SYNTHETIC_PREFIX = "BAĞLAM:"
_PROFILE_NOTE_RE = re.compile(r"(?<![\w])profile=([A-Za-z0-9_.-]+)")
_DEFAULT_PROFILES = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "lora" / "lora_profiles.yaml"
)


@dataclass(frozen=True)
class AdapterRecipe:
    """Sohbetin istem biçimi + bağlam bütçesi seçimi için gereken eğitim reçetesi."""

    prompt_format: PromptFormat
    max_seq_length: int | None
    profile: str | None
    source: str  # reçetenin okunduğu yer ("run_plan.json", …) ya da "bilinmiyor"
    warning: str | None = None  # dolu → reçete belirlenemedi, eski davranışa düşüldü

    @property
    def known(self) -> bool:
        return self.warning is None


def count_prompt_formats(rows: Iterable[dict]) -> dict[str, int]:
    """Eğitim satırlarının (``{"messages": [...]}``) ilk kullanıcı mesajı biçimini say.

    ``live_rag`` = canlı RAG istemi (geçmiş bloğu önde olabilir), ``synthetic_qa`` = "BAĞLAM:"
    biçimi, ``plain`` = bağlamsız soru. Bozuk satırlar sayılmaz.
    """
    counts = {LIVE_RAG: 0, SYNTHETIC_QA: 0, "plain": 0}
    for row in rows:
        msgs = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(msgs, list):
            continue
        user = next(
            (
                str(m.get("content") or "")
                for m in msgs
                if isinstance(m, dict) and m.get("role") == "user"
            ),
            None,
        )
        if user is None:
            continue
        if _LIVE_RAG_PREFIX in user and "QUESTION / SORU:" in user:
            counts[LIVE_RAG] += 1
        elif user.startswith(_SYNTHETIC_PREFIX):
            counts[SYNTHETIC_QA] += 1
        else:
            counts["plain"] += 1
    return counts


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _profile_entry(name: str, profiles_path: Path | None) -> dict | None:
    """Profilin YAML girdisi (yok/okunamaz → None). `load_lora_profile` ekstra anahtarları
    (``chat_prompt_format``) attığı için YAML doğrudan okunur."""
    import yaml

    try:
        data = yaml.safe_load((profiles_path or _DEFAULT_PROFILES).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    entry = data.get(name) if isinstance(data, dict) else None
    return entry if isinstance(entry, dict) else None


def _as_pos_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
        return int(value)
    return None


def _format_from_counts(counts: object) -> PromptFormat | None:
    if not isinstance(counts, dict):
        return None
    live = _as_pos_int(counts.get(LIVE_RAG)) or 0
    synth = _as_pos_int(counts.get(SYNTHETIC_QA)) or 0
    if live:
        return LIVE_RAG  # canlı istemi görmüş adapter'a canlı istem (öğrenilen atıf davranışı)
    if synth:
        return SYNTHETIC_QA
    return None


def _registry_profile(adapter_name: str, registry_path: Path | None) -> str | None:
    """Kayıt defterindeki SON eşleşen kaydın notundan ``profile=…`` (``-`` = profil yok)."""
    if registry_path is None or not registry_path.is_file():
        return None
    found: str | None = None
    try:
        lines = registry_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("adapter_name") != adapter_name:
            continue
        m = _PROFILE_NOTE_RE.search(str(row.get("notes") or ""))
        if m and m.group(1) != "-":
            found = m.group(1)
    return found


def _from_profile(
    profile: str, source: str, profiles_path: Path | None, seq_hint: int | None = None
) -> AdapterRecipe | None:
    entry = _profile_entry(profile, profiles_path)
    if entry is None:
        return None
    seq = seq_hint or _as_pos_int(entry.get("max_seq_length"))
    fmt = LIVE_RAG if entry.get("chat_prompt_format") == LIVE_RAG else SYNTHETIC_QA
    return AdapterRecipe(prompt_format=fmt, max_seq_length=seq, profile=profile, source=source)


def unknown_recipe(adapter_name: str, reason: str = "") -> AdapterRecipe:
    """Reçete belirlenemedi → eski (v11–v13) davranışı + görünür uyarı."""
    why = f" ({reason})" if reason else ""
    return AdapterRecipe(
        prompt_format=SYNTHETIC_QA,
        max_seq_length=None,
        profile=None,
        source="bilinmiyor",
        warning=(
            f"'{adapter_name}' adapter'ının eğitim reçetesi (profil / max_seq_length / istem "
            f"biçimi) belirlenemedi{why}; eski varsayım kullanıldı: sentetik QA biçimi "
            "('BAĞLAM: … SORU: …') ve 1024 token'a göre bağlam bütçesi. Adapter başka bir "
            "biçimle eğitildiyse cevap eğitim dağılımı dışında üretilmiş olabilir."
        ),
    )


def detect_adapter_recipe(
    adapter_dir: Path,
    *,
    registry_path: Path | None = None,
    profiles_path: Path | None = None,
) -> AdapterRecipe:
    """Adapter klasörü (+ kayıt defteri) → eğitim reçetesi. Hiçbir şey yazmaz."""
    name = adapter_dir.name

    # 1) run_plan.json — yeni koşular reçeteyi doğrudan yazar.
    plan = _read_json(adapter_dir / RUN_PLAN_FILE)
    plan_seq = _as_pos_int(plan.get("max_seq_length"))
    plan_profile = str(plan.get("profile") or "") or None
    plan_fmt = _format_from_counts(plan.get("prompt_formats"))
    if plan_seq and plan_fmt:
        return AdapterRecipe(
            prompt_format=plan_fmt,
            max_seq_length=plan_seq,
            profile=plan_profile,
            source=RUN_PLAN_FILE,
        )
    if plan_profile:
        rec = _from_profile(plan_profile, RUN_PLAN_FILE, profiles_path, plan_seq)
        if rec is not None:
            return rec

    # 2) Bitmiş koşunun arşivlenmiş durum kaydı — yalnız BU adapter'a aitse.
    final = _read_json(adapter_dir / FINAL_STATUS_FILE)
    if final.get("adapter") == name and final.get("profile"):
        rec = _from_profile(str(final["profile"]), FINAL_STATUS_FILE, profiles_path, plan_seq)
        if rec is not None:
            return rec

    # 3) Kayıt defteri notu.
    reg_profile = _registry_profile(name, registry_path)
    if reg_profile:
        rec = _from_profile(reg_profile, "registry", profiles_path, plan_seq)
        if rec is not None:
            return rec

    return unknown_recipe(name, "run_plan.json / train_status.final.json / kayıt defterinde yok")
