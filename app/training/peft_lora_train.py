"""PEFT-based LoRA trainer for Windows and Linux (CPU or CUDA).

Apple Silicon icin mlx_lora_train.py kullanin.
Bu modul Windows AMD64 ve Linux (CPU veya CUDA GPU) icin calisir.

Baslatma: yalnizca --run parametresiyle gercek egitim yapilir (dry-run varsayilan).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# NOT: HF 'datasets' kütüphanesi kullanılmaz — proje kökündeki yerel 'datasets/'
# klasörü onu gölgeler (ImportError). Tokenizasyon düz Python ile yapılır.
REQUIRED_PACKAGES = ["torch", "transformers", "peft"]

# target_modules AÇIKÇA sınırlı: Qwen3 tied-embeddings kullanır; lm_head/embed
# deltaları GGUF dönüşümünde sessizce atlanır ve adapter bozulur. Yalnız
# attention + MLP projeksiyonları hedeflenir (Ollama/llama.cpp uyumlu).
TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# PEFT init_lora_weights için geçerli string stratejileri (bool dışında).
# Kaynak: peft 0.19 LoraConfig — gaussian/pissa/olora/eva/loftq/corda/orthogonal.
_INIT_STRATEGIES: frozenset[str] = frozenset(
    {"gaussian", "pissa", "olora", "eva", "loftq", "corda", "orthogonal"}
)
# Base ağırlıkları init'te DEĞİŞTİREN stratejiler — GGUF/merge için adapter, ORİJİNAL
# base'e karşı residual'a çevrilmeli (PeftModel.save_pretrained
# path_initial_model_for_weight_conversion); aksi halde Ollama'da delta çift sayılır.
_GGUF_UNSAFE_INIT: frozenset[str] = frozenset({"pissa", "olora", "corda"})


def normalize_init_lora_weights(value: object) -> bool | str:
    """init_lora_weights değerini PEFT'in beklediği bool|str'e çevir + doğrula.

    "true"/"" → True, "false" → False; bilinen strateji adı (veya "pissa_niter_N")
    aynen döner. Bilinmeyen değer ValueError (sessiz yanlış-init yerine erken hata).
    """
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"true", ""}:
        return True
    if v == "false":
        return False
    if v in _INIT_STRATEGIES or v.startswith("pissa_niter_"):
        return v
    valid = "true, false, " + ", ".join(sorted(_INIT_STRATEGIES)) + ", pissa_niter_N"
    raise ValueError(f"Bilinmeyen init_lora_weights: {value!r}. Geçerli: {valid}")


def init_is_gguf_unsafe(value: object) -> bool:
    """init stratejisi base ağırlıkları değiştirip GGUF dönüşümünde dikkat ister mi?"""
    norm = normalize_init_lora_weights(value)
    if isinstance(norm, bool):
        return False
    base = norm.split("_niter_")[0]
    return base in _GGUF_UNSAFE_INIT


@dataclass
class PeftTrainConfig:
    base_model: str
    train_jsonl: Path
    valid_jsonl: Path
    adapter_output_path: Path
    iterations: int = 300
    batch_size: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Dinamik padding ile maliyet gerçek uzunluğa bağlı; 1024 sessiz kırpılmayı önler.
    # RAFT (golden+distractor bağlam) verisine geçişte 2048'e çıkarılmalı.
    max_seq_length: int = 1024
    # --- İleri LoRA teknikleri (araştırma entegrasyonu, doküman v1.2) ---
    # rsLoRA: ölçek alpha/r yerine alpha/sqrt(r) → yüksek r'de stabil, daha iyi öğrenme.
    # Varsayılan KAPALI (efektif büyüklüğü değiştirir; açılırsa lr yeniden ayarlanmalı).
    use_rslora: bool = False
    # DoRA: ağırlığı yön + büyüklüğe ayırır; düşük r'de LoRA'yı sık geçer ama ~2× yavaş.
    use_dora: bool = False
    # init_lora_weights: "true"|"gaussian"|"pissa"|"olora"|"eva"|"loftq"|"corda"...
    init_lora_weights: str = "true"
    # LoRA+: B matrisine lr * ratio (A'dan hızlı) → daha iyi/hızlı yakınsama. 0 = kapalı.
    loraplus_lr_ratio: float = 0.0
    # --- Regularizasyon (catastrophic forgetting + degenerasyon azaltma) ---
    # Yerel trainer bulut reçetesiyle hizalandı (warmup + cosine + weight_decay + clip).
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    # NEFTune: embedding'e eğitimde gürültü → overfit/ezber azalır, yönerge takibi korunur.
    # >0 açar (tipik 5). Yalnız eğitim-zamanı; mimariyi değiştirmez → GGUF-güvenli.
    neftune_noise_alpha: float = 0.0
    # assistant_only_loss: kaybı YALNIZ asistan yanıt token'larından hesaplar; sistem/
    # kullanıcı turları maskelenir. v5 disiplin-regresyonunun (degenerate-tekrar) doğrudan
    # adayı: prompt kalıbı (hep aynı uzun yönerge) full-text loss'ta ezberleniyordu. OPT-IN.
    # YEREL: train() bunu artık GERÇEKTEN uygular (build_masked_labels + _MaskedDataCollator;
    # prompt token'ları -100). Yalnız chat_template varken etkin; yoksa maskesiz yola düşer.
    # BULUT: templates/stage2_lora_colab.ipynb (Hücre 10) maskelemeyi ZATEN unsloth
    # `train_on_responses_only` ile uyguluyor → notebook'a TRL bayrağı EKLENMEZ (çift-maskeleme).
    # Bkz docs/egitim/LORA_ARASTIRMA_LOG.md 2026-06-22.
    assistant_only_loss: bool = False
    # --- KL-regularized fine-tuning (araştırma turu 3, 2026-07-03) ---
    # Kaynak: Riemer ve ark. (IBM Research), "The Effectiveness of Approximate Regularized
    # Replay for Efficient SFT of LLMs", arXiv:2512.22337 — Qwen2.5-Instruct (1.5B/3B/7B/14B)
    # üzerinde LoRA ile ölçüldü: standart LoRA SFT bile ciddi catastrophic forgetting
    # yaratıyor (β=0 satırı); β>0 KL cezası bunu büyük ölçüde ortadan kaldırıyor
    # (β=0.01: forgetting ~kaybolur, plasticity hafif düşer; β=0.001: plasticity KORUNUR
    # + forgetting ortalama >7× azalır — replay ile birlikte). v5 regresyonuyla DOĞRUDAN
    # ilgili: adapter'ın base'den KL-sapması cezalandırılınca disiplin/refusal davranışı
    # ezberden daha az etkileniyor. LoRA ile hesaplama/bellek sinerjisi VAR: base-model
    # forward-pass'i `model.disable_adapter()` ile AYNI ağırlıklar üzerinden alınır — ikinci
    # kopya modele gerek yok (makalenin ana bulgusu: LoRA'da ek bellek maliyeti sıfır).
    # PEFT/TRL'de NATIVE bayrak YOK (yalnız RLHF/DPO trainer'larında KL var, SFT'de değil) →
    # Trainer.compute_loss override'ı (_KLRegTrainer) ile uygulanır. GGUF-güvenli: yalnız
    # eğitim-zamanı ek loss terimi; mimari/ağırlık şekli değişmez. OPT-IN (0.0=kapalı).
    kl_reg_beta: float = 0.0
    seed: int = 42
    # --- Checkpoint'ten devam (resume) — VARSAYILAN KAPALI, AÇIK TERCİH ---
    # Eskiden train() çıktı klasöründeki en son checkpoint'ten KOŞULSUZ devam ederdi:
    # (a) aynı adapter adıyla ikinci koşu eski adapter ağırlıklarını SESSİZCE sürdürüyor,
    # (b) daha kötüsü eski global_step >= yeni max_steps ise SIFIR adım eğitip ok=True
    # dönüyordu — "eğitim yapıldı" denip hiçbir şey öğrenilmemiş oluyordu. Bu tam olarak
    # v5 sessiz-başarısızlık sınıfıdır ve CLAUDE.md Kural 2'yi ihlal eder. Artık devam
    # AÇIKÇA istenmeli: bu alan True ya da ortam değişkeni HEKTOR_TRAIN_RESUME=1.
    resume_from_checkpoint: bool = False
    # max_examples: yalnız N örnekle eğit (0 = hepsi). CPU'da makul süre için ZORUNLU
    # kaldıraç — 4B/1.5B'de tam set (~2000) tek epoch'ta bile saatlerce sürer; bu yüzden
    # yerel eğitim temsilî bir alt-kümeyle yapılır. Determinist örnekleme (seed).
    max_examples: int = 0


def _check_deps() -> list[str]:
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def build_lora_kwargs(cfg: PeftTrainConfig) -> dict:
    """PEFT ``LoraConfig`` için kwargs sözlüğü kur (torch/peft import'suz → offline test).

    İleri teknikleri (rsLoRA / DoRA / init stratejisi) config'ten taşır. ``train()``
    bunu doğrudan ``LoraConfig(**build_lora_kwargs(cfg))`` ile kullanır.
    """
    return {
        "task_type": "CAUSAL_LM",  # TaskType str-enum; "CAUSAL_LM" == TaskType.CAUSAL_LM
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "bias": "none",
        "target_modules": list(TARGET_MODULES),
        "use_rslora": cfg.use_rslora,
        "use_dora": cfg.use_dora,
        "init_lora_weights": normalize_init_lora_weights(cfg.init_lora_weights),
    }


def build_training_kwargs(
    cfg: PeftTrainConfig, *, num_epochs: int, output_dir: str, on_cuda: bool, max_steps: int = 0
) -> dict:
    """transformers ``TrainingArguments`` için kwargs sözlüğü kur (saf → offline test).

    Regularizasyon (warmup + cosine + weight_decay + grad-clip) ve NEFTune burada
    bağlanır; bunlar v5 catastrophic-forgetting/degenerasyon dersinin doğrudan yanıtı.

    ``max_steps>0`` ise adım sayısını TAM kapar (HF Trainer bunu ``num_train_epochs``'un
    önüne alır). ``cfg.iterations`` gerçekte ADIM sayısıdır; epoch'a çevirmek küçük
    değerlerde (iterations<steps_per_epoch) tüm-epoch'a kaçırıyordu (kök bug).
    """
    kwargs: dict = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "max_grad_norm": cfg.max_grad_norm,
        "fp16": on_cuda,
        "logging_steps": 5,
        # CPU eğitimleri saatler sürer; web/Windows çökmesinde sıfırdan başlamamak için
        # sık ve dönen checkpoint tut.
        "save_strategy": "steps",
        "save_steps": 25,
        "save_total_limit": 3,
        "eval_strategy": "no",
        "report_to": "none",
        "dataloader_pin_memory": False,
        "seed": cfg.seed,
    }
    if cfg.neftune_noise_alpha and cfg.neftune_noise_alpha > 0:
        kwargs["neftune_noise_alpha"] = cfg.neftune_noise_alpha
    if max_steps and max_steps > 0:
        kwargs["max_steps"] = max_steps
    return kwargs


def recipe_summary(cfg: PeftTrainConfig) -> dict:
    """Eğitim reçetesinin (aktif teknikler) insan-okur özeti — dry-run/log için."""
    techniques: list[str] = []
    if cfg.use_rslora:
        techniques.append("rsLoRA (alpha/sqrt(r) ölçek)")
    if cfg.use_dora:
        techniques.append("DoRA (ağırlık ayrıştırma)")
    init = normalize_init_lora_weights(cfg.init_lora_weights)
    if init is not True:
        techniques.append(f"init={init}")
    if cfg.loraplus_lr_ratio and cfg.loraplus_lr_ratio > 0:
        techniques.append(f"LoRA+ (lr_ratio={cfg.loraplus_lr_ratio})")
    if cfg.neftune_noise_alpha and cfg.neftune_noise_alpha > 0:
        techniques.append(f"NEFTune (alpha={cfg.neftune_noise_alpha})")
    if cfg.assistant_only_loss:
        techniques.append("assistant_only_loss (yalnız asistan token kaybı — yerelde aktif)")
    if cfg.kl_reg_beta and cfg.kl_reg_beta > 0:
        techniques.append(f"kl_reg (β={cfg.kl_reg_beta}, base'e KL cezası — forgetting azaltma)")
    return {
        "r": cfg.lora_r,
        "alpha": cfg.lora_alpha,
        "dropout": cfg.lora_dropout,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler": cfg.lr_scheduler_type,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
        "advanced_techniques": techniques or ["(yok — vanilya LoRA)"],
        "gguf_unsafe_init": init_is_gguf_unsafe(cfg.init_lora_weights),
    }


def load_lora_profile(name: str, profiles_path: Path | None = None) -> dict:
    """``configs/lora/lora_profiles.yaml``'tan bir profili PeftTrainConfig alanlarına çevir.

    Yalnız tanınan alanları döndürür (ekstra YAML anahtarları yok sayılır). Profil
    ``epochs`` içeriyorsa ``iterations``'a haritalanmaz (epoch↔iterasyon ayrı kavram;
    çağıran ``epochs``'u ayrı kullanır). Bilinmeyen profil → KeyError.
    """
    import yaml

    path = profiles_path or (
        Path(__file__).resolve().parent.parent.parent / "configs" / "lora" / "lora_profiles.yaml"
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if name not in data:
        raise KeyError(f"Profil bulunamadı: {name!r}. Mevcut: {sorted(data)}")
    prof = data[name] or {}
    field_map = {
        "r": "lora_r",
        "alpha": "lora_alpha",
        "dropout": "lora_dropout",
        "max_seq_length": "max_seq_length",
        "learning_rate": "learning_rate",
        "use_rslora": "use_rslora",
        "use_dora": "use_dora",
        "init_lora_weights": "init_lora_weights",
        "loraplus_lr_ratio": "loraplus_lr_ratio",
        "weight_decay": "weight_decay",
        "warmup_ratio": "warmup_ratio",
        "lr_scheduler_type": "lr_scheduler_type",
        "max_grad_norm": "max_grad_norm",
        "neftune_noise_alpha": "neftune_noise_alpha",
        "assistant_only_loss": "assistant_only_loss",
        "kl_reg_beta": "kl_reg_beta",
    }
    out: dict = {}
    for yaml_key, cfg_field in field_map.items():
        if yaml_key in prof and prof[yaml_key] is not None:
            out[cfg_field] = prof[yaml_key]
    # epochs/target_modules/max_examples/note çağırana ayrı bilgi olarak verilebilir.
    for extra in ("epochs", "max_examples"):
        if extra in prof:
            out[extra] = prof[extra]
    return out


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class ResumeDecision:
    """``train()``'in checkpoint kararı: nereden devam edilecek, ne uyarılacak, hata var mı.

    ``checkpoint is None`` → eğitim SIFIRDAN başlar. ``error`` doluysa eğitim HİÇ
    başlatılmaz (çağıran ``ok=False`` döndürür) — sessizce sıfır adım koşmak yerine.
    """

    checkpoint: str | None = None
    last_step: int = 0
    error: str | None = None
    warnings: tuple[str, ...] = ()


def find_last_checkpoint(output_dir: Path | str) -> tuple[str | None, int]:
    """``output_dir`` içindeki en yüksek adımlı ``checkpoint-<N>`` klasörünü bul.

    transformers ``get_last_checkpoint`` ile aynı kuralı uygular ama o import'u
    GEREKTİRMEZ → çevrimdışı test edilebilir. ``(yol, adım)`` döner; checkpoint yoksa
    ``(None, 0)``. Adım sayısı öncelikle ``trainer_state.json``'daki ``global_step``'ten,
    okunamazsa klasör adından alınır.
    """
    d = Path(output_dir)
    if not d.is_dir():
        return None, 0
    best: tuple[int, Path] | None = None
    for child in sorted(d.iterdir()):
        m = _CHECKPOINT_RE.match(child.name)
        if not m or not child.is_dir():
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, child)
    if best is None:
        return None, 0
    step, path = best
    try:
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        step = int(state["global_step"])
    except Exception:  # trainer_state yok/bozuk → klasör adındaki adım geçerli
        pass
    return str(path), step


def zero_step_error(checkpoint: str, last_step: int, max_steps: int) -> str:
    """Devam edilecek checkpoint hedefi zaten karşılıyorsa gösterilecek HATA metni.

    Bu durumda HF Trainer sıfır adım atar; eskiden bu "başarılı eğitim" olarak
    raporlanıyordu (Kural 2 ihlali — sessiz başarısızlık).
    """
    return (
        f"Devam edilecek checkpoint hedefi zaten karşılıyor: {checkpoint} "
        f"(global_step={last_step} >= hedef adım={max_steps}). Eğitim SIFIR adım atardı; "
        "bu 'başarılı eğitim' DEĞİLDİR (CLAUDE.md Kural 2) — yeni veri modele hiç girmez. "
        f"Ya adım hedefini {last_step}'ten büyük seç, ya yeni bir adapter adı kullan, "
        "ya da devamı kapat (resume_from_checkpoint=False → sıfırdan eğitim)."
    )


def resolve_resume_checkpoint(
    output_dir: Path | str, *, resume: bool, max_steps: int = 0
) -> ResumeDecision:
    """Checkpoint'ten devam kararını ver (saf fonksiyon → çevrimdışı test edilebilir).

    Kurallar:
    * ``resume=False`` (VARSAYILAN): mevcut checkpoint KULLANILMAZ. Klasörde checkpoint
      varsa bu sessizce geçilmez, GÖRÜNÜR uyarı üretilir (eski adapter üzerine yazılacak).
    * ``resume=True`` ama checkpoint yok: uyarı + sıfırdan eğitim (hata değil).
    * ``resume=True`` ve checkpoint'in adımı hedefi karşılıyorsa: ``error`` — eğitim
      başlatılmaz, "başarılı" DENMEZ.

    ``max_steps<=0`` hedef henüz bilinmiyor demektir (``cfg.iterations<=0`` iken hedef
    ancak veri tokenize edildikten sonra belli olur); bu halde sıfır-adım kıyası atlanır
    ve çağıran hedef netleştiğinde ``zero_step_error`` guard'ını yeniden uygular.
    """
    checkpoint, last_step = find_last_checkpoint(output_dir)
    if not resume:
        if checkpoint:
            return ResumeDecision(
                checkpoint=None,
                last_step=last_step,
                warnings=(
                    f"Çıktı klasöründe checkpoint VAR ({checkpoint}, global_step={last_step}) "
                    "ama devam KAPALI (varsayılan) → eğitim SIFIRDAN başlıyor ve klasördeki "
                    "adapter ÜZERİNE yazılacak. Devam etmek için resume_from_checkpoint=True "
                    "(veya HEKTOR_TRAIN_RESUME=1). Not: eski checkpoint klasörleri silinmez; "
                    "temiz bir koşu için yeni bir adapter adı verin.",
                ),
            )
        return ResumeDecision()
    if not checkpoint:
        return ResumeDecision(
            warnings=(f"Devam istendi ama {output_dir} içinde checkpoint yok → sıfırdan eğitim.",)
        )
    if max_steps > 0 and last_step >= max_steps:
        return ResumeDecision(
            checkpoint=None,
            last_step=last_step,
            error=zero_step_error(checkpoint, last_step, max_steps),
        )
    return ResumeDecision(
        checkpoint=checkpoint,
        last_step=last_step,
        warnings=(
            f"Eğitim son checkpoint'ten SÜRDÜRÜLÜYOR (açık tercih): {checkpoint} "
            f"(global_step={last_step}, hedef adım={max_steps or 'veriden'}).",
        ),
    )


def resume_requested(cfg: PeftTrainConfig) -> bool:
    """Checkpoint'ten devam AÇIKÇA istendi mi? config alanı VEYA HEKTOR_TRAIN_RESUME=1.

    Ortam değişkeni, henüz ``--resume`` bayrağı geçirmeyen çağıranların (web butonu,
    detached launch) çökme sonrası kurtarma için devamı bilinçli açabilmesi içindir.
    """
    if cfg.resume_from_checkpoint:
        return True
    return os.environ.get("HEKTOR_TRAIN_RESUME", "").strip().lower() in {"1", "true", "yes", "evet"}


def dry_run(cfg: PeftTrainConfig) -> dict:
    missing = _check_deps()
    return {
        "dry_run": True,
        "resume_from_checkpoint": resume_requested(cfg),
        "base_model": cfg.base_model,
        "train_jsonl": str(cfg.train_jsonl),
        "valid_jsonl": str(cfg.valid_jsonl),
        "adapter_output": str(cfg.adapter_output_path),
        "iterations": cfg.iterations,
        "max_examples": cfg.max_examples,
        "recipe": recipe_summary(cfg),
        "missing_packages": missing,
        "install_cmd": f"uv pip install {' '.join(missing)}" if missing else None,
    }


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_messages(row: dict) -> list[dict]:
    """Satırı chat mesaj listesine çevir.

    Desteklenen formatlar (öncelik sırası): messages > prompt/completion >
    system/user/assistant > text. ``prompt``/``completion`` MLX-LM 'completions'
    formatıdır ve ``DatasetBuilder`` bunu üretir — eksikse PEFT eğitimi tüm
    örnekleri sessizce atıp boş veriyle çökerdi.
    """
    if "messages" in row:
        return [m for m in row["messages"] if m.get("content")]
    if row.get("prompt") or row.get("completion"):
        msgs = []
        if row.get("prompt"):
            msgs.append({"role": "user", "content": row["prompt"]})
        if row.get("completion"):
            msgs.append({"role": "assistant", "content": row["completion"]})
        return msgs
    msgs = []
    for role in ("system", "user", "assistant"):
        if row.get(role):
            msgs.append({"role": role, "content": row[role]})
    if msgs:
        return msgs
    if row.get("text"):
        return [{"role": "user", "content": row["text"]}]
    return []


def _row_to_text(row: dict) -> str:
    """Yedek düz-metin formatı (yalnız chat template yoksa kullanılır)."""
    if "text" in row:
        return row["text"]
    return "\n".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in _row_to_messages(row))


def sample_rows(rows: list[dict], max_examples: int, seed: int) -> list[dict]:
    """Determinist alt-küme seç (seed). max_examples<=0 veya yetersizse rows aynen döner.

    Yerel CPU eğitimi tam seti (~2000) işleyemez; temsilî bir alt-küme alınır. random
    yerine sabit seed → tekrar üretilebilir (Kural 6: determinizm).
    """
    if not max_examples or max_examples <= 0 or len(rows) <= max_examples:
        return rows
    import random

    return random.Random(seed).sample(rows, max_examples)


def _chat_input_ids(tokenizer: Any, msgs: list[dict], *, add_generation_prompt: bool) -> list[int]:
    """``apply_chat_template(tokenize=True)`` çıktısını sürümden bağımsız list[int]'e çevir.

    transformers sürümüne göre düz ``list[int]`` veya ``BatchEncoding`` (UserDict; ``dict``
    alt-sınıfı DEĞİL → ``isinstance(.., dict)`` yakalamaz) döner. Her iki halde de düz
    token-id listesi döndürülür. (Bu ayrım atlanınca maskeleme Encoding nesnesi kıyaslar
    ve TÜM örnekler sessizce atılır — smoke'ta görülen "0 örnek" hatası.)
    """
    enc = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if not isinstance(enc, list):  # BatchEncoding/UserDict → input_ids
        enc = enc["input_ids"]
    return [int(t) for t in enc]


def build_masked_labels(prompt_ids: list[int], full_ids: list[int]) -> list[int] | None:
    """assistant_only_loss için label dizisi kur: prompt token'larını -100 ile maskele.

    Yalnız asistan cevabı (``full_ids``'in prompt sonrası kısmı) loss'a girer; sistem/
    kullanıcı turları öğrenilmez. Bu, v5'teki degenerate-tekrar/disiplin-regresyonunun
    doğrudan ilacı: prompt kalıbı (hep aynı uzun yönerge) artık ezberlenmez.

    ``full_ids`` prompt ile başlamıyorsa (chat-template tutarsızlığı) ya da öğrenilebilir
    token kalmıyorsa ``None`` döner → çağıran o örneği ATAR (sessiz yanlış-maskeleme yerine
    görünür veri kaybı). notebook'taki ``assert n_loss > 0`` guard'ının yerel karşılığı.
    """
    n = len(prompt_ids)
    if n == 0 or n >= len(full_ids) or list(full_ids[:n]) != list(prompt_ids):
        return None
    return [-100] * n + list(full_ids[n:])


class _MaskedDataCollator:
    """input_ids/attention_mask/labels'ı sağdan doldurur; labels -100 ile (loss'tan muaf).

    ``DataCollatorForLanguageModeling`` labels'ı input'a eşitler (maskeleme yok); bu
    collator ise ``build_masked_labels``'ın ürettiği maskeli labels'ı korur.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        maxlen = max(len(f["input_ids"]) for f in features)
        out: dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = maxlen - len(f["input_ids"])
            out["input_ids"].append(list(f["input_ids"]) + [self.pad_token_id] * pad)
            out["attention_mask"].append(list(f["attention_mask"]) + [0] * pad)
            out["labels"].append(list(f["labels"]) + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


class _KLRegTrainer:  # transformers.Trainer alt sınıfı; import torch/transformers gerektirir.
    """``compute_loss``'a base-model'e karşı KL cezası ekleyen Trainer mixin'i.

    Kaynak: arXiv:2512.22337 (Riemer ve ark., IBM Research) — "approximate regularized
    replay". Bu sınıf yalnız KL-regularizasyon parçasını uygular (replay/corpus karışımı
    KAPSAM DIŞI — ayrı veri hattı gerektirir, bu turda entegre edilmedi; bkz.
    docs/egitim/LORA_ARASTIRMA_LOG.md Tur 3). LoRA'nın PEFT ``disable_adapter()`` context
    manager'ı sayesinde base-model forward-pass'i İKİNCİ bir model kopyası YÜKLEMEDEN
    aynı ağırlıklar üzerinden alınır (makalenin bellek-sinerji bulgusunun uygulanışı).

    ``kl_reg_beta`` config'te 0 ise bu sınıf hiç devreye girmez (``train()`` düz
    ``Trainer`` kullanır) — opt-in, varsayılan davranış değişmez.
    """

    kl_reg_beta: float = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        import torch
        import torch.nn.functional as F

        outputs = model(**inputs)
        sft_loss = outputs.loss
        labels = inputs.get("labels")
        if self.kl_reg_beta <= 0 or labels is None:
            return (sft_loss, outputs) if return_outputs else sft_loss

        # Loss'a giren (maskelenmemiş) token pozisyonlarını izole et.
        mask = (labels != -100)[:, 1:].contiguous()
        if not mask.any():
            return (sft_loss, outputs) if return_outputs else sft_loss

        adapter_logits = outputs.logits[:, :-1, :][mask]
        with torch.no_grad():
            base_model = getattr(model, "disable_adapter", None)
            if base_model is None:  # adapter sarmalanmamışsa (beklenmez) sessizce atla
                return (sft_loss, outputs) if return_outputs else sft_loss
            with model.disable_adapter():
                base_outputs = model(
                    input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
                )
            base_logits = base_outputs.logits[:, :-1, :][mask].detach()

        # KL(adapter || base): base'e göre "sürpriz" cezalandırılır (Bayesian prior — β
        # yüksekse forgetting azalır ama plastisite de azalır; makale β=0.001-0.01 önerir).
        adapter_logp = F.log_softmax(adapter_logits.float(), dim=-1)
        base_logp = F.log_softmax(base_logits.float(), dim=-1)
        kl = F.kl_div(adapter_logp, base_logp, log_target=True, reduction="batchmean")
        loss = sft_loss + self.kl_reg_beta * kl
        if return_outputs:
            return loss, outputs
        return loss


def _make_trainer_cls(kl_reg_beta: float) -> type:
    """``kl_reg_beta>0`` ise KL-regularized Trainer sınıfı, aksi halde düz Trainer döner.

    torch/transformers import'u burada (fonksiyon-içi) yapılır — modül top-level'ı
    torch'suz ortamda (yalnız ``--extra dev`` testleri) import edilebilir kalır. Sınıf
    ``type()`` ile RUNTIME'da kurulur (statik ``class X(A, B):`` bildirimi DEĞİL) — mypy
    sürümüne/ortamına göre değişen çoklu-miras ``[misc]`` uyarısını (Windows'ta gerekli,
    CI/Linux'ta "unused ignore") her iki ortamda da hiç TETİKLEMEZ.
    """
    from transformers import Trainer

    if kl_reg_beta <= 0:
        return Trainer

    trainer_cls = type("_KLRegTrainerImpl", (_KLRegTrainer, Trainer), {"kl_reg_beta": kl_reg_beta})
    return trainer_cls


def train(cfg: PeftTrainConfig) -> dict:
    missing = _check_deps()
    if missing:
        return {
            "ok": False,
            "error": f"Eksik paketler: {missing}. Kur: uv pip install {' '.join(missing)}",
        }

    # Devam (resume) kararı EN BAŞTA verilir: hatalıysa GB'lık model yüklenmeden dönülür.
    # Varsayılan KAPALI → mevcut checkpoint sessizce kullanılmaz (bkz. PeftTrainConfig).
    resume_plan = resolve_resume_checkpoint(
        cfg.adapter_output_path, resume=resume_requested(cfg), max_steps=cfg.iterations
    )
    for _msg in resume_plan.warnings:
        logger.warning("%s", _msg)
    if resume_plan.error:
        logger.error("%s", resume_plan.error)
        return {"ok": False, "error": resume_plan.error}

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU base dtype: fp32 (varsayılan, kararlı) veya bf16 (HEKTOR_TRAIN_DTYPE=bf16).
    # bf16: model 16→8GB, bellek trafiği yarı → bellek-bağımlı CPU'da potansiyel ~2×.
    # Uyarı: AVX512-BF16'sız CPU'da (örn. Tiger Lake) bf16 emüle edilir; önce ölç.
    _want_bf16 = os.environ.get("HEKTOR_TRAIN_DTYPE", "fp32") == "bf16"
    cpu_dtype = torch.bfloat16 if _want_bf16 else torch.float32
    dtype = torch.float16 if device == "cuda" else cpu_dtype
    logger.info("PEFT LoRA egitimi basladi. Cihaz: %s, dtype: %s", device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model: Any → PEFT/torch yokken (CI: yalnız --extra dev) get_peft_model/
    # print_trainable_parameters satırları "unused type: ignore" üretmesin; torch
    # varken de gereksiz ignore kalmasın. Runtime/eğitim davranışı DEĞİŞMEZ.
    model: Any = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )

    # LoraConfig kwargs'ı saf builder'dan gelir (rsLoRA/DoRA/init dahil). target_modules
    # TARGET_MODULES ile sınırlı (Qwen3 tied-embeddings → GGUF uyumu; bkz. modül başı).
    if init_is_gguf_unsafe(cfg.init_lora_weights):
        logger.warning(
            "init_lora_weights=%s base ağırlıkları değiştirir; GGUF/merge öncesi adapter'ı "
            "ORİJİNAL base'e karşı residual'a çevir (path_initial_model_for_weight_conversion).",
            cfg.init_lora_weights,
        )
    logger.info("LoRA reçetesi: %s", recipe_summary(cfg))
    peft_config = LoraConfig(**build_lora_kwargs(cfg))
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_rows = _load_jsonl(cfg.train_jsonl)
    _n_all = len(train_rows)
    train_rows = sample_rows(train_rows, cfg.max_examples, cfg.seed)
    if len(train_rows) != _n_all:
        logger.info(
            "max_examples=%d → %d/%d örnek (determinist, seed=%d).",
            cfg.max_examples,
            len(train_rows),
            _n_all,
            cfg.seed,
        )

    def _render(r: dict) -> str:
        # Modelin GERÇEK chat şablonu kullanılır (eğitim ↔ çıkarım format eşleşmesi
        # şart; uydurma <|role|> formatı adapter'ı sessizce bozar). Şablon yoksa
        # düz-metin yedeğe düşülür.
        msgs = _row_to_messages(r)
        if msgs and getattr(tokenizer, "chat_template", None):
            try:
                return str(tokenizer.apply_chat_template(msgs, tokenize=False))
            except Exception:
                pass
        return _row_to_text(r).strip()

    def _tokenize(rows: list[dict]) -> list[dict]:
        # Her metni TEK TEK tokenize et (batch yolu transformers'ta overflow
        # edge-case'inde IndexError verebiliyor). Boş örnekleri atla.
        # HIZ: padding YOK — collator dinamik doldurur (batch=1 → sıfır padding,
        # her adım yalnız gerçek token sayısını işler; 512'ye doldurmaktan ~%40 hızlı).
        out: list[dict] = []
        for r in rows:
            text = _render(r)
            if not text:
                continue
            enc = tokenizer(text, truncation=True, max_length=cfg.max_seq_length)
            out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
        return out

    def _tokenize_masked(rows: list[dict]) -> list[dict]:
        # assistant_only_loss: prompt'u chat-template ile AYRI render edip token sınırını
        # bul; labels'ta prompt token'larını -100'le → yalnız asistan cevabı öğrenilir.
        # Maskelenemeyen örnek (final asistan turu yok / template tutarsız) ATILIR.
        out: list[dict] = []
        skipped = 0
        for r in rows:
            msgs = _row_to_messages(r)
            if not msgs or msgs[-1].get("role") != "assistant":
                skipped += 1
                continue
            try:
                prompt_ids = _chat_input_ids(tokenizer, msgs[:-1], add_generation_prompt=True)
                full_ids = _chat_input_ids(tokenizer, msgs, add_generation_prompt=False)[
                    : cfg.max_seq_length
                ]
            except Exception:
                skipped += 1
                continue
            labels = build_masked_labels(prompt_ids, full_ids)
            if labels is None:
                skipped += 1
                continue
            out.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
        if skipped:
            logger.warning("assistant_only_loss: %d örnek maskelenemedi/atlandı.", skipped)
        return out

    # assistant_only_loss yalnız chat_template varken anlamlı (prompt sınırı template'ten
    # bulunur). Şablon yoksa istek loglanır ama maskesiz (eski) yola düşülür.
    use_mask = bool(cfg.assistant_only_loss and getattr(tokenizer, "chat_template", None))
    collator: Any
    if use_mask:
        train_ds = _tokenize_masked(train_rows)
        collator = _MaskedDataCollator(tokenizer.pad_token_id)
        logger.info("assistant_only_loss AKTİF — prompt maskelendi (%d örnek).", len(train_ds))
    else:
        if cfg.assistant_only_loss:
            logger.warning("assistant_only_loss istendi ama chat_template yok → maskesiz eğitim.")
        train_ds = _tokenize(train_rows)
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    if not train_ds:
        return {"ok": False, "error": "Eğitim verisi boş (tokenize sonrası 0 örnek)."}

    steps_per_epoch = max(1, len(train_ds) // cfg.batch_size)
    # cfg.iterations = TOPLAM ADIM hedefi (epoch DEĞİL). Eski kod `iterations // steps_per_epoch`
    # ile epoch'a çeviriyordu → iterations < steps_per_epoch olunca 0→1 epoch (TÜM veri) kaçağı:
    # "iterations=200" gerçekte 1 tam epoch (ör. 1919 adım) koşuyor, eğitim hiç bitmiyordu.
    # Artık max_steps adım sayısını TAM kapar; num_epochs yalnız tavan (max_steps onu keser).
    max_steps = cfg.iterations if cfg.iterations > 0 else steps_per_epoch
    num_epochs = max(1, -(-max_steps // steps_per_epoch))  # ceil(max_steps/steps_per_epoch)

    output_dir = str(cfg.adapter_output_path)
    # HIZ ayarları (CPU): eval kapalı, tek checkpoint, pin_memory kapalı, dinamik padding.
    # Regularizasyon (warmup/cosine/weight_decay/grad-clip/NEFTune) build_training_kwargs'ta.
    args = TrainingArguments(
        **build_training_kwargs(
            cfg,
            num_epochs=num_epochs,
            output_dir=output_dir,
            on_cuda=(device == "cuda"),
            max_steps=max_steps,
        )
    )

    # LoRA+ (opt-in): B matrisine A'dan `loraplus_lr_ratio` kat hızlı lr → daha iyi yakınsama.
    optimizers: tuple = (None, None)
    if cfg.loraplus_lr_ratio and cfg.loraplus_lr_ratio > 0:
        from peft.optimizers import create_loraplus_optimizer

        opt = create_loraplus_optimizer(
            model=model,
            optimizer_cls=torch.optim.AdamW,
            lr=cfg.learning_rate,
            loraplus_lr_ratio=cfg.loraplus_lr_ratio,
            weight_decay=cfg.weight_decay,
        )
        optimizers = (opt, None)
        logger.info("LoRA+ optimizer aktif (lr_ratio=%s)", cfg.loraplus_lr_ratio)

    # kl_reg_beta>0 ise KL-regularized Trainer alt sınıfı (base'e karşı KL cezası);
    # 0 ise düz Trainer (davranış değişmez). Bkz. _KLRegTrainer docstring (arXiv:2512.22337).
    trainer_cls = _make_trainer_cls(cfg.kl_reg_beta)
    if cfg.kl_reg_beta and cfg.kl_reg_beta > 0:
        logger.info(
            "KL-regularizasyon AKTİF (β=%s) — base-model'e karşı KL cezası.", cfg.kl_reg_beta
        )
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        optimizers=optimizers,
    )

    import datetime as _dt

    started_at = _dt.datetime.now().isoformat(timespec="seconds")

    # Sıfır-adım guard'ının SON uygulanışı: cfg.iterations<=0 iken hedef (max_steps) ancak
    # burada belli olur, dolayısıyla baştaki kontrol kıyası atlamıştı. Sıfır adım eğitip
    # "başarılı" demektense burada açıkça hata döneriz (Kural 2).
    if resume_plan.checkpoint and resume_plan.last_step >= max_steps:
        err = zero_step_error(resume_plan.checkpoint, resume_plan.last_step, max_steps)
        logger.error("%s", err)
        return {"ok": False, "error": err}
    trainer.train(resume_from_checkpoint=resume_plan.checkpoint)
    finished_at = _dt.datetime.now().isoformat(timespec="seconds")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Loss eğrisini reports/training/<adapter>_loss.json'a yaz → web "Eğitim grafiği"
    # bu dosyaları okur (/api/learning/training-runs). CLI eğitimi de artık kaydeder.
    _write_loss_curve(cfg, trainer.state.log_history, started_at, finished_at)

    logger.info("Adapter kaydedildi: %s", output_dir)
    return {"ok": True, "adapter_path": output_dir, "device": device}


def _write_loss_curve(
    cfg: PeftTrainConfig, log_history: list[dict], started_at: str, finished_at: str
) -> None:
    """Trainer log_history'den loss eğrisi çıkar; web grafiğinin okuduğu JSON'u yaz."""
    curve: list[dict] = []
    for entry in log_history:
        if "loss" in entry:  # eğitim loss'u (logging_steps'te)
            curve.append(
                {
                    "step": int(entry.get("step", len(curve) + 1)),
                    "train_loss": round(float(entry["loss"]), 4),
                    "val_loss": (
                        round(float(entry["eval_loss"]), 4) if "eval_loss" in entry else None
                    ),
                }
            )
    report = {
        "adapter_name": cfg.adapter_output_path.name,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_iters": cfg.iterations,
        "base_model": cfg.base_model,
        "curve": curve,
    }
    out_dir = Path("reports/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.adapter_output_path.name}_loss.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def generate_colab_notebook(
    cfg: PeftTrainConfig,
    out_path: Path,
    *,
    hf_dataset_repo: str = "KULLANICI/hektor-lora-sft",
    num_epochs: int = 2,
) -> Path:
    """Stage 2 bulut-GPU (Kaggle/Colab) eğitim notebook'u + Ollama Modelfile üretir.

    Doğrulanmış unsloth şablonunu (Qwen3-4B-Instruct-2507 → GGUF Q4_K_M → Ollama)
    `app/training/cloud_notebook.py` üzerinden doldurur. Eski düz-transformers
    notebook'u (5 bilinen hata: target_modules eksik, {messages} okunmuyor, uydurma
    chat formatı, padding='max_length', GGUF export yok) tamamen değiştirildi.
    Detay: docs/PROTOKOL_BULUT_EGITIM.md.
    """
    from app.training.cloud_notebook import build_stage2_notebook, write_modelfile

    build_stage2_notebook(
        base_model=cfg.base_model,
        adapter_name=cfg.adapter_output_path.name,
        hf_dataset_repo=hf_dataset_repo,
        max_seq_length=cfg.max_seq_length,
        lora_r=cfg.lora_r,
        learning_rate=cfg.learning_rate,
        num_epochs=num_epochs,
        out_path=out_path,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_rslora=cfg.use_rslora,
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
    )
    write_modelfile(out_path.parent)
    logger.info("Stage 2 bulut notebook + Modelfile olusturuldu: %s", out_path)
    return out_path


def build_command(cfg: PeftTrainConfig) -> list[str]:
    """TrainingManager.start() için subprocess komutu üretir.

    ``--resume`` YALNIZ config'te açıkça istendiğinde eklenir; varsayılan komut eskisiyle
    birebir aynıdır (çağıran sözleşmesi korunur).
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train",
        str(cfg.train_jsonl),
        "--valid",
        str(cfg.valid_jsonl),
        "--output",
        str(cfg.adapter_output_path),
        "--model",
        cfg.base_model,
        "--iters",
        str(cfg.iterations),
        "--run",
    ]
    if cfg.resume_from_checkpoint:
        cmd.append("--resume")
    return cmd


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PEFT LoRA trainer (Windows/Linux)")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B-Instruct-2507"
    )  # tek beyin (Ollama qwen3:4b)
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", default="models/adapters/hektor_lora_peft")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Cikti klasorundeki son checkpoint'ten devam et (varsayilan: KAPALI - sifirdan).",
    )
    parsed = parser.parse_args()

    cfg = PeftTrainConfig(
        base_model=parsed.model,
        train_jsonl=Path(parsed.train),
        valid_jsonl=Path(parsed.valid),
        adapter_output_path=Path(parsed.output),
        iterations=parsed.iters,
        resume_from_checkpoint=parsed.resume,
    )

    if parsed.colab:
        nb = generate_colab_notebook(cfg, Path(parsed.output) / "hektor_colab.ipynb")
        print(f"Colab notebook: {nb}")
        sys.exit(0)

    if not parsed.run:
        result = dry_run(cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["missing_packages"]:
            print(f"\nKur: {result['install_cmd']}")
        sys.exit(0)

    result = train(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("ok") else 1)
