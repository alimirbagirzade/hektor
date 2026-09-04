"""MLX-LM LoRA training launcher (Apple Silicon).

Per spec: does NOT auto-start heavy training. It validates inputs, builds the
exact ``mlx_lm.lora`` command, and only runs it when ``--run`` is passed.
After a real run it registers the adapter in the AdapterRegistry.

This module is import-safe on non-Apple platforms; it only requires mlx-lm at
actual run time.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.training.adapter_registry import AdapterRegistry


@dataclass
class TrainConfig:
    base_model: str
    train_jsonl: Path
    valid_jsonl: Path
    adapter_output_path: Path
    iterations: int = 300
    # 8GB Apple Silicon için güvenli varsayılanlar:
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_layers: int = 8
    # Her iterasyonda değil, eğitim sonunda checkpoint al (OOM önleme)
    save_every: int = 0  # 0 → iterations sonuna ertele
    lora_phase: int = 0  # 0 = faz yok, 1-4 = faz numarası
    from_phase: int = 0  # 0 = sıfırdan başla, 1-4 = bu fazın adapter'ından devam


def build_command(cfg: TrainConfig) -> list[str]:
    data_dir = cfg.train_jsonl.parent
    save_every = cfg.save_every if cfg.save_every > 0 else cfg.iterations
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        cfg.base_model,
        "--train",
        "--data",
        str(data_dir),
        "--iters",
        str(cfg.iterations),
        "--batch-size",
        str(cfg.batch_size),
        "--learning-rate",
        str(cfg.learning_rate),
        "--num-layers",
        str(cfg.num_layers),
        "--adapter-path",
        str(cfg.adapter_output_path),
        "--save-every",
        str(save_every),
        "--steps-per-eval",
        str(save_every),
    ]
    # Önceki fazın adapter'ından devam et
    if cfg.from_phase > 0:
        resume_path = (
            Path("models")
            / "adapters"
            / f"achilles_lora_v3_phase{cfg.from_phase}"
            / "adapters.safetensors"
        )
        cmd += ["--resume-adapter-file", str(resume_path)]
    return cmd


def validate(cfg: TrainConfig) -> list[str]:
    problems = []
    if not cfg.train_jsonl.exists():
        problems.append(f"train_jsonl yok: {cfg.train_jsonl}")
    if not cfg.valid_jsonl.exists():
        problems.append(f"valid_jsonl yok: {cfg.valid_jsonl}")
    if shutil.which("python") is None and sys.executable is None:
        problems.append("python yorumlayıcısı bulunamadı")
    return problems


def run(cfg: TrainConfig, *, content_hash: str | None = None, notes: str | None = None) -> None:
    cmd = build_command(cfg)
    print("Çalıştırılıyor:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Adapter versiyonunu faz'a göre isimlendir
    version = (
        f"achilles_lora_v3_phase{cfg.lora_phase}"
        if cfg.lora_phase > 0
        else cfg.adapter_output_path.name
    )
    AdapterRegistry().register(
        version=version,
        base_model=cfg.base_model,
        adapter_path=cfg.adapter_output_path,
        training_data_hash=content_hash,
        notes=notes,
    )
    print(f"Adapter kaydedildi: {version}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MLX-LM LoRA training launcher")
    p.add_argument("--base-model", required=True)
    p.add_argument("--train-jsonl", required=True, type=Path)
    p.add_argument("--valid-jsonl", required=True, type=Path)
    p.add_argument("--adapter-output-path", required=True, type=Path)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--run", action="store_true", help="gerçekten eğitimi başlat")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        base_model=args.base_model,
        train_jsonl=args.train_jsonl,
        valid_jsonl=args.valid_jsonl,
        adapter_output_path=args.adapter_output_path,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_layers=args.num_layers,
    )

    problems = validate(cfg)
    if problems:
        print("DOĞRULAMA HATALARI:")
        for pr in problems:
            print(f"  - {pr}")
        return 1

    if not args.run:
        print("[dry-run] Eğitim KOMUTU (çalıştırmak için --run ekleyin):")
        print("  " + " ".join(build_command(cfg)))
        return 0

    run(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
